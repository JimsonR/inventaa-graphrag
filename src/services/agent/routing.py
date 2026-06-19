"""
routing.py — Deterministic intent → tool selection mapping.

Design principles:
1. The routing is PURELY rule-based. No LLM calls, no probabilistic scoring.
2. Each keyword belongs to EXACTLY ONE intent (no overlap). This prevents regressions.
3. The caller can pass an explicit `intent` to bypass classification entirely.
4. Intents map to a fixed set of tools via a lookup table.

Intent hierarchy (checked in order to avoid ambiguity):
  SEARCH    → Show / find / list / filter products
  DETAIL    → One named product's specs, wattage, warranty, material
  POLICY    → Shipping, returns, warranty claims, bulk pricing
  ADVICE    → Installation, suitability, lifespan FAQs
  KNOWLEDGE → Educational / comparison / concept articles
"""

import logging
from typing import Tuple, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# ─── Intent constants ─────────────────────────────────────────────────────────
INTENT_SEARCH    = "search"
INTENT_DETAIL    = "detail"
INTENT_POLICY    = "policy"
INTENT_ADVICE    = "advice"
INTENT_KNOWLEDGE = "knowledge"

# ─── Mapping: external upstream intent strings → internal intents ─────────────
EXTERNAL_INTENT_MAP = {}

# ─── Per-intent compact system prompts ────────────────────────────────────────
_BASE_RULE = (
    "You are an AI sales assistant for Inventaa, an Indian LED lighting brand.\n"
    "RULES:\n"
    "1. ALWAYS use tools to query the database before answering. \n"
    "   - Always use `SearchProductsDatabase` to search for products. If the tool returns `needs_clarification` with a list of `available_collections`, you MUST present those collections to the user and ask them to narrow down their choice.\n"
    "2. If the tool returns no data, say: \"I'm sorry, I don't have that information in our database.\"\n"
    "3. NEVER hallucinate product names, prices, specs, or policies.\n"
    "4. CRITICAL: NEVER manually list or type out product options as text. If you need to recommend or show products, you MUST call the `SearchProductsDatabase` tool so the UI can render them with images. Do not summarize products from conversation history into text.\n"
    "5. When answering questions about policies, offers, or discounts, you MUST explicitly list out the exact percentages, tiers, and details provided by the tool. Do not just summarize that they exist.\n"
    "6. If the tool returns store-wide or generic discounts (e.g. 'Extra 5% OFF on Rs 7500'), state those exact tiers. NEVER invent a specific percentage discount (like '50% off') for a product category if the tool does not explicitly state it.\n\n"
)


def get_intent_prompts():
    from src.services.agent.config import AgentConfig
    collections_str = " | ".join(AgentConfig.collections) if AgentConfig.collections else "3 in 1 gate light | Divine Light For Home Entrance | Indoor Commercial Lights | Indoor Domestic Lights | LED Outdoor Wall Light | Outdoor Commercial Lights | Outdoor Garden Bollard Light | Outdoor LED Gate Lamp Lights | Solar Lights"
    
    return {
        INTENT_SEARCH: (
            _BASE_RULE +
            "Use SearchProductsDatabase to find products.\\n"
            "Pass the user's natural language as the 'query' param. You could also incorporate any long-term user preferences or context provided to you if relevant. For example, if memory says the user prefers garden lights, you could set the `category` parameter to 'Bollard & Garden Lights'.\\n"
            "Use max_price for budget limits. Use sort_by for cheapest/best-rated.\\n"
            f"Available categories (collections): {collections_str}"
        ),
        INTENT_DETAIL: (
            _BASE_RULE +
            "CRITICAL: ALWAYS use the ProductDetailsDatabase tool to look up the specific named product before answering. "
            "Do NOT rely on conversational memory for specs or warranty. "
            "Pass the product name as 'product_name'. "
            "The tool returns wattage options, colour options, specs, and warranty directly from the Neo4j database."
        ),
        INTENT_POLICY: (
            _BASE_RULE +
            "If the user is asking about the warranty for a specific product from the conversation, use ProductDetailsDatabase.\\n"
            "Use GeneralKnowledgeDatabase to search for general coupon codes, active offers, discounts, and shipping/replacement policies. IMPORTANT: Provide ONLY the core policy keywords as the query (e.g., 'offers', 'discount', 'shipping'), not the product names.\\n"
            "Otherwise, use PolicyVectorDatabase to answer questions about general company policies.\\n"
            "Topics for PolicyVectorDatabase: shipping, delivery time, return/replacement, warranty claims, "
            "bulk pricing, dealer rates, damaged/wrong items."
        ),
        INTENT_ADVICE: (
            _BASE_RULE +
            "Use ProductAdviceDatabase to answer general product FAQs NOT tied to a specific product.\\n"
            "Topics: installation, mounting, smart switch/timer compatibility, "
            "coastal suitability, LED lifespan, electricity savings, maintenance."
        ),
        INTENT_KNOWLEDGE: (
            _BASE_RULE +
            "Use GeneralKnowledgeDatabase to answer educational or comparison questions about lighting.\\n"
            "Topics: LED vs fluorescent, wave-free vs traditional, what is IP rating, "
            "how to choose outdoor lighting, benefits of solar, CRI, lumens guide."
        ),
    }

# ─── Per-intent allowed tool names ────────────────────────────────────────────
INTENT_TOOLS = {
    INTENT_SEARCH:    ["SearchProductsDatabase"],
    INTENT_DETAIL:    ["ProductDetailsDatabase", "SearchProductsDatabase"],
    INTENT_POLICY:    ["PolicyVectorDatabase", "ProductDetailsDatabase", "GeneralKnowledgeDatabase"],
    INTENT_ADVICE:    ["ProductAdviceDatabase", "GeneralKnowledgeDatabase"],
    INTENT_KNOWLEDGE: ["GeneralKnowledgeDatabase", "ProductAdviceDatabase"],
}



# ─── Agentic Intent Classifier ────────────────────────────────────────────────

class IntentClassification(BaseModel):
    """Schema for routing a user query to the correct intent."""
    intent: str = Field(
        ...,
        description="The classified intent. Must be one of: 'search', 'detail', 'policy', 'advice', or 'knowledge'."
    )

_ROUTER_SYSTEM_PROMPT = """You are an intent classification router for an LED lighting company.
Classify the user's query into exactly ONE of the following intents:

- "search" : Browsing, finding, recommending, or filtering products by budget/rating/type.
- "detail" : Asking for specific specs (wattage, dimensions, warranty, material) of a product, OR asking a follow-up question (like "warranty", "price") about a product recently mentioned in the conversation context.
- "policy" : Questions about discounts, offers, deals, coupon codes (e.g. "offers on solar lights"), shipping, delivery, returns, general warranty claims procedure, or bulk pricing/dealer rates.
- "advice" : Questions about installation, durability (waterproof, coastal, weather), or maintenance (NO named product).
- "knowledge" : Educational concepts (what is IP rating/CRI/lumens), comparisons (LED vs solar, warm vs cool), or general buying guides.

If the query does not perfectly match one, select the closest fit. If completely unrelated to lighting or company operations, default to "search".
"""

def classify_intent(query: str, llm=None, history_context: str = "") -> str:
    """
    Classifies intent using a fast LLM call.
    """

    # 2. Agentic Classification
    if llm is None:
        logger.warning("[Router] No LLM provided to classifier, defaulting to 'search'.")
        return INTENT_SEARCH

    try:
        router = llm.with_structured_output(IntentClassification)
        
        system_content = _ROUTER_SYSTEM_PROMPT
        if history_context:
            system_content += f"\n\nRecent Conversation Context to help disambiguate:\n{history_context}"
            
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=query)
        ]
        
        logger.info("--- PROMPT INJECTED TO ROUTER LLM ---")
        for idx, m in enumerate(messages):
            role = "SYSTEM" if isinstance(m, SystemMessage) else "USER" if isinstance(m, HumanMessage) else "AGENT"
            logger.info(f"[{idx}] {role}:\n{m.content}\n")
        logger.info("---------------------------------------")

        result = router.invoke(messages)
        intent = result.intent.lower()
        
        # Validate LLM output against known intents
        if intent not in get_intent_prompts():
            logger.warning(f"[Router] LLM returned unknown intent '{intent}', defaulting to 'search'.")
            intent = INTENT_SEARCH
            
        logger.info(f"[Router] intent={intent} (agentic)  query={query!r}")
        return intent

    except Exception as e:
        logger.error(f"[Router] Agentic classification failed: {e}. Defaulting to 'search'.")
        return INTENT_SEARCH


def get_intent_config(
    query: str,
    all_tools: list,
    llm=None,
    explicit_intent: Optional[str] = None,
    history_context: str = ""
) -> Tuple[str, list, str]:
    """
    Returns (system_prompt, filtered_tools, intent) for the given query.

    Args:
        query: The user's message text.
        all_tools: Full list of Tool objects from get_tools().
        llm: The language model used for agentic intent classification.
        explicit_intent: If provided, bypasses keyword classification entirely.
    """
    if explicit_intent and explicit_intent in get_intent_prompts():
        intent = explicit_intent
        logger.info(f"[Router] intent={intent} (explicit override)")
    else:
        intent = classify_intent(query, llm=llm, history_context=history_context)

    prompt = get_intent_prompts()[intent]
    allowed_names = set(INTENT_TOOLS[intent])
    filtered = [t for t in all_tools if t.name in allowed_names]

    if not filtered:
        logger.warning(f"[Router] No tools matched intent={intent}, using all tools.")
        filtered = all_tools

    logger.info(f"[Router] intent={intent} | tools={[t.name for t in filtered]}")
    return prompt, filtered, intent
