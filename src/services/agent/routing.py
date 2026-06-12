"""
routing.py — Intent classification and dynamic prompt/tool selection.

Instead of one massive system prompt listing all 5 tools, we:
1. Classify the query into one of 5 intents using fast keyword heuristics
2. Build a compact, focused system prompt for that intent only
3. Return only the tool(s) relevant to that intent

This keeps the per-request system prompt small (~200 tokens vs ~700 tokens).
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ─── Intent enum ──────────────────────────────────────────────────────────────
INTENT_SEARCH   = "search"    # Browse / filter / recommend products
INTENT_DETAIL   = "detail"    # Single named product specs / warranty / wattage
INTENT_POLICY   = "policy"    # Shipping / returns / warranty claim / bulk pricing
INTENT_ADVICE   = "advice"    # Installation / suitability / lifespan FAQs
INTENT_KNOWLEDGE = "knowledge" # Educational / comparison / how-to / concepts


# ─── Per-intent compact system prompts ────────────────────────────────────────
_BASE_RULE = (
    "You are an AI sales assistant for Inventaa, an Indian LED lighting brand.\n"
    "RULES:\n"
    "1. ALWAYS call the provided tool first. Never answer from your own knowledge.\n"
    "2. If the tool returns no data, say: \"I'm sorry, I don't have that information in our database.\"\n"
    "3. NEVER hallucinate product names, prices, specs, or policies.\n\n"
)

INTENT_PROMPTS = {
    INTENT_SEARCH: (
        _BASE_RULE +
        "Use SearchProductsDatabase to find products.\n"
        "Pass the user's natural language as the 'query' param.\n"
        "Use max_price for budget limits. Use sort_by for cheapest/best-rated.\n"
        "Available categories: Gate & Pillar Lights | Solar Lights | Outdoor Wall Lights | "
        "Bollard & Garden Lights | Street Lights | Flood Lights | Indoor & Ceiling Lights | "
        "Panel Lights | Pathway & Step Lights | Bulkhead Lights | Divine & Temple Lights | General Purpose Lights"
    ),
    INTENT_DETAIL: (
        _BASE_RULE +
        "Use ProductDetailsDatabase to look up ONE specific named product.\n"
        "Pass the product name as 'product_name'. "
        "The tool returns wattage options, colour options, specs, and warranty."
    ),
    INTENT_POLICY: (
        _BASE_RULE +
        "Use PolicyVectorDatabase to answer questions about company policies.\n"
        "Topics: shipping, delivery time, return/replacement, warranty claims, "
        "bulk pricing, dealer rates, damaged/wrong items."
    ),
    INTENT_ADVICE: (
        _BASE_RULE +
        "Use ProductAdviceDatabase to answer general product FAQs NOT tied to a specific product.\n"
        "Topics: installation, mounting, smart switch/timer compatibility, "
        "coastal suitability, LED lifespan, electricity savings, maintenance."
    ),
    INTENT_KNOWLEDGE: (
        _BASE_RULE +
        "Use GeneralKnowledgeDatabase to answer educational or comparison questions about lighting.\n"
        "Topics: LED vs fluorescent, wave-free vs traditional, what is IP rating, "
        "how to choose outdoor lighting, benefits of solar, CRI, lumens guide."
    ),
}

# ─── Per-intent allowed tool names (in priority order) ────────────────────────
INTENT_TOOLS = {
    INTENT_SEARCH:    ["SearchProductsDatabase"],
    INTENT_DETAIL:    ["ProductDetailsDatabase", "SearchProductsDatabase"],
    INTENT_POLICY:    ["PolicyVectorDatabase"],
    INTENT_ADVICE:    ["ProductAdviceDatabase", "GeneralKnowledgeDatabase"],
    INTENT_KNOWLEDGE: ["GeneralKnowledgeDatabase", "ProductAdviceDatabase"],
}


# ─── Keyword-based intent classifier ──────────────────────────────────────────
def classify_intent(query: str) -> str:
    """
    Fast heuristic intent classification.
    Returns one of the INTENT_* constants.
    """
    q = query.lower()

    # --- POLICY: operational / business questions ---
    _policy = [
        "return", "refund", "ship", "deliver", "dispatch",
        "warranty claim", "claim warranty", "how do i claim",
        "replace", "replacement", "exchange", "track order",
        "cancel", "cancellation", "bulk order", "bulk price",
        "dealer", "distributor", "contractor", "wholesale",
        "damaged", "wrong item", "wrong product", "broken on arrival",
        "how long does", "how much does delivery", "cash on delivery", "cod",
        "emi", "payment", "invoice",
    ]
    if any(kw in q for kw in _policy):
        logger.info(f"[Router] intent=policy  query={query!r}")
        return INTENT_POLICY

    # --- KNOWLEDGE: educational / comparison / concept ---
    _knowledge = [
        " vs ", " versus ", "compare", "comparison", "difference between",
        "which is better", "what is better", "wave-free", "wave free",
        "what is a ", "what are ", "how to choose", "how do i choose",
        "benefits of", "advantages of", "disadvantages of",
        "cri", "colour rendering", "color rendering",
        "lumens", "lumen guide", "how many lumens",
        "ip rating", "what is ip", "ip65", "ip66",
        "led vs", "fluorescent", "halogen", "incandescent",
        "solar vs", "wired vs", "types of lighting", "lighting guide",
        "why led", "how does led", "what makes",
    ]
    if any(kw in q for kw in _knowledge):
        logger.info(f"[Router] intent=knowledge  query={query!r}")
        return INTENT_KNOWLEDGE

    # --- ADVICE: installation / suitability / general FAQ ---
    _advice = [
        "install", "installation", "mounting", "mount it",
        "can i install", "easy to install", "diy", "self install",
        "smart switch", "timer", "dimmer", "dimmable",
        "coastal", "near sea", "salt air", "humidity",
        "lifespan", "how long do", "how long will", "last for",
        "electricity bill", "save electricity", "energy saving",
        "maintenance", "maintain", "clean",
        "suitable for", "can it be used", "commercial use",
        "does it come with", "included in the box", "package include",
        "recommended height", "mounting height",
    ]
    if any(kw in q for kw in _advice):
        logger.info(f"[Router] intent=advice  query={query!r}")
        return INTENT_ADVICE

    # --- DETAIL: user mentions a product name + asks a specific spec ---
    _detail_specs = [
        "wattage", "watt", "dimension", "size", "weight",
        "material", "aluminium", "aluminum", "polycarbonate", "stainless",
        "beam angle", "lumen output",
        "colour temperature", "color temperature", "warm white", "cool white",
        "neutral white", "3-in-1", "3 in 1",
        "ip rating of", "ip of",
        "warranty of", "warranty for", "does it have warranty",
        "available in", "come in", "specification of", "spec of",
        "made of", "body material",
    ]
    if any(kw in q for kw in _detail_specs):
        logger.info(f"[Router] intent=detail  query={query!r}")
        return INTENT_DETAIL

    # --- Default: product search / recommendation ---
    logger.info(f"[Router] intent=search (default)  query={query!r}")
    return INTENT_SEARCH


def get_intent_config(query: str, all_tools: list) -> Tuple[str, list]:
    """
    Returns (system_prompt, filtered_tools) for the given query.

    all_tools: the full list of StructuredTool/Tool objects from get_tools()
    """
    intent = classify_intent(query)
    prompt = INTENT_PROMPTS[intent]
    allowed_names = set(INTENT_TOOLS[intent])
    filtered = [t for t in all_tools if t.name in allowed_names]

    # Safety: if filtering left zero tools, fall back to all tools
    if not filtered:
        logger.warning(f"[Router] No tools matched intent={intent}, using all tools.")
        filtered = all_tools

    logger.info(f"[Router] intent={intent} | tools={[t.name for t in filtered]}")
    return prompt, filtered
