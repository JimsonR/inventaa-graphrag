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

    Priority order (highest wins):
      1. Explicit search trigger words → SEARCH (prevent false positives downstream)
      2. POLICY operational keywords
      3. KNOWLEDGE educational/comparison keywords
      4. ADVICE installation/suitability keywords
      5. DETAIL product-specific spec keywords
      6. Default → SEARCH
    """
    q = query.lower()

    # ── 1. EXPLICIT SEARCH SIGNALS (highest priority) ───────────────────────
    # If query is clearly asking to show/find/list/recommend products, always SEARCH
    _explicit_search = [
        "show me", "show ", "find me", "find ",
        "list ", "get me", "display", "browse",
        "recommend", "suggest", "which lights", "what lights",
        "lights for", "light for", "lamps for",
        "under rs", "under inr", "under rupees", "within rs",
        "cheapest", "lowest rated", "highest rated", "best rated",
        "lowest price", "best price", "affordable",
        "ip65 rated", "ip66 rated",          # "X rated products" = search filter
        "products under", "lights under",
    ]
    if any(kw in q for kw in _explicit_search):
        logger.info(f"[Router] intent=search (explicit trigger)  query={query!r}")
        return INTENT_SEARCH

    # ── 2. POLICY: operational / business questions ──────────────────────────
    _policy = [
        "return policy", "refund", "ship", "deliver", "dispatch",
        "warranty claim", "claim warranty", "how do i claim",
        "replacement policy", "exchange policy",
        "track order", "order status",
        "cancel order", "cancellation",
        "bulk order", "bulk price", "bulk discount",
        "dealer", "distributor", "contractor", "wholesale",
        "damaged", "wrong item", "wrong product", "broken on arrival",
        "how much does delivery", "cash on delivery", "cod",
        "emi", "payment mode", "invoice",
    ]
    if any(kw in q for kw in _policy):
        logger.info(f"[Router] intent=policy  query={query!r}")
        return INTENT_POLICY

    # ── 3. KNOWLEDGE: educational / comparison / concept ────────────────────
    # Only specific educational phrases — NOT bare spec terms like ip65 or lumens
    _knowledge = [
        " vs ", " versus ", "compare to", "comparison between",
        "difference between", "which is better", "what is better",
        "wave-free", "wave free",
        "how to choose", "how do i choose", "guide to",
        "benefits of", "advantages of", "disadvantages of",
        "what is ip rating", "what is ip", "what is cri",
        "colour rendering index", "color rendering index",
        "what is a lumen", "lumen guide", "how many lumens",
        "led vs ", "vs led", "fluorescent vs", "halogen vs", "incandescent vs",
        "solar vs wired", "wired vs solar",
        "types of led", "types of outdoor", "types of lighting",
        "lighting guide", "lighting 101",
        "why use led", "how does led work", "how led works",
    ]
    if any(kw in q for kw in _knowledge):
        logger.info(f"[Router] intent=knowledge  query={query!r}")
        return INTENT_KNOWLEDGE

    # ── 4. ADVICE: installation / suitability / FAQ ──────────────────────────
    _advice = [
        "install", "installation", "how to mount", "mounting procedure",
        "can i install", "easy to install", "diy", "self install",
        "smart switch", "timer", "dimmer", "dimmable",
        "coastal", "near sea", "salt air",
        "lifespan", "how long do led", "how long will it last", "last for years",
        "electricity bill", "save electricity", "energy saving",
        "maintenance", "maintain", "clean the light",
        "suitable for", "can it be used", "commercial use",
        "does it come with", "included in the box", "package include",
        "mounting height", "recommended height",
    ]
    if any(kw in q for kw in _advice):
        logger.info(f"[Router] intent=advice  query={query!r}")
        return INTENT_ADVICE

    # ── 5. DETAIL: named product + specific spec question ────────────────────
    _detail_specs = [
        "wattage of", "wattage for", "what wattage",
        "dimension", "size of", "weight of",
        "material", "aluminium", "aluminum", "polycarbonate", "stainless",
        "beam angle", "lumen output",
        "colour temperature", "color temperature", "warm white", "cool white",
        "neutral white", "3-in-1", "3 in 1",
        "ip rating of", "ip of the", "what ip",
        "warranty of", "warranty for", "has warranty", "does it have warranty",
        "available in", "come in", "specification of", "spec of",
        "made of", "body material", "fixture material",
    ]
    if any(kw in q for kw in _detail_specs):
        logger.info(f"[Router] intent=detail  query={query!r}")
        return INTENT_DETAIL

    # ── 6. Default: product search / recommendation ──────────────────────────
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
