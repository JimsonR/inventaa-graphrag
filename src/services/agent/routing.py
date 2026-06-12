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

logger = logging.getLogger(__name__)

# ─── Intent constants ─────────────────────────────────────────────────────────
INTENT_SEARCH    = "search"
INTENT_DETAIL    = "detail"
INTENT_POLICY    = "policy"
INTENT_ADVICE    = "advice"
INTENT_KNOWLEDGE = "knowledge"

# ─── Mapping: external upstream intent strings → internal intents ─────────────
# The upstream router sends coarse intents like "FAQ_KNOWLEDGE".
# All of them flow to the agent, so we classify within the agent.
# If the upstream ever sends finer-grained intents, add them here:
EXTERNAL_INTENT_MAP = {
    # Future: "PRODUCT_SEARCH": INTENT_SEARCH,
    # Future: "POLICY_QUESTION": INTENT_POLICY,
}

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

# ─── Per-intent allowed tool names ────────────────────────────────────────────
INTENT_TOOLS = {
    INTENT_SEARCH:    ["SearchProductsDatabase"],
    INTENT_DETAIL:    ["ProductDetailsDatabase", "SearchProductsDatabase"],
    INTENT_POLICY:    ["PolicyVectorDatabase"],
    INTENT_ADVICE:    ["ProductAdviceDatabase", "GeneralKnowledgeDatabase"],
    INTENT_KNOWLEDGE: ["GeneralKnowledgeDatabase", "ProductAdviceDatabase"],
}

# ─── Deterministic keyword lookup tables ──────────────────────────────────────
# IMPORTANT: Each keyword must appear in AT MOST ONE table.
# If a term is ambiguous (e.g., "install"), place it in the LOWER-priority table
# and let higher-priority guards (SEARCH signals) override it first.

# Step 1 — Hard SEARCH overrides: query clearly asks to find/list products
# These take precedence over ALL other intents.
_SEARCH_OVERRIDES = frozenset([
    "show me", "show ", "find me", "find products",
    "list products", "list lights", "get me",
    "display products", "browse",
    "lights for ", "light for ", "lamps for ",
    "lights under ", "products under ",
    "under rs", "under ₹", "under inr", "under rupees", "within rs",
    "cheapest", "lowest rated", "highest rated", "best rated",
    "lowest price", "best price", "affordable",
    "ip65 rated", "ip66 rated", "ip67 rated",
    "recommend lighting", "suggest lighting",
    "what solar lights", "what gate lights", "what outdoor lights",
    "which lights should", "which light should",
])

# Step 2 — POLICY: strictly operational/business topics
_POLICY_EXACT = frozenset([
    "return policy", "returns policy",
    "replacement policy",
    "exchange policy",
    "refund policy",
    "warranty claim", "claim warranty", "how do i claim", "how to claim",
    "track order", "order status", "track my order",
    "cancel order", "cancel my order", "cancellation policy",
    "bulk order", "bulk price", "bulk discount", "bulk purchase",
    "dealer pricing", "distributor pricing", "contractor pricing",
    "wholesale price", "wholesale pricing",
    "damaged product", "damaged item", "damaged on arrival",
    "wrong item", "wrong product", "received wrong",
    "broken on arrival",
    "how much does delivery", "delivery charge", "delivery cost",
    "shipping charge", "shipping cost",
    "cash on delivery", " cod ", "pay on delivery",
    "payment method", "payment mode",
    "invoice request",
    "do you deliver to", "deliver to",
    "express delivery", "fast delivery",
])

# Step 3 — KNOWLEDGE: educational/comparison articles (no product-specific name)
_KNOWLEDGE_EXACT = frozenset([
    " vs ", " versus ",
    "compare to", "comparison between", "difference between",
    "which is better", "what is better", "which one is better",
    "wave-free", "wave free",
    "how to choose lighting", "how do i choose lighting", "guide to lighting",
    "benefits of solar", "advantages of solar",
    "benefits of led", "advantages of led",
    "disadvantages of", "drawbacks of",
    "what is ip rating", "what does ip rating mean",
    "what is cri", "colour rendering index", "color rendering index",
    "what is a lumen", "lumen guide", "how many lumens",
    "led vs ", "vs led",
    "fluorescent vs", "halogen vs", "incandescent vs",
    "solar vs wired", "wired vs solar",
    "types of led lighting", "types of outdoor lighting",
    "lighting guide", "lighting 101",
    "why use led", "how does led work", "how led works",
    "what is kelvin", "colour temperature guide",
    "what is beam angle",
])

# Step 4 — ADVICE: installation / suitability / general FAQ (no product name)
_ADVICE_EXACT = frozenset([
    "is installation easy", "easy to install",
    "can i install it myself", "can i install myself", "diy installation",
    "self install", "install it myself",
    "how to install", "installation guide", "mounting guide",
    "how to mount",
    "can it be connected to a timer", "can it be connected to smart",
    "smart switch compatible", "timer compatible", "works with timer",
    # Suitability — general (safe: search queries use "lights for" → hits SEARCH first)
    "suitable for",
    "suitable for coastal", "near coastal", "near sea", "salt air",
    "suitable for harsh", "suitable for weather", "suitable for heavy",
    "suitable for wet", "suitable for outdoor", "suitable for indoor",
    "suitable for commercial", "can it be used for commercial",
    "can it be used near", "can it be used in",
    # Durability
    "can it withstand", "can this withstand", "withstand rain",
    "withstand weather", "withstand harsh", "withstand heavy",
    "waterproof rating", "is it waterproof", "is it rustproof",
    "is it uv", "uv resistant", "fade resistant", "fade proof",
    "how long do leds last", "lifespan of led", "led lifespan",
    "how long will it last", "how many years",
    "will it reduce electricity", "reduce electricity bill",
    "save on electricity", "energy saving benefit",
    "maintenance required", "how to maintain", "how to clean the light",
    "does it come with mounting", "included in the package",
    "what is the mounting height", "recommended mounting height",
    "can i use it outdoors", "outdoor safe", "weather resistant",
])

# Step 5 — DETAIL: named product + spec question
# These are spec keywords that only make sense with a named product.
_DETAIL_EXACT = frozenset([
    "what wattages are available", "what wattage", "wattage available",
    "available wattages", "wattage for this",
    "what are the dimensions", "dimensions of", "size of the",
    "what material", "material of the", "body material", "fixture material",
    "made of", "what is it made of",
    "beam angle of", "what beam angle",
    "lumens does it", "lumen output of",
    "colour temperature of", "color temperature of",
    "available in warm white", "available in cool white", "available in colour",
    "is it available in", "does it come in",
    "ip rating of", "what ip rating does", "what ip does",
    "warranty of", "warranty for this", "does it have warranty",
    "has warranty", "warranty on the",
    "specification of", "specs of", "spec sheet for",
])


def classify_intent(query: str) -> str:
    """
    Deterministic keyword-based intent classification.

    Each lookup table has non-overlapping keywords.
    Tables are checked in priority order — higher priority tables win.
    Returns one of: INTENT_SEARCH, INTENT_DETAIL, INTENT_POLICY,
                    INTENT_ADVICE, INTENT_KNOWLEDGE.
    """
    q = query.lower()

    # ── Priority 1: Hard SEARCH overrides ────────────────────────────────────
    if any(kw in q for kw in _SEARCH_OVERRIDES):
        logger.info(f"[Router] intent=search (override)  query={query!r}")
        return INTENT_SEARCH

    # ── Priority 2: POLICY ────────────────────────────────────────────────────
    if any(kw in q for kw in _POLICY_EXACT):
        logger.info(f"[Router] intent=policy  query={query!r}")
        return INTENT_POLICY

    # ── Priority 3: KNOWLEDGE ─────────────────────────────────────────────────
    if any(kw in q for kw in _KNOWLEDGE_EXACT):
        logger.info(f"[Router] intent=knowledge  query={query!r}")
        return INTENT_KNOWLEDGE

    # ── Priority 4: ADVICE ────────────────────────────────────────────────────
    if any(kw in q for kw in _ADVICE_EXACT):
        logger.info(f"[Router] intent=advice  query={query!r}")
        return INTENT_ADVICE

    # ── Priority 5: DETAIL ────────────────────────────────────────────────────
    if any(kw in q for kw in _DETAIL_EXACT):
        logger.info(f"[Router] intent=detail  query={query!r}")
        return INTENT_DETAIL

    # ── Default: SEARCH ───────────────────────────────────────────────────────
    logger.info(f"[Router] intent=search (default)  query={query!r}")
    return INTENT_SEARCH


def get_intent_config(
    query: str,
    all_tools: list,
    explicit_intent: Optional[str] = None,
) -> Tuple[str, list]:
    """
    Returns (system_prompt, filtered_tools) for the given query.

    Args:
        query: The user's message text.
        all_tools: Full list of Tool objects from get_tools().
        explicit_intent: If provided, bypasses keyword classification entirely.
                         Use this when the upstream router sends a reliable sub-intent.
    """
    if explicit_intent and explicit_intent in INTENT_PROMPTS:
        intent = explicit_intent
        logger.info(f"[Router] intent={intent} (explicit override)")
    else:
        intent = classify_intent(query)

    prompt = INTENT_PROMPTS[intent]
    allowed_names = set(INTENT_TOOLS[intent])
    filtered = [t for t in all_tools if t.name in allowed_names]

    # Safety: if filtering somehow left zero tools, fall back to all
    if not filtered:
        logger.warning(f"[Router] No tools matched intent={intent}, using all tools.")
        filtered = all_tools

    logger.info(f"[Router] intent={intent} | tools={[t.name for t in filtered]}")
    return prompt, filtered
