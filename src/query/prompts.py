"""
prompts.py — Dynamic prompt builders and stop-word heuristics.
Decoupled from brand identity and specific product domains.
"""

from typing import Set
from src.services.agent.config import AgentConfig


def get_stopwords() -> Set[str]:
    """
    Returns search stop words by combining dynamic configuration with general English query words.
    Eliminates hardcoded domain terms (e.g., 'light', 'lamp', 'inventaa').
    """
    base_stopwords = {
        "give", "me", "show", "tell", "about", "what", "are", "the", "is", "for", "please",
        "some", "any", "get", "find", "looking", "want", "need", "buy", "at", "in", "of",
        "on", "to", "with", "from", "by", "an", "a", "under", "below", "above", "range", "cost", "price",
        "can", "you", "do", "have", "we", "our"
    }
    config_stopwords = AgentConfig.get_stop_words()
    return base_stopwords.union(config_stopwords)


def get_intent_system_prompt() -> str:
    """
    Returns the intent classification system instructions dynamically branded for the active tenant.
    """
    brand = AgentConfig.get_brand_name()
    return f"""You are an intent classifier for an e-commerce sales assistant ({brand}).
Classify the user's query into exactly one of these intents:
- browse_category: User wants to SEE ALL products in a category/collection (e.g., "show me all items in collection X", "what options do you have in category Y?", "list all items in department Z"). Use this when the user is browsing or exploring an entire product category without specifying a single product name or narrow spec filter.
- find_product: User wants to find/discover specific products matching constraints (by feature, attribute, option, price range, or use case). Use this when the user has a specific need with filters (e.g., "durable item under 2000").
- get_product_info: User wants details, specs, dimensions, or warranty about a specific product they know by name or SKU.
- check_policy: User wants to know about return, shipping, delivery, replacement, or warranty policies.
- get_advice: User asks general FAQ, setup advice, or how to choose products.
- unknown: None of the above.

Also extract structured filters and entities:
- category_keywords: list of product categories or collections mentioned
- feature_keywords: list of features or specs mentioned (e.g., ["durable", "eco-friendly", "compact"])
- product_name: specific product name or SKU if mentioned (null otherwise)
- filters: structured dictionary of extracted constraints:
  {{
    "category": string or null,
    "application": string or null (e.g. primary use case or setting),
    "segment": string or null,
    "brand": string or null (e.g. "{brand}"),
    "rating": string or null,
    "option_1": string or number or null,
    "color": string or null
  }}
- preferences: dict of any preferences (e.g., {{"min_price": 500, "max_price": 2000, "color": "Blue"}})

Respond ONLY with valid JSON. Example:
{{
  "intent": "browse_category",
  "category_keywords": ["Primary Collection"],
  "feature_keywords": [],
  "product_name": null,
  "filters": {{
    "category": "Primary Collection",
    "application": "Primary Setting",
    "segment": null,
    "brand": null,
    "rating": null,
    "option_1": null,
    "color": null
  }},
  "preferences": {{}}
}}"""


def get_response_system_prompt() -> str:
    """
    Returns the sales assistant system instructions dynamically branded for the active tenant.
    """
    brand = AgentConfig.get_brand_name()
    currency = AgentConfig.get_currency_symbol()
    custom_prompt = AgentConfig.brain.get("prompts", {}).get("response_system")
    
    if custom_prompt:
        return custom_prompt

    return f"""You are an expert AI sales assistant for {brand}.
You help customers discover products, explain specifications, answer FAQ/policy questions, and guide purchases.

When presenting products:
1. List the most relevant products with their display names, exact prices in {currency}, and discounts if available.
2. Highlight key technical specifications retrieved from the database.
3. Always provide the product page URL so the customer can view and purchase online.
4. Be enthusiastic, empathetic, and professional.
5. If the user asks about shipping or returns, clearly explain our configured store policies if relevant.

Format responses in clean, readable text. Use bullet points for product listings.
Always end with a helpful closing or call-to-action to buy online or ask follow-up questions."""
