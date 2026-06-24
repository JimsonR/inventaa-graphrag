"""
Deterministic query analysis BEFORE the LLM.
Handles ONLY two cases:
  1. Ultra-broad queries (all noise words) → show top-level CategoryGroups
  2. Pure umbrella terms matching a CategoryGroup name → show that group's collections

Everything else (specific queries, typos, non-English) is passed to the LLM + Vector taxonomy.
No token-level string matching against collection names.
"""
import re

_NOISE = {
    "show", "me", "the", "all", "your", "available", "list", "what", 
    "do", "you", "have", "products", "product", "lights", "light", 
    "led", "please", "can", "i", "see", "get", "want", "need",
    "looking", "for", "some", "any", "browse", "view", "give",
    "lighting", "type", "types", "category", "categories"
}

def preprocess_search(query: str, collections: list[str], category_groups: dict = None, top_level_groups: list = None) -> dict:
    """
    Lightweight deterministic check for browse/navigation queries.
    Only intercepts umbrella terms via CategoryGroup hierarchy.
    Returns {"action": "pass"} for everything else → falls through to LLM.
    """
    # Remove punctuation
    cleaned_query = re.sub(r'[^\w\s]', '', query)
    query_tokens = set([t.lower().strip() for t in cleaned_query.split()])
    
    
    # Meaningful tokens are those not in the noise list and length > 2
    meaningful_query_tokens = {t for t in query_tokens if t not in _NOISE and len(t) > 2}
    
    _NEGATION = {"not", "except", "other", "besides", "without", "no", "non"}
    if any(t in _NEGATION for t in query_tokens):
        return {"action": "pass"}
        
    # Case 0: All noise → ultra-broad query (e.g. "show me products", "show all lights")
    if not meaningful_query_tokens:
        if top_level_groups:
            return {"action": "clarify", "collections": top_level_groups}
        return {"action": "clarify", "collections": collections}
    
    # Case 1: Pure umbrella term matching a CategoryGroup name
    # Only triggers when the group name is the ONLY meaningful token
    if category_groups:
        for group_name, group_collections in category_groups.items():
            if group_name.lower() in meaningful_query_tokens:
                remaining_tokens = meaningful_query_tokens - {group_name.lower()}
                if not remaining_tokens:
                    if len(group_collections) == 1:
                        return {"action": "search", "category": group_collections[0]}
                    else:
                        return {"action": "clarify", "collections": group_collections}
    
    # Everything else → pass to LLM + Vector taxonomy
    return {"action": "pass"}
