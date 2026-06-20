"""
Deterministic query analysis BEFORE the LLM.
Tenant-agnostic: works with any tenant's collection set.
"""
import re
from difflib import SequenceMatcher

_NOISE = {
    "show", "me", "the", "all", "your", "available", "list", "what", 
    "do", "you", "have", "products", "product", "lights", "light", 
    "led", "please", "can", "i", "see", "get", "want", "need",
    "looking", "for", "some", "any", "browse", "view"
}

def preprocess_search(query: str, collections: list[str]) -> dict:
    """
    Analyzes the query and determines if it maps to a specific collection,
    or if it is too broad and needs clarification.
    """
    # Remove punctuation
    cleaned_query = re.sub(r'[^\w\s]', '', query)
    tokens = [t.lower().strip() for t in cleaned_query.split()]
    
    # Meaningful tokens are those not in the noise list and length > 2
    meaningful = [t for t in tokens if t not in _NOISE and len(t) > 2]
    cleaned_meaningful = " ".join(meaningful)
    
    # Case 1: All noise -> too broad
    if not meaningful:
        return {"action": "clarify", "collections": collections}
    
    # Case 2: Exact/fuzzy match against a collection name
    best_match, best_score = None, 0
    for col in collections:
        # Score against the full cleaned query (in case noise words are part of collection name)
        score_full = SequenceMatcher(None, cleaned_query.lower(), col.lower()).ratio()
        # Score against only meaningful words
        score_meaningful = SequenceMatcher(None, cleaned_meaningful, col.lower()).ratio()
        
        score = max(score_full, score_meaningful)
        if score > best_score:
            best_score = score
            best_match = col
            
    # Empirically, 0.55 is a decent threshold for fuzzy matching short strings
    if best_score >= 0.55:
        return {"action": "search", "category": best_match}
    
    # Case 3: Umbrella term -> filter matching collections
    # e.g., "indoor" matches "Indoor Commercial" and "Indoor Domestic"
    matching = [c for c in collections if any(t in c.lower() for t in meaningful)]
    
    if len(matching) > 1:
        return {"action": "clarify", "collections": matching}
    if len(matching) == 1:
        return {"action": "search", "category": matching[0]}
    
    # Case 4: Unknown specific term -> let LLM + tool handle it
    return {"action": "search", "category": None}
