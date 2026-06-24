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
    query_tokens = set([t.lower().strip() for t in cleaned_query.split()])
    
    # Meaningful tokens are those not in the noise list and length > 2
    meaningful_query_tokens = {t for t in query_tokens if t not in _NOISE and len(t) > 2}
    
    _NEGATION = {"not", "except", "other", "besides", "without", "no", "non"}
    if any(t in _NEGATION for t in query_tokens):
        # Negation requires semantic reasoning that fuzzy matching cannot do.
        # Fall back to the LLM immediately so it doesn't force the excluded category.
        return {"action": "search", "category": None}
        
    # Case 1: All noise -> too broad
    if not meaningful_query_tokens:
        return {"action": "clarify", "collections": collections}
    
    best_matches = []
    best_score = 0
    
    # Case 2: Token-level overlap match
    for col in collections:
        col_tokens = set([t.lower().strip() for t in re.sub(r'[^\w\s]', '', col).split()])
        meaningful_col_tokens = {t for t in col_tokens if t not in _NOISE and len(t) > 2}
        
        if not meaningful_col_tokens:
            continue
            
        intersection = meaningful_query_tokens.intersection(meaningful_col_tokens)
        
        if intersection:
            # Score is based on how much of the collection's core keywords were matched
            score = len(intersection) / len(meaningful_col_tokens)
            if score > best_score:
                best_score = score
                best_matches = [col]
            elif score == best_score:
                best_matches.append(col)
                
    # If the user perfectly hit some collection keywords (score >= 0.5 means at least half of the collection's unique keywords matched)
    if best_score >= 0.5:
        if len(best_matches) == 1:
            return {"action": "search", "category": best_matches[0]}
        else:
            # It tied across multiple collections (e.g. "outdoor" matches "Outdoor Wall" and "Outdoor Commercial" equally)
            # We want to clarify among the tied matches!
            return {"action": "clarify", "collections": best_matches}
    
    # Case 3: Umbrella term mapping (Fallback if strict intersection failed)
    matching = [c for c in collections if any(t in c.lower() for t in meaningful_query_tokens)]
    
    if len(matching) > 1:
        return {"action": "clarify", "collections": matching}
    if len(matching) == 1:
        return {"action": "search", "category": matching[0]}
    
    # Case 4: Unknown specific term -> let LLM + tool handle it
    return {"action": "search", "category": None}
