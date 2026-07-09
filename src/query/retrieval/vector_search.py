"""
retrieval/vector_search.py — Similarity search across FAQ and policy vector stores.
Decoupled from hardcoded e-commerce URL routing patterns (/products/).
"""

import logging
from typing import List, Dict, Any
from src.services.agent.config import AgentConfig
from src.query.models import QueryIntent

logger = logging.getLogger(__name__)


def vector_search(intent_data: dict, query: str) -> List[Dict[str, Any]]:
    """Synchronous vector search across FAQ, policy, and general knowledge vector indexes."""
    try:
        intent = intent_data.get("intent")
        results: List[Dict[str, Any]] = []

        # 1. Policy check intent -> search policy vector store
        if intent == QueryIntent.CHECK_POLICY:
            if AgentConfig.policy_vector_store:
                docs = AgentConfig.policy_vector_store.similarity_search_with_score(query, k=3)
                for doc, score in docs:
                    results.append({"type": "policy", "text": doc.page_content, "score": score})
            return results

        # 2. Advice or general FAQ intent -> search FAQ vector store
        if intent == QueryIntent.GET_ADVICE:
            if AgentConfig.product_faq_vector_store:
                docs = AgentConfig.product_faq_vector_store.similarity_search_with_score(query, k=3)
                for doc, score in docs:
                    results.append({"type": "faq", "text": doc.page_content, "score": score, "metadata": doc.metadata})
            return results

        # 3. Product info / detail documentation intent -> search FAQ vectors for spec sheets and manual references
        if intent == QueryIntent.GET_PRODUCT_INFO and AgentConfig.product_faq_vector_store:
            logger.info(f"[DEBUG-VECTOR] Executing similarity search over FAQ vector store for query: '{query}'")
            docs = AgentConfig.product_faq_vector_store.similarity_search_with_score(query, k=5)
            for doc, score in docs:
                meta = doc.metadata or {}
                # Prioritize explicit sku or item ID metadata if present
                sku_slug = meta.get("sku") or meta.get("product_sku") or meta.get("product_id") or meta.get("sku_slug")
                
                # Fall back to URL parsing without assuming fixed e-commerce prefixes
                if not sku_slug:
                    url = meta.get("product_url", "") or meta.get("url", "")
                    if url:
                        # Extract the trailing path segment (e.g. /products/foo -> foo, /item/bar -> bar)
                        parts = [p for p in url.split("/") if p and not p.startswith("http") and "." not in p]
                        if parts:
                            sku_slug = parts[-1]

                if sku_slug:
                    results.append({
                        "type": "product_vector",
                        "sku_slug": str(sku_slug).strip(),
                        "text": doc.page_content,
                        "score": score
                    })
            logger.info(f"[DEBUG-VECTOR] FAQ search matched {len(results)} references. Top slugs: {[r.get('sku_slug') for r in results[:5]]}")
        
        return results
    except Exception as e:
        logger.error(f"Vector search error: {e}", exc_info=True)
        return []
