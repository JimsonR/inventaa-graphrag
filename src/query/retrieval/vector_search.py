"""
retrieval/vector_search.py — Neo4j native vector similarity search across FAQ, Blog (:Chunk), and Policy nodes.
"""

import logging
from typing import List, Dict, Any
from src.services.agent.config import TenantConfig

logger = logging.getLogger(__name__)


def vector_search(intent_data: dict, query: str) -> List[Dict[str, Any]]:
    """
    Executes vector similarity search in Neo4j across blog articles (:Chunk) and FAQ items (:FAQ).
    Prioritizes execution when intent is 'faq_knowledge'.
    """
    if not query or not TenantConfig.graph or not TenantConfig.embeddings:
        return []

    intent = str(intent_data.get("intent", "")).lower()
    if intent not in ("faq", "find_item", "advice", "unknown"):
        return []

    results: List[Dict[str, Any]] = []
    try:
        query_emb = TenantConfig.embeddings.embed_query(query)
        threshold = 0.65 if intent == "faq" else 0.75

        # 1. Search Blog Articles / Knowledge Chunks (:Chunk)
        try:
            faq_idx = TenantConfig.get_faq_index()
            chunk_matches = TenantConfig.graph.query("""
            CALL db.index.vector.queryNodes($faq_idx, 5, $emb)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node.id AS id, node.text AS text, score
            """, params={"faq_idx": faq_idx, "emb": query_emb, "threshold": threshold})
            for m in (chunk_matches or []):
                if m.get("text"):
                    results.append({
                        "type": "chunk",
                        "text": m["text"],
                        "score": m["score"],
                        "source": "blog_knowledge"
                    })
        except Exception as e:
            logger.debug(f"[VectorSearch] Chunk vector search skipped/error: {e}")

        # 2. Search Product FAQs (:FAQ via product_faq_vector)
        # Skipped for faq_knowledge - that intent should return only (:Chunk)
        # blog/knowledge articles, not per-product FAQ entries.
        if intent != "faq":
            try:
                faq_matches = TenantConfig.graph.query("""
                CALL db.index.vector.queryNodes($prod_faq_idx, 4, $emb)
                YIELD node, score
                WHERE score >= $threshold
                RETURN node.question AS q, node.answer AS a, score
                """, params={"prod_faq_idx": TenantConfig.get_product_faq_index(), "emb": query_emb, "threshold": threshold})
                for m in (faq_matches or []):
                    if m.get("q") and m.get("a"):
                        text_block = f"Q: {m['q']}\nA: {m['a']}"
                        results.append({
                            "type": "faq",
                            "text": text_block,
                            "score": m["score"],
                            "source": "product_faq"
                        })
            except Exception as e:
                logger.debug(f"[VectorSearch] FAQ vector search skipped/error: {e}")

        if results:
            logger.info(f"[VectorSearch] Found {len(results)} vector matches across FAQ/Blog (intent={intent})")
        return results
    except Exception as e:
        logger.error(f"[VectorSearch] Failed during vector search execution: {e}", exc_info=True)
        return []
