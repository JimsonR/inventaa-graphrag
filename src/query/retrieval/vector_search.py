"""
retrieval/vector_search.py — Neo4j native vector similarity search across FAQ, Blog (:Chunk), and Policy nodes.
"""

import logging
from typing import List, Dict, Any
from src.services.agent.config import TenantConfig

logger = logging.getLogger(__name__)


def vector_search(intent_data: dict, query: str) -> List[Dict[str, Any]]:
    """
    Executes vector similarity search in Neo4j across blog articles (:Chunk), FAQ items (:FAQ), and policy terms (:Policy).
    Prioritizes execution when intent is 'faq_knowledge' or 'check_policy'.
    """
    if not query or not TenantConfig.graph or not TenantConfig.embeddings:
        return []

    intent = str(intent_data.get("intent", "")).lower()
    if intent not in ("faq_knowledge", "check_policy", "find_product", "get_advice", "unknown"):
        return []

    results: List[Dict[str, Any]] = []
    try:
        query_emb = TenantConfig.embeddings.embed_query(query)
        threshold = 0.65 if intent in ("faq_knowledge", "check_policy") else 0.75

        # 1. Search Blog Articles / Knowledge Chunks (:Chunk via inventaa_faq_vector)
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
        try:
            faq_matches = TenantConfig.graph.query("""
            CALL db.index.vector.queryNodes('product_faq_vector', 4, $emb)
            YIELD node, score
            WHERE score >= $threshold
            RETURN node.question AS q, node.answer AS a, score
            """, params={"emb": query_emb, "threshold": threshold})
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

        # 3. Search Policies (:Policy via policy_vector)
        if intent in ("faq_knowledge", "check_policy") or any(w in query.lower() for w in ["policy", "return", "exchange", "warranty", "shipping"]):
            try:
                policy_matches = TenantConfig.graph.query("""
                CALL db.index.vector.queryNodes('policy_vector', 3, $emb)
                YIELD node, score
                WHERE score >= $threshold
                RETURN node.title AS title, node.content AS content, score
                """, params={"emb": query_emb, "threshold": threshold})
                for m in (policy_matches or []):
                    if m.get("title") and m.get("content"):
                        text_block = f"Policy: {m['title']}\n{m['content']}"
                        results.append({
                            "type": "policy",
                            "text": text_block,
                            "score": m["score"],
                            "source": "store_policy"
                        })
            except Exception as e:
                logger.debug(f"[VectorSearch] Policy vector search skipped/error: {e}")

        if results:
            logger.info(f"[VectorSearch] Found {len(results)} vector matches across FAQ/Blog/Policy (intent={intent})")
        return results
    except Exception as e:
        logger.error(f"[VectorSearch] Failed during vector search execution: {e}", exc_info=True)
        return []
