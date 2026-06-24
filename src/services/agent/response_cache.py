"""
response_cache.py — Semantic response cache using Pinecone.

When the Judge validates a response as VALID, the query embedding + response
are stored in Pinecone (in the 'semantic-cache' namespace). Future semantically 
similar queries (cosine ≥ threshold) return the cached response instantly, 
bypassing the entire LangGraph pipeline.
"""

import os
import json
import time
import logging
import hashlib
from typing import Optional

logger = logging.getLogger(__name__)

_index = None

def _get_index():
    """Lazy-initialize the Pinecone index."""
    global _index
    if _index is None:
        try:
            from pinecone import Pinecone
            api_key = os.getenv("PINECONE_API_KEY")
            index_name = os.getenv("PINECONE_INDEX_NAME")
            if not api_key or not index_name:
                logger.warning("[Cache] PINECONE_API_KEY or PINECONE_INDEX_NAME not set. Cache disabled.")
                return None
            pc = Pinecone(api_key=api_key)
            _index = pc.Index(index_name)
            logger.info("[Cache] Pinecone index initialized for semantic cache.")
        except ImportError:
            logger.warning("[Cache] pinecone not installed. Cache disabled.")
            return None
        except Exception as e:
            logger.error(f"[Cache] Failed to initialize Pinecone: {e}", exc_info=True)
            return None
    return _index

def cache_lookup(
    embedding: list,
    tenant_id: str,
    intent: str,
    threshold: float = 0.99,
    skip_intents: list = None,
) -> Optional[dict]:
    """
    Search the cache for a semantically similar query.

    Returns:
        dict with keys {response, response_type, query_text} if cache HIT,
        None if cache MISS.
    """
    if skip_intents and intent in skip_intents:
        logger.info(f"[Cache] Skipping cache lookup for intent={intent}")
        return None

    index = _get_index()
    if index is None:
        return None

    try:
        # Build metadata filter for tenant isolation
        filter_dict = {}
        if tenant_id:
            filter_dict["tenant_id"] = tenant_id

        results = index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True,
            filter=filter_dict,
            namespace="semantic-cache"
        )

        if not results or not results.matches:
            logger.info("[Cache] MISS — no results from Pinecone.")
            return None

        top = results.matches[0]
        if top.score >= threshold:
            metadata = top.metadata or {}

            # Check TTL: if cached_at + ttl < now, treat as expired
            cached_at = metadata.get("cached_at", 0)
            ttl = metadata.get("ttl", 86400)  # default 24h
            if time.time() - cached_at > ttl:
                logger.info(f"[Cache] MISS — entry expired (age={int(time.time() - cached_at)}s, ttl={ttl}s).")
                # Optionally delete the expired entry
                try:
                    index.delete(ids=[top.id], namespace="semantic-cache")
                except Exception:
                    pass
                return None

            # Parse response
            response_raw = metadata.get("response", "")
            response_type = metadata.get("response_type", "text")
            if response_type == "products":
                try:
                    response = json.loads(response_raw)
                except (json.JSONDecodeError, TypeError):
                    response = response_raw
            else:
                response = response_raw

            logger.info(
                f"[Cache] HIT — score={top.score:.4f}, "
                f"original_query={metadata.get('query_text', '?')!r}, "
                f"response_type={response_type}"
            )
            return {
                "response": response,
                "response_type": response_type,
                "query_text": metadata.get("query_text", ""),
            }
        else:
            logger.info(f"[Cache] MISS — top score={top.score:.4f} < threshold={threshold}")
            return None

    except Exception as e:
        logger.error(f"[Cache] Lookup error: {e}", exc_info=True)
        return None

def cache_store(
    embedding: list,
    query_text: str,
    response,
    tenant_id: str,
    intent: str,
    response_type: str = "text",
    ttl: int = 86400,
    skip_intents: list = None,
):
    """
    Store a validated response in the cache.

    Args:
        embedding: The query's embedding vector.
        query_text: The original query text.
        response: The response (str for text, list/dict for products).
        tenant_id: Tenant ID for isolation.
        intent: The classified intent.
        response_type: "text" or "products".
        ttl: Time-to-live in seconds (default 24h).
        skip_intents: List of intents to skip caching for.
    """
    if skip_intents and intent in skip_intents:
        logger.info(f"[Cache] Skipping cache store for intent={intent}")
        return

    index = _get_index()
    if index is None:
        return

    try:
        # Serialize response
        if isinstance(response, (dict, list)):
            response_str = json.dumps(response, ensure_ascii=False)
        else:
            response_str = str(response)

        # Generate a deterministic ID from query + tenant
        cache_id = hashlib.sha256(f"{tenant_id}:{query_text}".encode()).hexdigest()[:32]

        metadata = {
            "query_text": query_text,
            "response": response_str,
            "response_type": response_type,
            "intent": intent,
            "tenant_id": tenant_id or "",
            "cached_at": time.time(),
            "ttl": ttl,
        }

        index.upsert(vectors=[(cache_id, embedding, metadata)], namespace="semantic-cache")
        logger.info(f"[Cache] STORED — id={cache_id}, intent={intent}, ttl={ttl}s, query={query_text!r}")

    except Exception as e:
        logger.error(f"[Cache] Store error: {e}", exc_info=True)
