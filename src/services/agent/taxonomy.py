import logging
import os
from pinecone import Pinecone
from src.services.agent.config import TenantConfig, AgentConfig

logger = logging.getLogger(__name__)

# ── Module-level Pinecone singleton ──
# Warm-connected during sync_taxonomy() at server boot so that
# fetch_taxonomy_candidates() never pays the cold-start handshake.
_pinecone_client = None
_pinecone_index = None


def _ensure_pinecone():
    """Initialise the Pinecone client + index exactly once per process."""
    global _pinecone_client, _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not api_key or not index_name:
        raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be configured.")

    _pinecone_client = Pinecone(api_key=api_key)
    _pinecone_index = _pinecone_client.Index(index_name)
    logger.info(f"[Pinecone] Connected to index '{index_name}'")
    return _pinecone_index


def sync_taxonomy():
    """
    Embeds categories/collections, features, and use_cases and upserts them to Pinecone.
    Uses the 'taxonomy-cache' namespace.
    Also warm-connects the Pinecone index for future query calls.
    """
    if not TenantConfig.categories and not TenantConfig.features and not TenantConfig.use_cases:
        # Still warm-connect even if no taxonomy to sync
        try:
            _ensure_pinecone()
        except Exception as e:
            logger.warning(f"Could not warm-connect Pinecone: {e}")
        return

    try:
        index = _ensure_pinecone()
        namespace = "taxonomy-cache"

        # Check if namespace already has data to avoid re-embedding on every boot
        stats = index.describe_index_stats()
        if namespace in stats.namespaces and stats.namespaces[namespace].vector_count > 0:
            logger.info(f"Taxonomy already synced ({stats.namespaces[namespace].vector_count} vectors). Skipping.")
            return

        logger.info("Syncing taxonomy to Pinecone 'taxonomy-cache' namespace...")

        vectors = []

        def process_items(items, tag_type):
            nonlocal vectors
            if not items:
                return

            embeddings = TenantConfig.embeddings.embed_documents(items)
            for text, emb in zip(items, embeddings):
                safe_id = f"{tag_type}_{text}".replace(" ", "_").replace("/", "_").lower()
                vectors.append({
                    "id": safe_id,
                    "values": emb,
                    "metadata": {"type": tag_type, "name": text}
                })

        process_items(TenantConfig.categories or TenantConfig.collections, "category")
        process_items(TenantConfig.features, "feature")
        process_items(TenantConfig.use_cases, "use_case")

        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
            logger.info(f"Successfully upserted {len(vectors)} taxonomy tags.")
    except Exception as e:
        logger.error(f"Failed to sync taxonomy: {e}", exc_info=True)


def fetch_taxonomy_candidates(query_embedding: list, threshold: float = 0.80) -> dict:
    """
    Queries the taxonomy-cache and returns matched tags grouped by type.
    Example return: {'feature': ['waterproof'], 'use_case': ['gate-pillar', 'garden-pathway']}
    """
    try:
        index = _ensure_pinecone()

        res = index.query(
            namespace="taxonomy-cache",
            vector=query_embedding,
            top_k=7,
            include_metadata=True
        )

        matched_tags = {}
        for match in res.matches:
            if match.score >= threshold:
                tag_type = match.metadata.get("type")
                tag_name = match.metadata.get("name")
                if tag_type and tag_name:
                    if tag_type not in matched_tags:
                        matched_tags[tag_type] = []
                    if tag_name not in matched_tags[tag_type] and len(matched_tags[tag_type]) < 10:
                        matched_tags[tag_type].append(tag_name)

        if matched_tags:
            logger.info(f"[Taxonomy] Fetched candidate tags: {matched_tags} (threshold={threshold})")
        return matched_tags
    except Exception as e:
        logger.error(f"[Taxonomy] Error fetching taxonomy candidates: {e}")
        return {}


def fetch_taxonomy_candidates_fast(query: str) -> dict:
    """Instant in-memory taxonomy match without network calls to Pinecone or Azure OpenAI."""
    if not query or len(query.strip()) < 2:
        return {}
    q_lower = query.lower().strip()
    q_tokens = set([t for t in q_lower.split() if len(t) > 2])
    matched_tags = {}

    from src.services.agent.config import AgentConfig
    # Check collections/categories
    for cat in (AgentConfig.categories or []):
        c_lower = str(cat).lower().strip()
        if c_lower in q_lower or q_lower in c_lower:
            matched_tags.setdefault("category", []).append(str(cat))
            if len(matched_tags["category"]) >= 6: break
    # Check category groups
    for g_name in (AgentConfig.category_groups.keys() if AgentConfig.category_groups else []):
        g_lower = str(g_name).lower().strip()
        if g_lower in q_lower or q_lower == f"{g_lower} lights" or q_lower == f"{g_lower} collections":
            matched_tags.setdefault("category", []).append(str(g_name))
            if len(matched_tags["category"]) >= 6: break
    # Check features
    for feat in (AgentConfig.features or []):
        f_lower = str(feat).lower().strip()
        if len(f_lower) > 2 and (f_lower in q_lower or any(t == f_lower for t in q_tokens)):
            matched_tags.setdefault("feature", []).append(str(feat))
            if len(matched_tags.get("feature", [])) >= 5: break
    # Check use cases
    for uc in (AgentConfig.use_cases or []):
        uc_lower = str(uc).lower().strip()
        if len(uc_lower) > 2 and (uc_lower in q_lower or any(t in uc_lower for t in q_tokens)):
            matched_tags.setdefault("use_case", []).append(str(uc))
            if len(matched_tags.get("use_case", [])) >= 5: break

    if matched_tags:
        logger.info(f"[Taxonomy-Fast] Instant matched candidate tags: {matched_tags}")
    return matched_tags

