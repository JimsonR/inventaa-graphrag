import logging
from src.services.agent.config import TenantConfig, AgentConfig

logger = logging.getLogger(__name__)


def _item_plural() -> str:
    """Domain-agnostic plural noun from tenant config."""
    return AgentConfig.get_item_noun_plural()


def _ensure_neo4j_vector_index():
    """Ensure the Neo4j vector index for TaxonomyTag exists."""
    if not TenantConfig.graph:
        return False
    try:
        TenantConfig.graph.query("""
        CREATE VECTOR INDEX taxonomy_vector_index IF NOT EXISTS
        FOR (t:TaxonomyTag) ON (t.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 1536,
          `vector.similarity_function`: 'cosine'
        }}
        """)
        return True
    except Exception as e:
        logger.warning(f"[Neo4j-Taxonomy] Could not verify/create vector index: {e}")
        return False


def sync_taxonomy():
    """
    Embeds categories/collections, features, and use_cases and upserts them to Neo4j as (:TaxonomyTag) nodes.
    Uses Neo4j native vector indexing ('taxonomy_vector_index').
    """
    if not TenantConfig.graph or not TenantConfig.embeddings:
        return

    try:
        _ensure_neo4j_vector_index()

        # Check if we already have TaxonomyTag nodes synced
        res = TenantConfig.graph.query("MATCH (t:TaxonomyTag) WHERE t.embedding IS NOT NULL RETURN count(t) AS count")
        count = res[0]["count"] if res else 0
        if count > 0:
            logger.info(f"Taxonomy already synced to Neo4j ({count} TaxonomyTag nodes). Skipping.")
            return

        logger.info("Syncing taxonomy to Neo4j (:TaxonomyTag) nodes and vector index...")

        def process_items(items, tag_type):
            if not items:
                return
            embeddings = TenantConfig.embeddings.embed_documents(items)
            for text, emb in zip(items, embeddings):
                safe_id = f"{tag_type}_{text}".replace(" ", "_").replace("/", "_").lower()
                TenantConfig.graph.query("""
                MERGE (t:TaxonomyTag {id: $safe_id})
                SET t.name = $name,
                    t.type = $tag_type,
                    t.embedding = $embedding
                """, params={"safe_id": safe_id, "name": text, "tag_type": tag_type, "embedding": emb})

        process_items(TenantConfig.categories or TenantConfig.collections, "category")
        process_items(TenantConfig.features, "feature")
        process_items(TenantConfig.use_cases, "use_case")

        logger.info("Successfully upserted taxonomy tags to Neo4j.")
    except Exception as e:
        logger.error(f"Failed to sync taxonomy to Neo4j: {e}", exc_info=True)


def fetch_taxonomy_candidates(query_embedding: list, threshold: float = 0.80) -> dict:
    """
    Queries Neo4j vector index ('taxonomy_vector_index') and returns matched tags grouped by type.
    Example return: {'feature': ['waterproof'], 'use_case': ['gate-pillar', 'garden-pathway']}
    """
    if not TenantConfig.graph or not query_embedding:
        return {}

    try:
        _ensure_neo4j_vector_index()

        res = TenantConfig.graph.query("""
        CALL db.index.vector.queryNodes('taxonomy_vector_index', 7, $embedding)
        YIELD node, score
        WHERE score >= $threshold
        RETURN node.type AS type, node.name AS name, score
        """, params={"embedding": query_embedding, "threshold": threshold})

        matched_tags = {}
        for match in (res or []):
            tag_type = match.get("type")
            tag_name = match.get("name")
            if tag_type and tag_name:
                matched_tags.setdefault(tag_type, [])
                if tag_name not in matched_tags[tag_type] and len(matched_tags[tag_type]) < 10:
                    matched_tags[tag_type].append(tag_name)

        if matched_tags:
            logger.info(f"[Taxonomy-Neo4j] Fetched candidate tags: {matched_tags} (threshold={threshold})")
        return matched_tags
    except Exception as e:
        logger.error(f"[Taxonomy-Neo4j] Error fetching taxonomy candidates: {e}")
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
        if g_lower in q_lower or q_lower == f"{g_lower} {_item_plural()}" or q_lower == f"{g_lower} collections":
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

