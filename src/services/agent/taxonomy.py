import logging
import os
from pinecone import Pinecone
from src.services.agent.config import AgentConfig

logger = logging.getLogger(__name__)

def sync_taxonomy():
    """
    Embeds collections, features, and use_cases and upserts them to Pinecone.
    Uses the 'taxonomy-cache' namespace.
    """
    if not AgentConfig.collections and not AgentConfig.features and not AgentConfig.use_cases:
        logger.warning("No taxonomy loaded to sync.")
        return

    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "inventaa")
        
        if index_name not in pc.list_indexes().names():
            logger.warning(f"Pinecone index '{index_name}' not found. Skipping taxonomy sync.")
            return
            
        index = pc.Index(index_name)
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
            if not items: return
            
            embeddings = AgentConfig.embeddings.embed_documents(items)
            for text, emb in zip(items, embeddings):
                safe_id = f"{tag_type}_{text}".replace(" ", "_").replace("/", "_").lower()
                vectors.append({
                    "id": safe_id,
                    "values": emb,
                    "metadata": {"type": tag_type, "name": text}
                })

        process_items(AgentConfig.collections, "category")
        process_items(AgentConfig.features, "feature")
        process_items(AgentConfig.use_cases, "use_case")
        
        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
            logger.info(f"Successfully upserted {len(vectors)} taxonomy tags.")
    except Exception as e:
        logger.error(f"Failed to sync taxonomy: {e}", exc_info=True)


def extract_taxonomy(query_embedding: list, threshold: float = 0.85) -> dict:
    """
    Queries the taxonomy-cache and returns matched tags grouped by type.
    Example return: {'feature': 'waterproof', 'use_case': 'gate-pillar'}
    We only return the TOP match per type to avoid conflicting parameters.
    """
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "inventaa")
        index = pc.Index(index_name)
        
        res = index.query(
            namespace="taxonomy-cache",
            vector=query_embedding,
            top_k=9,
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
                    # Keep up to top 3 unique candidates per type
                    if tag_name not in matched_tags[tag_type] and len(matched_tags[tag_type]) < 3:
                        matched_tags[tag_type].append(tag_name)
                    
        if matched_tags:
            logger.info(f"[Taxonomy] Matched candidate tags: {matched_tags} (threshold={threshold})")
        return matched_tags
    except Exception as e:
        logger.error(f"[Taxonomy] Error extracting taxonomy: {e}")
        return {}
