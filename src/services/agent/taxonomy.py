import logging
import os
from pinecone import Pinecone
from src.services.agent.config import TenantConfig, AgentConfig

logger = logging.getLogger(__name__)

def sync_taxonomy():
    """
    Embeds categories/collections, features, and use_cases and upserts them to Pinecone.
    Uses the 'taxonomy-cache' namespace.
    """
    if not TenantConfig.categories and not TenantConfig.features and not TenantConfig.use_cases:
        logger.warning("No taxonomy loaded to sync.")
        return

    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = TenantConfig.get_pinecone_index()
        
        if not index_name:
            logger.warning("PINECONE_INDEX_NAME not configured. Skipping taxonomy sync.")
            return

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


from pydantic import BaseModel, Field
from typing import Optional

class TaxonomyExtraction(BaseModel):
    category: Optional[str] = Field(None, description="The specific collection name, if explicitly requested or matched. Must be exact string from candidates.")
    use_case: Optional[str] = Field(None, description="The specific use case. Must be exact string from candidates.")
    feature: Optional[str] = Field(None, description="The specific feature. Must be exact string from candidates.")
    clarify: bool = Field(False, description="Set to true ONLY IF multiple conflicting collections/categories apply equally and you need the user to clarify.")

_pinecone_index = None

def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = AgentConfig.get_pinecone_index()
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME is not configured in environment or brain.")
        _pinecone_index = pc.Index(index_name)
    return _pinecone_index

def fetch_taxonomy_candidates(query_embedding: list, threshold: float = 0.80) -> dict:
    """
    Queries the taxonomy-cache and returns matched tags grouped by type.
    Example return: {'feature': ['waterproof'], 'use_case': ['gate-pillar', 'garden-pathway']}
    """
    try:
        index = _get_pinecone_index()
        
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
                    # Keep up to top 10 unique candidates per type
                    if tag_name not in matched_tags[tag_type] and len(matched_tags[tag_type]) < 10:
                        matched_tags[tag_type].append(tag_name)
                    
        if matched_tags:
            logger.info(f"[Taxonomy] Fetched candidate tags: {matched_tags} (threshold={threshold})")
        return matched_tags
    except Exception as e:
        logger.error(f"[Taxonomy] Error fetching taxonomy candidates: {e}")
        return {}

def extract_taxonomy_parameters(query_text: str, candidate_tags: dict) -> TaxonomyExtraction:
    """
    Sub-Agent: Uses LLM Structured Outputs to filter the messy candidate_tags 
    into an exact set of tool parameters for the Main Agent.
    """
    if not candidate_tags:
        return TaxonomyExtraction()

    try:
        structured_llm = TenantConfig.llm.with_structured_output(TaxonomyExtraction)
        
        prompt = (
            f"User Query: '{query_text}'\n\n"
            f"Candidate Tags from Vector DB:\n{candidate_tags}\n\n"
            "Task: Act as a strict filter. Read the candidate tags and select the exact correct category, use_case, or feature that matches the user's query.\n"
            "Rules:\n"
            "1. You MUST pick the exact string from the Candidate Tags provided. Do not invent tags.\n"
            "2. If the user's query exactly or near-exactly matches a Candidate Tag, select it immediately and DO NOT set clarify=True.\n"
            "3. Reject false-positives (e.g. if query asks for one category/feature and candidate lists suggest an incompatible one, ignore it).\n"
            "4. If and only if the query is very broad and multiple Candidate Tags apply equally without any one being a clear best match, set clarify=True.\n"
            "5. Return null for fields that have no perfect or clear match."
        )
        
        result = structured_llm.invoke(prompt)
        logger.info(f"[Taxonomy Sub-Agent] Extracted Parameters: {result}")
        return result
        
    except Exception as e:
        logger.error(f"[Taxonomy Sub-Agent] Error extracting parameters: {e}", exc_info=True)
        return TaxonomyExtraction()
