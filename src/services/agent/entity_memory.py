import os
import logging
from mem0 import Memory

logger = logging.getLogger(__name__)

_memory_instance = None

def get_entity_memory() -> Memory:
    """
    Initializes and returns the Mem0 Memory instance backed by Pinecone and Azure OpenAI.
    """
    global _memory_instance
    if _memory_instance is None:
        try:
            # Azure OpenAI variables
            azure_endpoint = os.getenv("AZURE_AI_ENDPOINT")
            azure_api_key = os.getenv("AZURE_AI_API_KEY")
            azure_api_version = "2024-02-15-preview"
            llm_deployment = os.getenv("GPT_4_1_MINI_DEPLOYMENT", "gpt-4.1-mini")
            embed_deployment = os.getenv("TEXT_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

            # Pinecone variables
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_index = os.getenv("PINECONE_INDEX_NAME", "inventaa")

            if not pinecone_api_key or not azure_api_key:
                logger.error("Missing Pinecone or Azure API keys. Mem0 extraction will fail.")

            config = {
                "vector_store": {
                    "provider": "pinecone",
                    "config": {
                        "api_key": pinecone_api_key,
                        "collection_name": pinecone_index,
                        "embedding_model_dims": 1536,
                        "metric": "cosine",
                        "serverless_config": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                },
                "llm": {
                    "provider": "azure_openai",
                    "config": {
                        "model": llm_deployment,
                        "azure_kwargs": {
                            "azure_deployment": llm_deployment,
                            "api_version": azure_api_version,
                            "azure_endpoint": azure_endpoint,
                            "api_key": azure_api_key
                        }
                    }
                },
                "embedder": {
                    "provider": "azure_openai",
                    "config": {
                        "model": embed_deployment,
                        "azure_kwargs": {
                            "azure_deployment": embed_deployment,
                            "api_version": azure_api_version,
                            "azure_endpoint": azure_endpoint,
                            "api_key": azure_api_key
                        }
                    }
                }
            }
            
            _memory_instance = Memory.from_config(config)
            logger.info("Mem0 initialized with Pinecone and Azure OpenAI.")
        except Exception as e:
            logger.error(f"Error initializing Mem0 Memory: {e}", exc_info=True)
            
    return _memory_instance

def extract_and_store_entities(text: str, user_id: str, metadata: dict = None):
    """
    Passes the incoming message to Mem0 to extract and store user entities.
    """
    try:
        mem = get_entity_memory()
        mem.add(text, user_id=user_id, metadata=metadata or {})
        logger.info(f"Stored entities for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to extract and store entities in Mem0: {e}", exc_info=True)

def get_user_entities(user_id: str) -> str:
    """
    Retrieves the concatenated list of stored user entities/preferences.
    """
    try:
        mem = get_entity_memory()
        # Use get_all instead of search to retrieve all memories without logging a search query in the dashboard
        results = mem.get_all(filters={'user_id': user_id})
        if not results:
            return "No specific preferences recorded."
        
        # Results is usually a list of dicts with 'memory' or 'text'
        preferences = []
        for r in results:
            # Mem0 result format handles dicts or strings depending on version, safely extract
            text = r.get("memory") if isinstance(r, dict) else str(r)
            if text:
                preferences.append(f"- {text}")
                
        return "\n".join(preferences) if preferences else "No specific preferences recorded."
    except Exception as e:
        logger.error(f"Failed to retrieve user entities from Mem0: {e}", exc_info=True)
        return "Failed to retrieve preferences."
