import os
os.environ["MEM0_TELEMETRY"] = "false"

import logging
from typing import Optional
from mem0 import Memory

logger = logging.getLogger(__name__)

_mem0_client: Optional[Memory] = None

def get_mem0_client() -> Optional[Memory]:
    """
    Initializes and returns the Mem0 client using Pinecone and Azure OpenAI.
    Returns None if the required environment variables are missing.
    """
    global _mem0_client
    if _mem0_client is not None:
        return _mem0_client

    # Required for Azure OpenAI
    azure_api_key = os.getenv("AZURE_AI_API_KEY")
    azure_endpoint = os.getenv("AZURE_AI_ENDPOINT")
    
    # Required for Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index = os.getenv("PINECONE_INDEX_NAME")
    
    # Optional defaults
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    embedding_deployment = os.getenv("TEXT_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
    llm_deployment = os.getenv("GPT_4_1_MINI_DEPLOYMENT", "gpt-4.1-mini")
    
    if not all([azure_api_key, azure_endpoint, pinecone_api_key, pinecone_index]):
        logger.warning("Mem0 environment variables missing. Mem0 will not be initialized.")
        return None

    try:
        config = {
            "llm": {
                "provider": "azure_openai",
                "config": {
                    "model": llm_deployment,
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "azure_kwargs": {
                        "api_key": azure_api_key,
                        "azure_endpoint": azure_endpoint,
                        "api_version": azure_api_version,
                        "azure_deployment": llm_deployment
                    }
                }
            },
            "embedder": {
                "provider": "azure_openai",
                "config": {
                    "model": embedding_deployment,
                    "azure_kwargs": {
                        "api_key": azure_api_key,
                        "azure_endpoint": azure_endpoint,
                        "api_version": azure_api_version,
                        "azure_deployment": embedding_deployment
                    }
                }
            },
            "vector_store": {
                "provider": "pinecone",
                "config": {
                    "api_key": pinecone_api_key,
                    "collection_name": pinecone_index,
                    "embedding_model_dims": 1536, # text-embedding-ada-002 is 1536 dims
                    "metric": "cosine"
                }
            }
        }
        
        _mem0_client = Memory.from_config(config)
        logger.info("Mem0 client initialized successfully with Pinecone and Azure OpenAI.")
        return _mem0_client
    except Exception as e:
        logger.error(f"Failed to initialize Mem0 client: {e}", exc_info=True)
        return None

def fetch_long_term_context(query: str, user_id: str) -> str:
    """
    Searches Mem0 for long-term facts about the user relevant to the query.
    """
    if not user_id:
        return ""
        
    client = get_mem0_client()
    if not client:
        return ""
        
    try:
        results = client.search(query, filters={"user_id": user_id})
        if not results:
            return ""
            
        if isinstance(results, dict) and "results" in results:
            results = results["results"]
            
        facts = []
        for res in results:
            if isinstance(res, dict):
                facts.append(f"- {res.get('memory', res)}")
            elif hasattr(res, 'memory'):
                facts.append(f"- {res.memory}")
            else:
                facts.append(f"- {res}")
        
        if not facts:
            return ""
            
        context = "\n".join(facts)
        return f"\n\nHere are some relevant facts from the user's previous sessions:\n{context}\n"
    except Exception as e:
        logger.error(f"Error fetching Mem0 context: {e}", exc_info=True)
        return ""

def store_long_term_context(query: str, response: str, user_id: str):
    """
    Stores the conversation turn into Mem0 to update the user's long-term profile.
    """
    if not user_id or not query or not response:
        return
        
    client = get_mem0_client()
    if not client:
        return
        
    try:
        # We only pass the user's query to Mem0. 
        # Passing the agent's response (especially if it's a huge product JSON)
        # pollutes Mem0 with facts about our products rather than facts about the user.
        client.add(query, user_id=user_id)
        logger.info(f"Successfully stored user query in Mem0 for user {user_id}")
    except Exception as e:
        logger.error(f"Error storing Mem0 context: {e}", exc_info=True)
