import os
import logging
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

logger = logging.getLogger(__name__)

class AgentConfig:
    """Singleton holding all initialized connections."""
    _initialized = False
    llm = None
    embeddings = None
    graph = None
    
    # Cache for dynamically loaded vector stores
    _vector_stores = {}

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return

        logger.info("Initializing AgentConfig dependencies...")

        NEO4J_URI = os.getenv("NEO4J_URI", "").replace("neo4j+s://", "neo4j+ssc://")
        NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

        # 1. Initialize Azure OpenAI LLM
        cls.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
            api_key=os.getenv("AZURE_AI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_deployment=os.getenv("GPT_4_1_MINI_DEPLOYMENT", "gpt-4.1-mini"),
            temperature=0
        )

        # 2. Initialize Embeddings
        cls.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
            api_key=os.getenv("AZURE_AI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_deployment=os.getenv("TEXT_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        )

        # 3. Connect to Graph
        cls.graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        cls.graph.refresh_schema()

        cls._initialized = True
        logger.info("AgentConfig initialized.")

    @classmethod
    def get_vector_store(cls, index_name: str, text_node_property: str = "text", retrieval_query: str = None):
        """
        Lazily initialize and return a Neo4jVector for the given index.
        This allows multi-tenancy where different tenants use different indices.
        """
        if not index_name:
            raise ValueError("index_name cannot be empty")
            
        cache_key = f"{index_name}_{text_node_property}_{retrieval_query or 'default'}"
        
        if cache_key not in cls._vector_stores:
            logger.info(f"Initializing Neo4jVector for index: {index_name}")
            
            NEO4J_URI = os.getenv("NEO4J_URI", "").replace("neo4j+s://", "neo4j+ssc://")
            NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
            NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
            
            cls._vector_stores[cache_key] = Neo4jVector.from_existing_index(
                embedding=cls.embeddings, 
                url=NEO4J_URI, 
                username=NEO4J_USERNAME, 
                password=NEO4J_PASSWORD,
                index_name=index_name, 
                text_node_property=text_node_property,
                retrieval_query=retrieval_query
            )
            
        return cls._vector_stores[cache_key]
