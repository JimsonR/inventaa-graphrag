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
    general_vector_store = None
    policy_vector_store = None
    product_faq_vector_store = None

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

        # 3. Connect to Graph (disable schema refresh for extreme cold start boost)
        cls.graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, refresh_schema=False)
        
        # 4. We defer vector store initialization to tools.py to prevent blocking startup

        cls._initialized = True
        logger.info("AgentConfig initialized.")
