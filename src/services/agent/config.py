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
    
    # Dynamic schema from graph
    collections = []
    use_cases = []
    features = []

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
        # However, we DO want to eagerly initialize Mem0 to load Spacy NLP models
        # and prevent a 15 second cold start delay on the first user query.
        try:
            from src.services.agent.mem0_client import get_mem0_client
            get_mem0_client()
        except Exception as e:
            logger.warning(f"Could not eagerly initialize Mem0 during startup: {e}")

        cls._initialized = True
        
        # 5. Fetch dynamic graph schema for LLM routing and tool prompt generation
        try:
            query = """
            CALL () { MATCH (c:Collection) RETURN collect(DISTINCT c.name) AS cols }
            CALL () { MATCH (uc:UseCase) RETURN collect(DISTINCT uc.name) AS ucs }
            CALL () { MATCH (f:Feature) RETURN collect(DISTINCT f.name) AS feats }
            RETURN cols, ucs, feats
            """
            res = cls.graph.query(query)
            if res:
                row = res[0]
                cls.collections = sorted(row.get("cols", []))
                cls.use_cases = sorted(row.get("ucs", []))
                cls.features = sorted(row.get("feats", []))
            
            logger.info(f"Loaded schema dynamically: {len(cls.collections)} Collections, {len(cls.use_cases)} UseCases, {len(cls.features)} Features")
        except Exception as e:
            logger.error(f"Failed to fetch dynamic graph schema: {e}")

        logger.info("AgentConfig initialized.")
