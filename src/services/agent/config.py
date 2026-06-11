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

        # 3. Connect to Graph
        cls.graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
        cls.graph.refresh_schema()

        # 4. Connect to Vector Stores
        cls.general_vector_store = Neo4jVector.from_existing_index(
            embedding=cls.embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
            index_name="inventaa_faq_vector", text_node_property="text"
        )

        cls.policy_vector_store = Neo4jVector.from_existing_index(
            embedding=cls.embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
            index_name="policy_vector", text_node_property="text"
        )

        cls.product_faq_vector_store = Neo4jVector.from_existing_index(
            embedding=cls.embeddings, url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
            index_name="product_faq_vector", text_node_property="question",
            retrieval_query='''
            MATCH (node)<-[:HAS_FAQ]-(p:Product)
            RETURN "FAQ Match: " + node.question + "\\nAnswer: " + node.answer + 
                   "\\n--> This belongs to Product: " + p.name + " (Price: ₹" + toString(p.price_num) + ")" AS text,
                   score, {product_url: p.url} AS metadata
            '''
        )

        cls._initialized = True
        logger.info("AgentConfig initialized.")
