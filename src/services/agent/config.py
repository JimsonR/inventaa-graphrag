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
    category_groups = {}  # {"Outdoor": ["LED Outdoor Wall Light", ...], ...}
    top_level_groups = []  # ["Outdoor", "Indoor", "Solar"]
    product_options = [] # Dynamic options like [{'rel_type': 'AVAILABLE_IN_WATTAGE', 'target_label': 'WattageOption', 'alias': 'wattage'}, ...]
    
    # YAML Configuration
    brain = {}
    
    # Dependencies
    memory_provider = None

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return

        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "agent_config.yaml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cls.brain = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load agent_config.yaml at {config_path}: {e}")
            cls.brain = {}

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
            
            # Load hand-curated category groups for browse/navigation queries
            group_res = cls.graph.query("""
                MATCH (cg:CategoryGroup)-[:CONTAINS]->(c:Collection)
                RETURN cg.name AS group_name, cg.is_top_level AS is_top_level, collect(c.name) AS collections
            """)
            cls.category_groups = {r['group_name']: r['collections'] for r in group_res}
            cls.top_level_groups = [r['group_name'] for r in group_res if r.get('is_top_level')]
            
            logger.info(f"Loaded schema dynamically: {len(cls.collections)} Collections, {len(cls.use_cases)} UseCases, {len(cls.features)} Features, {len(cls.category_groups)} CategoryGroups ({len(cls.top_level_groups)} top-level)")
            
            # 6. Discover Dynamic Product Options (e.g., AVAILABLE_IN_COLOR)
            try:
                rel_query = """
                MATCH (p:Product)-[r]->(target)
                WHERE type(r) STARTS WITH 'AVAILABLE_IN_'
                RETURN DISTINCT type(r) AS rel_type, labels(target)[0] AS target_label
                """
                rel_res = cls.graph.query(rel_query)
                cls.product_options = []
                for row in rel_res:
                    rel_type = row.get("rel_type")
                    if not rel_type: continue
                    # Extract the attribute name from the relationship, e.g. AVAILABLE_IN_WATTAGE -> wattage
                    alias = rel_type.replace("AVAILABLE_IN_", "").lower()
                    cls.product_options.append({
                        "rel_type": rel_type,
                        "target_label": row.get("target_label"),
                        "alias": alias
                    })
                logger.info(f"Discovered {len(cls.product_options)} dynamic product options: {[o['alias'] for o in cls.product_options]}")
            except Exception as e:
                logger.error(f"Failed to discover dynamic product options: {e}")

            # 7. Initialize Memory Provider
            from src.services.agent.memory import SupabaseMemoryProvider, InMemoryProvider
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
            if supabase_url and supabase_key:
                try:
                    cls.memory_provider = SupabaseMemoryProvider(supabase_url, supabase_key)
                    logger.info("Initialized SupabaseMemoryProvider.")
                except Exception as e:
                    logger.error(f"Failed to initialize SupabaseMemoryProvider: {e}. Falling back to InMemoryProvider.")
                    cls.memory_provider = InMemoryProvider()
            else:
                cls.memory_provider = InMemoryProvider()
                logger.info("Initialized InMemoryProvider (no SUPABASE credentials found).")

            # 7. Sync taxonomy to vector database for semantic parameter mapping
            from src.services.agent.taxonomy import sync_taxonomy
            sync_taxonomy()
            
        except Exception as e:
            logger.error(f"Failed to fetch dynamic schema: {e}", exc_info=True)

        logger.info("AgentConfig initialized.")
