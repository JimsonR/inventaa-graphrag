import os
import logging
from typing import Optional, List, Dict, Any, Set
from langchain_neo4j import Neo4jGraph
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
    category_groups = {}  # Map of top-level category groups to child collections
    top_level_groups = []  # Top-level category groups
    product_options = [] # Dynamic options discovered from graph schema
    collection_to_skus = {} # Map of collection names to SKUs
    collection_to_sqlite_cats = {} # Map of collection names to SQLite categories
    
    # YAML Configuration
    brain = {}
    
    # Dependencies
    memory_provider = None

    @classmethod
    def get_brand_name(cls) -> str:
        return cls.brain.get("tenant", {}).get("name", os.getenv("TENANT_NAME", "The Brand"))

    @classmethod
    def get_brand_description(cls) -> str:
        return cls.brain.get("tenant", {}).get("description", os.getenv("TENANT_DESCRIPTION", "an AI assistant"))

    @classmethod
    def get_currency_symbol(cls) -> str:
        return cls.brain.get("tenant", {}).get("currency_symbol", os.getenv("CURRENCY_SYMBOL", "$"))

    @classmethod
    def get_stop_words(cls) -> set:
        default_stop = ["the", "a", "an", "and", "or", "for", "of", "in", "with", "by", "from", "show", "me", "any", "on", "product", "products"]
        return set(cls.brain.get("search_heuristics", {}).get("stop_words", default_stop))

    @classmethod
    def get_detail_stop_words(cls) -> set:
        default_stop = ["the", "a", "an", "and", "or", "for", "of", "in", "with", "by", "from"]
        return set(cls.brain.get("search_heuristics", {}).get("detail_stop_words", default_stop))

    @classmethod
    def get_fulltext_index(cls) -> str:
        return os.getenv("NEO4J_FULLTEXT_INDEX", cls.brain.get("neo4j", {}).get("fulltext_index", "product_name_ft"))

    @classmethod
    def get_faq_index(cls) -> str:
        return os.getenv("NEO4J_FAQ_INDEX", cls.brain.get("neo4j", {}).get("faq_index", "faq_vector"))

    @classmethod
    def get_pinecone_index(cls) -> Optional[str]:
        return os.getenv("PINECONE_INDEX_NAME", cls.brain.get("pinecone", {}).get("index_name", None))

    @classmethod
    def get_cache_skip_intents(cls) -> list:
        return cls.brain.get("cache", {}).get("skip_intents", ["detail", "search", "policy"])

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return

        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "agent_config.yaml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cls.brain = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load agent_config.yaml at {config_path}: {e}")
            cls.brain = {}

        domain = os.getenv("DOMAIN") or cls.brain.get("tenant", {}).get("domain") or cls.brain.get("domain", "ecommerce")
        domain_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "domains", f"{domain}.yaml")
        if os.path.exists(domain_path):
            try:
                with open(domain_path, "r", encoding="utf-8") as f:
                    domain_config = yaml.safe_load(f) or {}
                for key, val in domain_config.items():
                    if isinstance(val, dict):
                        if key not in cls.brain or not isinstance(cls.brain[key], dict):
                            cls.brain[key] = val
                        else:
                            merged = val.copy()
                            merged.update(cls.brain[key])
                            cls.brain[key] = merged
                    elif key not in cls.brain:
                        cls.brain[key] = val
                logger.info(f"Loaded domain configuration from {domain_path} (domain: {domain})")
            except Exception as e:
                logger.error(f"Failed to load domain config at {domain_path}: {e}")

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
            
            # Load top-level category groups and their child collections
            group_res = cls.graph.query("""
                MATCH (cg:CategoryGroup)-[:CONTAINS]->(c:Collection)
                WHERE cg.is_top_level = true
                RETURN cg.name AS group_name, cg.is_top_level AS is_top_level, collect(c.name) AS collections
            """)
            cls.category_groups = {r['group_name']: r['collections'] for r in group_res}
            cls.top_level_groups = sorted(list(cls.category_groups.keys()))
            
            # Dynamically map Neo4j collections to SKUs and SQLite categories via BELONGS_TO_COLLECTION
            try:
                col_sku_res = cls.graph.query("""
                    MATCH (c:Collection)<-[:BELONGS_TO_COLLECTION]-(p:Product)
                    RETURN c.name AS collection, collect(DISTINCT toLower(p.sku)) AS skus
                """)
                cls.collection_to_skus = {r['collection']: r['skus'] for r in col_sku_res if r.get('collection')}
                
                from src.db.database import get_session
                from src.db.models import Product
                from sqlalchemy import func
                with get_session() as session:
                    for col, skus in cls.collection_to_skus.items():
                        if not skus: continue
                        prods = session.query(Product).filter(func.lower(Product.sku).in_(skus)).all()
                        sqlite_cats = {p.categories for p in prods if p.categories}
                        cls.collection_to_sqlite_cats[col] = sqlite_cats
                logger.info(f"Dynamically mapped {len(cls.collection_to_sqlite_cats)} Neo4j collections to SQLite categories via graph.")
            except Exception as e:
                logger.warning(f"Could not dynamically map Neo4j collections to SQLite categories: {e}")
            
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
                    # Extract attribute name from relationship, e.g. AVAILABLE_IN_<ATTR> -> <attr>
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
