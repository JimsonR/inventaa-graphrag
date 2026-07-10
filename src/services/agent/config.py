import os
import logging
from typing import Optional, List, Dict, Any, Set
from langchain_neo4j import Neo4jGraph
from langchain_openai import AzureOpenAIEmbeddings

logger = logging.getLogger(__name__)

class TenantConfig:
    """Singleton holding all initialized connections and tenant schema configuration for GraphRAG."""
    _initialized = False
    embeddings = None
    graph = None

    # Dynamic schema from graph (generalized across domains: e-commerce collections, hospital departments, etc.)
    categories = []        # Primary domain categories (collections, departments, specialties)
    collections = []       # Backwards-compatible alias for categories
    departments = []       # Backwards-compatible alias for categories
    use_cases = []
    features = []
    category_groups = {}   # Map of top-level category groups to child collections/categories
    top_level_groups = []  # Top-level category groups
    product_options = []   # Dynamic options discovered from graph schema
    category_to_skus = {}
    collection_to_skus = {}
    category_to_sqlite_cats = {}
    collection_to_sqlite_cats = {}

    # YAML Configuration
    brain = {}

    # SQLite session factory (set during initialize)
    SessionLocal = None

    @classmethod
    def get_brand_name(cls) -> str:
        return cls.brain.get("tenant", {}).get("name", os.getenv("TENANT_NAME", "The Brand"))

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

        logger.info("Initializing TenantConfig dependencies...")

        NEO4J_URI = os.getenv("NEO4J_URI", "").replace("neo4j+s://", "neo4j+ssc://")
        NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

        # 1. Initialize Embeddings (for taxonomy semantic search)
        cls.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
            api_key=os.getenv("AZURE_AI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_deployment=os.getenv("TEXT_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        )

        # 2. Connect to Neo4j Graph (disable schema refresh for fast cold start)
        cls.graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, refresh_schema=False)

        cls._initialized = True

        # 3. Fetch dynamic graph schema (domain agnostic: categories/departments/collections)
        try:
            query = """
            CALL () { MATCH (c) WHERE 'Collection' IN labels(c) OR 'Category' IN labels(c) OR 'Department' IN labels(c) RETURN collect(DISTINCT c.name) AS cols }
            CALL () { MATCH (uc:UseCase) RETURN collect(DISTINCT uc.name) AS ucs }
            CALL () { MATCH (f:Feature) RETURN collect(DISTINCT f.name) AS feats }
            RETURN cols, ucs, feats
            """
            res = cls.graph.query(query)
            if res:
                row = res[0]
                cls.categories = sorted(row.get("cols", []))
                cls.collections = cls.categories
                cls.departments = cls.categories
                cls.use_cases = sorted(row.get("ucs", []))
                cls.features = sorted(row.get("feats", []))

            # Load top-level category groups and their child collections/categories
            group_res = cls.graph.query("""
            CALL () {
                MATCH (cg:CategoryGroup)-[:CONTAINS]->(c)
                WHERE cg.is_top_level = true AND ('Collection' IN labels(c) OR 'Category' IN labels(c) OR 'Department' IN labels(c))
                RETURN cg.name AS group_name, cg.is_top_level AS is_top_level, collect(c.name) AS collections
            }
            RETURN group_name, is_top_level, collections
            """)
            cls.category_groups = {r['group_name']: r['collections'] for r in group_res}
            cls.top_level_groups = sorted(list(cls.category_groups.keys()))

            # Dynamically map Neo4j categories/collections to SKUs and SQLite categories
            try:
                col_sku_res = cls.graph.query("""
                    MATCH (c)<-[r]-(p:Product)
                    WHERE ('Collection' IN labels(c) OR 'Category' IN labels(c) OR 'Department' IN labels(c))
                      AND type(r) IN ['BELONGS_TO_COLLECTION', 'BELONGS_TO_CATEGORY', 'BELONGS_TO_DEPARTMENT', 'IN_CATEGORY']
                    RETURN c.name AS collection, collect(DISTINCT toLower(p.sku)) AS skus
                """)
                cls.category_to_skus = {r['collection']: r['skus'] for r in col_sku_res if r.get('collection')}
                cls.collection_to_skus = cls.category_to_skus

                from src.db.database import get_session
                from src.db.models import Product
                from sqlalchemy import func
                with get_session() as session:
                    for col, skus in cls.category_to_skus.items():
                        if not skus:
                            continue
                        prods = session.query(Product).filter(func.lower(Product.sku).in_(skus)).all()
                        sqlite_cats = {p.categories for p in prods if p.categories}
                        cls.category_to_sqlite_cats[col] = sqlite_cats
                cls.collection_to_sqlite_cats = cls.category_to_sqlite_cats
                logger.info(f"Dynamically mapped {len(cls.category_to_sqlite_cats)} Neo4j categories to SQLite categories via graph.")
            except Exception as e:
                logger.warning(f"Could not dynamically map Neo4j categories to SQLite categories: {e}")

            logger.info(f"Loaded schema dynamically: {len(cls.categories)} Categories/Collections, {len(cls.use_cases)} UseCases, {len(cls.features)} Features, {len(cls.category_groups)} CategoryGroups ({len(cls.top_level_groups)} top-level)")

            # 4. Discover Dynamic Product Options (e.g., AVAILABLE_IN_COLOR, AVAILABLE_IN_OPTION)
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
                    if not rel_type:
                        continue
                    alias = rel_type.replace("AVAILABLE_IN_", "").lower()
                    cls.product_options.append({
                        "rel_type": rel_type,
                        "target_label": row.get("target_label"),
                        "alias": alias
                    })
                logger.info(f"Discovered {len(cls.product_options)} dynamic product options: {[o['alias'] for o in cls.product_options]}")
            except Exception as e:
                logger.error(f"Failed to discover dynamic product options: {e}")

            # 5. Sync taxonomy to vector database for semantic parameter mapping
            from src.services.agent.taxonomy import sync_taxonomy
            sync_taxonomy()

        except Exception as e:
            logger.error(f"Failed to fetch dynamic schema: {e}", exc_info=True)

        logger.info("TenantConfig initialized.")


# Backwards-compatible alias for existing code referencing AgentConfig
AgentConfig = TenantConfig
