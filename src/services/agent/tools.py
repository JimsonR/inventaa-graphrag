import json
import logging
import re
from typing import Optional
from pydantic import Field
from langchain_core.tools import StructuredTool, Tool
from src.services.agent.config import AgentConfig

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH SCHEMA (for reference / prompt engineering)
# Categories: Gate & Pillar Lights | Solar Lights | Outdoor Wall Lights |
#             Bollard & Garden Lights | Street Lights | Flood Lights |
#             Indoor & Ceiling Lights | Panel Lights | Pathway & Step Lights |
#             Bulkhead Lights | Divine & Temple Lights | General Purpose Lights
#
# UseCases: gate-pillar | indoor-ceiling | outdoor-wall | garden-pathway |
#           pathway-step | street-road | flood-area | solar-outdoor |
#           religious-decorative
#
# Features: outdoor | indoor | solar-powered | waterproof | IP65-rated |
#           IP66-rated | motion-sensor | dimmable | energy-efficient |
#           warm-white | cool-white | neutral-white | 3-in-1-colour |
#           aluminium-body | polycarbonate-body | surface-mount | wall-mount |
#           post-top-mount | UV-protected | rustproof
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    # Gate & Pillar
    "gate": "Gate & Pillar Lights",
    "pillar": "Gate & Pillar Lights",
    "post": "Gate & Pillar Lights",
    "compound": "Gate & Pillar Lights",
    "entrance": "Gate & Pillar Lights",
    "boundary": "Gate & Pillar Lights",
    # Solar
    "solar": "Solar Lights",
    # Outdoor Wall
    "wall": "Outdoor Wall Lights",
    "sconce": "Outdoor Wall Lights",
    "elevation": "Outdoor Wall Lights",
    # Bollard & Garden
    "bollard": "Bollard & Garden Lights",
    "garden": "Bollard & Garden Lights",
    "lawn": "Bollard & Garden Lights",
    "landscape": "Bollard & Garden Lights",
    "driveway": "Bollard & Garden Lights",
    "yard": "Bollard & Garden Lights",
    "terrace": "Bollard & Garden Lights",
    "balcony": "Bollard & Garden Lights",
    "resort": "Bollard & Garden Lights",
    "hotel": "Bollard & Garden Lights",
    "villa": "Bollard & Garden Lights",
    "community": "Bollard & Garden Lights",
    # Street
    "street": "Street Lights",
    "road": "Street Lights",
    "parking": "Street Lights",
    # Flood
    "flood": "Flood Lights",
    "stadium": "Flood Lights",
    # Indoor & Ceiling
    "indoor": "Indoor & Ceiling Lights",
    "ceiling": "Indoor & Ceiling Lights",
    "downlight": "Indoor & Ceiling Lights",
    # Panel
    "panel": "Panel Lights",
    # Pathway & Step
    "pathway": "Pathway & Step Lights",
    "step": "Pathway & Step Lights",
    "stairway": "Pathway & Step Lights",
    "walkway": "Pathway & Step Lights",
    "stair": "Pathway & Step Lights",
    # Bulkhead
    "bulkhead": "Bulkhead Lights",
    # Divine & Temple
    "divine": "Divine & Temple Lights",
    "temple": "Divine & Temple Lights",
    "religious": "Divine & Temple Lights",
    "god": "Divine & Temple Lights",
    "pooja": "Divine & Temple Lights",
    # General
    "general": "General Purpose Lights",
}

USECASE_KEYWORDS = {
    "gate": "gate-pillar",
    "pillar": "gate-pillar",
    "entrance": "gate-pillar",
    "compound": "gate-pillar",
    "post": "gate-pillar",
    "indoor": "indoor-ceiling",
    "ceiling": "indoor-ceiling",
    "downlight": "indoor-ceiling",
    "wall": "outdoor-wall",
    "sconce": "outdoor-wall",
    "elevation": "outdoor-wall",
    "garden": "garden-pathway",
    "lawn": "garden-pathway",
    "landscape": "garden-pathway",
    "driveway": "garden-pathway",
    "villa": "garden-pathway",
    "terrace": "garden-pathway",
    "balcony": "garden-pathway",
    "resort": "garden-pathway",
    "hotel": "garden-pathway",
    "bollard": "garden-pathway",
    "pathway": "pathway-step",
    "walkway": "pathway-step",
    "step": "pathway-step",
    "stairway": "pathway-step",
    "stair": "pathway-step",
    "street": "street-road",
    "road": "street-road",
    "parking": "street-road",
    "flood": "flood-area",
    "stadium": "flood-area",
    "solar": "solar-outdoor",
    "temple": "religious-decorative",
    "divine": "religious-decorative",
    "religious": "religious-decorative",
    "pooja": "religious-decorative",
}

FEATURE_KEYWORDS = {
    "solar": "solar-powered",
    "waterproof": "waterproof",
    "weatherproof": "waterproof",
    "rain": "waterproof",
    "water": "waterproof",
    "ip65": "IP65-rated",
    "ip66": "IP66-rated",
    "coastal": "rustproof",
    "motion": "motion-sensor",
    "sensor": "motion-sensor",
    "dimmable": "dimmable",
    "dim": "dimmable",
    "warm": "warm-white",
    "cool": "cool-white",
    "neutral": "neutral-white",
    "3-in-1": "3-in-1-colour",
    "colour": "3-in-1-colour",
    "color": "3-in-1-colour",
    "aluminium": "aluminium-body",
    "aluminum": "aluminium-body",
    "metal": "aluminium-body",
    "polycarbonate": "polycarbonate-body",
    "plastic": "polycarbonate-body",
    "surface": "surface-mount",
    "rustproof": "rustproof",
    "rust": "rustproof",
    "uv": "UV-protected",
    "fade": "UV-protected",
    "energy": "energy-efficient",
    "efficient": "energy-efficient",
}

# Stop words to ignore when tokenizing queries
_STOP_WORDS = {
    "light", "lights", "lamp", "lamps", "led", "product", "products",
    "show", "me", "get", "find", "list", "give", "want", "need",
    "rated", "rating", "lowest", "highest", "best", "top", "some",
    "a", "an", "the", "for", "with", "of", "in", "and", "or",
    "is", "are", "what", "which", "how", "do", "can", "does",
    "suggest", "recommend", "suitable", "use", "buy", "choose",
    "my", "i", "we", "our", "this", "that", "under", "budget",
    "within", "rs", "inr", "rupees", "good", "looking", "modern",
}


def _classify_query(query: str):
    """Return (matched_category, matched_usecase, matched_features, remaining_tokens)."""
    tokens = [t.lower().strip(".,?!") for t in query.split() if t.lower().strip(".,?!") not in _STOP_WORDS]
    matched_category = None
    matched_usecase = None
    matched_features = []
    remaining = []

    for token in tokens:
        cat = CATEGORY_KEYWORDS.get(token)
        uc = USECASE_KEYWORDS.get(token)
        feat = FEATURE_KEYWORDS.get(token)
        if cat and not matched_category:
            matched_category = cat
        if uc and not matched_usecase:
            matched_usecase = uc
        if feat and feat not in matched_features:
            matched_features.append(feat)
        if not cat and not uc and not feat:
            remaining.append(token)

    return matched_category, matched_usecase, matched_features, remaining


def search_products_db(
    query: Optional[str] = None,
    category: Optional[str] = None,
    collection: Optional[str] = None,
    use_case: Optional[str] = None,
    feature: Optional[str] = None,
    spec: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    sort_by: Optional[str] = None,
    limit: int = 100
):
    from src.services.agent.utils import track_time
    tool_logger = logging.getLogger("Tool.SearchProducts")
    with track_time("Tool: SearchProductsDatabase", custom_logger=tool_logger):
        """
        Search and filter products from the Neo4j graph database.
        Uses Category, UseCase, and Feature graph nodes for precise filtering.
        sort_by values: price_asc, price_desc, rating_desc, rating_asc, reviews_desc
        """
    try:
        params = {"limit": limit}

        # Auto-classify the query into category/usecase/feature
        auto_category, auto_usecase, auto_features, remaining_tokens = _classify_query(query or "")

        # Prefer explicit parameters over auto-detected ones
        final_category = category or auto_category
        final_usecase = use_case or auto_usecase
        final_features = ([feature] if feature else []) + [f for f in auto_features if f != feature]

        logger.info(f"SearchProducts | query={query!r} | auto_category={auto_category!r} "
                    f"| auto_usecase={auto_usecase!r} | auto_features={auto_features} "
                    f"| remaining_tokens={remaining_tokens}")

        # Build Cypher: start from Category for maximum precision when category is known
        if final_category:
            cypher_query = """
MATCH (cat:Category {name: $category})-[:HAS_PRODUCT]->(p:Product)
"""
            params["category"] = final_category
        else:
            cypher_query = "MATCH (p:Product)\n"

        from src.services.agent.context import tenant_context
        tenant_id = tenant_context.get()
        if tenant_id:
            where_clauses = ["p.tenant = $tenant_id"]
            params["tenant_id"] = tenant_id
        else:
            where_clauses = []

        # UseCase filter
        if final_usecase:
            cypher_query += "MATCH (p)-[:SUITABLE_FOR]->(uc:UseCase {name: $use_case})\n"
            params["use_case"] = final_usecase
            
        # Collection filter
        if collection:
            cypher_query += "MATCH (p)-[:BELONGS_TO_COLLECTION]->(col:Collection {name: $collection})\n"
            params["collection"] = collection

        # Feature filter
        if final_features:
            for i, feat in enumerate(final_features[:2]):  # max 2 feature filters
                feat_param = f"feature_{i}"
                cypher_query += f"MATCH (p)-[:HAS_FEATURE]->(f{i}:Feature {{name: ${feat_param}}})\n"
                params[feat_param] = feat

        # Spec filter
        if spec:
            cypher_query += "MATCH (p)-[:HAS_SPEC]->(s:Spec)\n"
            where_clauses.append("(toLower(s.key) CONTAINS toLower($spec) OR toLower(s.value) CONTAINS toLower($spec))")
            params["spec"] = spec

        # Price filters
        if min_price is not None:
            where_clauses.append("p.price_num >= $min_price")
            params["min_price"] = min_price
        if max_price is not None:
            where_clauses.append("p.price_num <= $max_price")
            params["max_price"] = max_price

        # Apply remaining tokens (like "7w") as text filters on the main query
        if remaining_tokens:
            for i, token in enumerate(remaining_tokens[:3]): # max 3 token filters
                tok_param = f"rem_{i}"
                params[tok_param] = token
                where_clauses.append(
                    f"(toLower(p.name) CONTAINS toLower(${tok_param}) "
                    f"OR toLower(p.feature_descriptions) CONTAINS toLower(${tok_param}) "
                    f"OR EXISTS {{ MATCH (p)-[:HAS_SPEC]->(s:Spec) WHERE toLower(s.value) CONTAINS toLower(${tok_param}) }})"
                )

        if where_clauses:
            cypher_query += "WHERE " + " AND ".join(where_clauses) + "\n"

        # Sort
        sort_map = {
            "price_asc": "ORDER BY p.price_num ASC",
            "price_low": "ORDER BY p.price_num ASC",
            "price_desc": "ORDER BY p.price_num DESC",
            "price_high": "ORDER BY p.price_num DESC",
            "rating_desc": "ORDER BY p.rating_score DESC",
            "rating_asc": "ORDER BY p.rating_score ASC",
            "reviews_desc": "ORDER BY p.review_count DESC",
        }
        sort_clause = sort_map.get((sort_by or "").lower(), "ORDER BY p.rating_score DESC")

        cypher_query += f"""
OPTIONAL MATCH (p)-[:HAS_WARRANTY]->(w:Warranty)
OPTIONAL MATCH (p)-[:HAS_POLICY]->(pol:Policy) WHERE toLower(pol.title) CONTAINS 'replacement' OR toLower(pol.title) CONTAINS 'exchange'
OPTIONAL MATCH (p)-[:BELONGS_TO_COLLECTION]->(col2:Collection)
RETURN DISTINCT p.sku AS sku, p.name AS name, p.price_num AS price_num,
       p.regular_price AS regular_price, p.discount_percentage AS discount_percentage,
       p.image_url AS image_url, p.url AS url, p.rating_score AS rating,
       p.review_count AS review_count, p.tenant AS tenant, p.feature_descriptions AS feature_descriptions,
       p.installation_url AS installation_url, w.description AS warranty, pol.content AS replacement_exchange_policy,
       collect(DISTINCT col2.name) AS collections
{sort_clause}
LIMIT $limit
"""

        logger.info(f"SearchProductsDatabase Cypher:\n{cypher_query.strip()}\nParams: {params}")
        res = AgentConfig.graph.query(cypher_query, params=params)

        if not res:
            return "[]"

        return json.dumps(res, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in SearchProductsDatabase: {e}", exc_info=True)
        return f"Error querying graph: {e}"


def get_product_details_db(product_name: str):
    try:
        # --- Smarter tokenization ---
        # Strip noise tokens (stop words, single chars, numbers-only, punctuation like "-")
        _DETAIL_STOP = {
            "led", "light", "lights", "lamp", "lamps", "the", "a", "an",
            "and", "or", "for", "of", "in", "with", "by", "from",
            "frontgate", "lighting", "design", "outdoor", "indoor",
        }
        raw_tokens = [t.strip(".,?!-–—/|") for t in product_name.split()]
        # Keep tokens that are meaningful: length > 1, not stop, not pure numbers
        good_tokens = [
            t for t in raw_tokens
            if t and len(t) > 1 and t.lower() not in _DETAIL_STOP and not t.isdigit()
        ]

        if not good_tokens:
            return "Please provide a valid product name."

        # Build Lucene query: use first 3 good tokens with fuzzy (~), join remaining as plain phrases
        lucene_tokens = [t + "~" for t in good_tokens[:3]]
        lucene_query = " AND ".join(lucene_tokens)
        logger.info(f"ProductDetailsDatabase | product_name={product_name!r} | lucene={lucene_query!r}")

        cypher_query = """
        CALL db.index.fulltext.queryNodes("product_name_ft", $lucene_query) YIELD node AS p, score
        WHERE p.tenant = $tenant_id OR p.tenant IS NULL
        WITH p, score
        ORDER BY score DESC LIMIT 1
        OPTIONAL MATCH (p)-[:HAS_WARRANTY]->(w:Warranty)
        OPTIONAL MATCH (p)-[:HAS_POLICY]->(pol:Policy) WHERE toLower(pol.title) CONTAINS 'replacement' OR toLower(pol.title) CONTAINS 'exchange'
        OPTIONAL MATCH (p)-[:HAS_SPEC]->(s:Spec)
        OPTIONAL MATCH (p)-[:AVAILABLE_IN_WATTAGE]->(wo:WattageOption)
        OPTIONAL MATCH (p)-[:AVAILABLE_IN_COLOR]->(co:ColorOption)
        OPTIONAL MATCH (p)-[:BELONGS_TO_COLLECTION]->(col:Collection)
        RETURN p.name AS name, p.price_num AS price,
               p.feature_descriptions AS feature_descriptions,
               p.installation_url AS installation_url,
               pol.content AS replacement_exchange_policy,
               w.description AS warranty_info, w.duration_years AS warranty_duration,
               collect(DISTINCT s.key + ': ' + s.value) AS specs,
               collect(DISTINCT wo.name) AS wattages,
               collect(DISTINCT co.name) AS colors,
               collect(DISTINCT col.name) AS collections
        """
        from src.services.agent.context import tenant_context
        params = {"lucene_query": lucene_query, "tenant_id": tenant_context.get()}
        logger.info(f"ProductDetailsDatabase Cypher: {cypher_query.strip()} | Params: {params}")
        res = AgentConfig.graph.query(cypher_query, params=params)

        if not res:
            return "Product not found."

        product = res[0]
        output = f"Product Name: {product.get('name')}\n"
        output += f"Price: Rs. {product.get('price')}\n"

        # Wattage options
        wattages = [w for w in (product.get('wattages') or []) if w]
        if wattages:
            output += f"Available Wattages: {', '.join(sorted(wattages))}\n"
        else:
            # Fallback: look in specs for wattage
            watt_specs = [s for s in (product.get('specs') or []) if 'watt' in s.lower() or 'power' in s.lower()]
            if watt_specs:
                output += f"Wattage Info: {'; '.join(watt_specs)}\n"

        # Color options
        colors = [c for c in (product.get('colors') or []) if c]
        if colors:
            output += f"Available Colors: {', '.join(sorted(colors))}\n"

        # Collections
        collections = [c for c in (product.get('collections') or []) if c]
        if collections:
            output += f"Collections: {', '.join(sorted(collections))}\n"

        # Warranty & Policies
        if product.get('warranty_info'):
            output += f"Warranty: {product.get('warranty_info')}\n"
        else:
            output += "Warranty: This product carries Inventaa's standard 1-Year replacement warranty. Contact support to claim.\n"
            
        if product.get('replacement_exchange_policy'):
            output += f"Replacement Policy: {product.get('replacement_exchange_policy')}\n"
            
        if product.get('installation_url'):
            output += f"Installation Image: {product.get('installation_url')}\n"

        # All specs
        if product.get('specs'):
            output += f"Specifications: {', '.join(product.get('specs'))}\n"

        if product.get('feature_descriptions'):
            output += f"Features: {product.get('feature_descriptions')}\n"

        return output.encode("ascii", errors="ignore").decode("ascii")
    except Exception as e:
        logger.error(f"Error in ProductDetailsDatabase: {e}", exc_info=True)
        return f"Error getting product details: {e}"



def get_categories_db(*args, **kwargs):
    """
    Returns all available product categories currently in the database.
    """
    try:
        from src.services.agent.context import tenant_context
        tenant_id = tenant_context.get()
        
        cypher = "MATCH (c:Category)-[:HAS_PRODUCT]->(p:Product)\n"
        params = {}
        if tenant_id:
            cypher += "WHERE p.tenant = $tenant_id\n"
            params["tenant_id"] = tenant_id
            
        cypher += "RETURN DISTINCT c.name AS category ORDER BY category"
        res = AgentConfig.graph.query(cypher, params=params)
        
        if not res:
            return "No categories found in the database."
            
        cats = [r["category"] for r in res]
        return "Available Categories in DB: " + ", ".join(cats)
    except Exception as e:
        logger.error(f"Error in GetCategoriesDatabase: {e}", exc_info=True)
        return f"Error getting categories: {e}"


def query_policies(query: str):
    from langchain_neo4j import Neo4jVector
    import os
    
    if AgentConfig.policy_vector_store is None:
        NEO4J_URI = os.getenv("NEO4J_URI", "").replace("neo4j+s://", "neo4j+ssc://")
        AgentConfig.policy_vector_store = Neo4jVector.from_existing_index(
            embedding=AgentConfig.embeddings, url=NEO4J_URI, 
            username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"),
            index_name="policy_vector", text_node_property="text"
        )
        
    from src.services.agent.context import tenant_context
    tenant_id = tenant_context.get()
    filter_dict = {"tenant": tenant_id} if tenant_id else None
    
    results = AgentConfig.policy_vector_store.similarity_search_with_score(query, k=2, filter=filter_dict)
    if not results:
        if AgentConfig.general_vector_store is None:
            NEO4J_URI = os.getenv("NEO4J_URI", "").replace("neo4j+s://", "neo4j+ssc://")
            AgentConfig.general_vector_store = Neo4jVector.from_existing_index(
                embedding=AgentConfig.embeddings, url=NEO4J_URI, 
                username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"),
                index_name="inventaa_faq_vector", text_node_property="text"
            )
        results = AgentConfig.general_vector_store.similarity_search_with_score(query, k=2, filter=filter_dict)
    if not results:
        return "No relevant policy found."
    text = "\n\n".join([doc.page_content for doc, _ in results])
    return text.encode("ascii", errors="ignore").decode("ascii")


def query_product_faqs(query: str):
    from langchain_neo4j import Neo4jVector
    import os
    
    if AgentConfig.product_faq_vector_store is None:
        NEO4J_URI = os.getenv("NEO4J_URI", "").replace("neo4j+s://", "neo4j+ssc://")
        AgentConfig.product_faq_vector_store = Neo4jVector.from_existing_index(
            embedding=AgentConfig.embeddings, url=NEO4J_URI, 
            username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"),
            index_name="product_faq_vector", text_node_property="question",
            retrieval_query='''
            MATCH (node)<-[:HAS_FAQ]-(p:Product)
            RETURN "FAQ Match: " + node.question + "\\nAnswer: " + node.answer + 
                   "\\n--> This belongs to Product: " + p.name + " (Price: ₹" + toString(p.price_num) + ")" AS text,
                   score, {product_url: p.url} AS metadata
            '''
        )
        
    from src.services.agent.context import tenant_context
    tenant_id = tenant_context.get()
    filter_dict = {"tenant": tenant_id} if tenant_id else None
    
    results = AgentConfig.product_faq_vector_store.similarity_search_with_score(query, k=2, filter=filter_dict)
    if not results:
        return "No relevant product FAQ found."
    text = "\n\n".join([doc.page_content for doc, _ in results])
    return text.encode("ascii", errors="ignore").decode("ascii")


def query_general_knowledge(query: str):
    """
    Search blog articles and general lighting knowledge stored as Chunk nodes.
    Uses full-text search on the chunk_text index.
    """
    try:
        _STOP = {"the", "a", "an", "is", "are", "which", "one", "better", "vs",
                 "or", "and", "for", "of", "in", "with", "what", "how", "do",
                 "does", "can", "will", "between", "difference"}
        # Strip Lucene special chars: + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
        _LUCENE_SPECIAL = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/.,?!\'"–—]')
        raw_tokens = []
        for word in query.split():
            # First split on hyphens (Wave-Free → Wave, Free), then clean each part
            parts = word.split("-")
            for part in parts:
                clean = _LUCENE_SPECIAL.sub("", part).strip()
                if clean:
                    raw_tokens.append(clean)
        good = [t + "~" for t in raw_tokens if t and len(t) > 2 and t.lower() not in _STOP]
        if not good:
            return "No relevant articles found."
        lucene_query = " AND ".join(good[:5])
        logger.info(f"GeneralKnowledgeDatabase | lucene={lucene_query!r}")

        cypher = """
        CALL db.index.fulltext.queryNodes("chunk_text", $q) YIELD node AS c, score
        WHERE (c.tenant = $tenant_id OR c.tenant IS NULL) AND score > 0.5
        WITH c, score
        RETURN c.text AS text, score
        ORDER BY score DESC
        LIMIT 3
        """
        from src.services.agent.context import tenant_context
        tenant_id = tenant_context.get()
        res = AgentConfig.graph.query(cypher, params={"q": lucene_query, "tenant_id": tenant_id})

        if not res:
            # Fallback: try with only the first distinctive token (no fuzzy)
            key_token = good[0].rstrip("~")
            logger.info(f"GeneralKnowledgeDatabase fallback: trying single token '{key_token}'")
            res = AgentConfig.graph.query(cypher, params={"q": key_token, "tenant_id": tenant_id})

        if not res:
            return "No relevant articles found."

        # Concatenate results, strip markdown images/links to reduce noise
        combined = []
        for row in res:
            text = row.get("text") or ""
            # Strip markdown image/link syntax
            text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
            text = re.sub(r"\[.*?\]\(.*?\)", "", text)
            text = re.sub(r"---.*?---", "", text, flags=re.DOTALL)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 50:
                combined.append(text[:1500])  # limit per chunk

        if not combined:
            return "No relevant articles found."

        output = "\n\n---\n".join(combined)
        return output.encode("ascii", errors="ignore").decode("ascii")
    except Exception as e:
        logger.error(f"Error in GeneralKnowledgeDatabase: {e}", exc_info=True)
        return f"Error searching knowledge base: {e}"


def get_tools():
    return [
        StructuredTool.from_function(
            name="SearchProductsDatabase",
            func=search_products_db,
            description=(
                "Search, list, filter, or get product recommendations from the graph database. "
                "Use this for: product listings, budget-based queries, application-based recommendations, "
                "comparison queries (e.g. warm vs cool white), or any query that requires showing multiple products. "
                "The tool auto-detects category, use case, and features from the query. If the user's long-term memory or context specifies a preference (like a specific category), you could explicitly set the corresponding parameter (e.g. `category` or `feature`) rather than relying purely on the current conversational text."
                "\n\nParameters:"
                "\n- query (str): natural language query. Examples: 'indoor ceiling lights', 'solar gate lights', "
                "'garden bollard', 'waterproof outdoor lights', 'driveway lights', 'landscape lights for villa'"
                "\n- category (str): explicit category override, one of: "
                "'Gate & Pillar Lights', 'Solar Lights', 'Outdoor Wall Lights', 'Bollard & Garden Lights', "
                "'Street Lights', 'Flood Lights', 'Indoor & Ceiling Lights', 'Panel Lights', "
                "'Pathway & Step Lights', 'Bulkhead Lights', 'Divine & Temple Lights', 'General Purpose Lights'"
                "\n- collection (str): filter by collection name. Available collections: '3 in 1 gate light', 'Divine Light For Home Entrance', 'Indoor Commercial Lights', 'Indoor Domestic Lights', 'LED Outdoor Wall Light', 'Outdoor Commercial Lights', 'Outdoor Garden Bollard Light', 'Outdoor LED Gate Lamp Lights', 'Outdoor LED Solar Powered Garden Or Street Light Online'"
                "\n- feature (str): one of: solar-powered, waterproof, IP65-rated, IP66-rated, motion-sensor, "
                "dimmable, warm-white, cool-white, neutral-white, 3-in-1-colour, aluminium-body, "
                "polycarbonate-body, surface-mount, wall-mount, rustproof, UV-protected, energy-efficient"
                "\n- spec (str): technical spec filter. Examples: 'IP65', '12W', '18W', 'aluminium', 'beam angle'"
                "\n- min_price / max_price (int): price range in INR (e.g. max_price=10000 for '₹10,000 budget')"
                "\n- sort_by (str): rating_desc, rating_asc, price_asc, price_desc, reviews_desc"
                "\n- limit (int): number of results (default 100)"
                "\n\nEXAMPLES:"
                "\n- 'show me indoor lights' → query='indoor lights'"
                "\n- 'cheapest solar gate light' → query='solar gate', sort_by='price_asc'"
                "\n- 'best rated panel lights' → query='panel lights', sort_by='rating_desc'"
                "\n- 'lights for garden under ₹2000' → query='garden', max_price=2000"
                "\n- 'waterproof outdoor lights with warm white' → query='outdoor warm waterproof'"
                "\n- 'lights for a villa entrance and driveway' → query='gate driveway entrance'"
                "\n- 'recommend lights for a hotel landscape' → query='landscape garden bollard'"
                "\n- 'lights under ₹10000' → max_price=10000, sort_by='rating_desc'"
                "\n- 'IP65 rated street lights' → query='street', spec='IP65'"
                "\n- 'lights for heavy rainfall area' → feature='waterproof'"
                "\n- 'warm white pathway lights' → query='pathway', feature='warm-white'"
                "\n- 'energy efficient gate lights' → query='gate', feature='energy-efficient'"
            ),
            return_direct=False
        ),
        Tool(
            name="GetCategoriesDatabase",
            func=get_categories_db,
            description=(
                "Use this to fetch the list of available product categories from the database. "
                "Call this when the user's request is extremely broad (e.g. 'show me products') "
                "to find out what options exist before asking them a clarifying question."
            )
        ),
        StructuredTool.from_function(
            name="ProductDetailsDatabase",
            func=get_product_details_db,
            description=(
                "Use this when the user asks about ONE specific named product's details: "
                "warranty, wattage, dimensions, material, IP rating, beam angle, lumens, "
                "mounting type, colour temperature, or any other technical specification. "
                "Examples: 'What is the warranty of the Artoo light?', "
                "'Is the Athena light available in warm white?', "
                "'What material is the Tacita fixture made of?', "
                "'What are the dimensions of the Mini Olivia light?'"
            ),
            return_direct=False
        ),
        Tool(
            name="PolicyVectorDatabase",
            func=query_policies,
            description=(
                "Use this for company-wide policies and general operational questions: "
                "shipping, delivery time, delivery charges, order tracking, return policy, "
                "replacement process, exchange process, warranty claim procedure, "
                "bulk pricing, dealer/distributor pricing, contractor rates, "
                "discounts, offers, promotions, coupon codes, "
                "damaged product on arrival, wrong item received, required documents for claims. "
                "Do NOT use for product-specific specs or features."
            )
        ),
        Tool(
            name="ProductAdviceDatabase",
            func=query_product_faqs,
            description=(
                "Use this for general product FAQs, installation guidance, and suitability advice "
                "NOT tied to a specific named product: "
                "'Is installation easy?', 'Can I install it myself?', "
                "'Does the package include mounting hardware?', "
                "'Can it be connected to a timer or smart switch?', "
                "'Is it suitable for coastal areas?', "
                "'How long do LEDs last?', 'What is the expected lifespan?', "
                "'Will switching to LED reduce electricity bill?', "
                "'Which is better: warm white or cool white?', "
                "'Which light requires the least maintenance?', "
                "'Can this be used for commercial spaces?'"
            )
        ),
        Tool(
            name="GeneralKnowledgeDatabase",
            func=query_general_knowledge,
            description=(
                "Use this for educational, comparison, and 'how-to' questions about lighting concepts "
                "that are NOT about a specific product and NOT a company policy. "
                "This searches Inventaa's blog articles and knowledge base. "
                "Examples: "
                "'Wave-Free LED Panel Lights vs Traditional LED Panel Lights', "
                "'What is the difference between bollard and pathway lights?', "
                "'How to choose outdoor lighting?', "
                "'Benefits of solar lights', "
                "'What is colour rendering index (CRI)?', "
                "'How many lumens do I need for outdoor lighting?', "
                "'LED vs fluorescent lights comparison', "
                "'How to reduce electricity bill with LED lighting', "
                "'What is IP rating in outdoor lights?'"
            )
        )
    ]
