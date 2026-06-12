import json
import logging
from typing import Optional
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
    "gate": "Gate & Pillar Lights",
    "pillar": "Gate & Pillar Lights",
    "solar": "Solar Lights",
    "wall": "Outdoor Wall Lights",
    "bollard": "Bollard & Garden Lights",
    "garden": "Bollard & Garden Lights",
    "street": "Street Lights",
    "road": "Street Lights",
    "flood": "Flood Lights",
    "indoor": "Indoor & Ceiling Lights",
    "ceiling": "Indoor & Ceiling Lights",
    "panel": "Panel Lights",
    "pathway": "Pathway & Step Lights",
    "step": "Pathway & Step Lights",
    "stairway": "Pathway & Step Lights",
    "bulkhead": "Bulkhead Lights",
    "divine": "Divine & Temple Lights",
    "temple": "Divine & Temple Lights",
    "religious": "Divine & Temple Lights",
    "god": "Divine & Temple Lights",
    "general": "General Purpose Lights",
}

USECASE_KEYWORDS = {
    "gate": "gate-pillar",
    "pillar": "gate-pillar",
    "indoor": "indoor-ceiling",
    "ceiling": "indoor-ceiling",
    "wall": "outdoor-wall",
    "garden": "garden-pathway",
    "pathway": "pathway-step",
    "step": "pathway-step",
    "stairway": "pathway-step",
    "street": "street-road",
    "road": "street-road",
    "flood": "flood-area",
    "solar": "solar-outdoor",
    "temple": "religious-decorative",
    "divine": "religious-decorative",
    "religious": "religious-decorative",
}

FEATURE_KEYWORDS = {
    "solar": "solar-powered",
    "waterproof": "waterproof",
    "motion": "motion-sensor",
    "sensor": "motion-sensor",
    "dimmable": "dimmable",
    "warm": "warm-white",
    "cool": "cool-white",
    "neutral": "neutral-white",
    "3-in-1": "3-in-1-colour",
    "aluminium": "aluminium-body",
    "aluminum": "aluminium-body",
    "polycarbonate": "polycarbonate-body",
    "surface": "surface-mount",
    "rustproof": "rustproof",
    "uv": "UV-protected",
    "indoor": "indoor",
    "outdoor": "outdoor",
}

# Stop words to ignore when tokenizing queries
_STOP_WORDS = {"light", "lights", "lamp", "lamps", "led", "product", "products",
               "show", "me", "get", "find", "list", "give", "want", "need",
               "rated", "rating", "lowest", "highest", "best", "top", "some",
               "a", "an", "the", "for", "with", "of", "in", "and", "or"}


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
    use_case: Optional[str] = None,
    feature: Optional[str] = None,
    spec: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    sort_by: Optional[str] = None,
    limit: int = 5
):
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

        where_clauses = []

        # UseCase filter
        if final_usecase:
            cypher_query += "MATCH (p)-[:SUITABLE_FOR]->(uc:UseCase {name: $use_case})\n"
            params["use_case"] = final_usecase

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

        # Fuzzy name search on remaining tokens (only if no category matched)
        lucene_query = None
        if remaining_tokens and not final_category:
            lucene_tokens = [t + "~" for t in remaining_tokens if t]
            if lucene_tokens:
                lucene_query = " AND ".join(lucene_tokens)

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
RETURN DISTINCT p.sku AS sku, p.name AS name, p.price_num AS price_num,
       p.regular_price AS regular_price, p.discount_percentage AS discount_percentage,
       p.image_url AS image_url, p.url AS url, p.rating_score AS rating,
       p.review_count AS review_count, p.tenant AS tenant, p.feature_descriptions AS feature_descriptions
{sort_clause}
LIMIT $limit
"""

        logger.info(f"SearchProductsDatabase Cypher:\n{cypher_query.strip()}\nParams: {params}")
        res = AgentConfig.graph.query(cypher_query, params=params)

        # Fallback: if category matched but no results, try full-text on remaining tokens
        if not res and lucene_query:
            logger.info(f"Category search returned 0 results, falling back to full-text: {lucene_query}")
            ft_query = f"""
CALL db.index.fulltext.queryNodes("product_name_ft", $lucene_query) YIELD node AS p, score
WITH p, score
{('WHERE ' + ' AND '.join(where_clauses)) if where_clauses else ''}
RETURN DISTINCT p.sku AS sku, p.name AS name, p.price_num AS price_num,
       p.regular_price AS regular_price, p.discount_percentage AS discount_percentage,
       p.image_url AS image_url, p.url AS url, p.rating_score AS rating,
       p.review_count AS review_count, p.tenant AS tenant, p.feature_descriptions AS feature_descriptions
ORDER BY score DESC
LIMIT $limit
"""
            params["lucene_query"] = lucene_query
            res = AgentConfig.graph.query(ft_query, params=params)

        if not res:
            return "[]"

        return json.dumps(res, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in SearchProductsDatabase: {e}", exc_info=True)
        return f"Error querying graph: {e}"


def get_product_details_db(product_name: str):
    try:
        tokens = [t.strip() + "~" for t in product_name.split() if t.strip()]
        if not tokens:
            return "Please provide a valid product name."
        lucene_query = " AND ".join(tokens)

        cypher_query = """
        CALL db.index.fulltext.queryNodes("product_name_ft", $lucene_query) YIELD node AS p, score
        WITH p, score
        ORDER BY score DESC LIMIT 1
        OPTIONAL MATCH (p)-[:HAS_WARRANTY]->(w:Warranty)
        OPTIONAL MATCH (p)-[:HAS_SPEC]->(s:Spec)
        RETURN p.name AS name, p.price_num AS price, p.feature_descriptions AS feature_descriptions,
               w.description AS warranty_info, w.duration_years AS warranty_duration,
               collect(s.key + ': ' + s.value) AS specs
        """
        params = {"lucene_query": lucene_query}
        logger.info(f"ProductDetailsDatabase executing Cypher: {cypher_query} | Params: {params}")
        res = AgentConfig.graph.query(cypher_query, params=params)

        if not res:
            return "Product not found."

        product = res[0]
        output = f"Product Name: {product.get('name')}\n"
        output += f"Price: {product.get('price')}\n"
        output += f"Features: {product.get('feature_descriptions')}\n"
        if product.get('warranty_info'):
            output += f"Warranty: {product.get('warranty_info')} ({product.get('warranty_duration')} years)\n"
        if product.get('specs'):
            output += f"Specifications: {', '.join(product.get('specs'))}\n"

        return output.encode("ascii", errors="ignore").decode("ascii")
    except Exception as e:
        return f"Error getting product details: {e}"


def query_policies(query: str):
    results = AgentConfig.policy_vector_store.similarity_search_with_score(query, k=2)
    if not results:
        results = AgentConfig.general_vector_store.similarity_search_with_score(query, k=2)
    if not results:
        return "No relevant policy found."
    text = "\n\n".join([doc.page_content for doc, _ in results])
    return text.encode("ascii", errors="ignore").decode("ascii")


def query_product_faqs(query: str):
    results = AgentConfig.product_faq_vector_store.similarity_search_with_score(query, k=2)
    if not results:
        return "No relevant product FAQ found."
    text = "\n\n".join([doc.page_content for doc, _ in results])
    return text.encode("ascii", errors="ignore").decode("ascii")


def get_tools():
    return [
        StructuredTool.from_function(
            name="SearchProductsDatabase",
            func=search_products_db,
            description=(
                "Search, list, or filter products from the graph database. "
                "The tool auto-detects category, use case, and features from the query — you usually only need to pass `query`. "
                "Parameters: "
                "query (str) - natural language product query (e.g. 'indoor ceiling lights', 'solar gate lights', 'garden bollard'); "
                "category (str) - explicit category override, one of: "
                "'Gate & Pillar Lights', 'Solar Lights', 'Outdoor Wall Lights', 'Bollard & Garden Lights', "
                "'Street Lights', 'Flood Lights', 'Indoor & Ceiling Lights', 'Panel Lights', "
                "'Pathway & Step Lights', 'Bulkhead Lights', 'Divine & Temple Lights', 'General Purpose Lights'; "
                "use_case (str) - one of: gate-pillar, indoor-ceiling, outdoor-wall, garden-pathway, "
                "pathway-step, street-road, flood-area, solar-outdoor, religious-decorative; "
                "feature (str) - one of: solar-powered, waterproof, IP65-rated, motion-sensor, dimmable, "
                "warm-white, cool-white, 3-in-1-colour, aluminium-body, polycarbonate-body, surface-mount; "
                "spec (str) - technical spec filter (e.g. 'IP65', '12W'); "
                "min_price / max_price (int) - price range in INR; "
                "sort_by (str) - rating_desc, rating_asc, price_asc, price_desc, reviews_desc; "
                "limit (int) - number of results (default 5). "
                "EXAMPLES: "
                "- 'show me indoor lights' → query='indoor lights' "
                "- 'cheapest solar gate light' → query='solar gate', sort_by='price_asc' "
                "- 'best rated panel lights' → query='panel lights', sort_by='rating_desc'"
            ),
            return_direct=False
        ),
        StructuredTool.from_function(
            name="ProductDetailsDatabase",
            func=get_product_details_db,
            description="Use this when the user asks a specific question about ONE product requiring a conversational sentence (e.g., 'What is the warranty of the Artoo light?', 'Tell me about its features.').",
            return_direct=False
        ),
        Tool(
            name="PolicyVectorDatabase",
            func=query_policies,
            description="Use this ONLY for general company-wide policies (e.g., general shipping, return, replacement, exchange, or warranty rules). DO NOT use this for product-specific features."
        ),
        Tool(
            name="ProductAdviceDatabase",
            func=query_product_faqs,
            description="Use this to answer conversational FAQs, installation instructions, usage suitability, or troubleshooting for a specific product."
        )
    ]
