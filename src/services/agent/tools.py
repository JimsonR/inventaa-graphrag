import json
import logging
from typing import Optional
from langchain_core.tools import StructuredTool, Tool
from src.services.agent.config import AgentConfig

logger = logging.getLogger(__name__)

def search_products_db(query: Optional[str] = None, spec: Optional[str] = None, min_price: Optional[int] = None, max_price: Optional[int] = None, sort_by: Optional[str] = None, limit: int = 5):
    """
    Search and filter products from the graph database.
    sort_by values: price_asc, price_desc, rating_desc, rating_asc, reviews_desc
    """
    try:
        cypher_query = ""
        params = {"limit": limit}

        tokens = []
        if query:
            tokens = [t.strip() + "~" for t in query.split() if t.strip() and t.lower() not in ["light", "lights", "lamp", "lamps", "product", "products", "show", "rated", "rating", "lowest", "highest", "best", "top"]]
            if tokens:
                lucene_query = " AND ".join(tokens)
                cypher_query += 'CALL db.index.fulltext.queryNodes("product_name_ft", $lucene_query) YIELD node AS p, score\n'
                cypher_query += 'WITH p, score\n'
                params["lucene_query"] = lucene_query
            else:
                cypher_query += "MATCH (p:Product)\n"
        else:
            cypher_query += "MATCH (p:Product)\n"

        where_clauses = []
        if spec:
            cypher_query += "MATCH (p)-[:HAS_SPEC]->(s:Spec)\n"
            where_clauses.append("(toLower(s.key) CONTAINS toLower($spec) OR toLower(s.value) CONTAINS toLower($spec))")
            params["spec"] = spec

        if min_price is not None:
            where_clauses.append("p.price_num >= $min_price")
            params["min_price"] = min_price
        if max_price is not None:
            where_clauses.append("p.price_num <= $max_price")
            params["max_price"] = max_price

        if where_clauses:
            cypher_query += "WHERE " + " AND ".join(where_clauses) + "\n"

        # Normalise sort_by synonym → canonical ORDER BY clause
        sort_clause = ""
        sort_by_lower = (sort_by or "").lower().strip()
        if sort_by_lower in ["price_asc", "price_low", "cheapest", "lowest_price", "low_price"]:
            sort_clause = "ORDER BY p.price_num ASC"
        elif sort_by_lower in ["price_desc", "price_high", "expensive", "highest_price", "high_price"]:
            sort_clause = "ORDER BY p.price_num DESC"
        elif sort_by_lower in ["rating_desc", "rating", "highest_rated", "best_rated", "top_rated", "high_rating"]:
            sort_clause = "ORDER BY p.rating_score DESC"
        elif sort_by_lower in ["rating_asc", "lowest_rated", "low_rating", "worst_rated", "least_rated"]:
            sort_clause = "ORDER BY p.rating_score ASC"
        elif sort_by_lower in ["reviews_desc", "reviews", "most_reviewed", "most_popular"]:
            sort_clause = "ORDER BY p.review_count DESC"
        elif tokens:  # default: relevance score from full-text search
            sort_clause = "ORDER BY score DESC"

        cypher_query += f"""
        RETURN p.sku AS sku, p.name AS name, p.price_num AS price_num,
               p.regular_price AS regular_price, p.discount_percentage AS discount_percentage,
               p.image_url AS image_url, p.url AS url, p.rating_score AS rating,
               p.review_count AS review_count, p.tenant AS tenant, p.feature_descriptions AS feature_descriptions
        {sort_clause}
        LIMIT $limit
        """

        logger.info(f"SearchProductsDatabase | query={query!r} spec={spec!r} sort_by={sort_by!r} limit={limit} | Cypher: {cypher_query.strip()}")
        res = AgentConfig.graph.query(cypher_query, params=params)
        
        if not res:
            return "[]"
            
        return json.dumps(res, indent=2, ensure_ascii=False)
    except Exception as e:
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
    # We search the dedicated policy index first
    results = AgentConfig.policy_vector_store.similarity_search_with_score(query, k=2)
    if not results:
        # Fallback to the general chunk vector store if no policy matches
        results = AgentConfig.general_vector_store.similarity_search_with_score(query, k=2)
        
    if not results:
        return "No relevant policy found."
    text = "\n\n".join([doc.page_content for doc, _ in results])
    # Encode to ASCII-safe to prevent UnicodeEncodeError on Windows console
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
                "Use this to SEARCH, LIST, or FILTER products. "
                "Parameters: "
                "query (str) - product name or category keywords (e.g. 'garden bollard', 'gate light'); "
                "spec (str) - technical specification to filter by (e.g. 'IP65', '12W', 'aluminum'); "
                "min_price / max_price (int) - price range in INR; "
                "sort_by (str) - one of: 'rating_desc' (highest rated), 'rating_asc' (lowest rated), "
                "'price_asc' (cheapest), 'price_desc' (most expensive), 'reviews_desc' (most reviewed); "
                "limit (int) - number of results (default 5). "
                "IMPORTANT: for 'lowest rated' use sort_by='rating_asc'; for 'highest rated' use sort_by='rating_desc'. "
                "Always pass spec separately from query — do NOT put spec values inside query."
            ),
            return_direct=True
        ),
        StructuredTool.from_function(
            name="ProductDetailsDatabase",
            func=get_product_details_db,
            description="Use this when the user asks a specific question about a product that requires a conversational sentence (e.g., 'What is the warranty of the Artoo light?', 'Tell me about its features.').",
            return_direct=False
        ),
        Tool(
            name="PolicyVectorDatabase",
            func=query_policies,
            description="Use this ONLY for general company-wide policies (e.g., general shipping, return, replacement, exchange, or warranty rules). DO NOT use this for finding product-specific features or warranties."
        ),
        Tool(
            name="ProductAdviceDatabase",
            func=query_product_faqs,
            description="Use this to answer conversational questions, FAQs, installation instructions, usage suitability (e.g. 'Is it suitable for commercial properties?'), or troubleshooting for a specific product."
        )
    ]
