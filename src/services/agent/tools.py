import json
import logging
import re
import time
import os
from typing import Optional, List, Dict, Any
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
    """
    Search and filter products from the Neo4j graph database.
    Use this tool to find, show, list, filter, or recommend products.
    """
    from src.services.agent.utils import track_time
    tool_logger = logging.getLogger("Tool.SearchProducts")
    with track_time("Tool: SearchProductsDatabase", custom_logger=tool_logger):
        pass

    try:
        params = {"limit": limit}

        logger.info(f"SearchProducts | query={query!r} | category={category!r} "
                    f"| use_case={use_case!r} | feature={feature!r}")

        cypher_query = ""
        where_clauses = []
        
        from src.services.agent.context import tenant_context
        tenant_id = tenant_context.get()
        if tenant_id:
            where_clauses.append("(p.tenant = $tenant_id OR p.tenant IS NULL)")
            params["tenant_id"] = tenant_id

        # ─── 1. Full-Text Search on Product Name for raw query ──────────────────
        if query and query.strip():
            _STOP = {
                "led", "light", "lights", "lamp", "lamps", "the", "a", "an", "and", "or", "for", "of", "in", "with", 
                "by", "from", "lighting", "offers", "offer", "discount", "discounts", "sale", "deal", "deals", 
                "best", "top", "cheap", "cheapest", "buy", "show", "me", "any", "on"
            }
            raw = [t.strip(".,?!-–—/|") for t in query.split()]
            good_tokens = [t.lower() for t in raw if t and len(t) > 2 and t.lower() not in _STOP]
            
            if good_tokens:
                lucene_query = " AND ".join([t + "~" for t in good_tokens])
                cypher_query += 'CALL db.index.fulltext.queryNodes("product_name_ft", $lucene_query) YIELD node AS p, score\n'
                params["lucene_query"] = lucene_query
            else:
                cypher_query += "MATCH (p:Product)\n"
        else:
            cypher_query += "MATCH (p:Product)\n"

        # ─── 2. Structural Graph Matching ───────────────────────────────────────
        target_col = category or collection
        if target_col:
            cypher_query += "MATCH (p)-[:BELONGS_TO_COLLECTION]->(c:Collection {name: $collection})\n"
            params["collection"] = target_col
            
        if feature:
            cypher_query += "MATCH (p)-[:HAS_FEATURE]->(f:Feature {name: $feature})\n"
            params["feature"] = feature
            
        if use_case:
            cypher_query += "MATCH (p)-[:SUITABLE_FOR]->(u:UseCase {name: $use_case})\n"
            params["use_case"] = use_case

        # ─── 3. Add Technical Spec filter if provided ───────────────────────────
        if spec:
            cypher_query += "MATCH (p)-[:HAS_SPEC]->(s:Spec)\n"
            where_clauses.append("(toLower(s.key) CONTAINS toLower($spec) OR toLower(s.value) CONTAINS toLower($spec))")
            params["spec"] = spec

        # ─── 4. Add Price filters if provided ───────────────────────────────────
        if min_price is not None:
            where_clauses.append("p.price_num >= $min_price")
            params["min_price"] = min_price
        if max_price is not None:
            where_clauses.append("p.price_num <= $max_price")
            params["max_price"] = max_price

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
            "relevance": "ORDER BY score DESC"
        }
        order_clause = sort_map.get((sort_by or "").lower(), "")
        if not order_clause:
            order_clause = "ORDER BY score DESC, p.rating_score DESC" if "lucene_query" in params else "ORDER BY p.rating_score DESC"

        cypher_query += f"""
WITH p
{order_clause}
LIMIT $limit
RETURN p.sku AS sku, p.name AS name, p.price_num AS price_num,
       p.regular_price AS regular_price, p.discount_percentage AS discount_percentage,
       p.image_url AS image_url, p.url AS url, p.rating_score AS rating,
       p.review_count AS review_count, p.tenant AS tenant, p.feature_descriptions AS feature_descriptions,
       p.installation_url AS installation_url,
       [(p)-[:HAS_WARRANTY]->(w:Warranty) | w.description][0] AS warranty,
       [(p)-[:HAS_POLICY]->(pol:Policy) WHERE toLower(pol.title) CONTAINS 'replacement' OR toLower(pol.title) CONTAINS 'exchange' | pol.content][0] AS replacement_exchange_policy,
       [(p)-[:BELONGS_TO_COLLECTION]->(col2:Collection) | col2.name] AS collections
"""

        logger.info(f"SearchProductsDatabase Cypher:\n{cypher_query.strip()}\nParams: {params}")
        res = AgentConfig.graph.query(cypher_query, params=params)

        if not res:
            return "[]"

        # Graph-driven routing logic
        all_collections = set()
        for row in res:
            cols = row.get("collections") or []
            all_collections.update(cols)
        
        # Intercept if the query matches products across multiple distinct collections and is large
        if len(all_collections) > 2 and len(res) >= 10 and not category and not collection:
            return json.dumps({
                "status": "needs_clarification",
                "message": "The query matched many products across multiple different collections. Ask the user to narrow down their choice from the available collections.",
                "available_collections": sorted(list(all_collections))
            }, indent=2, ensure_ascii=False)

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
        RETURN p.name AS name, p.price_num AS price,
               p.feature_descriptions AS feature_descriptions,
               p.installation_url AS installation_url,
               [(p)-[:HAS_POLICY]->(pol:Policy) WHERE toLower(pol.title) CONTAINS 'replacement' OR toLower(pol.title) CONTAINS 'exchange' | pol.content][0] AS replacement_exchange_policy,
               [(p)-[:HAS_WARRANTY]->(w:Warranty) | w.description][0] AS warranty_info,
               [(p)-[:HAS_WARRANTY]->(w:Warranty) | w.duration_years][0] AS warranty_duration,
               [(p)-[:HAS_SPEC]->(s:Spec) | s.key + ': ' + s.value] AS specs,
               [(p)-[:AVAILABLE_IN_WATTAGE]->(wo:WattageOption) | wo.name] AS wattages,
               [(p)-[:AVAILABLE_IN_COLOR]->(co:ColorOption) | co.name] AS colors,
               [(p)-[:BELONGS_TO_COLLECTION]->(col:Collection) | col.name] AS collections
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
    cols_str = "', '".join(AgentConfig.collections) if AgentConfig.collections else "Solar Lights"
    feats_str = ", ".join(AgentConfig.features) if AgentConfig.features else "solar-powered"
    
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
                f"\n- category (str): explicit collection override, one of: '{cols_str}'"
                f"\n- collection (str): filter by collection name. Available collections: '{cols_str}'"
                f"\n- feature (str): one of: {feats_str}"
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
