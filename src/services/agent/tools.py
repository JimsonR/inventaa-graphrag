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
    collections: Optional[list[str]] = None,
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

        # 1. Full-Text Search
        if query:
            _STOP = {
                "led", "light", "lights", "lamp", "lamps", "the", "a", "an", "and", "or", "for", "of", "in", "with", 
                "by", "from", "lighting", "offers", "offer", "discount", "discounts", "sale", "deal", "deals", 
                "best", "top", "cheap", "cheapest", "buy", "show", "me", "any", "on", "indoor", "outdoor",
                "product", "products"
            }
            raw = [t.strip(".,?!-–—/|") for t in query.split()]
            good_tokens = [t.lower() for t in raw if t and len(t) > 2 and t.lower() not in _STOP]
            
            # ── Broad Query Defense ──────────────────────────────────────────────
            # If ALL tokens were stripped (query was entirely generic like
            # "show me products"), AND no category/collection filter was provided,
            # return a needs_clarification response instead of doing a blind
            # MATCH (p:Product) that returns everything or a broken Lucene search.
            if not good_tokens and not category and not collection:
                available = AgentConfig.collections if AgentConfig.collections else []
                logger.info(f"[BroadQueryDefense] All tokens stripped from query={query!r}. Returning needs_clarification.")
                return json.dumps({
                    "needs_clarification": True,
                    "message": "The query is too broad. Please ask the user which specific collection they'd like to browse.",
                    "available_collections": available
                })

            if good_tokens:
                # Use prefix matching instead of fuzzy to prevent 'indoor' from matching 'door' or 'outdoor'
                lucene_query = " AND ".join([t + "*" for t in good_tokens])
                cypher_query += 'CALL db.index.fulltext.queryNodes("product_name_ft", $lucene_query) YIELD node AS p, score\n'
                params["lucene_query"] = lucene_query
            else:
                cypher_query += "MATCH (p:Product)\n"
        else:
            cypher_query += "MATCH (p:Product)\n"

        # ─── 2. Structural Graph Matching ───────────────────────────────────────
        target_collections = collections or []
        target_col = category or collection
        if target_col and target_col not in target_collections:
            target_collections.append(target_col)
            
        if target_collections:
            cypher_query += "MATCH (p)-[:BELONGS_TO_COLLECTION]->(c:Collection)\n"
            where_clauses.append("toLower(c.name) IN [x IN $target_collections | toLower(x)]")
            params["target_collections"] = target_collections
            
        if feature:
            cypher_query += "MATCH (p)-[:HAS_FEATURE]->(f:Feature)\n"
            where_clauses.append("toLower(f.name) = toLower($feature)")
            params["feature"] = feature
            
        if use_case:
            cypher_query += "MATCH (p)-[:SUITABLE_FOR]->(u:UseCase)\n"
            where_clauses.append("toLower(u.name) = toLower($use_case)")
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
        sort_by_lower = (sort_by or "").lower()
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
        # We handle wattage_asc and wattage_desc in Python after the query
        order_clause = sort_map.get(sort_by_lower, "")
        if not order_clause and "wattage" not in sort_by_lower:
            order_clause = "ORDER BY score DESC, p.rating_score DESC" if "lucene_query" in params else "ORDER BY p.rating_score DESC"

        cypher_query += f"""
WITH p{", score" if "lucene_query" in params else ""}
{order_clause}
LIMIT $limit
RETURN p.sku AS sku, p.name AS name, p.price_num AS price_num,
       p.regular_price AS regular_price, p.discount_percentage AS discount_percentage,
       p.image_url AS image_url, p.url AS url, p.rating_score AS rating,
       p.review_count AS review_count, p.tenant AS tenant, p.feature_descriptions AS feature_descriptions,
       p.installation_url AS installation_url,
       [(p)-[:HAS_WARRANTY]->(w:Warranty) | w.description][0] AS warranty,
       [(p)-[:HAS_POLICY]->(pol:Policy) WHERE toLower(pol.title) CONTAINS 'replacement' OR toLower(pol.title) CONTAINS 'exchange' | pol.content][0] AS replacement_exchange_policy,
       [(p)-[:HAS_SPEC]->(s:Spec) WHERE toLower(s.key) CONTAINS 'watt' OR toLower(s.key) CONTAINS 'power' | s.value] AS wattages,
       [(p)-[:BELONGS_TO_COLLECTION]->(col2:Collection) | col2.name] AS collections,
       [(p)-[:BELONGS_TO_TENANT]->(t:Tenant)-[:HAS_GLOBAL_OFFER]->(o:GlobalOffer) | o.text][0] AS global_offers
"""

        logger.info(f"SearchProductsDatabase Cypher:\n{cypher_query.strip()}\nParams: {params}")
        res = AgentConfig.graph.query(cypher_query, params=params)

        # ── Query Relaxation 1: AND to OR ─────────────────────────────────────────
        if not res and "lucene_query" in params and " AND " in params["lucene_query"]:
            relaxed_lucene = params["lucene_query"].replace(" AND ", " OR ")
            logger.info(f"[QueryRelaxation] 0 results for AND. Relaxing to OR query: {relaxed_lucene}")
            params["lucene_query"] = relaxed_lucene
            res = AgentConfig.graph.query(cypher_query, params=params)
            
        # ── Query Relaxation 2: Drop Lucene entirely if structural filters exist ──
        if not res and "lucene_query" in params and (category or collection or feature or use_case or spec):
            logger.info(f"[QueryRelaxation] 0 results for OR. Dropping full-text query entirely since structural filters exist.")
            # Remove the lucene query from the cypher string
            relaxed_cypher = cypher_query.replace('CALL db.index.fulltext.queryNodes("product_name_ft", $lucene_query) YIELD node AS p, score\n', 'MATCH (p:Product)\n')
            # Remove score from the WITH clause
            relaxed_cypher = relaxed_cypher.replace('WITH p, score', 'WITH p')
            # Remove score from the ORDER BY clause if present
            relaxed_cypher = relaxed_cypher.replace('score DESC, ', '')
            relaxed_cypher = relaxed_cypher.replace('ORDER BY score DESC', 'ORDER BY p.rating_score DESC')
            
            res = AgentConfig.graph.query(relaxed_cypher, params=params)
            
        import re
        if res and "wattage" in (sort_by or "").lower():
            def extract_wattage(row):
                wattages = row.get("wattages") or []
                if not wattages:
                    return float('inf') if "asc" in (sort_by or "").lower() else float('-inf')
                match = re.search(r'(\d+(?:\.\d+)?)', wattages[0])
                if match:
                    return float(match.group(1))
                return float('inf') if "asc" in (sort_by or "").lower() else float('-inf')
            res.sort(key=extract_wattage, reverse="desc" in (sort_by or "").lower())

        if not res:
            # ── Fulltext Fallback ─────────────────────────────────────────
            # Lucene matched 0 products. If we had a lucene_query, the tokens
            # may have been valid but not in any product *name*. Fall back to
            # asking for clarification rather than silently returning empty.
            if "lucene_query" in params and not category and not collection:
                available = AgentConfig.collections if AgentConfig.collections else []
                logger.info(f"[FulltextFallback] Lucene query={params['lucene_query']!r} returned 0 results. Returning needs_clarification.")
                return json.dumps({
                    "needs_clarification": True,
                    "message": f"No products matched '{query}'. Please ask the user to pick a specific collection.",
                    "available_collections": available
                })
            return "[]"

        # Graph-driven routing logic
        all_collections = set()
        for row in res:
            cols = row.get("collections") or []
            all_collections.update(cols)
        
        # Intercept if the query matches products across multiple distinct collections and is large
        if len(all_collections) > 2 and len(res) >= 10 and not category and not collection and not collections:
            logger.info(f"[MultiCollectionFallback] Query matched {len(res)} products across {len(all_collections)} collections. Returning needs_clarification.")
            return json.dumps({
                "needs_clarification": True,
                "message": "I found many products matching your request across several different collections. Could you please specify which type of light you're looking for?",
                "available_collections": sorted(list(all_collections))
            })

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
               [(p)-[:BELONGS_TO_COLLECTION]->(col:Collection) | col.name] AS collections,
               [(p)-[:BELONGS_TO_TENANT]->(t:Tenant)-[:HAS_GLOBAL_OFFER]->(o:GlobalOffer) | o.text][0] AS global_offers
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

        # Global Offers
        global_offers = product.get('global_offers')
        if global_offers:
            output += f"Global Offers: {global_offers}\n"

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
    
    from concurrent.futures import ThreadPoolExecutor
    
    def fetch_offers():
        try:
            from src.services.agent.context import tenant_context
            tenant_id = tenant_context.get()
            offer_res = AgentConfig.graph.query(
                "MATCH (t:Tenant)-[:HAS_GLOBAL_OFFER]->(o:GlobalOffer) WHERE (t.id = $tenant_id OR $tenant_id IS NULL) RETURN o.text AS text LIMIT 1",
                params={"tenant_id": tenant_id}
            )
            if offer_res and offer_res[0].get("text"):
                return offer_res[0]["text"] + "\n\n---\n"
        except Exception as e:
            logger.error(f"Error fetching global offers: {e}")
        return ""

    def fetch_vectors():
        res = AgentConfig.policy_vector_store.similarity_search_with_score(query, k=2, filter=filter_dict)
        if not res:
            if AgentConfig.general_vector_store is None:
                NEO4J_URI = os.getenv("NEO4J_URI", "").replace("neo4j+s://", "neo4j+ssc://")
                AgentConfig.general_vector_store = Neo4jVector.from_existing_index(
                    embedding=AgentConfig.embeddings, url=NEO4J_URI, 
                    username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"),
                    index_name="inventaa_faq_vector", text_node_property="text"
                )
            res = AgentConfig.general_vector_store.similarity_search_with_score(query, k=2, filter=filter_dict)
        return res

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_offers = executor.submit(fetch_offers)
        future_vectors = executor.submit(fetch_vectors)
        
        global_offers_text = future_offers.result()
        results = future_vectors.result()

    if not results:
        return global_offers_text + "No relevant policy found."
        
    text = "\n\n".join([doc.page_content for doc, _ in results])
    text = global_offers_text + text

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
    Search blog articles, offers, and general lighting knowledge stored as Chunk nodes.
    Uses vector search on the inventaa_faq_vector index.
    """
    try:
        from langchain_neo4j import Neo4jVector
        import os
        
        if AgentConfig.general_vector_store is None:
            NEO4J_URI = os.getenv("NEO4J_URI", "").replace("neo4j+s://", "neo4j+ssc://")
            AgentConfig.general_vector_store = Neo4jVector.from_existing_index(
                embedding=AgentConfig.embeddings, url=NEO4J_URI, 
                username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"),
                index_name="inventaa_faq_vector", text_node_property="text"
            )
            
        from src.services.agent.context import tenant_context
        tenant_id = tenant_context.get()
        filter_dict = {"tenant": tenant_id} if tenant_id else None
        
        logger.info(f"GeneralKnowledgeDatabase | vector_search={query!r}")
        from concurrent.futures import ThreadPoolExecutor
        
        def fetch_offers():
            try:
                offer_res = AgentConfig.graph.query(
                    "MATCH (t:Tenant)-[:HAS_GLOBAL_OFFER]->(o:GlobalOffer) WHERE (t.id = $tenant_id OR $tenant_id IS NULL) RETURN o.text AS text LIMIT 1",
                    params={"tenant_id": tenant_id}
                )
                if offer_res and offer_res[0].get("text"):
                    return offer_res[0]["text"] + "\n\n---\n"
            except Exception as e:
                logger.error(f"Error fetching global offers: {e}")
            return ""

        def fetch_vectors():
            return AgentConfig.general_vector_store.similarity_search_with_score(query, k=3, filter=filter_dict)
            
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_offers = executor.submit(fetch_offers)
            future_vectors = executor.submit(fetch_vectors)
            
            global_offers_text = future_offers.result()
            results = future_vectors.result()
        
        if not results:
            return global_offers_text + "No relevant articles found."
            
        combined = []
        for doc, score in results:
            text = doc.page_content or ""
            text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
            text = re.sub(r"\[.*?\]\(.*?\)", "", text)
            text = re.sub(r"---.*?---", "", text, flags=re.DOTALL)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 50:
                combined.append(text[:1500])
                
        output = "\n\n---\n".join(combined)
        output = global_offers_text + output

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
                "The tool auto-detects category, use case, and features from the query. If the user's long-term memory or context specifies a preference (like a specific category), you could explicitly set the corresponding parameter (e.g. `category` or `feature`) rather than relying purely on the current conversational text. "
                "CRITICAL: If the user asks to search 'in all collections' or 'across all categories', DO NOT call this tool multiple times. Simply call it ONCE with `category=None` and `collection=None`."
                "\n\nParameters:"
                "\n- query (str): ONLY use this for exact PROPER NOUNS representing specific product names or brands (e.g. 'oxana', 'fabra'). DO NOT pass generic words, locations, applications, or features here (like 'pathway', 'walkway', 'garden', 'indoor', 'waterproof'). If you have a generic term, use the `feature` or `use_case` parameters instead and LEAVE THIS EMPTY."
                "\n- category (str): explicit collection override."
                "\n- collection (str): filter by collection name."
                "\n- collections (list[str]): filter by multiple collections at once (e.g. for a broad category group like 'Outdoor')."
                f"\n- feature (str): explicit filter for a physical feature. Valid options: {', '.join(AgentConfig.features) if AgentConfig.features else 'waterproof, dimmable'}."
                f"\n- use_case (str): explicit filter for where the light will be used. Valid options: {', '.join(AgentConfig.use_cases) if AgentConfig.use_cases else 'Garden, Exterior'}."
                "\n- spec (str): technical spec filter. Examples: 'IP65', '12W', '18W', 'aluminium', 'beam angle'"
                "\n- min_price / max_price (int): price range in INR (e.g. max_price=10000 for '₹10,000 budget')"
                "\n- sort_by (str): rating_desc, rating_asc, price_asc, price_desc, reviews_desc, wattage_asc, wattage_desc"
                "\n- limit (int): number of results (default 100)"
                "\n\nEXAMPLES:"
                "\n- 'show me waterproof lights for the garden' → query=None, feature='Waterproof', use_case='Garden'"
                "\n- 'low electricity exterior lights' → query=None, use_case='Exterior', sort_by='wattage_asc'"
                "\n- 'cheapest solar gate light' → query=None, category='Solar Lights', use_case='Gate-Pillar', sort_by='price_asc'"
                "\n- 'lights for garden under ₹2000' → query=None, use_case='Garden', max_price=2000"
                "\n- 'oxana wall light' → query='oxana', category='LED Outdoor Wall Light', use_case=None"
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
                "Use this for company-wide operational policies: "
                "Use this for company-wide operational policies: "
                "shipping, delivery time, delivery charges, order tracking, return policy, "
                "replacement process, exchange process, warranty claim procedure, "
                "bulk pricing, dealer/distributor pricing, contractor rates, "
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
                "Use this to search for active offers, discounts, promotions, and coupon codes. "
                "Also use this for educational, comparison, and 'how-to' questions about lighting concepts "
                "that are NOT about a specific product and NOT a company policy. "
                "This searches Inventaa's blog articles and knowledge base. "
                "Examples: "
                "'Are there any offers on solar lights?', "
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
