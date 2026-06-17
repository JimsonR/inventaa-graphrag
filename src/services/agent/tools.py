import json
import logging
import re
from typing import Optional
from langchain_core.tools import StructuredTool, Tool
from src.services.agent.config import AgentConfig

logger = logging.getLogger(__name__)


def filter_graph_entities(
    filters: Optional[list] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    sort_by: Optional[str] = None,
    limit: int = 100
):
    """
    Filter the central entity (e.g. Product) using connected metadata labels.
    filters should be a list of dicts: [{"label": "Category", "value": "Solar Lights"}, ...]
    """
    try:
        params = {"limit": limit}
        where_clauses = []
        
        from src.services.agent.context import tenant_context
        tenant_id = tenant_context.get()
        if tenant_id:
            where_clauses.append("target.tenant = $tenant_id")
            params["tenant_id"] = tenant_id

        from src.services.agent.routing import load_tenant_configs
        configs = load_tenant_configs()
        tenant_conf = configs.get(tenant_id, configs.get("default", {}))
        
        primary_entity = tenant_conf.get("primary_entity", "Product")
        cypher_query = f"MATCH (target:{primary_entity})\n"
        
        if filters:
            for i, f in enumerate(filters):
                label = f.get("label")
                val = f.get("value")
                if label and val:
                    # Generic undirected match dynamically finds the metadata node
                    cypher_query += f"MATCH (target)--(f{i}:{label} {{name: $val_{i}}})\n"
                    params[f"val_{i}"] = val

        # Price filters using config-driven property name
        price_prop = tenant_conf.get("price_property", "price_num")
        if min_price is not None:
            where_clauses.append(f"target.{price_prop} >= $min_price")
            params["min_price"] = min_price
        if max_price is not None:
            where_clauses.append(f"target.{price_prop} <= $max_price")
            params["max_price"] = max_price

        if where_clauses:
            cypher_query += "WHERE " + " AND ".join(where_clauses) + "\n"


        # Sort
        sort_map = tenant_conf.get("sort_map", {})
        default_sort = tenant_conf.get("default_sort", "")
        sort_clause = sort_map.get((sort_by or "").lower(), default_sort)

        cypher_query += f"""
WITH DISTINCT target
{sort_clause}
RETURN properties(target) AS details
LIMIT $limit
"""
        logger.info(f"FilterGraphEntities Cypher:\n{cypher_query.strip()}\nParams: {params}")
        res = AgentConfig.graph.query(cypher_query, params=params)
        
        if not res:
            return "[]"
            
        # Flatten the 'details' dict so the final JSON is a flat list of properties 
        # just like the old hardcoded query format, which frontends/clients expect.
        flat_res = [row.get("details", {}) for row in res]
            
        return json.dumps(flat_res, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error in filter_graph_entities: {e}", exc_info=True)
        return f"Error querying graph: {e}"


def get_entity_details_db(entity_name: str):
    try:
        from src.services.agent.context import tenant_context
        from src.services.agent.routing import load_tenant_configs
        tenant_id = tenant_context.get()
        
        configs = load_tenant_configs()
        tenant_conf = configs.get(tenant_id, configs.get("default", {}))
        
        # Get stop words and index from config, or fallback to defaults
        stop_words = set(tenant_conf.get("stop_words", []))
        _DETAIL_STOP = stop_words.union({"the", "a", "an", "and", "or", "for", "of", "in", "with", "by", "from"})
        
        raw_tokens = [t.strip(".,?!-–—/|") for t in entity_name.split()]
        # Keep tokens that are meaningful: length > 1, not stop, not pure numbers
        good_tokens = [
            t for t in raw_tokens
            if t and len(t) > 1 and t.lower() not in _DETAIL_STOP and not t.isdigit()
        ]

        if not good_tokens:
            return "Please provide a valid entity name."

        # Build Lucene query: use first 3 good tokens with fuzzy (~), join remaining as plain phrases
        lucene_tokens = [t + "~" for t in good_tokens[:3]]
        lucene_query = " AND ".join(lucene_tokens)
        logger.info(f"EntityDetailsDatabase | entity_name={entity_name!r} | lucene={lucene_query!r}")

        ft_index = tenant_conf.get("indexes", {}).get("entity_fulltext", "entity_name_ft")
        
        primary_entity = tenant_conf.get("primary_entity", "Product")
        
        cypher_query = f"""
        CALL db.index.fulltext.queryNodes("{ft_index}", $lucene_query) YIELD node AS p, score
        WHERE p.tenant = $tenant_id OR p.tenant IS NULL
        WITH p, score
        ORDER BY score DESC LIMIT 1
        OPTIONAL MATCH (p)--(m)
        WHERE labels(m)[0] <> '{primary_entity}'
        RETURN properties(p) AS details,
               collect(DISTINCT labels(m)[0] + ': ' + coalesce(m.name, m.value, m.description, '')) AS metadata
        """
        from src.services.agent.context import tenant_context
        params = {"lucene_query": lucene_query, "tenant_id": tenant_context.get()}
        logger.info(f"EntityDetailsDatabase Cypher: {cypher_query.strip()} | Params: {params}")
        res = AgentConfig.graph.query(cypher_query, params=params)

        if not res:
            return "Entity not found."

        row = res[0]
        details = row.get('details', {})
        metadata_list = row.get('metadata', [])

        output = "Entity Details:\n"
        output += json.dumps(details, indent=2, ensure_ascii=False) + "\n"

        if metadata_list:
            output += "\nConnected Information:\n"
            for item in sorted(set(metadata_list)):
                if not item.endswith(': '):  # filter out empty names
                    output += f"- {item}\n"

        return output.encode("ascii", errors="ignore").decode("ascii")
    except Exception as e:
        logger.error(f"Error in EntityDetailsDatabase: {e}", exc_info=True)
        return f"Error getting entity details: {e}"






def query_policies(query: str):
    from src.services.agent.context import tenant_context
    from src.services.agent.routing import load_tenant_configs
    tenant_id = tenant_context.get()
    filter_dict = {"tenant": tenant_id} if tenant_id else None
    
    configs = load_tenant_configs()
    tenant_conf = configs.get(tenant_id, configs.get("default", {}))
    
    policy_idx = tenant_conf.get("indexes", {}).get("policy", "policy_vector")
    general_faq_idx = tenant_conf.get("indexes", {}).get("general_faq", "general_faq_vector")
    
    results = AgentConfig.get_vector_store(policy_idx).similarity_search_with_score(query, k=2, filter=filter_dict)
    if not results:
        results = AgentConfig.get_vector_store(general_faq_idx).similarity_search_with_score(query, k=2, filter=filter_dict)
    if not results:
        return "No relevant policy found."
    text = "\n\n".join([doc.page_content for doc, _ in results])
    return text.encode("ascii", errors="ignore").decode("ascii")


def query_product_faqs(query: str):
    from src.services.agent.context import tenant_context
    from src.services.agent.routing import load_tenant_configs
    tenant_id = tenant_context.get()
    filter_dict = {"tenant": tenant_id} if tenant_id else None
    
    configs = load_tenant_configs()
    tenant_conf = configs.get(tenant_id, configs.get("default", {}))
    
    product_faq_idx = tenant_conf.get("indexes", {}).get("product_faq", "product_faq_vector")
    retrieval_query = tenant_conf.get("retrieval_queries", {}).get("product_faq")

    faq_node_property = tenant_conf.get("faq_node_property", "question")
    results = AgentConfig.get_vector_store(product_faq_idx, text_node_property=faq_node_property, retrieval_query=retrieval_query).similarity_search_with_score(query, k=2, filter=filter_dict)
    if not results:
        return "No relevant product FAQ found."
    text = "\n\n".join([doc.page_content for doc, _ in results])
    return text.encode("ascii", errors="ignore").decode("ascii")


def query_general_knowledge(query: str):
    """
    Search knowledge articles stored as Chunk nodes using full-text search.
    All configuration (index name, stop words) is driven by the tenant config.
    """
    try:
        from src.services.agent.context import tenant_context
        from src.services.agent.routing import load_tenant_configs
        tenant_id = tenant_context.get()
        
        configs = load_tenant_configs()
        tenant_conf = configs.get(tenant_id, configs.get("default", {}))
        
        chunk_idx = tenant_conf.get("indexes", {}).get("chunk_fulltext", "chunk_text")
        
        # Merge universal function-word stop list with tenant-specific domain stop words
        generic_stop = {"the", "a", "an", "is", "are", "which", "one", "better", "vs",
                        "or", "and", "for", "of", "in", "with", "what", "how", "do",
                        "does", "can", "will", "between", "difference"}
        tenant_stop = set(tenant_conf.get("stop_words", []))
        _STOP = generic_stop.union(tenant_stop)
        
        # Strip Lucene special chars: + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
        _LUCENE_SPECIAL = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/.,?!\'"–—]')
        raw_tokens = []
        for word in query.split():
            # First split on hyphens, then clean each part
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

        cypher = f"""
        CALL db.index.fulltext.queryNodes("{chunk_idx}", $q) YIELD node AS c, score
        WHERE (c.tenant = $tenant_id OR c.tenant IS NULL) AND score > 0.5
        WITH c, score
        RETURN properties(c) AS details, score
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
            details = row.get("details", {})
            text = details.get("text") or details.get("content") or ""
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
            name="FilterGraphEntities",
            func=filter_graph_entities,
            description=(
                "Search, list, and filter the primary entity (e.g. Products) from the graph database. "
                "You MUST use the exact labels and values provided in the GRAPH METADATA schema. "
                "\n\nParameters:"
                "\n- filters (list): list of dictionaries, e.g., [{'label': 'Category', 'value': 'Solar Lights'}, {'label': 'Feature', 'value': 'waterproof'}]."
                "\n- min_price / max_price (int): price range constraints."
                "\n- sort_by (str): price_asc, price_desc, rating_desc."
                "\n- limit (int): number of results (default 100)."
            ),
            return_direct=False
        ),
        StructuredTool.from_function(
            name="EntityDetailsDatabase",
            func=get_entity_details_db,
            description=(
                "Use this when the user asks about ONE specific named entity's details (e.g., a specific product): "
                "warranty, specs, dimensions, material, or any other connected metadata. "
                "Examples: 'What is the warranty of Product X?', "
                "'Is Product Y available in blue?', "
                "'What material is Product Z made of?'"
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
                "'Which is better: Option A or Option B?', "
                "'Which product requires the least maintenance?', "
                "'Can this be used for commercial spaces?'"
            )
        ),
        Tool(
            name="GeneralKnowledgeDatabase",
            func=query_general_knowledge,
            description=(
                "Use this for educational, comparison, and 'how-to' questions about domain concepts "
                "that are NOT about a specific product and NOT a company policy. "
                "This searches the company's blog articles and knowledge base. "
                "Examples: "
                "'Concept A vs Concept B', "
                "'What is the difference between category X and category Y?', "
                "'How to choose the right product?', "
                "'Benefits of feature Z'"
            )
        )
    ]
