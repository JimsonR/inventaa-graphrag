"""
retrieval/graph_search.py — Neo4j relational traversal and full-text search.
Decoupled from static tenant namespaces and hardcoded schema relationship names.
"""

import os
import logging
from typing import List, Dict, Any
from src.services.agent.config import AgentConfig
from src.services.agent.context import tenant_context
from src.query.models import QueryIntent

logger = logging.getLogger(__name__)


def graph_search(intent_data: dict, query: str) -> List[Dict[str, Any]]:
    """Synchronous Neo4j relational graph traversal for semantic product matching."""
    try:
        intent = intent_data.get("intent")
        prod_name = intent_data.get("product_name")

        if not AgentConfig.graph:
            return []

        # 1. Full-text search for exact named product queries
        if intent == QueryIntent.GET_PRODUCT_INFO or prod_name:
            search_target = prod_name or query
            lucene_q = " AND ".join([f"{term}~" for term in search_target.split() if len(term) > 2])
            if not lucene_q:
                lucene_q = search_target + "~"
            
            ft_index = AgentConfig.get_fulltext_index()
            cypher = f"""
            CALL db.index.fulltext.queryNodes("{ft_index}", $lucene_q) YIELD node AS p, score
            RETURN p.sku AS sku, p.name AS name, score AS graph_score
            ORDER BY score DESC LIMIT 10
            """
            try:
                res = AgentConfig.graph.query(cypher, params={"lucene_q": lucene_q})
                if not res and len(query.split()) > 0:
                    # Fallback to simple first word prefix/fuzzy
                    cypher_fb = f"""
                    CALL db.index.fulltext.queryNodes("{ft_index}", $q + "~") YIELD node AS p, score
                    RETURN p.sku AS sku, p.name AS name, score AS graph_score
                    ORDER BY score DESC LIMIT 10
                    """
                    res = AgentConfig.graph.query(cypher_fb, params={"q": query.split()[0]})
                return res if res else []
            except Exception as e:
                logger.warning(f"Full-text search error on index {ft_index}: {e}")
                return []

        # 2. Relational Traversal: Category -> Product -> Dynamic Options
        tenant_id = tenant_context.get() or os.getenv("TENANT_ID") or AgentConfig.brain.get("tenant", {}).get("id", "default")
        filters = intent_data.get("filters", {}) or {}
        cats_list = intent_data.get("category_keywords", []) or []
        cat_filter = filters.get("category")
        cats = list(set([c for c in cats_list if c] + ([cat_filter] if cat_filter else [])))
        feats = intent_data.get("feature_keywords", []) or []

        cypher_parts = ["MATCH (p:Product) WHERE p.tenant = $tenant_id"]
        params: Dict[str, Any] = {"tenant_id": tenant_id}

        if cats:
            cypher_parts.append("""
            OPTIONAL MATCH (cat:Category)-[:HAS_PRODUCT]->(p)
            OPTIONAL MATCH (col:Collection)<-[:BELONGS_TO_COLLECTION]-(p)
            OPTIONAL MATCH (p)-[:SUITABLE_FOR]->(uc:UseCase)
            """)
        
        # Dynamically append option traversals from discovered schema
        opt_aliases = []
        if feats or AgentConfig.product_options:
            cypher_parts.append("OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)")
            for opt in AgentConfig.product_options:
                rel = opt.get("rel_type")
                lbl = opt.get("target_label")
                alias = opt.get("alias")
                if rel and lbl and alias:
                    cypher_parts.append(f"OPTIONAL MATCH (p)-[:{rel}]->({alias}:{lbl})")
                    opt_aliases.append(alias)

        # Build filtering conditions
        conds = []
        if cats:
            conds.append("(any(c IN $cats WHERE toLower(cat.name) CONTAINS toLower(c) OR toLower(col.name) CONTAINS toLower(c) OR toLower(uc.name) CONTAINS toLower(c)))")
            params["cats"] = cats
        if feats:
            feat_checks = ["toLower(f.name) CONTAINS toLower(f_tag)"]
            for alias in opt_aliases:
                feat_checks.append(f"toLower({alias}.name) CONTAINS toLower(f_tag)")
            feat_cond_str = " OR ".join(feat_checks)
            conds.append(f"(any(f_tag IN $feats WHERE {feat_cond_str}))")
            params["feats"] = feats

        preferences = intent_data.get("preferences", {}) or {}
        max_p = preferences.get("max_price")
        if max_p and isinstance(max_p, (int, float)):
            conds.append("(p.price_num <= $max_price AND p.price_num > 0)")
            params["max_price"] = max_p
        min_p = preferences.get("min_price")
        if min_p and isinstance(min_p, (int, float)):
            conds.append("(p.price_num >= $min_price)")
            params["min_price"] = min_p

        if conds:
            cypher_parts.append("WITH p, " + " AND ".join(conds) + " AS match_flag WHERE match_flag")
        elif not conds:
            logger.info("[DEBUG-NEO4J] No category/feature/option conditions specified for relational search; returning empty list.")
            return []

        cypher_parts.append("RETURN DISTINCT p.sku AS sku, p.name AS name, 1.0 AS graph_score LIMIT 15")
        cypher = "\n".join(cypher_parts)

        logger.info(f"[DEBUG-NEO4J] Executing relational graph traversal for tenant {tenant_id}")
        res = AgentConfig.graph.query(cypher, params=params)
        return res if res else []
    except Exception as e:
        logger.error(f"Neo4j graph search error: {e}", exc_info=True)
        return []
