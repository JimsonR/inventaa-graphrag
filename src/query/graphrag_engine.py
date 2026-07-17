"""
graphrag_engine.py — Lightweight MCP Retrieval Engine.
Receives pre-classified intent_data from the client, runs parallel retrieval
(Neo4j Lucene + SQLite FTS + optional Pinecone), fuses via RRF, hydrates from
SQLite, and returns structured product data. No LLM calls.
"""

import os
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any

from src.services.agent.config import AgentConfig
from src.query.models import QueryIntent, QueryResult
from src.query.retrieval import parallel_retrieve, category_browse_from_sqlite, vector_search
from src.query.fusion import fuse_results, hydrate_from_sqlite, build_context
from src.utils.timing import log_timing, async_time_it

logger = logging.getLogger(__name__)

__all__ = ["GraphRAGEngine", "QueryIntent", "QueryResult"]


class GraphRAGEngine:
    """Lightweight MCP retrieval engine — no LLM calls, no memory management."""

    def __init__(self):
        AgentConfig.initialize()

    @log_timing("GraphRAGEngine.retrieve")
    async def retrieve(self, query: str, intent_data: dict):
        """Delegates parallel retrieval to the modular retrieval sub-package."""
        return await parallel_retrieve(query, intent_data)

    @log_timing("GraphRAGEngine.query")
    async def query(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        intent_data: Optional[dict] = None,
    ) -> QueryResult:
        """
        Lightweight MCP pipeline: receive intent_data → retrieve → fuse → hydrate → return.
        No LLM classification, no response synthesis, no memory read/write.
        """
        if tenant_id:
            from src.services.agent.context import tenant_context
            tenant_context.set(tenant_id)

        logger.info(f"GraphRAG query: {user_query!r} | session_id: {session_id} | tenant_id: {tenant_id}")

        # ── Normalise intent_data (client MUST provide it; fallback to find_product) ──
        if not intent_data or not isinstance(intent_data, dict) or not intent_data.get("intent"):
            intent_data = {
                "intent": "find_product",
                "category_keywords": [],
                "feature_keywords": [],
                "filters": {},
                "preferences": {},
            }
        else:
            logger.info(f"[GraphRAG] Using pre-classified client intent_data: {intent_data.get('intent')}")
            intent_data.setdefault("category_keywords", [])
            intent_data.setdefault("feature_keywords", [])
            intent_data.setdefault("filters", {})
            intent_data.setdefault("preferences", {})

        # ── Enrich category keywords from filters ──
        cat_kws = intent_data["category_keywords"]
        filters = intent_data.get("filters", {}) or {}
        if filters.get("category") and filters["category"] not in cat_kws:
            cat_kws.append(filters["category"])
        if not cat_kws:
            for k in ["segment", "application"]:
                val = filters.get(k)
                if val and isinstance(val, str) and len(val.strip()) > 2 and not any(
                    g.lower() == val.strip().lower() for g in AgentConfig.top_level_groups
                ):
                    cat_kws.append(val.strip())

        # ── Direct collection match from raw query ──
        query_clean = user_query.strip().lower()
        for col in AgentConfig.collections:
            if col.lower() == query_clean or col.lower() in query_clean:
                if col not in cat_kws:
                    cat_kws.append(col)
                filters["category"] = col
                logger.info(f"[GraphRAG] Direct collection match from user_query: '{col}'")
                break

        # ── Resolve intent enum ──
        try:
            intent = QueryIntent(intent_data.get("intent", "unknown"))
        except ValueError:
            intent_str = str(intent_data.get("intent", "unknown")).lower()
            if any(w in intent_str for w in ["faq", "knowledge", "blog", "idea"]):
                intent = QueryIntent.FAQ_KNOWLEDGE
            elif "browse" in intent_str or "category" in intent_str:
                intent = QueryIntent.BROWSE_CATEGORY
            else:
                intent = QueryIntent.FIND_PRODUCT

        logger.info(f"Classified Intent: {intent} | Keywords: {intent_data.get('category_keywords')} {intent_data.get('feature_keywords')}")

        # ── Detect broad / top-level navigation ──
        query_lower = user_query.strip().lower()
        # Domain-agnostic item noun (e.g. 'light', 'furniture', 'item') so
        # navigation matching works across verticals, not just lighting.
        noun = AgentConfig.get_item_noun().lower()
        plural = noun if noun.endswith("s") else f"{noun}s"
        # Generic browse nouns that signal "show me your catalog/menu" rather than
        # a specific product search, even though they are not stopwords.
        nav_generic = {
            "collection", "collections", "catalog", "catalogue", "products",
            "product", "range", "menu", "categories", "category", "offers",
            "offer", "deals", "deal", "shop", "store", "items", "item",
            "your", "my", "our", "their",
        }
        # A query still carries a *specific* search term (e.g. a product name like
        # "athena") when, after removing stopwords, the generic item noun, and
        # generic browse nouns, at least one meaningful token remains. Such
        # queries are real product searches and must NOT be swallowed by broad
        # navigation.
        try:
            from src.query.retrieval.text_search import get_stopwords
            _stop = get_stopwords()
        except Exception:
            _stop = set()
        searchable_tokens = [
            t for t in query_lower.split()
            if len(t) > 2 and t not in _stop and t != noun and t != plural and t not in nav_generic
        ]
        is_broad_query = (
            intent in (QueryIntent.FIND_PRODUCT, QueryIntent.BROWSE_CATEGORY, QueryIntent.UNKNOWN)
            and not cat_kws
            and not intent_data.get("feature_keywords")
            and not intent_data.get("product_name")
            and not filters.get("category")
            and not filters.get("brand")
            and not searchable_tokens
        )

        if intent == QueryIntent.BROWSE_CATEGORY or is_broad_query:
            top_groups = [g.lower() for g in AgentConfig.top_level_groups]
            is_top_level = not filters.get("category") and (
                any(
                    query_lower == g or query_lower == f"{g} {plural}" or query_lower == f"{g} {noun}"
                    or query_lower == f"show {g}" or query_lower == f"show {g} {plural}"
                    or f"{g} {plural}" in query_lower
                    for g in top_groups
                )
                or (
                    str(filters.get("application") or "").lower() in top_groups
                    or any(k.strip().lower() in top_groups for k in cat_kws)
                )
            )

            if is_top_level or (is_broad_query and not filters.get("category")):
                logger.info(f"Detected TOP-LEVEL / BROAD category navigation for query='{user_query}'")
                app = str(filters.get("application") or "").lower()
                greeting = AgentConfig.brain.get("prompts", {}).get(
                    "browse_greeting",
                    "Hello! We have a variety of collections available. Which category would you like to explore?",
                )
                friendly_lines = [greeting]
                matched_groups = [
                    g for g in AgentConfig.top_level_groups
                    if g.lower() in query_lower or g.lower() in app or any(g.lower() in k.lower() for k in cat_kws)
                ]
                if not matched_groups:
                    matched_groups = AgentConfig.top_level_groups or list(AgentConfig.category_groups.keys())
                for g in matched_groups:
                    cols = AgentConfig.category_groups.get(g, [])
                    if cols:
                        friendly_lines.append(f"\n*{g} Collections:*")
                        for c in cols:
                            friendly_lines.append(f"• {c}")

                response = "\n".join(friendly_lines)
                return QueryResult(
                    intent=QueryIntent.BROWSE_CATEGORY,
                    products=[],
                    context_text=response,
                    response=response,
                    product_links=[],
                    chunks=[],
                )

        # ── Category browse via SQLite ──
        if intent == QueryIntent.BROWSE_CATEGORY:
            hydrated_products = category_browse_from_sqlite(cat_kws, intent_data.get("preferences", {}))
            if hydrated_products:
                context_text = build_context(hydrated_products, [])
                product_links = [
                    {"sku": p["sku"], "name": p["name"], "url": p["url"], "price": p["price_num"], "image_url": p["image_url"]}
                    for p in hydrated_products if p.get("url")
                ]
                return QueryResult(
                    intent=intent,
                    products=hydrated_products,
                    context_text=context_text,
                    response=f"Found {len(hydrated_products)} products matching your category.",
                    product_links=product_links,
                    chunks=[],
                )
            logger.info("BROWSE_CATEGORY found 0 products; returning empty result.")
            return QueryResult(intent=intent, products=[], context_text="", response="No products found in this category.", product_links=[], chunks=[])

        # ── Parallel retrieval ──
        # For pure knowledge queries (faq_knowledge) we only need the vector
        # channel (Neo4j faq_vector / product_faq_vector over :Chunk / :FAQ).
        # Running graph_search / text_search here is noise: graph_search bails
        # without a category, and text_search's token fallback leaks products.
        if intent == QueryIntent.FAQ_KNOWLEDGE:
            async with async_time_it("retrieve.vector_only"):
                vector_results = await asyncio.to_thread(vector_search, intent_data, user_query)
                graph_results: List[Dict[str, Any]] = []
                text_results: List[Dict[str, Any]] = []
        else:
            async with async_time_it("retrieve.parallel"):
                vector_results, graph_results, text_results = await self.retrieve(user_query, intent_data)
        logger.info(f"Retrieved: {len(vector_results)} vector, {len(graph_results)} graph, {len(text_results)} text")

        # ── Reciprocal Rank Fusion ──
        async with async_time_it("fusion.fuse_results"):
            fused_skus, non_prod_contexts = fuse_results(vector_results, graph_results, text_results)

        # ── SQLite Hydration ──
        async with async_time_it("fusion.hydrate_from_sqlite"):
            hydrated_products = hydrate_from_sqlite(
                fused_skus, intent_data.get("preferences", {}),
                query=user_query, category_keywords=intent_data.get("category_keywords"),
            )
        logger.info(f"Hydrated {len(hydrated_products)} authoritative product cards from SQLite")

        # ── Build context and product links ──
        async with async_time_it("fusion.build_context"):
            context_text = build_context(hydrated_products, non_prod_contexts)
        product_links = [
            {"sku": p["sku"], "name": p["name"], "url": p["url"], "price": p["price_num"], "image_url": p["image_url"]}
            for p in hydrated_products if p.get("url")
        ]

        resp_parts = []
        if hydrated_products:
            resp_parts.append(f"Found {len(hydrated_products)} matching products.")
        if non_prod_contexts:
            resp_parts.append(f"Retrieved {len(non_prod_contexts)} knowledge/policy entries.")
        if not resp_parts:
            resp_parts.append("No matching products or knowledge entries found.")

        return QueryResult(
            intent=intent,
            products=hydrated_products,
            context_text=context_text,
            response=" ".join(resp_parts),
            product_links=product_links,
            chunks=non_prod_contexts,
        )

    def _fuse_results(self, vector_results: list, graph_results: list, text_results: list = None):
        """Delegates RRF scoring to the fusion module."""
        return fuse_results(vector_results, graph_results, text_results)

    def _hydrate_from_sqlite(self, skus: list, preferences: dict = None, query: str = "", category_keywords: list = None):
        """Delegates SQLite hydration to the fusion module."""
        return hydrate_from_sqlite(skus, preferences or {}, query, category_keywords)

    def _build_context(self, products: list, non_prod_contexts: list = None):
        """Delegates markdown context formatting to the fusion module."""
        return build_context(products, non_prod_contexts or [])
