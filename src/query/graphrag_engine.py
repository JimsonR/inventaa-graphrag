"""
graphrag_engine.py — Linear GraphRAG Query Engine Coordinator.
Decomposed into modular components (retrieval, fusion, prompts, models) and decoupled from brand/domain specifics.
"""

import os
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any

from src.services.agent.config import AgentConfig
from src.query.models import QueryIntent, QueryResult
from src.query.prompts import get_intent_system_prompt, get_response_system_prompt
from src.query.retrieval import parallel_retrieve, category_browse_from_sqlite
from src.query.fusion import fuse_results, hydrate_from_sqlite, build_context

logger = logging.getLogger(__name__)

# Re-export models for backwards compatibility with existing importers
__all__ = ["GraphRAGEngine", "QueryIntent", "QueryResult"]


class GraphRAGEngine:
    """Main coordinator for the multi-channel RAG retrieval and synthesis pipeline."""
    
    def __init__(self):
        AgentConfig.initialize()

    async def classify_intent(self, query: str, history_context: str = "", taxonomy_hints: dict = None) -> dict:
        """Use LLM to classify query intent and extract entities, factoring in conversation flow and taxonomy."""
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            prompt_content = f"User Query: {query}"
            if history_context:
                prompt_content = (
                    f"Previous Conversation History:\n{history_context}\n\n"
                    f"Current User Query: {query}\n\n"
                    "IMPORTANT: If the Current User Query is short or vague, infer the categories, feature keywords, and preferences from the Previous Conversation History so the conversation flow continues seamlessly."
                )

            taxonomy_section = ""
            if taxonomy_hints:
                taxonomy_section = "\n\nCANDIDATE TAXONOMY HINTS (CRITICAL RULE: Select any category/collection below that matches the user's explicit request. Do NOT blindly copy unrelated candidate categories):"
                if taxonomy_hints.get("category"):
                    taxonomy_section += f"\n  Matched Categories: {taxonomy_hints['category']}"
                if taxonomy_hints.get("use_case"):
                    taxonomy_section += f"\n  Matched Use Cases: {taxonomy_hints['use_case']}"
                if taxonomy_hints.get("feature"):
                    taxonomy_section += f"\n  Matched Features: {taxonomy_hints['feature']}"

            system_prompt = get_intent_system_prompt()
            if taxonomy_section:
                system_prompt += taxonomy_section

            dyn_cats = "\n".join(AgentConfig.collections) if AgentConfig.collections else ""
            if dyn_cats:
                system_prompt += f"\n\nAVAILABLE PRODUCT COLLECTIONS IN DATABASE (Select ONLY the exact collection matching the user's query; do not list collections that were not requested):\n{dyn_cats}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt_content)
            ]
            response = await asyncio.to_thread(AgentConfig.llm.invoke, messages)
            content = response.content.strip()
            logger.info(f"[DEBUG-LLM] Intent Classifier raw JSON response:\n{content}")
            
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except Exception as e:
            logger.error(f"Intent classification error: {e}", exc_info=True)
            return {
                "intent": "find_product",
                "category_keywords": [],
                "feature_keywords": [],
                "product_name": None,
                "filters": {},
                "preferences": {},
            }

    async def synthesize_response(self, query: str, context: str, intent_data: dict, history_context: str = "", taxonomy_hints: dict = None) -> str:
        """Use LLM to generate a helpful, professional sales response."""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        history_part = f"Previous Conversation Flow:\n{history_context}\n\n" if history_context else ""
        currency = AgentConfig.get_currency_symbol()
        
        taxonomy_part = ""
        if taxonomy_hints and any(taxonomy_hints.values()):
            taxonomy_part = f"Taxonomy Agent Results (Available Categories/Collections in store): {json.dumps(taxonomy_hints)}\n\n"

        prompt = (
            f"{history_part}User Query: {query}\n\n"
            f"Detected Intent: {intent_data.get('intent')}\n"
            f"Extracted Preferences: {json.dumps(intent_data.get('preferences', {}))}\n\n"
            f"{taxonomy_part}"
            f"{context}\n\n"
            "Please write a helpful, professional sales response based on the above catalog and policy context, continuing the conversation flow seamlessly.\n"
            "CRITICAL INSTRUCTION: If no products are retrieved in the catalog context (or if the requested category does not exist), DO NOT hallucinate or recommend random/unrelated products. Instead, politely inform the customer that we do not have items matching that exact request, and use the 'Taxonomy Agent Results' above to offer resolutions, suggesting the most relevant candidate categories they can explore.\n"
            f"Always include product prices in {currency} and direct product URLs when recommending products."
        )

        try:
            messages = [
                SystemMessage(content=get_response_system_prompt()),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(AgentConfig.llm.invoke, messages)
            res_text = response.content.strip()
            logger.info(f"[DEBUG-LLM] Synthesized final response:\n{res_text}")
            return res_text
        except Exception as e:
            logger.error(f"Response synthesis error: {e}", exc_info=True)
            return f"Here are the details matching your query:\n\n{context}"

    async def retrieve(self, query: str, intent_data: dict):
        """Delegates parallel retrieval to the modular retrieval sub-package."""
        return await parallel_retrieve(query, intent_data)

    def _fuse_results(self, vector_results: list, graph_results: list, text_results: list = None):
        """Delegates RRF scoring to the fusion module."""
        return fuse_results(vector_results, graph_results, text_results)

    def _hydrate_from_sqlite(self, fused_skus: list, preferences: dict, query: str = "", category_keywords: list = None):
        """Delegates authoritative SQLite hydration to the fusion module."""
        return hydrate_from_sqlite(fused_skus, preferences, query=query, category_keywords=category_keywords)

    def _build_context(self, products: list, non_prod_contexts: list) -> str:
        """Delegates markdown context formatting to the fusion module."""
        return build_context(products, non_prod_contexts)

    def _category_browse_from_sqlite(self, category_keywords: list, preferences: dict) -> list:
        """Delegates category browsing to the retrieval text search module."""
        return category_browse_from_sqlite(category_keywords, preferences)

    async def query(self, user_query: str, session_id: Optional[str] = None, tenant_id: Optional[str] = None) -> QueryResult:
        """Full GraphRAG pipeline: check DB history -> taxonomy -> classify -> retrieve -> fuse -> hydrate -> synthesize."""
        if tenant_id:
            from src.services.agent.context import tenant_context
            tenant_context.set(tenant_id)
        logger.info(f"GraphRAG query: {user_query!r} | session_id: {session_id} | tenant_id: {tenant_id}")

        # Step 0: Check conversation history from DB if session_id is provided
        history_context = ""
        if session_id:
            try:
                from src.services.agent.memory import get_recent_messages
                from langchain_core.messages import HumanMessage
                recent_msgs = await asyncio.to_thread(get_recent_messages, session_id=session_id, limit=5)
                if recent_msgs:
                    lines = [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in recent_msgs]
                    history_context = "\n".join(lines)
                    logger.info(f"Loaded {len(recent_msgs)} previous messages from DB for session {session_id}")
            except Exception as e:
                logger.warning(f"Could not load conversation history from DB for session {session_id}: {e}")

        # Step 0.5: Taxonomy resolution
        taxonomy_hints = {}
        try:
            from src.services.agent.taxonomy import fetch_taxonomy_candidates
            query_embedding = await asyncio.to_thread(AgentConfig.embeddings.embed_query, user_query)
            taxonomy_hints = await asyncio.to_thread(fetch_taxonomy_candidates, query_embedding, 0.80)
            if taxonomy_hints:
                logger.info(f"Taxonomy resolved: {taxonomy_hints}")
        except Exception as e:
            logger.warning(f"Taxonomy resolution failed (non-fatal): {e}")

        # Step 1: Classify intent
        intent_data = await self.classify_intent(user_query, history_context=history_context, taxonomy_hints=taxonomy_hints)
        cat_kws = intent_data.setdefault("category_keywords", [])
        filters = intent_data.get("filters", {}) or {}
        if filters.get("category") and filters["category"] not in cat_kws:
            cat_kws.append(filters["category"])
        if not cat_kws:
            for k in ["segment", "application"]:
                val = filters.get(k)
                if val and isinstance(val, str) and len(val.strip()) > 2 and not any(g.lower() == val.strip().lower() for g in AgentConfig.top_level_groups):
                    cat_kws.append(val.strip())

        intent = QueryIntent(intent_data.get("intent", "unknown"))
        logger.info(f"Classified Intent: {intent} | Keywords: {intent_data.get('category_keywords')} {intent_data.get('feature_keywords')}")

        is_broad_query = (
            intent in (QueryIntent.FIND_PRODUCT, QueryIntent.BROWSE_CATEGORY, QueryIntent.UNKNOWN)
            and not cat_kws
            and not intent_data.get("feature_keywords")
            and not intent_data.get("product_name")
            and not filters.get("category")
            and not filters.get("brand")
        )

        if intent == QueryIntent.BROWSE_CATEGORY or is_broad_query:

            top_groups = [g.lower() for g in AgentConfig.top_level_groups]
            is_top_level = not filters.get("category") and (
                str(filters.get("application") or "").lower() in top_groups
                or any(k.strip().lower() in top_groups for k in cat_kws)
            )
            if is_top_level or is_broad_query:
                logger.info(f"Detected TOP-LEVEL / BROAD category navigation query for keywords: {cat_kws} | Intent: {intent}")
                app = str(filters.get("application") or "").lower()

                lines = ["AVAILABLE SPECIALIZED COLLECTIONS (Do NOT list individual items or prices; instead, introduce these available collections clearly and ask the customer which collection they would like to explore):"]
                matched_groups = [g for g in AgentConfig.top_level_groups if g.lower() in app or any(g.lower() in k.lower() for k in cat_kws)]
                if not matched_groups:
                    matched_groups = AgentConfig.top_level_groups or list(AgentConfig.category_groups.keys())
                for g in matched_groups:
                    cols = AgentConfig.category_groups.get(g, [])
                    if cols:
                        lines.append(f"\n{g} Collections:")
                        for c in cols:
                            lines.append(f"• {c}")

                context_text = "\n".join(lines)
                response = await self.synthesize_response(user_query, context_text, intent_data, history_context=history_context, taxonomy_hints=taxonomy_hints)

                if session_id and AgentConfig.memory_provider and hasattr(AgentConfig.memory_provider, "add_message"):
                    try:
                        from langchain_core.messages import HumanMessage as HM, AIMessage
                        AgentConfig.memory_provider.add_message(session_id, HM(content=user_query))
                        AgentConfig.memory_provider.add_message(session_id, AIMessage(content=response))
                    except Exception as e:
                        logger.warning(f"Could not store interaction: {e}")

                return QueryResult(intent=QueryIntent.BROWSE_CATEGORY, products=[], context_text=context_text, response=response, product_links=[])

        if intent == QueryIntent.BROWSE_CATEGORY:

            hydrated_products = self._category_browse_from_sqlite(cat_kws, intent_data.get("preferences", {}))
            if hydrated_products:
                context_text = self._build_context(hydrated_products, [])
                response = await self.synthesize_response(user_query, context_text, intent_data, history_context=history_context, taxonomy_hints=taxonomy_hints)

                if session_id and AgentConfig.memory_provider and hasattr(AgentConfig.memory_provider, "add_message"):
                    try:
                        from langchain_core.messages import HumanMessage as HM, AIMessage
                        AgentConfig.memory_provider.add_message(session_id, HM(content=user_query))
                        AgentConfig.memory_provider.add_message(session_id, AIMessage(content=response))
                    except Exception as e:
                        logger.warning(f"Could not store interaction: {e}")

                product_links = [
                    {"sku": p["sku"], "name": p["name"], "url": p["url"], "price": p["price_num"], "image_url": p["image_url"]}
                    for p in hydrated_products if p.get("url")
                ]
                return QueryResult(intent=intent, products=hydrated_products, context_text=context_text, response=response, product_links=product_links)
            logger.info("BROWSE_CATEGORY found 0 products; returning empty list with taxonomy resolution instead of falling through to RAG pipeline.")
            response = await self.synthesize_response(user_query, "", intent_data, history_context=history_context, taxonomy_hints=taxonomy_hints)
            if session_id and AgentConfig.memory_provider and hasattr(AgentConfig.memory_provider, "add_message"):
                try:
                    from langchain_core.messages import HumanMessage as HM, AIMessage
                    AgentConfig.memory_provider.add_message(session_id, HM(content=user_query))
                    AgentConfig.memory_provider.add_message(session_id, AIMessage(content=response))
                except Exception as e:
                    logger.warning(f"Could not store interaction: {e}")
            return QueryResult(intent=intent, products=[], context_text="", response=response, product_links=[])

        # Step 2: Parallel retrieval (Vector + Graph + Text/SQL)
        vector_results, graph_results, text_results = await self.retrieve(user_query, intent_data)
        logger.info(f"Retrieved: {len(vector_results)} vector items, {len(graph_results)} graph items, {len(text_results)} text items")

        # Step 3: Reciprocal Rank Fusion
        fused_skus, non_prod_contexts = self._fuse_results(vector_results, graph_results, text_results)

        # Step 4: SQL Hydration
        hydrated_products = self._hydrate_from_sqlite(fused_skus, intent_data.get("preferences", {}), query=user_query, category_keywords=intent_data.get("category_keywords"))
        logger.info(f"Hydrated {len(hydrated_products)} authoritative product cards from SQLite")

        # Step 5: Build context
        context_text = self._build_context(hydrated_products, non_prod_contexts)

        # Step 6: Response synthesis
        response = await self.synthesize_response(user_query, context_text, intent_data, history_context=history_context, taxonomy_hints=taxonomy_hints)

        # Step 7: Save interaction to memory provider
        if session_id and AgentConfig.memory_provider and hasattr(AgentConfig.memory_provider, "add_message"):
            try:
                from langchain_core.messages import HumanMessage, AIMessage
                AgentConfig.memory_provider.add_message(session_id, HumanMessage(content=user_query))
                AgentConfig.memory_provider.add_message(session_id, AIMessage(content=response))
            except Exception as e:
                logger.warning(f"Could not store interaction to memory provider for session {session_id}: {e}")

        # Step 8: Extract product links for UI
        product_links = [
            {
                "sku": p["sku"],
                "name": p["name"],
                "url": p["url"],
                "price": p["price_num"],
                "image_url": p["image_url"]
            }
            for p in hydrated_products if p.get("url")
        ]

        return QueryResult(
            intent=intent,
            products=hydrated_products,
            context_text=context_text,
            response=response,
            product_links=product_links,
        )
