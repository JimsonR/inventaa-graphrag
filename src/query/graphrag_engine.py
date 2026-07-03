"""
Linear GraphRAG Query Engine for Inventaa Outdoor Lighting Assistant.

Architecture (matching Knowledge-Base-main):
  User Query
    ↓
  Intent Classifier (Azure OpenAI LLM)  →  identifies: find_product | get_product_info | check_policy | get_advice
    ↓
  Parallel Retrieval (Concurrent via asyncio.gather):
    ├── Vector Search (Neo4j Vector / FAQ / Policy)   → semantic text similarity
    └── Graph Traversal (Neo4j Graph)                 → structured relational traversal
    ↓
  Reciprocal Rank Fusion (RRF)
    ↓
  SQL Hydration (SQLite Tri-Store)                    → authoritative prices, discounts, specs, URLs
    ↓
  Response Synthesis (Azure OpenAI LLM)             → structured sales recommendation
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Dict, Optional

import logging
import re
from sqlalchemy import func, or_
from src.services.agent.config import AgentConfig
from src.db.database import get_session
from src.db.models import Product, ProductSpec, ProductVariant

logger = logging.getLogger(__name__)

STOPWORDS = {
    "give", "me", "show", "tell", "about", "what", "are", "the", "is", "for", "please", 
    "some", "any", "get", "find", "looking", "want", "need", "buy", "at", "in", "of", 
    "on", "to", "with", "from", "by", "an", "a", "lights", "light", "lighting", 
    "lamp", "lamps", "fixture", "fixtures", "inventaa", "can", "you", "do", "have", "we", "our",
    "watt", "watts", "volt", "volts", "meter", "meters", "inch", "inches", "rs", "rupee", "rupees", "price", "under", "below", "above", "range", "cost"
}

class QueryIntent(str, Enum):
    FIND_PRODUCT = "find_product"
    GET_PRODUCT_INFO = "get_product_info"
    BROWSE_CATEGORY = "browse_category"
    CHECK_POLICY = "check_policy"
    GET_ADVICE = "get_advice"
    UNKNOWN = "unknown"


@dataclass
class QueryResult:
    intent: QueryIntent
    products: list[dict]
    context_text: str
    response: str
    product_links: list[dict]


INTENT_SYSTEM = """You are an intent classifier for an e-commerce outdoor LED lighting assistant (Inventaa).
Classify the user's query into exactly one of these intents:
- browse_category: User wants to SEE ALL products in a category/collection (e.g., "show me gate lights", "what indoor lights do you have?", "list all solar lights", "ceiling downlights"). Use this when the user is browsing or exploring an entire product category without specifying a single product name or narrow spec filter.
- find_product: User wants to find/discover specific products matching constraints (by feature, color, wattage, price range, or use case). Use this when the user has a specific need with filters (e.g., "12W waterproof gate light under 2000").
- get_product_info: User wants details, specs, dimensions, or warranty about a specific product they know by name or SKU.
- check_policy: User wants to know about return, shipping, delivery, replacement, or warranty policies.
- get_advice: User asks general lighting FAQ, installation advice, or how to choose lighting.
- unknown: None of the above.

Also extract structured filters and entities:
- category_keywords: list of product categories or collections mentioned (e.g., ["Gate & Pillar Lights", "Divine & Temple Lights", "Indoor & Ceiling Lights", "Solar Lights", "Outdoor Commercial Lights"])
- feature_keywords: list of features or specs mentioned (e.g., ["waterproof", "IP65", "motion-sensor", "warm white", "aluminium"])
- product_name: specific product name or SKU if mentioned (null otherwise)
- filters: structured dictionary of extracted constraints:
  {
    "category": string or null (e.g. "Divine & Temple Lights", "Flood Lights"),
    "application": string or null (e.g. "Outdoor", "Indoor", "Garden"),
    "segment": string or null (e.g. "Commercial", "Residential", "Temple"),
    "manufacturer": string or null (e.g. "Inventaa"),
    "ip_rating": string or null (e.g. "IP65", "IP66"),
    "wattage": string or number or null (e.g. "12W" or 12),
    "color": string or null (e.g. "Cool White", "Warm White")
  }
- preferences: dict of any preferences (e.g., {"min_price": 500, "max_price": 2000, "wattage": 12, "color": "Cool White"})

Respond ONLY with valid JSON. Example:
{
  "intent": "browse_category",
  "category_keywords": ["Gate & Pillar Lights"],
  "feature_keywords": [],
  "product_name": null,
  "filters": {
    "category": "Gate & Pillar Lights",
    "application": "Outdoor",
    "segment": null,
    "manufacturer": null,
    "ip_rating": null,
    "wattage": null,
    "color": null
  },
  "preferences": {}
}"""

RESPONSE_SYSTEM = """You are an expert AI sales assistant for Inventaa, a premier Indian LED outdoor lighting manufacturer.
You help customers discover lighting solutions, explain technical specifications, answer FAQ/policy questions, and guide purchases.

When presenting products:
1. List the most relevant products with their display names, exact prices in ₹ (INR), and MRP/discounts if available.
2. Highlight key technical specifications (Wattage, IP Rating, Material, Warranty) retrieved from the database.
3. Always provide the product page URL so the customer can view and purchase online.
4. Be enthusiastic, empathetic, and professional.
5. If the user asks about shipping or returns, clearly state our 7-day replacement guarantee and free shipping policies if relevant.

Format responses in clean, readable text. Use bullet points for product listings.
Always end with a helpful closing or call-to-action to buy online or ask follow-up questions."""


class GraphRAGEngine:
    def __init__(self):
        AgentConfig.initialize()

    async def classify_intent(self, query: str, history_context: str = "", taxonomy_hints: dict = None) -> dict:
        """Use Azure OpenAI LLM to classify query intent and extract entities, factoring in conversation flow and taxonomy."""
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            prompt_content = query
            if history_context:
                prompt_content = f"Previous Conversation History:\n{history_context}\n\nCurrent User Query: {query}\n\nIMPORTANT: If the Current User Query is short or vague (e.g., 'show me products', 'what options do you have?', 'yes', 'show me more', 'how much?'), infer the product categories, feature keywords, and preferences from the Previous Conversation History so the conversation flow continues seamlessly."
            
            # Inject taxonomy hints so the LLM knows our exact category/feature vocabulary
            taxonomy_section = ""
            if taxonomy_hints:
                taxonomy_section = "\n\nCANDIDATE TAXONOMY HINTS (CRITICAL RULE: Select any category/collection below that matches the user's explicit request. If the user asks for a category or product type not listed below, extract the user's exact category words into category_keywords. Do NOT blindly copy unrelated candidate categories into category_keywords):"
                if taxonomy_hints.get("category"):
                    taxonomy_section += f"\n  Matched Categories: {taxonomy_hints['category']}"
                if taxonomy_hints.get("use_case"):
                    taxonomy_section += f"\n  Matched Use Cases: {taxonomy_hints['use_case']}"
                if taxonomy_hints.get("feature"):
                    taxonomy_section += f"\n  Matched Features: {taxonomy_hints['feature']}"
            
            system_prompt = INTENT_SYSTEM
            if taxonomy_section:
                system_prompt += taxonomy_section
            
            # Also inject the dynamically discovered categories so the LLM can map to exact names
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
            # Clean markdown code block if present
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return {
                "intent": "find_product",
                "category_keywords": [],
                "feature_keywords": [],
                "product_name": None,
                "filters": {},
                "preferences": {},
            }

    def _graph_search(self, intent_data: dict, query: str) -> list[dict]:
        """Synchronous Neo4j relational graph traversal for semantic product matching."""
        try:
            cats = intent_data.get("category_keywords", [])
            feats = intent_data.get("feature_keywords", [])
            prod_name = intent_data.get("product_name")

            # Extract clean keyword tokens
            raw_text = f"{query} {prod_name or ''} {' '.join(cats)} {' '.join(feats)}"
            tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9-]{2,}\b", raw_text) if w.lower() not in STOPWORDS]

            if tokens:
                # Use Neo4j's Lucene fulltext search index for ultra-fast fuzzy matching
                lucene_query = " OR ".join([f"{t}~" for t in tokens[:6]])
                logger.info(f"[DEBUG-NEO4J] Executing Lucene fulltext search with query: '{lucene_query}'")
                cypher = """
                CALL db.index.fulltext.queryNodes("product_name_ft", $lucene_q) YIELD node AS p, score
                RETURN DISTINCT p.sku AS sku, p.name AS name, score AS graph_score
                ORDER BY score DESC
                LIMIT 15
                """
                try:
                    res = AgentConfig.graph.query(cypher, params={"lucene_q": lucene_query})
                    if res:
                        logger.info(f"[DEBUG-NEO4J] Lucene search matched {len(res)} nodes. Top SKUs: {[r.get('sku') for r in res[:5]]}")
                        return res
                except Exception as e:
                    logger.warning(f"Lucene fulltext search failed: {e}")

            if not cats and not feats:
                # Fallback to Lucene search with first word if no tokens matched
                cypher = """
                CALL db.index.fulltext.queryNodes("product_name_ft", $q + "~") YIELD node AS p, score
                RETURN p.sku AS sku, p.name AS name, score AS graph_score
                ORDER BY score DESC LIMIT 10
                """
                try:
                    res = AgentConfig.graph.query(cypher, params={"q": query.split()[0]})
                    return res if res else []
                except Exception:
                    return []

            # Relational Traversal: Category -> Product -> Feature / UseCase
            cypher_parts = ["MATCH (p:Product) WHERE p.tenant = 'inventaa' OR p.tenant IS NULL"]
            params = {}

            if cats:
                cypher_parts.append("""
                OPTIONAL MATCH (cat:Category)-[:HAS_PRODUCT]->(p)
                OPTIONAL MATCH (p)-[:SUITABLE_FOR]->(uc:UseCase)
                """)
            if feats:
                cypher_parts.append("""
                OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
                OPTIONAL MATCH (p)-[:AVAILABLE_IN_COLOR]->(co:ColorOption)
                OPTIONAL MATCH (p)-[:AVAILABLE_IN_WATTAGE]->(wo:WattageOption)
                """)

            # Build filtering conditions
            conds = []
            if cats:
                conds.append("(any(c IN $cats WHERE toLower(cat.name) CONTAINS toLower(c) OR toLower(uc.name) CONTAINS toLower(c)))")
                params["cats"] = cats
            if feats:
                conds.append("(any(f_tag IN $feats WHERE toLower(f.name) CONTAINS toLower(f_tag) OR toLower(co.name) CONTAINS toLower(f_tag) OR toLower(wo.name) CONTAINS toLower(f_tag)))")
                params["feats"] = feats

            if conds:
                cypher_parts.append("WITH p, " + " OR ".join(conds) + " AS match_flag WHERE match_flag")

            cypher_parts.append("RETURN DISTINCT p.sku AS sku, p.name AS name, 1.0 AS graph_score LIMIT 15")
            cypher = "\n".join(cypher_parts)
            
            logger.info(f"[DEBUG-NEO4J] Executing relational graph traversal with params: {params}")
            res = AgentConfig.graph.query(cypher, params=params)
            logger.info(f"[DEBUG-NEO4J] Relational traversal matched {len(res) if res else 0} nodes. Top SKUs: {[r.get('sku') for r in (res or [])[:5]]}")
            return res if res else []
        except Exception as e:
            logger.error(f"Neo4j graph search error: {e}")
            return []

    def _vector_search(self, intent_data: dict, query: str) -> list[dict]:
        """Synchronous Vector search across product FAQs, policies, or general knowledge."""
        try:
            intent = intent_data.get("intent")
            results = []

            # 1. If policy check, search policy vector store or general knowledge
            if intent == QueryIntent.CHECK_POLICY:
                if AgentConfig.policy_vector_store:
                    docs = AgentConfig.policy_vector_store.similarity_search_with_score(query, k=3)
                    for doc, score in docs:
                        results.append({"type": "policy", "text": doc.page_content, "score": score})
                return results

            # 2. If advice or general FAQ, search product FAQ or general knowledge
            if intent == QueryIntent.GET_ADVICE:
                if AgentConfig.product_faq_vector_store:
                    docs = AgentConfig.product_faq_vector_store.similarity_search_with_score(query, k=3)
                    for doc, score in docs:
                        results.append({"type": "faq", "text": doc.page_content, "score": score, "metadata": doc.metadata})
                return results

            # 3. For product search, search product FAQs to find matching product URLs/SKUs
            if AgentConfig.product_faq_vector_store:
                logger.info(f"[DEBUG-VECTOR] Executing similarity search over product FAQ vector store for query: '{query}'")
                docs = AgentConfig.product_faq_vector_store.similarity_search_with_score(query, k=5)
                for doc, score in docs:
                    # Extract product URL from metadata if available
                    meta = doc.metadata or {}
                    url = meta.get("product_url", "")
                    sku = None
                    if url and "/products/" in url:
                        sku_slug = url.split("/products/")[-1].strip("/")
                        results.append({"type": "product_vector", "sku_slug": sku_slug, "text": doc.page_content, "score": score})
                logger.info(f"[DEBUG-VECTOR] FAQ vector search matched {len(results)} product references. Top slugs: {[r.get('sku_slug') for r in results[:5]]}")
            return results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def _text_search(self, intent_data: dict, query: str) -> list[dict]:
        """Synchronous Text/SQL Keyword search across SQLite catalog using structured filters and tokens."""
        try:
            filters = intent_data.get("filters", {}) or {}
            cats = intent_data.get("category_keywords", []) or []
            feats = intent_data.get("feature_keywords", []) or []
            prod_name = intent_data.get("product_name")

            # Combine category from filters and category_keywords
            cat_filter = filters.get("category")
            all_cats = list(set([c for c in cats if c] + ([cat_filter] if cat_filter else [])))

            # Extract text tokens
            raw_text = f"{query} {prod_name or ''} {' '.join(all_cats)} {' '.join(feats)} {filters.get('application') or ''} {filters.get('segment') or ''}"
            tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9-]{2,}\b", raw_text) if w.lower() not in STOPWORDS]

            with get_session() as session:
                q = session.query(Product)

                # If specific category or application/segment filter is present
                if all_cats:
                    cat_conds = [
                        or_(
                            Product.categories.ilike(f"%{c}%"),
                            Product.use_cases.ilike(f"%{c}%"),
                            Product.description.ilike(f"%{c}%")
                        )
                        for c in all_cats
                    ]
                    q = q.filter(or_(*cat_conds))

                # Match tokens across product fields
                if tokens:
                    token_conds = []
                    for t in tokens[:6]:
                        token_conds.append(or_(
                            Product.name.ilike(f"%{t}%"),
                            Product.sku.ilike(f"%{t}%"),
                            Product.categories.ilike(f"%{t}%"),
                            Product.features.ilike(f"%{t}%"),
                            Product.use_cases.ilike(f"%{t}%"),
                            Product.color_options.ilike(f"%{t}%"),
                            Product.wattage_options.ilike(f"%{t}%")
                        ))
                    if token_conds and not all_cats:
                        q = q.filter(or_(*token_conds))

                # Check price preferences
                prefs = intent_data.get("preferences", {}) or {}
                max_p = prefs.get("max_price")
                if max_p and isinstance(max_p, (int, float)):
                    q = q.filter(Product.price_num <= max_p, Product.price_num > 0)
                min_p = prefs.get("min_price")
                if min_p and isinstance(min_p, (int, float)):
                    q = q.filter(Product.price_num >= min_p)

                logger.info(f"[DEBUG-TEXT/SQL] Executing SQLite filter query | Categories: {all_cats} | Tokens: {tokens[:6]} | Price range: {min_p} - {max_p}")
                prods = q.order_by(Product.rating_score.desc()).limit(15).all()
                logger.info(f"[DEBUG-TEXT/SQL] SQLite filter query matched {len(prods)} products. Top SKUs: {[p.sku for p in prods[:5]]}")
                return [{"type": "text", "sku": p.sku, "name": p.name, "score": 1.0} for p in prods]
        except Exception as e:
            logger.error(f"Text/SQL search error: {e}")
            return []

    async def retrieve(self, query: str, intent_data: dict) -> tuple[list[dict], list[dict], list[dict]]:
        """Parallel retrieval from Neo4j Graph (Lucene), Vector Store (BM25/Semantic), and SQLite (Text/SQL)."""
        vector_task = asyncio.create_task(asyncio.to_thread(self._vector_search, intent_data, query))
        graph_task = asyncio.create_task(asyncio.to_thread(self._graph_search, intent_data, query))
        text_task = asyncio.create_task(asyncio.to_thread(self._text_search, intent_data, query))

        vector_results, graph_results, text_results = await asyncio.gather(vector_task, graph_task, text_task)
        return vector_results, graph_results, text_results

    def _fuse_results(self, vector_results: list[dict], graph_results: list[dict], text_results: list[dict], n: int = 8) -> tuple[list[str], list[str]]:
        """
        Reciprocal Rank Fusion (RRF) across Vector/BM25 + Graph/Lucene + Text/SQL results.
        Returns (top_skus_to_hydrate, non_product_contexts).
        """
        scores: dict[str, float] = {}
        non_product_contexts: list[str] = []
        k = 60  # RRF constant

        # Process Vector / BM25 Results
        for rank, r in enumerate(vector_results):
            rtype = r.get("type")
            if rtype in ("policy", "faq"):
                non_product_contexts.append(r.get("text", ""))
            elif rtype == "product_vector":
                slug = r.get("sku_slug")
                if slug:
                    scores[slug.lower()] = scores.get(slug.lower(), 0) + 1 / (k + rank + 1)

        # Process Graph / Lucene Results
        for rank, r in enumerate(graph_results):
            sku = r.get("sku")
            if sku:
                scores[sku.lower()] = scores.get(sku.lower(), 0) + 1 / (k + rank + 1)

        # Process Text / SQL Results
        for rank, r in enumerate(text_results):
            sku = r.get("sku")
            if sku:
                scores[sku.lower()] = scores.get(sku.lower(), 0) + 1 / (k + rank + 1)

        sorted_skus = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        top_n = sorted_skus[:n]
        logger.info(f"[DEBUG-RRF] Fused ranks across 3 channels. Top SKUs: {top_n} | Scores: {{k: round(scores[k], 4) for k in top_n}}")
        return top_n, non_product_contexts

    def _hydrate_from_sqlite(self, skus: list[str], preferences: dict, query: str = "", category_keywords: list[str] = None) -> list[dict]:
        """SQL Hydration step: Fetch authoritative product details, prices, and specs from SQLite based on RRF rank."""
        logger.info(f"[DEBUG-HYDRATE] Hydrating SKUs from SQLite: {skus} | Category constraints: {category_keywords}")
        if not skus:
            # If no specific SKUs ranked, fetch top rated products from SQL matching preferences
            with get_session() as session:
                q = session.query(Product)
                if category_keywords:
                    cat_conds = [
                        or_(
                            Product.categories.ilike(f"%{c}%"),
                            Product.use_cases.ilike(f"%{c}%")
                        )
                        for c in category_keywords
                    ]
                    q = q.filter(or_(*cat_conds))
                max_p = preferences.get("max_price")
                if max_p and isinstance(max_p, (int, float)):
                    q = q.filter(Product.price_num <= max_p, Product.price_num > 0)
                min_p = preferences.get("min_price")
                if min_p and isinstance(min_p, (int, float)):
                    q = q.filter(Product.price_num >= min_p)
                products = q.order_by(Product.rating_score.desc()).limit(5).all()
                skus = [p.sku.lower() for p in products]

        hydrated = []
        with get_session() as session:
            for sku_key in skus:
                # Match by SKU or by url slug
                prod = session.query(Product).filter(
                    (func.lower(Product.sku) == sku_key) | (func.lower(Product.url).contains(sku_key))
                ).first()
                if not prod:
                    continue

                if category_keywords:
                    # Strict category enforcement: ensure product category matches requested category keywords
                    cat_match = False
                    for c_kw in category_keywords:
                        c_clean = c_kw.lower().strip()
                        if prod.categories and (c_clean in prod.categories.lower() or prod.categories.lower() in c_clean):
                            cat_match = True
                        elif prod.use_cases and c_clean in prod.use_cases.lower() and not any(noisy in (prod.categories or "").lower() for noisy in ["gate", "outdoor", "street", "bollard", "solar"]):
                            cat_match = True
                        else:
                            # Dynamically check if product's SQLite category matches Neo4j collection mapping
                            for col_name, sqlite_cats in AgentConfig.collection_to_sqlite_cats.items():
                                if (c_clean == col_name.lower() or c_clean in col_name.lower() or col_name.lower() in c_clean) and prod.categories in sqlite_cats:
                                    cat_match = True
                                    break
                            if not cat_match:
                                # Also check category groups
                                for group_name, col_names in AgentConfig.category_groups.items():
                                    if c_clean == group_name.lower() or c_clean in group_name.lower():
                                        if any(prod.categories in AgentConfig.collection_to_sqlite_cats.get(col, set()) for col in col_names):
                                            cat_match = True
                                            break
                    if not cat_match:
                        continue

                # Check price preferences
                max_p = preferences.get("max_price")
                if max_p and isinstance(max_p, (int, float)) and prod.price_num > max_p:
                    continue
                min_p = preferences.get("min_price")
                if min_p and isinstance(min_p, (int, float)) and prod.price_num < min_p:
                    continue

                # Check wattage preference against product wattage, options, variants, or specs
                pref_w = preferences.get("wattage")
                if pref_w:
                    pref_w_str = str(pref_w).lower().replace("w", "").strip()
                    has_w = False
                    if prod.wattage and str(prod.wattage) == pref_w_str:
                        has_w = True
                    elif prod.wattage_options and f"{pref_w_str}w" in prod.wattage_options.lower():
                        has_w = True
                    elif any(v.wattage_option and f"{pref_w_str}w" in str(v.wattage_option).lower() for v in prod.variants):
                        has_w = True
                    elif any("wattage" in s.spec_key.lower() and f"{pref_w_str}w" in s.spec_value.lower() for s in prod.specs):
                        has_w = True
                    if not has_w and prod.wattage_options and prod.wattage_options.strip():
                        # If product has strict wattage options and does not match requested wattage, exclude
                        continue

                # Check color preference against product color options, variants, or specs
                pref_c = preferences.get("color")
                if pref_c and isinstance(pref_c, str):
                    pref_c_lower = pref_c.lower().strip()
                    has_c = False
                    if prod.color_options and pref_c_lower in prod.color_options.lower():
                        has_c = True
                    elif any(v.color_option and pref_c_lower in str(v.color_option).lower() for v in prod.variants):
                        has_c = True
                    elif any("color" in s.spec_key.lower() and pref_c_lower in s.spec_value.lower() for s in prod.specs):
                        has_c = True
                    elif prod.description and pref_c_lower in prod.description.lower():
                        has_c = True
                    if not has_c and prod.color_options and prod.color_options.strip():
                        # If product lists specific color options and does not match requested color, exclude
                        continue

                specs_dict = {s.spec_key: s.spec_value for s in prod.specs[:6]}
                variants_list = [
                    {"sku": v.variant_sku, "color": v.color_option, "wattage": v.wattage_option, "price": v.price_num}
                    for v in prod.variants[:4]
                ]

                hydrated.append({
                    "sku": prod.sku,
                    "name": prod.name,
                    "price_num": prod.price_num,
                    "regular_price": prod.regular_price,
                    "discount_percentage": prod.discount_percentage,
                    "rating_score": prod.rating_score,
                    "review_count": prod.review_count,
                    "url": prod.url,
                    "image_url": prod.image_url,
                    "description": prod.feature_descriptions or prod.description,
                    "categories": prod.categories,
                    "features": prod.features,
                    "specs": specs_dict,
                    "variants": variants_list
                })

        return hydrated

    def _build_context(self, products: list[dict], non_prod_contexts: list[str]) -> str:
        """Build a structured context string for Azure OpenAI synthesis."""
        lines = []

        if non_prod_contexts:
            lines.append("RELEVANT FAQ & POLICY INFORMATION:")
            for idx, ctx in enumerate(non_prod_contexts, 1):
                lines.append(f"[{idx}] {ctx}\n")
            lines.append("---")

        if products:
            lines.append("AUTHORITATIVE PRODUCTS FROM SQLITE CATALOG:")
            for i, p in enumerate(products, 1):
                lines.append(f"{i}. {p['name']} (SKU: {p['sku']})")
                lines.append(f"   Price: Rs. {p['price_num']} (MRP: Rs. {p['regular_price']}, {p['discount_percentage']}% off)")
                lines.append(f"   Rating: {p['rating_score']} stars ({p['review_count']} reviews)")
                if p['categories']:
                    lines.append(f"   Categories: {p['categories']}")
                if p['features']:
                    lines.append(f"   Features: {p['features']}")
                if p['specs']:
                    specs_str = ", ".join([f"{k}: {v}" for k, v in p['specs'].items()])
                    lines.append(f"   Specifications: {specs_str}")
                if p['url']:
                    lines.append(f"   Product URL: {p['url']}")
                lines.append("")
        elif not non_prod_contexts:
            lines.append("No specific matching products found in the catalog.")

        return "\n".join(lines)

    async def synthesize_response(self, query: str, context: str, intent_data: dict, history_context: str = "") -> str:
        """Use Azure OpenAI LLM to generate a persuasive, helpful response."""
        from langchain_core.messages import SystemMessage, HumanMessage
        history_part = f"Previous Conversation Flow:\n{history_context}\n\n" if history_context else ""
        prompt = f"""{history_part}User Query: {query}

Detected Intent: {intent_data.get('intent')}
Extracted Preferences: {json.dumps(intent_data.get('preferences', {}))}

{context}

Please write a helpful, professional sales response based on the above catalog and policy context, continuing the conversation flow seamlessly.
Always include product prices in ₹ and direct product URLs when recommending products."""

        try:
            messages = [
                SystemMessage(content=RESPONSE_SYSTEM),
                HumanMessage(content=prompt)
            ]
            response = await asyncio.to_thread(AgentConfig.llm.invoke, messages)
            res_text = response.content.strip()
            logger.info(f"[DEBUG-LLM] Synthesized final response:\n{res_text}")
            return res_text
        except Exception as e:
            logger.error(f"Response synthesis error: {e}")
            return f"Here are the lighting details matching your query:\n\n{context}"

    def _category_browse_from_sqlite(self, category_keywords: list[str], preferences: dict) -> list[dict]:
        """Return ALL products in a matched category for browse/listing queries (not limited to RAG top-k)."""
        with get_session() as session:
            q = session.query(Product)
            
            # Build OR conditions across all category keywords
            cat_conds = []
            for kw in category_keywords:
                kw_lower = kw.lower().strip()
                cat_conds.append(Product.categories.ilike(f"%{kw_lower}%"))
                cat_conds.append(Product.use_cases.ilike(f"%{kw_lower}%"))
                cat_conds.append(Product.name.ilike(f"%{kw_lower}%"))
                
                # Dynamically map Neo4j collection keywords to SQLite categories via graph schema
                for col_name, sqlite_cats in AgentConfig.collection_to_sqlite_cats.items():
                    if kw_lower == col_name.lower() or kw_lower in col_name.lower() or col_name.lower() in kw_lower:
                        for sc in sqlite_cats:
                            cat_conds.append(Product.categories.ilike(f"%{sc}%"))
                
                for group_name, col_names in AgentConfig.category_groups.items():
                    if kw_lower == group_name.lower() or kw_lower in group_name.lower():
                        for col_name in col_names:
                            for sc in AgentConfig.collection_to_sqlite_cats.get(col_name, set()):
                                cat_conds.append(Product.categories.ilike(f"%{sc}%"))
            
            if cat_conds:
                logger.info(f"[DEBUG-SQLITE-BROWSE] Executing category browse query with keywords: {category_keywords}")
                q = q.filter(or_(*cat_conds))
            else:
                logger.info(f"[DEBUG-SQLITE-BROWSE] No category filter conditions generated for keywords: {category_keywords}")
            
            # Apply price preferences
            max_p = preferences.get("max_price")
            if max_p and isinstance(max_p, (int, float)):
                q = q.filter(Product.price_num <= max_p, Product.price_num > 0)
            min_p = preferences.get("min_price")
            if min_p and isinstance(min_p, (int, float)):
                q = q.filter(Product.price_num >= min_p)
            
            products = q.order_by(Product.rating_score.desc()).all()
            matched_summary = [f"'{p.name}' (SKU: {p.sku}, Cat: '{p.categories}', UseCases: '{p.use_cases}')" for p in products[:5]]
            logger.info(f"[DEBUG-SQLITE-BROWSE] Matched {len(products)} total products. Top 5 samples: {matched_summary}")
            
            hydrated = []
            for prod in products:
                specs_dict = {s.spec_key: s.spec_value for s in prod.specs[:6]}
                variants_list = [
                    {"sku": v.variant_sku, "color": v.color_option, "wattage": v.wattage_option, "price": v.price_num}
                    for v in prod.variants[:4]
                ]
                hydrated.append({
                    "sku": prod.sku,
                    "name": prod.name,
                    "price_num": prod.price_num,
                    "regular_price": prod.regular_price,
                    "discount_percentage": prod.discount_percentage,
                    "rating_score": prod.rating_score,
                    "review_count": prod.review_count,
                    "url": prod.url,
                    "image_url": prod.image_url,
                    "description": prod.feature_descriptions or prod.description,
                    "categories": prod.categories,
                    "features": prod.features,
                    "specs": specs_dict,
                    "variants": variants_list
                })
            
            logger.info(f"Category browse returned {len(hydrated)} products for keywords: {category_keywords}")
            return hydrated

    async def query(self, user_query: str, session_id: Optional[str] = None) -> QueryResult:
        """Full Linear GraphRAG pipeline: check DB history → taxonomy → classify → retrieve → fuse → hydrate → synthesize."""
        logger.info(f"GraphRAG query: {user_query!r} | session_id: {session_id}")

        # Step 0: Check user's previous messages in DB if session_id is provided
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

        # Step 0.5: Taxonomy resolution - embed query and match against Pinecone taxonomy-cache
        taxonomy_hints = {}
        try:
            from src.services.agent.taxonomy import fetch_taxonomy_candidates
            query_embedding = await asyncio.to_thread(AgentConfig.embeddings.embed_query, user_query)
            taxonomy_hints = await asyncio.to_thread(fetch_taxonomy_candidates, query_embedding, 0.80)
            if taxonomy_hints:
                logger.info(f"Taxonomy resolved: {taxonomy_hints}")
        except Exception as e:
            logger.warning(f"Taxonomy resolution failed (non-fatal): {e}")

        # Step 1: Classify intent with conversation history AND taxonomy hints
        intent_data = await self.classify_intent(user_query, history_context=history_context, taxonomy_hints=taxonomy_hints)
        intent = QueryIntent(intent_data.get("intent", "unknown"))
        logger.info(f"Classified Intent: {intent} | Keywords: {intent_data.get('category_keywords')} {intent_data.get('feature_keywords')}")

        if intent == QueryIntent.BROWSE_CATEGORY:
            cat_kws = intent_data.get("category_keywords", [])
            filters = intent_data.get("filters", {}) or {}
            # Only enrich with structured filter category
            if filters.get("category") and filters["category"] not in cat_kws:
                cat_kws.append(filters["category"])
            
            # Top-level category navigation should ONLY trigger when the query is explicitly asking for broad level-1 category groups (Outdoor, Indoor, Solar) without specifying a particular sub-category or collection name.
            is_top_level = not filters.get("category") and (
                filters.get("application") in ["Outdoor", "Indoor", "Solar"]
                or any(k.strip().lower() in [g.lower() for g in AgentConfig.top_level_groups] or k.strip().lower() in ['outdoor lighting', 'indoor lighting', 'solar lighting', 'lighting', 'lights'] for k in cat_kws)
            )
            if is_top_level:
                logger.info(f"Detected TOP-LEVEL category navigation query for keywords: {cat_kws} | Application: {filters.get('application')}")
                app = str(filters.get("application") or "").lower()
                
                lines = ["AVAILABLE SPECIALIZED COLLECTIONS IN NEO4J (Do NOT list individual products/SKUs or prices; instead, introduce these available collections clearly and ask the customer which collection they would like to explore):"]
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
                response = await self.synthesize_response(user_query, context_text, intent_data, history_context=history_context)
                
                # Save to memory
                if session_id and AgentConfig.memory_provider and hasattr(AgentConfig.memory_provider, "add_message"):
                    try:
                        from langchain_core.messages import HumanMessage as HM, AIMessage
                        AgentConfig.memory_provider.add_message(session_id, HM(content=user_query))
                        AgentConfig.memory_provider.add_message(session_id, AIMessage(content=response))
                    except Exception as e:
                        logger.warning(f"Could not store interaction: {e}")
                
                return QueryResult(intent=intent, products=[], context_text=context_text, response=response, product_links=[])
            
            hydrated_products = self._category_browse_from_sqlite(cat_kws, intent_data.get("preferences", {}))
            
            if hydrated_products:
                context_text = self._build_context(hydrated_products, [])
                response = await self.synthesize_response(user_query, context_text, intent_data, history_context=history_context)
                
                # Save to memory
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
            # If no products matched the category, fall through to normal RAG pipeline
            logger.info("BROWSE_CATEGORY found 0 products, falling through to RAG pipeline")

        # Step 2: Parallel retrieval (Vector/BM25 + Graph/Lucene + Text/SQL)
        vector_results, graph_results, text_results = await self.retrieve(user_query, intent_data)
        logger.info(f"Retrieved: {len(vector_results)} vector items, {len(graph_results)} graph items, {len(text_results)} text items")

        # Step 3: Reciprocal Rank Fusion across all 3 channels
        fused_skus, non_prod_contexts = self._fuse_results(vector_results, graph_results, text_results)

        # Step 4: SQL Hydration from SQLite
        hydrated_products = self._hydrate_from_sqlite(fused_skus, intent_data.get("preferences", {}), query=user_query, category_keywords=intent_data.get("category_keywords"))
        logger.info(f"Hydrated {len(hydrated_products)} authoritative product cards from SQLite")

        # Step 5: Build context
        context_text = self._build_context(hydrated_products, non_prod_contexts)

        # Step 6: Response synthesis with conversation history
        response = await self.synthesize_response(user_query, context_text, intent_data, history_context=history_context)

        # Step 7: Save interaction back to memory provider if session_id is present
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
