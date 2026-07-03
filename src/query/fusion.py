"""
fusion.py — Reciprocal Rank Fusion (RRF), SQLite hydration, and context building.
Decoupled from hardcoded product variants and domain-specific attributes.
"""

import logging
from typing import List, Dict, Any, Tuple
from sqlalchemy import func
from src.db.database import get_session
from src.db.models import Product
from src.query.retrieval.text_search import _hydrate_product_model, _matches_preference

logger = logging.getLogger(__name__)


def fuse_results(vector_results: list, graph_results: list, text_results: list = None) -> Tuple[List[str], List[str]]:
    """
    Combines ranked results from vector, graph, and text channels using Reciprocal Rank Fusion (RRF).
    Returns a tuple of (ranked_skus, non_product_contexts).
    """
    rrf_scores: Dict[str, float] = {}
    non_prod_contexts: List[str] = []
    text_results = text_results or []
    k = 60  # Standard RRF smoothing parameter

    # 1. Process Vector Results
    for rank, item in enumerate(vector_results):
        item_type = item.get("type")
        if item_type in ("policy", "faq"):
            text = item.get("text", "")
            if text and text not in non_prod_contexts:
                non_prod_contexts.append(text)
        elif item_type == "product_vector":
            sku_slug = item.get("sku_slug")
            if sku_slug:
                sku_clean = str(sku_slug).lower().strip()
                rrf_scores[sku_clean] = rrf_scores.get(sku_clean, 0.0) + (1.0 / (k + rank + 1))

    # 2. Process Graph Results
    for rank, item in enumerate(graph_results):
        sku = item.get("sku")
        if sku:
            sku_clean = str(sku).lower().strip()
            rrf_scores[sku_clean] = rrf_scores.get(sku_clean, 0.0) + (1.0 / (k + rank + 1)) * 1.5

    # 3. Process Text/SQL Keyword Results
    for rank, item in enumerate(text_results):
        sku = item.get("sku")
        if sku:
            sku_clean = str(sku).lower().strip()
            rrf_scores[sku_clean] = rrf_scores.get(sku_clean, 0.0) + (1.0 / (k + rank + 1)) * 2.0

    sorted_skus = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    logger.info(f"[DEBUG-RRF] Fused {len(sorted_skus)} unique SKUs from 3 channels. Top 5: {sorted_skus[:5]}")
    return sorted_skus, non_prod_contexts


def hydrate_from_sqlite(fused_skus: List[str], preferences: dict, query: str = "", category_keywords: list = None) -> List[Dict[str, Any]]:
    """
    Fetches authoritative product records from SQLite for the top fused SKUs.
    Preserves RRF ranking order.
    """
    if not fused_skus:
        return []

    try:
        with get_session() as session:
            prods = session.query(Product).filter(func.lower(Product.sku).in_([s.lower() for s in fused_skus])).all()
            prod_map = {p.sku.lower().strip(): p for p in prods if p.sku}

            preferences = preferences or {}
            max_p = preferences.get("max_price")
            min_p = preferences.get("min_price")

            hydrated: List[Dict[str, Any]] = []
            for sku in fused_skus:
                prod = prod_map.get(sku.lower().strip())
                if prod:
                    # Enforce price preferences during hydration
                    if max_p and isinstance(max_p, (int, float)) and prod.price_num > max_p:
                        continue
                    if min_p and isinstance(min_p, (int, float)) and prod.price_num < min_p:
                        continue
                    
                    # Enforce option/attribute preferences during hydration
                    opt_mismatch = False
                    for pref_k, pref_v in preferences.items():
                        if pref_k in ("min_price", "max_price", "price", "sort_by", "limit"):
                            continue
                        if not _matches_preference(prod, pref_k, pref_v):
                            opt_mismatch = True
                            break
                    if opt_mismatch:
                        continue

                    hydrated.append(_hydrate_product_model(prod))
                if len(hydrated) >= 10:
                    break

            logger.info(f"[DEBUG-HYDRATION] Hydrated {len(hydrated)} products from SQLite out of {len(fused_skus)} candidate SKUs.")
            return hydrated
    except Exception as e:
        logger.error(f"SQLite hydration error: {e}", exc_info=True)
        return []


def build_context(products: List[Dict[str, Any]], non_prod_contexts: List[str]) -> str:
    """Formats retrieved product cards and FAQ/Policy text into a clean context block for LLM synthesis."""
    lines: List[str] = []

    if non_prod_contexts:
        lines.append("=== RETRIEVED KNOWLEDGE / POLICIES ===")
        for idx, text in enumerate(non_prod_contexts, 1):
            lines.append(f"[{idx}] {text}")
        lines.append("")

    if products:
        lines.append("=== RETRIEVED CATALOG PRODUCTS ===")
        for idx, p in enumerate(products, 1):
            lines.append(f"Product {idx}: {p['name']} (SKU: {p['sku']})")
            lines.append(f"   Price: {p['price_num']}")
            try:
                if p.get("regular_price") and float(p["regular_price"]) > float(p.get("price_num", 0)):
                    lines.append(f"   Regular Price: {p['regular_price']} ({p.get('discount_percentage')}% OFF)")
            except (ValueError, TypeError):
                pass
            if p.get("categories"):
                lines.append(f"   Categories: {p['categories']}")
            if p.get("features"):
                lines.append(f"   Features: {p['features']}")
            if p.get("specs"):
                specs_str = ", ".join([f"{k}: {v}" for k, v in p["specs"].items()])
                lines.append(f"   Specifications: {specs_str}")
            if p.get("url"):
                lines.append(f"   Product URL: {p['url']}")
            lines.append("")
    elif not non_prod_contexts:
        lines.append("No specific matching items found in the catalog.")

    return "\n".join(lines)
