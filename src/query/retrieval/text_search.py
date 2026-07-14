"""
retrieval/text_search.py — SQLite keyword/token search and category browsing.
Decoupled from hardcoded wattage/color algorithms and domain exclusion arrays.
"""

import logging
from typing import List, Dict, Any
from sqlalchemy import or_, func
from src.db.database import get_session
from src.db.models import Product
from src.services.agent.config import AgentConfig


def get_stopwords():
    """Returns search stop words combining base English query words with dynamic config."""
    base_stopwords = {
        "give", "me", "show", "tell", "about", "what", "are", "the", "is", "for", "please",
        "some", "any", "get", "find", "looking", "want", "need", "buy", "at", "in", "of",
        "on", "to", "with", "from", "by", "an", "a", "under", "below", "above", "range", "cost", "price",
        "can", "you", "do", "have", "we", "our"
    }
    config_stopwords = AgentConfig.get_stop_words()
    return base_stopwords.union(config_stopwords)

logger = logging.getLogger(__name__)


def _matches_preference(prod: Product, pref_key: str, pref_val: Any) -> bool:
    """
    Generic option/attribute matcher. Evaluates whether a product satisfies a user preference
    by checking product fields, option lists, variants, and key-value specifications.
    """
    if pref_val is None:
        return True
    
    val_str = str(pref_val).lower().strip()
    key_lower = pref_key.lower().strip()

    # 1. Check direct product attributes (e.g. prod.color, prod.wattage, prod.size)
    direct_val = getattr(prod, key_lower, None)
    if direct_val is not None and val_str in str(direct_val).lower():
        return True

    # 2. Check option list attributes (e.g. prod.color_options, prod.wattage_options)
    options_val = getattr(prod, f"{key_lower}_options", None)
    if options_val is not None and val_str in str(options_val).lower():
        return True

    # 3. Check variant options (e.g. variant.color_option, variant.wattage_option)
    for v in (prod.variants or []):
        v_opt = getattr(v, f"{key_lower}_option", None) or getattr(v, key_lower, None)
        if v_opt is not None and val_str in str(v_opt).lower():
            return True

    # 4. Check key-value specifications
    for s in (prod.specs or []):
        if key_lower in str(s.spec_key).lower() and val_str in str(s.spec_value).lower():
            return True

    # 5. Check description/features fallback
    if prod.description and val_str in prod.description.lower():
        return True
    if prod.features and val_str in prod.features.lower():
        return True

    return False


def _hydrate_product_model(prod: Product) -> Dict[str, Any]:
    """Converts an SQLAlchemy Product model into a standardized dictionary representation."""
    specs_dict = {s.spec_key: s.spec_value for s in (prod.specs or [])[:6]}
    variants_list = []
    for v in (prod.variants or [])[:4]:
        v_dict = {"sku": v.variant_sku, "price": v.price_num}
        if hasattr(v, "__table__"):
            for col in v.__table__.columns:
                if col.name.endswith("_option"):
                    v_dict[col.name.replace("_option", "")] = getattr(v, col.name, None)
        else:
            for attr in dir(v):
                if attr.endswith("_option") and not attr.startswith("_"):
                    v_dict[attr.replace("_option", "")] = getattr(v, attr, None)
        variants_list.append(v_dict)
    return {
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
    }


def _expand_sqlite_categories(categories: List[str]) -> List[str]:
    expanded = set()
    for cat in categories:
        if not cat:
            continue
        c_clean = str(cat).strip()
        expanded.add(c_clean)
        for k, v in AgentConfig.collection_to_sqlite_cats.items():
            if c_clean.lower() == k.lower() or c_clean.lower() in k.lower() or k.lower() in c_clean.lower():
                expanded.update(v)
        for k, v in AgentConfig.category_to_sqlite_cats.items():
            if c_clean.lower() == k.lower() or c_clean.lower() in k.lower() or k.lower() in c_clean.lower():
                expanded.update(v)
    return list(expanded)


def text_search(intent_data: dict, query: str) -> List[Dict[str, Any]]:
    """Synchronous keyword search across SQLite catalog using structured filters and tokens."""
    try:
        filters = intent_data.get("filters", {}) or {}
        cats = intent_data.get("category_keywords", []) or []
        feats = intent_data.get("feature_keywords", []) or []
        prod_name = intent_data.get("product_name")
        preferences = intent_data.get("preferences", {}) or {}

        cat_filter = filters.get("category")
        all_cats = list(set([c for c in cats if c] + ([cat_filter] if cat_filter else [])))
        expanded_cats = _expand_sqlite_categories(all_cats)

        raw_text = f"{query} {prod_name or ''} {' '.join(all_cats)} {' '.join(feats)} {filters.get('application') or ''} {filters.get('segment') or ''}"
        tokens = [t.lower().strip() for t in raw_text.split() if len(t) > 2 and t.lower() not in get_stopwords()]

        with get_session() as session:
            q = session.query(Product)
            if prod_name:
                q = q.filter(or_(Product.name.ilike(f"%{prod_name}%"), Product.sku.ilike(f"%{prod_name}%")))
            elif expanded_cats:
                cat_conds = []
                for c in all_cats:
                    if not c: continue
                    c_clean = str(c).strip().lower()
                    for col_name, skus in AgentConfig.collection_to_skus.items():
                        if c_clean == col_name.lower() or c_clean in col_name.lower() or col_name.lower() in c_clean:
                            cat_conds.append(Product.sku.in_([s.upper() for s in skus if s]))
                    for group_name, skus in AgentConfig.group_to_skus.items():
                        g_lower = group_name.lower()
                        if c_clean == g_lower or c_clean == f"{g_lower} lights" or c_clean == f"{g_lower} collections" or (len(c_clean.split()) <= 2 and g_lower in c_clean):
                            cat_conds.append(Product.sku.in_([s.upper() for s in skus if s]))
                for c in expanded_cats:
                    cat_conds.append(Product.categories.ilike(f"%{c}%"))
                    cat_conds.append(Product.use_cases.ilike(f"%{c}%"))
                if cat_conds:
                    q = q.filter(or_(*cat_conds))
            elif tokens:
                token_conds = []
                for t in tokens[:4]:
                    token_conds.append(Product.name.ilike(f"%{t}%"))
                    token_conds.append(Product.sku.ilike(f"%{t}%"))
                    token_conds.append(Product.categories.ilike(f"%{t}%"))
                    token_conds.append(Product.use_cases.ilike(f"%{t}%"))
                    token_conds.append(Product.features.ilike(f"%{t}%"))
                q = q.filter(or_(*token_conds))

            candidates = q.order_by(Product.rating_score.desc()).limit(30).all()
            hydrated: List[Dict[str, Any]] = []

            for prod in candidates:
                # 1. Enforce category matching if requested
                if expanded_cats:
                    cat_match = False
                    prod_cats = (prod.categories or "").lower()
                    prod_ucs = (prod.use_cases or "").lower()
                    prod_sku = (prod.sku or "").upper()
                    for c in all_cats:
                        if not c: continue
                        c_clean = str(c).strip().lower()
                        for col_name, skus in AgentConfig.collection_to_skus.items():
                            if c_clean == col_name.lower() or c_clean in col_name.lower() or col_name.lower() in c_clean:
                                if prod_sku in [s.upper() for s in skus if s]: cat_match = True
                        for group_name, skus in AgentConfig.group_to_skus.items():
                            g_lower = group_name.lower()
                            if c_clean == g_lower or c_clean == f"{g_lower} lights" or c_clean == f"{g_lower} collections" or (len(c_clean.split()) <= 2 and g_lower in c_clean):
                                if prod_sku in [s.upper() for s in skus if s]: cat_match = True
                    for c in expanded_cats:
                        c_lower = c.lower().strip()
                        if c_lower in prod_cats or c_lower in prod_ucs:
                            cat_match = True
                            break
                    if not cat_match:
                        continue
                else:
                    cat_match = True

                # 2. Enforce feature/token matching
                if tokens and not expanded_cats:
                    token_match = False
                    prod_text = f"{prod.name or ''} {prod.categories or ''} {prod.use_cases or ''} {prod.features or ''} {prod.sku or ''}".lower()
                    for t in tokens:
                        if t in prod_text:
                            token_match = True
                            break
                    if not token_match:
                        continue

                # 3. Check numeric price bounds
                max_p = preferences.get("max_price")
                if max_p and isinstance(max_p, (int, float)) and prod.price_num > max_p:
                    continue
                min_p = preferences.get("min_price")
                if min_p and isinstance(min_p, (int, float)) and prod.price_num < min_p:
                    continue

                # 4. Check dynamic attribute/option preferences (e.g. color, wattage, size, material)
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

            return hydrated
    except Exception as e:
        logger.error(f"Text search error: {e}", exc_info=True)
        return []


def category_browse_from_sqlite(category_keywords: List[str], preferences: dict) -> List[Dict[str, Any]]:
    """Return products in a matched category for browse/listing queries."""
    expanded_cats = _expand_sqlite_categories(category_keywords)
    target_skus = set()
    exact_collection_matched = False
    for kw in category_keywords:
        if not kw: continue
        kw_clean = str(kw).strip().lower()
        for col_name, skus in AgentConfig.collection_to_skus.items():
            if kw_clean == col_name.lower():
                target_skus.update([s.upper() for s in skus if s])
                exact_collection_matched = True
        for group_name, skus in AgentConfig.group_to_skus.items():
            g_lower = group_name.lower()
            if kw_clean == g_lower or kw_clean == f"{g_lower} lights" or kw_clean == f"{g_lower} collections":
                target_skus.update([s.upper() for s in skus if s])
                exact_collection_matched = True

    if not exact_collection_matched:
        for kw in category_keywords:
            if not kw: continue
            kw_clean = str(kw).strip().lower()
            for col_name, skus in AgentConfig.collection_to_skus.items():
                if kw_clean in col_name.lower() or col_name.lower() in kw_clean:
                    target_skus.update([s.upper() for s in skus if s])
            for group_name, skus in AgentConfig.group_to_skus.items():
                g_lower = group_name.lower()
                if (len(kw_clean.split()) <= 2 and g_lower in kw_clean):
                    target_skus.update([s.upper() for s in skus if s])

    with get_session() as session:
        q = session.query(Product)
        if exact_collection_matched and target_skus:
            q = q.filter(Product.sku.in_(list(target_skus)))
        else:
            cat_conds = []
            if target_skus:
                cat_conds.append(Product.sku.in_(list(target_skus)))
            for kw in expanded_cats:
                kw_lower = kw.lower().strip()
                cat_conds.append(Product.categories.ilike(f"%{kw_lower}%"))
                cat_conds.append(Product.use_cases.ilike(f"%{kw_lower}%"))
                cat_conds.append(Product.name.ilike(f"%{kw_lower}%"))

            if cat_conds:
                q = q.filter(or_(*cat_conds))

        max_p = preferences.get("max_price")
        if max_p and isinstance(max_p, (int, float)):
            q = q.filter(Product.price_num <= max_p, Product.price_num > 0)
        min_p = preferences.get("min_price")
        if min_p and isinstance(min_p, (int, float)):
            q = q.filter(Product.price_num >= min_p)

        products = q.order_by(Product.rating_score.desc()).all()
        return [_hydrate_product_model(p) for p in products]


