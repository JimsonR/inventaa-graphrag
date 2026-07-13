#!/usr/bin/env python3
"""
Data Migration Script: Neo4j -> SQLite (Tri-Store Architecture)

Extracts authoritative product details, specifications, and variants from Neo4j
and populates the local SQLite relational database (`inventaa_knowledge_base.db`).
This enables fast filtering, price range sorting, SKU lookups, and rich UI display cards
without burdening Neo4j semantic graph traversals.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load environment variables
with open('.env', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split('=', 1)
        os.environ[k.strip()] = v.strip()

from neo4j import GraphDatabase
from sqlalchemy.orm import Session

from src.db.database import get_session, init_db
from src.db.models import Product, ProductSpec, ProductVariant


def get_neo4j_driver():
    uri = os.getenv("NEO4J_URI", "").replace("neo4j+s://", "neo4j+ssc://")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not password:
        raise ValueError("NEO4J_URI and NEO4J_PASSWORD must be set in .env")
    return GraphDatabase.driver(uri, auth=(username, password))


def migrate():
    print("Initializing SQLite Database...")
    init_db()
    
    print("Connecting to Neo4j...")
    driver = get_neo4j_driver()
    
    query = """
    MATCH (p:Product)
    WHERE p.tenant = 'inventaa' OR p.tenant IS NULL
    OPTIONAL MATCH (t)-[]-(p)
    WHERE 'Category' IN labels(t) OR 'Collection' IN labels(t) OR 'Department' IN labels(t)
    OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
    OPTIONAL MATCH (p)-[:SUITABLE_FOR]->(uc:UseCase)
    OPTIONAL MATCH (p)-[:AVAILABLE_IN_COLOR]->(co:ColorOption)
    OPTIONAL MATCH (p)-[:AVAILABLE_IN_WATTAGE]->(wo:WattageOption)
    OPTIONAL MATCH (p)-[:HAS_SPEC]->(s:Spec)
    RETURN p,
           collect(DISTINCT t.name) AS categories,
           collect(DISTINCT f.name) AS features,
           collect(DISTINCT uc.name) AS use_cases,
           collect(DISTINCT co.name) AS color_options,
           collect(DISTINCT wo.name) AS wattage_options,
           collect(DISTINCT {key: s.key, val: s.value}) AS specs
    """
    
    with driver.session() as neo_session:
        print("Executing extraction query against Neo4j...")
        result = neo_session.run(query)
        records = list(result)
        print(f"Extracted {len(records)} products from Neo4j.")

    with get_session() as sql_session:
        print("Migrating products to SQLite...")
        
        # Clear existing tables for a clean idempotent migration
        sql_session.query(ProductVariant).delete()
        sql_session.query(ProductSpec).delete()
        sql_session.query(Product).delete()
        
        count_products = 0
        count_specs = 0
        count_variants = 0
        
        seen_skus = set()
        seen_ids = set()
        seen_variant_skus = set()
        
        for r in records:
            p_node = r["p"]
            raw_sku = p_node.get("sku") or p_node.get("id") or f"SKU_{count_products+1}"
            sku = str(raw_sku).strip()
            
            # If SKU is duplicated in Neo4j, append a suffix so all 125 products are imported
            base_sku = sku
            counter = 2
            while sku in seen_skus:
                sku = f"{base_sku} ({counter})"
                counter += 1
            seen_skus.add(sku)
            
            prod_id = str(p_node.get("id") or sku).strip()
            if prod_id in seen_ids:
                prod_id = f"{prod_id}_{sku}"
            seen_ids.add(prod_id)
            
            name = p_node.get("name") or f"Product {sku}"
            
            # Extract basic numeric properties
            try:
                price_num = int(float(p_node.get("price_num", 0)))
            except (ValueError, TypeError):
                price_num = 0
                
            try:
                discount = int(float(p_node.get("discount_percentage", 0)))
            except (ValueError, TypeError):
                discount = 0
                
            try:
                rating = float(p_node.get("rating_score", 0.0))
            except (ValueError, TypeError):
                rating = 0.0
                
            try:
                reviews = int(float(p_node.get("review_count", 0)))
            except (ValueError, TypeError):
                reviews = 0
                
            try:
                wattage = int(float(p_node.get("wattage", 0)))
            except (ValueError, TypeError):
                wattage = None
                
            # Clean CSV strings
            cats = sorted(list(set([c for c in r["categories"] if c])))
            feats = sorted(list(set([f for f in r["features"] if f])))
            ucs = sorted(list(set([u for u in r["use_cases"] if u])))
            colors = sorted(list(set([c for c in r["color_options"] if c])))
            watts = sorted(list(set([w for w in r["wattage_options"] if w])))
            
            product = Product(
                id=prod_id,
                sku=sku,
                name=name,
                price_num=price_num,
                regular_price=str(p_node.get("regular_price", "")),
                discount_percentage=discount,
                rating_score=rating,
                review_count=reviews,
                image_url=p_node.get("image_url"),
                url=p_node.get("url"),
                description=p_node.get("description"),
                feature_descriptions=p_node.get("feature_descriptions"),
                has_variants=bool(p_node.get("has_variants", False) or colors or watts),
                wattage=wattage,
                tenant=p_node.get("tenant", "inventaa"),
                categories=",".join(cats) if cats else None,
                features=",".join(feats) if feats else None,
                use_cases=",".join(ucs) if ucs else None,
                color_options=",".join(colors) if colors else None,
                wattage_options=",".join(watts) if watts else None,
            )
            sql_session.add(product)
            count_products += 1
            
            # Add Specs
            seen_spec_keys = set()
            for s in r["specs"]:
                if s.get("key") and s.get("val"):
                    spec_k = str(s["key"]).strip()
                    if spec_k in seen_spec_keys:
                        continue
                    seen_spec_keys.add(spec_k)
                    spec_entry = ProductSpec(
                        product_sku=sku,
                        spec_key=spec_k,
                        spec_value=str(s["val"]).strip()
                    )
                    sql_session.add(spec_entry)
                    count_specs += 1
                    
            # Create default variants if color or wattage options exist
            if colors or watts:
                for c in (colors or [None]):
                    for w in (watts or [None]):
                        v_sku = f"{sku}-{c or ''}-{w or ''}".strip('-')
                        if v_sku in seen_variant_skus:
                            continue
                        seen_variant_skus.add(v_sku)
                        variant = ProductVariant(
                            product_sku=sku,
                            variant_sku=v_sku,
                            color_option=c,
                            wattage_option=w,
                            price_num=price_num,
                            is_available=True
                        )
                        sql_session.add(variant)
                        count_variants += 1

        sql_session.commit()
        print("\n=== MIGRATION COMPLETE ===")
        print(f"  Migrated Products : {count_products}")
        print(f"  Migrated Specs    : {count_specs}")
        print(f"  Created Variants  : {count_variants}")
        print(f"  SQLite Database   : {os.path.join(os.getcwd(), 'data', 'db', 'inventaa_knowledge_base.db')}")


if __name__ == "__main__":
    migrate()
