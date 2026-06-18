"""
Inventaa Product Graph Retrieval Script
========================================
Usage:
    python retrieve_products.py "gate lights under 1000 rupees"
    python retrieve_products.py "solar garden light" --top 5
    python retrieve_products.py "10W outdoor" --max-price 1000
    python retrieve_products.py "pillar light" --category gate_pillar_lights
    python retrieve_products.py "LED flood" --wattage 20

Retrieval strategy:
  1. Full-text keyword search on Product name + description
  2. Graph traversal to enrich each match with Category, Features, Specs, UseCases
  3. Optional filters: max price, wattage, category slug
"""

import sys
import os
import argparse
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(r"c:\project SMB\.env")

NEO4J_URI      = os.getenv("NEO4J_URI").replace("neo4j+s://", "neo4j+ssc://")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

TENANT = "inventaa"

# ---------------------------------------------------------------------------
# Cypher Queries
# ---------------------------------------------------------------------------

# Full-text keyword search on product name + description (CONTAINS fallback)
# Neo4j CONTAINS is case-sensitive so we use toLower()
QUERY_PRODUCT_KEYWORD = """
MATCH (p:Product {tenant: $tenant})
WHERE toLower(p.name) CONTAINS toLower($keyword)
   OR toLower(p.description) CONTAINS toLower($keyword)
WITH p

// Optional filters
WHERE ($max_price IS NULL OR p.price_num <= $max_price)
  AND ($wattage    IS NULL OR p.wattage = $wattage)

// Graph traversal — enrich with connected nodes
OPTIONAL MATCH (b:Brand)-[:MAKES]->(p)
OPTIONAL MATCH (cat:Category)-[:HAS_PRODUCT]->(p)
OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
OPTIONAL MATCH (p)-[:HAS_SPEC]->(sp:Spec)
OPTIONAL MATCH (p)-[:SUITABLE_FOR]->(u:UseCase)
OPTIONAL MATCH (p)-[:HAS_WARRANTY]->(w:Warranty)
OPTIONAL MATCH (p)-[:HAS_POLICY]->(pol:Policy)

RETURN
    p.id            AS product_id,
    p.name          AS name,
    p.description   AS description,
    p.price_str     AS price,
    p.price_num     AS price_num,
    p.wattage       AS wattage,
    p.url           AS url,
    p.installation_url AS installation_url,
    b.name          AS brand,
    collect(DISTINCT cat.name) AS categories,
    collect(DISTINCT f.name)   AS features,
    collect(DISTINCT {key: sp.key, value: sp.value}) AS specs,
    collect(DISTINCT u.name)   AS use_cases,
    collect(DISTINCT w.description) AS warranties,
    collect(DISTINCT pol.title) AS policies
ORDER BY p.price_num ASC
LIMIT $top
"""

# Category-scoped search
QUERY_PRODUCT_BY_CATEGORY = """
MATCH (cat:Category {slug: $category_slug, tenant: $tenant})-[:HAS_PRODUCT]->(p:Product)
WHERE ($keyword IS NULL OR toLower(p.name) CONTAINS toLower($keyword)
                        OR toLower(p.description) CONTAINS toLower($keyword))
  AND ($max_price IS NULL OR p.price_num <= $max_price)
  AND ($wattage   IS NULL OR p.wattage = $wattage)

OPTIONAL MATCH (b:Brand)-[:MAKES]->(p)
OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
OPTIONAL MATCH (p)-[:HAS_SPEC]->(sp:Spec)
OPTIONAL MATCH (p)-[:SUITABLE_FOR]->(u:UseCase)
OPTIONAL MATCH (p)-[:HAS_WARRANTY]->(w:Warranty)
OPTIONAL MATCH (p)-[:HAS_POLICY]->(pol:Policy)

RETURN
    p.id            AS product_id,
    p.name          AS name,
    p.description   AS description,
    p.price_str     AS price,
    p.price_num     AS price_num,
    p.wattage       AS wattage,
    p.url           AS url,
    p.installation_url AS installation_url,
    b.name          AS brand,
    [cat.name]      AS categories,
    collect(DISTINCT f.name)   AS features,
    collect(DISTINCT {key: sp.key, value: sp.value}) AS specs,
    collect(DISTINCT u.name)   AS use_cases,
    collect(DISTINCT w.description) AS warranties,
    collect(DISTINCT pol.title) AS policies
ORDER BY p.price_num ASC
LIMIT $top
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_product(i: int, r: dict) -> None:
    specs_clean = [
        f"{sp['key']}: {sp['value']}"
        for sp in r["specs"]
        if sp.get("key") and sp.get("value")
    ]
    print(f"\n[{i}] {r['name']}")
    print(f"    Price    : {r['price']}")
    if r["wattage"]:
        print(f"    Wattage  : {r['wattage']}W")
    if r["brand"]:
        print(f"    Brand    : {r['brand']}")
    if r["categories"]:
        print(f"    Category : {', '.join(r['categories'])}")
    if r["features"]:
        print(f"    Features : {', '.join(r['features'])}")
    if specs_clean:
        print(f"    Specs    : {' | '.join(specs_clean[:5])}")
    if r["use_cases"]:
        print(f"    Use Cases: {', '.join(r['use_cases'])}")
    print(f"    URL      : {r['url']}")
    print(f"    ---")
    print(f"    {r['description'][:200]}...")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Query the Inventaa product graph")
    parser.add_argument("query", nargs="?", default=None, help="Product search query")
    parser.add_argument("--top",       "-n",  type=int,   default=5,    help="Max results (default: 5)")
    parser.add_argument("--max-price", "-p",  type=float, default=None, help="Maximum price in INR")
    parser.add_argument("--wattage",   "-w",  type=int,   default=None, help="Exact wattage filter")
    parser.add_argument("--category",  "-c",  type=str,   default=None, help="Category slug (e.g. gate_pillar_lights, solar_lights)")
    args = parser.parse_args()

    if not args.query:
        args.query = input("Enter product search query: ").strip()
    if not args.query:
        print("No query provided. Exiting.")
        sys.exit(0)

    print(f"\nQuery    : \"{args.query}\"")
    print(f"Top      : {args.top}")
    if args.max_price:
        print(f"Max Price: Rs. {args.max_price}")
    if args.wattage:
        print(f"Wattage  : {args.wattage}W")
    if args.category:
        print(f"Category : {args.category}")
    print("=" * 70)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Extract a clean keyword (first meaningful word/phrase)
    # For multi-word queries, try each word and union results
    keywords = [w for w in re.split(r'\s+', args.query.strip()) if len(w) > 2]
    # Use the full query as first keyword, fallback to individual words
    search_keywords = [args.query] + keywords

    all_results = {}

    with driver.session(database=NEO4J_DATABASE) as session:
        for kw in search_keywords:
            if args.category:
                rows = session.run(
                    QUERY_PRODUCT_BY_CATEGORY,
                    tenant=TENANT,
                    keyword=kw,
                    category_slug=args.category,
                    max_price=args.max_price,
                    wattage=args.wattage,
                    top=args.top * 2,
                ).data()
            else:
                rows = session.run(
                    QUERY_PRODUCT_KEYWORD,
                    tenant=TENANT,
                    keyword=kw,
                    max_price=args.max_price,
                    wattage=args.wattage,
                    top=args.top * 2,
                ).data()

            for r in rows:
                pid = r["product_id"]
                if pid not in all_results:
                    all_results[pid] = r

            if len(all_results) >= args.top:
                break

    driver.close()

    results = list(all_results.values())[:args.top]

    if not results:
        print("\nNo products found matching your query.")
        print("Try broader terms or remove filters.")
        sys.exit(0)

    print(f"\nFound {len(results)} product(s):\n")
    for i, r in enumerate(results, 1):
        _print_product(i, r)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
