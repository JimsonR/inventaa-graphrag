from dotenv import load_dotenv; load_dotenv()
from neo4j import GraphDatabase
driver = GraphDatabase.driver("neo4j+ssc://7feba1e0.databases.neo4j.io", auth=("7feba1e0", "mH-S-zGwlgMX3GujG1mFGtH8OwkdfNx7rjdc89Oe6yE"))
with driver.session() as s:
    # Check Athena 3 in 1 warranty
    res = s.run("""
        CALL db.index.fulltext.queryNodes("product_name_ft", "Athena~ AND Gate~ AND Post~") YIELD node AS p, score
        WITH p, score ORDER BY score DESC LIMIT 3
        OPTIONAL MATCH (p)-[:HAS_WARRANTY]->(w:Warranty)
        RETURN p.name as name, p.sku as sku, w.description as warranty, score
    """)
    print("=== Athena products and their warranty nodes ===")
    for r in res:
        print(f"  [{r['sku']}] {r['name']} (score={r['score']:.2f})")
        print(f"    Warranty: {r['warranty']}")

    # How many products in total have NO warranty node?
    res2 = s.run("""
        MATCH (p:Product)
        WHERE NOT (p)-[:HAS_WARRANTY]->(:Warranty)
        RETURN count(p) as no_warranty_count
    """)
    for r in res2:
        print(f"\nProducts with NO warranty node: {r['no_warranty_count']}")

    # List them
    res3 = s.run("""
        MATCH (p:Product)
        WHERE NOT (p)-[:HAS_WARRANTY]->(:Warranty)
        RETURN p.name as name, p.sku as sku
        ORDER BY p.name
        LIMIT 20
    """)
    print("\n=== Products missing warranty ===")
    for r in res3:
        print(f"  [{r['sku']}] {r['name']}")

driver.close()
