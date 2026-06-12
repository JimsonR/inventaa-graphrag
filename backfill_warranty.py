"""
Backfill warranty nodes for the 43 products that are missing HAS_WARRANTY relationships.
All Inventaa products carry a 1-year replacement warranty (sourced from the existing warranty nodes).
"""
from dotenv import load_dotenv; load_dotenv()
from neo4j import GraphDatabase

driver = GraphDatabase.driver("neo4j+ssc://7feba1e0.databases.neo4j.io", auth=("7feba1e0", "mH-S-zGwlgMX3GujG1mFGtH8OwkdfNx7rjdc89Oe6yE"))

with driver.session() as s:
    # First, check what warranty nodes exist and pick the standard one
    res = s.run("MATCH (w:Warranty) RETURN w.description as desc, w.duration_years as years LIMIT 5")
    print("=== Existing Warranty Nodes ===")
    for r in res:
        print(f"  [{r['years']} yr] {r['desc'][:100]}")

    # Find the standard 1-year replacement warranty node
    res2 = s.run("""
        MATCH (w:Warranty)
        WHERE toLower(w.description) CONTAINS '1 year' OR w.duration_years = 1
        RETURN elementId(w) as wid, w.description as desc
        LIMIT 1
    """)
    warranty_node = list(res2)
    if not warranty_node:
        print("\nNo 1-year warranty node found! Cannot backfill.")
        driver.close()
        exit(1)

    wid = warranty_node[0]['wid']
    print(f"\n=== Using Warranty Node: {warranty_node[0]['desc'][:100]} ===")

    # Link all products missing warranty to this warranty node
    result = s.run("""
        MATCH (p:Product)
        WHERE NOT (p)-[:HAS_WARRANTY]->(:Warranty)
        WITH p
        MATCH (w:Warranty) WHERE elementId(w) = $wid
        MERGE (p)-[:HAS_WARRANTY]->(w)
        RETURN count(p) as linked
    """, wid=wid)

    for r in result:
        print(f"\nLinked {r['linked']} products to the warranty node.")

    # Verify
    res3 = s.run("""
        MATCH (p:Product)
        WHERE NOT (p)-[:HAS_WARRANTY]->(:Warranty)
        RETURN count(p) as remaining
    """)
    for r in res3:
        print(f"Products still missing warranty: {r['remaining']}")

driver.close()
