from dotenv import load_dotenv; load_dotenv()
from neo4j import GraphDatabase
driver = GraphDatabase.driver("neo4j+ssc://7feba1e0.databases.neo4j.io", auth=("7feba1e0", "mH-S-zGwlgMX3GujG1mFGtH8OwkdfNx7rjdc89Oe6yE"))
with driver.session() as s:
    # Check warranty info in specs and description of products missing warranty nodes
    print("=== Warranty mentions in specs for products without HAS_WARRANTY ===")
    res = s.run("""
        MATCH (p:Product)
        WHERE NOT (p)-[:HAS_WARRANTY]->(:Warranty)
        OPTIONAL MATCH (p)-[:HAS_SPEC]->(s:Spec)
        WHERE toLower(s.key) CONTAINS 'warrant' OR toLower(s.value) CONTAINS 'warrant'
           OR toLower(s.key) CONTAINS '1-year' OR toLower(s.value) CONTAINS '1 year'
        WITH p, collect(DISTINCT s.key + ': ' + s.value) as wSpecs
        WHERE size(wSpecs) > 0
        RETURN p.sku as sku, p.name as name, wSpecs
        ORDER BY p.name
        LIMIT 20
    """)
    count = 0
    for r in res:
        count += 1
        print(f"  [{r['sku']}] {r['name']}")
        for spec in r['wSpecs']:
            print(f"    - {spec}")
    print(f"Total with warranty in specs: {count}")

    # Check if description has warranty info
    print("\n=== Checking Athena 3-in-1 specifically ===")
    res2 = s.run("""
        MATCH (p:Product {sku: '18M-2042'})
        OPTIONAL MATCH (p)-[:HAS_SPEC]->(s:Spec)
        RETURN p.description as desc,
               p.feature_descriptions as feat_desc,
               collect(s.key + ': ' + s.value) as all_specs
    """)
    for r in res2:
        specs = r['all_specs'] or []
        warranty_specs = [s for s in specs if 'warrant' in s.lower() or 'replac' in s.lower() or '1-year' in s.lower() or '1 year' in s.lower()]
        print(f"  Warranty-related specs: {warranty_specs}")
        desc = (r['desc'] or "")
        sentences = [s.strip() for s in desc.replace('\n', '.').split('.') if 'warrant' in s.lower() or 'replac' in s.lower()]
        print(f"  Warranty mentions in description: {sentences[:3]}")

driver.close()
