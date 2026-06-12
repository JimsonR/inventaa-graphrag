"""Check Chunk node structure and verify wave-free content exists."""
from dotenv import load_dotenv; load_dotenv()
from neo4j import GraphDatabase

driver = GraphDatabase.driver("neo4j+ssc://7feba1e0.databases.neo4j.io", auth=("7feba1e0", "mH-S-zGwlgMX3GujG1mFGtH8OwkdfNx7rjdc89Oe6yE"))
with driver.session() as s:
    # Check if Chunk nodes have embeddings
    res = s.run("MATCH (c:Chunk) RETURN keys(c) as props LIMIT 1")
    for r in res:
        props = [p for p in r['props'] if p != 'embedding']
        has_embedding = 'embedding' in r['props']
        print(f"Chunk properties (non-embedding): {props}")
        print(f"Has embedding: {has_embedding}")

    # Show wave-free chunk content
    print("\n=== Chunk nodes containing 'wave' ===")
    res2 = s.run("""
        MATCH (c:Chunk)
        WHERE toLower(c.text) CONTAINS 'wave'
        RETURN c.url as url, left(c.text, 400) as preview
        LIMIT 3
    """)
    for r in res2:
        print(f"\n  URL: {r['url']}")
        print(f"  Preview: {r['preview']}")

    # Check all existing vector indexes (full details)
    print("\n=== FULL INDEX DETAILS ===")
    res3 = s.run("SHOW INDEXES")
    for r in res3:
        if r['type'] in ('VECTOR', 'FULLTEXT'):
            print(f"  [{r['type']}] {r['name']} on {r['labelsOrTypes']}.{r['properties']}")

driver.close()
