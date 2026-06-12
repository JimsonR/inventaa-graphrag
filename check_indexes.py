"""Check what vector indexes exist and if blog/article content is indexed."""
from dotenv import load_dotenv; load_dotenv()
from neo4j import GraphDatabase

driver = GraphDatabase.driver("neo4j+ssc://7feba1e0.databases.neo4j.io", auth=("7feba1e0", "mH-S-zGwlgMX3GujG1mFGtH8OwkdfNx7rjdc89Oe6yE"))
with driver.session() as s:
    print("=== ALL VECTOR INDEXES ===")
    res = s.run("SHOW INDEXES WHERE type = 'VECTOR'")
    for r in res:
        print(f"  {r['name']} | labels={r['labelsOrTypes']} | props={r['properties']} | state={r['state']}")

    print("\n=== ALL NODE LABELS AND COUNTS ===")
    labels = [r['label'] for r in s.run("CALL db.labels() YIELD label RETURN label")]
    for label in labels:
        cnt = s.run(f"MATCH (n:`{label}`) RETURN count(n) as c").single()['c']
        print(f"  {label}: {cnt}")

    print("\n=== inventaa_faq_vector index sample (what text it holds) ===")
    res2 = s.run("MATCH (n) WHERE n.text IS NOT NULL RETURN labels(n) as labels, left(n.text, 200) as text LIMIT 5")
    for r in res2:
        print(f"  [{r['labels']}]: {r['text']}")

    print("\n=== Searching ALL text properties for 'wave-free' ===")
    # Check Chunk, FAQ, Policy, etc.
    for label in labels:
        try:
            res3 = s.run(
                f"MATCH (n:`{label}`) WHERE toLower(toString(n.text)) CONTAINS 'wave' OR toLower(toString(n.question)) CONTAINS 'wave' OR toLower(toString(n.answer)) CONTAINS 'wave' RETURN count(n) as c"
            )
            cnt = res3.single()['c']
            if cnt > 0:
                print(f"  Found {cnt} '{label}' nodes with 'wave'")
        except Exception:
            pass

driver.close()
