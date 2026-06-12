from dotenv import load_dotenv; load_dotenv()
from neo4j import GraphDatabase
driver = GraphDatabase.driver("neo4j+ssc://7feba1e0.databases.neo4j.io", auth=("7feba1e0", "mH-S-zGwlgMX3GujG1mFGtH8OwkdfNx7rjdc89Oe6yE"))
with driver.session() as s:
    # Try different Lucene queries for the Athena product
    for query in ["Athena~", "Athena~  Gate~", "Athena 3"]:
        print(f"\n=== Lucene query: {query!r} ===")
        res = s.run(
            'CALL db.index.fulltext.queryNodes("product_name_ft", $q) YIELD node AS p, score '
            'WITH p, score ORDER BY score DESC LIMIT 3 '
            'OPTIONAL MATCH (p)-[:AVAILABLE_IN_WATTAGE]->(w:WattageOption) '
            'OPTIONAL MATCH (p)-[:HAS_SPEC]->(s:Spec) WHERE toLower(s.key) CONTAINS "watt" '
            'RETURN p.name as name, p.sku as sku, '
            'collect(DISTINCT w.name) as watts, '
            'collect(DISTINCT (s.key + ": " + s.value)) as watt_specs, score',
            q=query
        )
        for r in res:
            print(f"  Product: {r['name']} (score={r['score']:.2f})")
            print(f"    WattageOptions: {r['watts']}")
            print(f"    Watt Specs: {r['watt_specs']}")

    # Also check raw tokenization of the problematic product name
    print("\n=== Tokenization test for 'Athena 3 in 1 Gate Post Lights - Frontgate Lighting' ===")
    tokens = [t.strip(".,?!-") for t in "Athena 3 in 1 Gate Post Lights - Frontgate Lighting".split()]
    stop = {"light","lights","lamp","lamps","led","product","products","show","me","get","find","list","give","want","need","a","an","the","for","with","of","in","and","or","is","are","what","which","how","do","can","does","suggest","recommend","suitable","use","buy","choose","my","i","we","our","this","that","under","budget","within","rs","inr","rupees","good","looking","modern"}
    meaningful = [t for t in tokens if t and len(t) > 1 and t.lower() not in stop]
    print(f"  All tokens: {tokens}")
    print(f"  Meaningful: {meaningful}")
    lucene = " AND ".join(t + "~" for t in meaningful)
    print(f"  Lucene query: {lucene!r}")

driver.close()
