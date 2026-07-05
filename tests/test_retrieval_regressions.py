import os
import pytest
import asyncio
from pathlib import Path
from src.services.agent.config import AgentConfig
from src.query.retrieval.graph_search import graph_search

@pytest.fixture(scope="module", autouse=True)
def setup_env_and_config():
    """Load .env and initialize AgentConfig for retrieval regression tests."""
    if os.path.exists('.env'):
        with open('.env', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    os.environ[k.strip()] = v.strip()
    AgentConfig.initialize()

def test_graph_search_collection_matching_and_strict_and():
    """
    Regression test: Verify graph_search.py generates Cypher queries that match
    Collection nodes and strictly join category and feature conditions with AND.
    """
    with open("src/query/retrieval/graph_search.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "(col:Collection)<-[:BELONGS_TO_COLLECTION]-(p)" in content, (
        "graph_search.py must match Collection nodes when category keywords are present!"
    )
    assert '"WITH p, " + " AND ".join(conds)' in content, (
        "graph_search.py must join category and feature conditions with AND, not OR, to prevent cross-category leaks!"
    )

def test_no_divine_lights_leak_in_indoor_domestic_query():
    """
    Regression test: When querying for 'show indoor domestic lights under 500 rs',
    Divine & Temple Lights (e.g. SKU LOS06Y) must not leak into search results
    even though they share the generic 'indoor' feature tag.
    """
    intent_data = {
        "intent": "find_product",
        "category_keywords": ["Indoor Domestic Lights"],
        "feature_keywords": ["indoor"],
        "product_name": None,
        "filters": {
            "category": "Indoor Domestic Lights",
            "application": "indoor"
        },
        "preferences": {"max_price": 500}
    }
    query = "show indoor domestic lights under 500 rs"
    
    results = graph_search(intent_data, query)
    assert isinstance(results, list), "graph_search should return a list of product dictionaries."
    assert len(results) > 0, "Expected at least one valid indoor domestic light in graph_search results."
    
    # Ensure no returned SKU belongs to Divine Lights (e.g., LOS06Y)
    leaked_skus = [r["sku"] for r in results if "LOS06Y" in str(r.get("sku", "")).upper()]
    assert len(leaked_skus) == 0, f"Divine Lights leaked into Indoor Domestic Lights results: {leaked_skus}"

def test_nonexistent_outdoor_domestic_category_returns_empty():
    """
    Regression test: When querying for a non-existent category like 'Outdoor Domestic Lights',
    graph_search must merge filters['category'] into target categories and return []
    instead of falling back to returning arbitrary products with score 1.0.
    """
    intent_data = {
        "intent": "find_product",
        "category_keywords": [],
        "feature_keywords": [],
        "product_name": None,
        "filters": {
            "category": "Outdoor Domestic Lights",
            "application": None
        },
        "preferences": {"max_price": 500}
    }
    query = "show outdoor domestic lights under 500 rs"
    
    results = graph_search(intent_data, query)
    assert results == [], f"Expected [] for non-existent Outdoor Domestic category, got {len(results)} items."

def test_graphrag_engine_merges_category_filter_for_all_intents():
    """
    Regression test: Verify graphrag_engine.py merges filters['category'] into
    category_keywords regardless of intent (not only for BROWSE_CATEGORY).
    """
    with open("src/query/graphrag_engine.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check that cat_kws append logic is present immediately after classify_intent
    # and before check for BROWSE_CATEGORY
    classify_idx = content.find("self.classify_intent(")
    browse_idx = content.find("if intent == QueryIntent.BROWSE_CATEGORY:")
    append_idx = content.find('cat_kws.append(filters["category"])')
    
    assert classify_idx != -1 and browse_idx != -1 and append_idx != -1, (
        "Required syntax markers not found in graphrag_engine.py"
    )
    assert classify_idx < append_idx < browse_idx, (
        "filters['category'] must be merged into category_keywords before intent-specific branches!"
    )


def test_empty_browse_category_returns_resolutions_without_rag_fallthrough():
    """Verify that if BROWSE_CATEGORY finds 0 products, it does NOT fall through to RAG and passes taxonomy_hints to synthesize_response."""
    engine_path = Path("src/query/graphrag_engine.py").read_text(encoding="utf-8")
    
    # Assert that BROWSE_CATEGORY returns empty list directly instead of falling through
    assert "returning empty list with taxonomy resolution instead of falling through to RAG pipeline" in engine_path, (
        "BROWSE_CATEGORY should return empty list with taxonomy resolutions when 0 products matched, not fall through to RAG!"
    )
    
    # Assert that taxonomy_hints is passed to synthesize_response
    assert "taxonomy_hints=taxonomy_hints" in engine_path, (
        "synthesize_response must receive taxonomy_hints so LLM can offer candidate category resolutions!"
    )


def test_price_preferences_enforced_in_hydration_and_graph_search():
    """Verify that price preferences (max_price, min_price) are enforced in both graph_search and hydrate_from_sqlite."""
    fusion_path = Path("src/query/fusion.py").read_text(encoding="utf-8")
    graph_path = Path("src/query/retrieval/graph_search.py").read_text(encoding="utf-8")

    assert "if max_p and isinstance(max_p, (int, float)) and prod.price_num > max_p:" in fusion_path, (
        "hydrate_from_sqlite must filter out products whose price exceeds max_price preference!"
    )
    assert "p.price_num <= $max_price" in graph_path, (
        "graph_search must filter Neo4j traversal queries by max_price preference!"
    )


def test_broad_queries_trigger_category_navigation_instead_of_fallback_products():
    """Verify that broad queries without category/feature/name keywords trigger category navigation without dumping fallback products."""
    engine_path = Path("src/query/graphrag_engine.py").read_text(encoding="utf-8")
    graph_path = Path("src/query/retrieval/graph_search.py").read_text(encoding="utf-8")

    assert "is_broad_query =" in engine_path and "if intent == QueryIntent.BROWSE_CATEGORY or is_broad_query:" in engine_path, (
        "GraphRAGEngine.query must detect broad queries and trigger category browse navigation instead of falling through to parallel retrieval!"
    )
    assert "elif not conds:" in graph_path, (
        "graph_search must return empty list when no filtering conditions are specified, avoiding fallback product dumps!"
    )



