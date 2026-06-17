import logging
from src.services.agent.config import AgentConfig

logger = logging.getLogger(__name__)

_cached_schema = None

def get_graph_schema(tenant_id: str = None) -> str:
    """
    Queries Neo4j on startup/demand to fetch distinct categories and features.
    Caches the result to avoid redundant queries during serverless execution.
    """
    global _cached_schema
    if _cached_schema:
        return _cached_schema

    try:
        # Extract labels of all nodes connected to Product that have a 'name' property
        cypher_labels = """
        MATCH (p:Product)--(m)
        WHERE m.name IS NOT NULL AND labels(m)[0] <> 'Product'
        RETURN DISTINCT labels(m)[0] AS label
        """
        res_labels = AgentConfig.graph.query(cypher_labels)
        
        schema_str = "GRAPH METADATA (Filterable Fields):\n"
        
        if res_labels:
            for row in res_labels:
                label = row["label"]
                # For each metadata label, fetch the available distinct names
                cypher_vals = f"MATCH (m:{label}) RETURN DISTINCT m.name AS val ORDER BY val"
                res_vals = AgentConfig.graph.query(cypher_vals)
                vals = [r["val"] for r in res_vals] if res_vals else []
                if vals:
                    schema_str += f"- {label}: {', '.join(vals)}\n"

        if schema_str == "GRAPH METADATA (Filterable Fields):\n":
            schema_str += "No metadata filters available.\n"

        _cached_schema = schema_str
        return _cached_schema
    except Exception as e:
        logger.error(f"Failed to fetch graph schema: {e}", exc_info=True)
        return "GRAPH METADATA UNAVAILABLE."
