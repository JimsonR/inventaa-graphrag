import os
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from fastmcp import FastMCP, Context

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inventaa-mcp")

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Runs ONCE when server starts to initialize Tri-Store GraphRAG dependencies."""
    logger.info("Initializing Inventaa Tri-Store GraphRAG dependencies...")

    # Ensure environment variables are loaded
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    from src.services.agent.config import AgentConfig
    AgentConfig.initialize()
    logger.info("Tri-Store GraphRAG dependencies (Neo4j Lucene, SQLite) initialized successfully.")

    try:
        yield {"config": AgentConfig}
    finally:
        logger.info("Shutting down Inventaa MCP server...")

# Create FastMCP server instance at module level (REQUIRED for cloud & CLI compatibility)
mcp = FastMCP("inventaa-graphrag", lifespan=app_lifespan)

# Enable CORS (Cross-Origin Resource Sharing) for browser and MCP Inspector compatibility
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
_original_http_app = mcp.http_app
def _cors_http_app(*args, **kwargs):
    mw = kwargs.get("middleware") or []
    mw.append(Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*", "mcp-session-id", "mcp-protocol-version", "x-session-id"]
    ))
    kwargs["middleware"] = mw
    return _original_http_app(*args, **kwargs)
mcp.http_app = _cors_http_app

# ==========================================
# TOOLS (Functions callable by MCP clients)
# ==========================================

@mcp.tool()
async def search_catalog(
    query: str,
    limit: int = 6,
    session_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    intent_data: Optional[Dict[str, Any]] = None,
    dialogue_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Lightweight catalog search endpoint using Tri-Store GraphRAG retrieval (no LLM calls).

    Receives pre-classified intent from the client, runs parallel retrieval across
    Neo4j Lucene Fulltext + SQLite FTS, fuses via Reciprocal Rank Fusion, and hydrates
    authoritative product data from SQLite.

    Args:
        query: User input query (e.g., 'divine lights', 'solar gate lights under 1500').
        limit: Maximum number of products to return (default: 6).
        session_id: Optional session identifier (for future use).
        tenant_id: Optional tenant identifier to scope search.
        intent_data: Pre-classified intent dictionary from the client (required).
        dialogue_state: Optional dialogue state dictionary from the client.
    """
    try:
        if dialogue_state and not intent_data:
            intent_data = {
                "intent": str(dialogue_state.get("intent", "find_product")).lower(),
                "category_keywords": [dialogue_state["category"]] if dialogue_state.get("category") else [],
                "feature_keywords": [],
                "filters": {}
            }
        from src.query.graphrag_engine import GraphRAGEngine
        engine = GraphRAGEngine()
        result = await engine.query(user_query=query, session_id=session_id, tenant_id=tenant_id, intent_data=intent_data)
        return {
            "status": "success",
            "intent": result.intent,
            "products": result.products[:limit],
            "product_links": result.product_links[:limit],
            "response": result.response
        }
    except Exception as e:
        logger.error(f"Error executing search_catalog: {e}", exc_info=True)
        return {
            "error": str(e),
            "status": "failed",
            "products": [],
            "response": "We encountered an issue searching the catalog. Please try again."
        }

@mcp.tool()
async def get_taxonomy_context(query: str, threshold: float = 0.80) -> Dict[str, Any]:
    """Retrieve database taxonomy candidates (categories, features, use_cases) matching a query.

    Used by external client agents (WhatsApp bot, UI) to map user language to exact
    database category/feature names before formulating search requests.

    Args:
        query: User query text or keywords to resolve against the database taxonomy.
        threshold: Similarity threshold between 0.0 and 1.0 (default: 0.80).
    """
    try:
        from src.services.agent.taxonomy import fetch_taxonomy_candidates, fetch_taxonomy_candidates_fast
        from src.services.agent.config import AgentConfig
        import asyncio
        hints = fetch_taxonomy_candidates_fast(query)
        if not hints:
            query_embedding = await asyncio.to_thread(AgentConfig.embeddings.embed_query, query)
            hints = await asyncio.to_thread(fetch_taxonomy_candidates, query_embedding, threshold)
        return {"status": "success", "taxonomy": hints or {}}
    except Exception as e:
        logger.error(f"Error executing get_taxonomy_context: {e}", exc_info=True)
        return {"status": "failed", "error": str(e), "taxonomy": {}}


@mcp.tool()
def get_product_details(sku: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """Get authoritative product details, pricing, discounts, ratings, and specs by SKU.

    Args:
        sku: The unique product identifier (e.g., '18C-2042', 'MAR03C', 'GAT05').
        tenant_id: Optional tenant identifier to scope lookup.
    """
    try:
        if tenant_id:
            from src.services.agent.context import tenant_context
            tenant_context.set(tenant_id)
        from src.services.agent.config import AgentConfig
        from src.db.database import Product

        session = AgentConfig.SessionLocal()
        try:
            prod = session.query(Product).filter(Product.sku.ilike(f"%{sku}%")).first()
            if not prod:
                return {"status": "not_found", "message": f"No product found matching SKU: {sku}"}

            specs = []
            if prod.variants:
                specs = [f"{v.key}: {', '.join(v.options)}" for v in prod.variants if v.options]

            return {
                "status": "success",
                "sku": prod.sku,
                "name": prod.name,
                "price": f"Rs. {prod.price}",
                "mrp": f"Rs. {prod.mrp}" if prod.mrp else None,
                "discount": f"{int((1 - prod.price/prod.mrp)*100)}% off" if prod.mrp and prod.mrp > prod.price else None,
                "rating": f"{prod.rating} stars ({prod.reviews_count} reviews)",
                "categories": [c.name for c in prod.categories] if prod.categories else [],
                "features": [f.name for f in prod.features] if prod.features else [],
                "specifications": specs
            }
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Error in get_product_details: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# ==========================================
# RESOURCES (Read-only static/dynamic data)
# ==========================================

@mcp.resource("data://catalog/status")
def get_catalog_status() -> Dict[str, Any]:
    """Get catalog statistics, database connectivity status, and Lucene index health."""
    try:
        from src.services.agent.config import AgentConfig
        from src.db.database import Product

        session = AgentConfig.SessionLocal()
        try:
            total_products = session.query(Product).count()
            sqlite_status = True
        except Exception:
            total_products = 0
            sqlite_status = False
        finally:
            session.close()

        lucene_indexes = []
        try:
            res = AgentConfig.graph.query("SHOW FULLTEXT INDEXES YIELD name, state RETURN name, state")
            lucene_indexes = res if res else []
        except Exception as e:
            logger.warning(f"Failed to fetch Lucene indexes: {e}")

        return {
            "status": "healthy" if sqlite_status else "degraded",
            "sqlite_connected": sqlite_status,
            "total_products_in_catalog": total_products,
            "neo4j_lucene_indexes": lucene_indexes
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.resource("data://catalog/sku/{sku}")
def get_sku_resource(sku: str) -> Dict[str, Any]:
    """Dynamic resource exposing raw product catalog data by SKU."""
    return get_product_details(sku)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inventaa Tri-Store GraphRAG FastMCP Server")
    parser.add_argument("--transport", "-t", type=str, default="streamable-http",
                        choices=["sse", "streamable-http", "http", "stdio"],
                        help="Transport protocol to use (default: streamable-http)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", "-p", type=int, default=8008, help="Port to bind to")
    parser.add_argument("--stateless", action="store_true", default=True, help="Run in stateless HTTP mode (default: True)")
    parser.add_argument("--stateful", action="store_true", help="Run in stateful HTTP mode with session tracking")
    args = parser.parse_args()

    stateless_mode = not args.stateful if args.transport in ("http", "streamable-http") else args.stateless
    logger.info(f"Starting Inventaa FastMCP server with transport='{args.transport}' (stateless={stateless_mode}) on http://{args.host}:{args.port}")
    mcp.run(transport=args.transport, host=args.host, port=args.port, stateless_http=stateless_mode)
