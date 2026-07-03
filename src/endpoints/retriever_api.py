"""
/search routes — Linear GraphRAG semantic + graph search for Inventaa outdoor lighting products.
"""

from __future__ import annotations

from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException, status
from pydantic import BaseModel, Field

from src.query.graphrag_engine import GraphRAGEngine
from src.db.database import get_session
from src.db.models import Product

router = APIRouter(tags=["Retrieval & Search"])

# Singleton engine instance
_engine: Optional[GraphRAGEngine] = None

def get_engine() -> GraphRAGEngine:
    global _engine
    if _engine is None:
        _engine = GraphRAGEngine()
    return _engine


class HealthStatus(BaseModel):
    status: str = Field(..., description="General API operational status")
    sqlite_connected: bool = Field(..., description="SQLite database connection status")
    total_products_in_catalog: int = Field(..., description="Total products indexed in SQLite")


class LinearSearchResponse(BaseModel):
    query: str
    intent: str
    products: List[dict]
    product_links: List[dict]
    ai_response: str


@router.get("/health", response_model=HealthStatus, tags=["System Health"])
def health_check():
    """
    Performs health checks on the SQLite catalog database and counts indexed products.
    """
    try:
        with get_session() as session:
            count = session.query(Product).count()
        return {
            "status": "healthy",
            "sqlite_connected": True,
            "total_products_in_catalog": count
        }
    except Exception as e:
        return {
            "status": "degraded",
            "sqlite_connected": False,
            "total_products_in_catalog": 0
        }


@router.get("/search/products", response_model=LinearSearchResponse, tags=["Retrieval & Search"])
async def search_products_linear(
    q: str = Query(..., min_length=1, description="Natural language query, e.g. 'waterproof solar gate light'"),
    n: int = Query(6, ge=1, le=20, description="Number of products to return"),
    session_id: Optional[str] = Query(None, description="Session ID to fetch previous messages from DB for conversational continuity")
):
    """
    Linear GraphRAG search: combines semantic Vector search with Neo4j Graph traversal,
    ranks via Reciprocal Rank Fusion (RRF), hydrates authoritative details from SQLite,
    and generates an AI sales response.
    """
    try:
        engine = get_engine()
        result = await engine.query(q, session_id=session_id)
        return {
            "query": q,
            "intent": result.intent,
            "products": result.products[:n],
            "product_links": result.product_links[:n],
            "ai_response": result.response
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Linear GraphRAG retrieval failed: {str(e)}"
        )
