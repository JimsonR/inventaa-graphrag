from fastapi import APIRouter, Query, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import chromadb

from src.services.retrieve import ask_agent

router = APIRouter()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# --- Pydantic Data Models ---
class SearchResponse(BaseModel):
    query: str = Field(..., description="The query string submitted for search")
    response_text: Any = Field(..., description="Graph product list (JSON array) or conversational answer string")

class HealthStatus(BaseModel):
    status: str = Field(..., description="General API operational status")
    chromadb_connected: bool = Field(..., description="Database container connection status")
    indexed_chunks: int = Field(..., description="Total number of active vector documents across collections")

# --- API Routes ---

@router.get("/health", response_model=HealthStatus, tags=["System Health"])
def health_check():
    """
    Performs diagnosis checks on connectivity to the local ChromaDB database and sums all indexed chunks.
    """
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
        
        # Get all collections and sum their counts
        collections = client.list_collections()
        total_items = sum([client.get_collection(c.name).count() for c in collections])
        
        return {
            "status": "healthy",
            "chromadb_connected": True,
            "indexed_chunks": total_items
        }
    except Exception as e:
        return {
            "status": "degraded",
            "chromadb_connected": False,
            "indexed_chunks": 0
        }

@router.get("/search", response_model=SearchResponse, tags=["Retrieval"])
def search_knowledge_base(
    q: str = Query(..., min_length=1, description="Semantic text query to search for")
):
    """
    Submits a natural language query to the Hybrid RAG LangChain agent and returns the conversational response.
    """
    try:
        answer = ask_agent(q)
        return {
            "query": q,
            "response_text": answer
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic retrieval search failed: {str(e)}"
        )
