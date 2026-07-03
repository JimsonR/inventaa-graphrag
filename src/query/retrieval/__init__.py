"""
retrieval/__init__.py — Re-exports retrieval methods for the GraphRAGEngine.
"""

import asyncio
import logging
from typing import Tuple, List, Dict, Any
from src.query.retrieval.graph_search import graph_search
from src.query.retrieval.vector_search import vector_search
from src.query.retrieval.text_search import text_search, category_browse_from_sqlite

logger = logging.getLogger(__name__)


async def parallel_retrieve(query: str, intent_data: dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Executes Vector, Graph, and Text/SQL keyword searches in parallel.
    """
    vector_task = asyncio.to_thread(vector_search, intent_data, query)
    graph_task = asyncio.to_thread(graph_search, intent_data, query)
    text_task = asyncio.to_thread(text_search, intent_data, query)

    vector_results, graph_results, text_results = await asyncio.gather(
        vector_task, graph_task, text_task, return_exceptions=True
    )

    if isinstance(vector_results, Exception):
        logger.error(f"Parallel vector search failed: {vector_results}")
        vector_results = []
    if isinstance(graph_results, Exception):
        logger.error(f"Parallel graph search failed: {graph_results}")
        graph_results = []
    if isinstance(text_results, Exception):
        logger.error(f"Parallel text search failed: {text_results}")
        text_results = []

    return vector_results, graph_results, text_results

__all__ = ["parallel_retrieve", "category_browse_from_sqlite", "graph_search", "vector_search", "text_search"]
