"""
retrieval/__init__.py — Re-exports retrieval methods for the GraphRAGEngine.
"""

import asyncio
import logging
import time
from typing import Tuple, List, Dict, Any
from src.query.retrieval.graph_search import graph_search
from src.query.retrieval.vector_search import vector_search
from src.query.retrieval.text_search import text_search, category_browse_from_sqlite

logger = logging.getLogger(__name__)


async def parallel_retrieve(query: str, intent_data: dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Executes Vector, Graph, and Text/SQL keyword searches in parallel.
    """
    async def _timed_channel(name: str, fn, *args):
        t0 = time.perf_counter()
        try:
            result = await asyncio.to_thread(fn, *args)
            elapsed = time.perf_counter() - t0
            print(f"[TIMING] channel.{name}: {elapsed:.3f}s ({len(result) if isinstance(result, list) else 0} results)")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"[TIMING] channel.{name}: {elapsed:.3f}s (FAILED: {e})")
            return []

    vector_task = _timed_channel("vector", vector_search, intent_data, query)
    graph_task = _timed_channel("graph", graph_search, intent_data, query)
    text_task = _timed_channel("text", text_search, intent_data, query)

    vector_results, graph_results, text_results = await asyncio.gather(
        vector_task, graph_task, text_task, return_exceptions=True
    )

    if isinstance(vector_results, Exception):
        vector_results = []
    if isinstance(graph_results, Exception):
        graph_results = []
    if isinstance(text_results, Exception):
        text_results = []

    return vector_results, graph_results, text_results

__all__ = ["parallel_retrieve", "category_browse_from_sqlite", "graph_search", "vector_search", "text_search"]
