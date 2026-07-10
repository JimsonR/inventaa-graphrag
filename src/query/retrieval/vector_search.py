"""
retrieval/vector_search.py — Pinecone/FAQ vector similarity search.
Currently a stub since vector stores (policy, FAQ) are not initialized
in the lightweight MCP configuration. Returns empty results.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def vector_search(intent_data: dict, query: str) -> List[Dict[str, Any]]:
    """Stub — returns empty list. Vector stores are not used in lightweight MCP mode."""
    return []
