"""
models.py — Data structures and enums for the RAG engine.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any


class QueryIntent(str, Enum):
    """Supported user query intents (domain-agnostic)."""
    FIND_ITEM = "find_item"
    GET_ITEM_INFO = "get_item_info"
    BROWSE = "browse"
    FAQ = "faq"
    ADVICE = "advice"
    UNKNOWN = "unknown"


@dataclass
class QueryResult:
    """Standardized output container for RAG execution results."""
    intent: QueryIntent
    items: List[Dict[str, Any]]
    context_text: str
    response: str
    links: List[Dict[str, Any]]
    chunks: List[str] = field(default_factory=list)
