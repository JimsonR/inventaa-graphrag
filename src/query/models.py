"""
models.py — Data structures and enums for the RAG engine.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any


class QueryIntent(str, Enum):
    """Supported user query intents."""
    FIND_PRODUCT = "find_product"
    GET_PRODUCT_INFO = "get_product_info"
    BROWSE_CATEGORY = "browse_category"
    CHECK_POLICY = "check_policy"
    FAQ_KNOWLEDGE = "faq_knowledge"
    GET_ADVICE = "get_advice"
    UNKNOWN = "unknown"


@dataclass
class QueryResult:
    """Standardized output container for RAG execution results."""
    intent: QueryIntent
    products: List[Dict[str, Any]]
    context_text: str
    response: str
    product_links: List[Dict[str, Any]]
