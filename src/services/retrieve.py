"""
Proxy module to maintain backwards compatibility.
The actual implementation has been refactored into the src.services.agent package.
"""

from src.services.agent import ask_agent, initialize_agent

__all__ = ["ask_agent", "initialize_agent"]
