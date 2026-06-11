from src.services.agent.graph import ask_agent
from src.services.agent.config import AgentConfig

def initialize_agent():
    AgentConfig.initialize()

__all__ = ["ask_agent", "initialize_agent"]
