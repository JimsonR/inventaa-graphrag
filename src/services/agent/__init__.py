from src.services.agent.config import TenantConfig, AgentConfig

def initialize_agent():
    TenantConfig.initialize()

__all__ = ["TenantConfig", "AgentConfig", "initialize_agent"]
