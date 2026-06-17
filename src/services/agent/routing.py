import logging
import yaml
import os
from typing import Tuple, Optional
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

_configs = None

def load_tenant_configs():
    global _configs
    if _configs is None:
        config_path = os.path.join(os.path.dirname(__file__), "tenant_configs.yaml")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                _configs = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load tenant_configs.yaml: {e}")
            _configs = {
                "default": {
                    "system_prompt": "You are a helpful assistant.",
                    "router_prompt": "Classify the intent.",
                    "fallback_message": "I cannot answer this right now."
                }
            }
    return _configs

def get_tenant_prompts(tenant_id: str) -> dict:
    """
    Retrieves the prompts for a specific tenant, falling back to 'default'.
    """
    configs = load_tenant_configs()
    tenant_key = tenant_id if tenant_id and tenant_id in configs else "default"
    return configs.get(tenant_key, configs.get("default"))

def get_intent_config(
    query: str,
    all_tools: list,
    llm=None,
    explicit_intent: Optional[str] = None,
) -> Tuple[str, list]:
    """
    Returns the system prompt and allowed tools for the given query.
    For the graph-aware implementation, we return the unified system prompt from the tenant config
    and all tools, allowing the LLM to use its metadata awareness to choose the right tool.
    """
    from src.services.agent.context import tenant_context
    tenant_id = tenant_context.get()
    
    prompts = get_tenant_prompts(tenant_id)
    system_prompt = prompts.get("system_prompt", "You are a helpful graph-aware assistant.")
    
    # We no longer strictly filter tools. The LLM has the graph schema and can choose.
    return system_prompt, all_tools
