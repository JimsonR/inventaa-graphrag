import pytest
import contextvars
from concurrent.futures import ThreadPoolExecutor
from src.services.agent.context import tenant_context
from src.services.agent.memory import InMemoryProvider
from langchain_core.messages import HumanMessage

def test_contextvar_propagation_with_copy_context():
    """Verify that copy_context().run properly propagates tenant_context across ThreadPoolExecutor."""
    tenant_context.set("test_tenant_alpha")
    
    def worker_task():
        return tenant_context.get()
        
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ctx.run, worker_task)
        result = future.result()
        
    assert result == "test_tenant_alpha", f"Expected 'test_tenant_alpha', got {result!r}"

def test_in_memory_provider_scoping():
    """Verify InMemoryProvider get_recent_messages works and accepts tenant_id."""
    provider = InMemoryProvider()
    msg = HumanMessage(content="Hello world")
    provider.add_message("session_123", msg)
    
    res = provider.get_recent_messages("session_123", tenant_id="inventaa")
    assert len(res) == 1
    assert res[0].content == "Hello world"

def test_no_insecure_cypher_null_fallback_in_tools():
    """Verify tools.py does not contain insecure 'OR p.tenant IS NULL' fallbacks in search queries."""
    with open("src/services/agent/tools.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "OR p.tenant IS NULL" not in content, "Insecure Cypher null fallback found in tools.py!"

def test_no_insecure_cypher_null_fallback_in_graphrag():
    """Verify graphrag_engine.py does not contain insecure 'OR p.tenant IS NULL' fallbacks."""
    with open("src/query/graphrag_engine.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "OR p.tenant IS NULL" not in content, "Insecure Cypher null fallback found in graphrag_engine.py!"
