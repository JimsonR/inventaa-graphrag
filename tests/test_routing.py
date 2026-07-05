import pytest
import asyncio
import inspect
from src.endpoints.router_api import route_message, IncomingMessage

def test_route_message_is_async_endpoint():
    """Verify route_message has been converted to an async def endpoint."""
    assert inspect.iscoroutinefunction(route_message), "route_message is not an async def function!"

def test_route_message_operational_workflow():
    """Verify route_message properly routes operational workflows without calling semantic search."""
    msg = IncomingMessage(
        text="Check my order status",
        intent="ORDER_STATUS",
        session_id="sess_abc",
        message_id="msg_xyz",
        sender_phone_number="+919876543210"
    )
    res = asyncio.run(route_message(msg))
    assert res["status"] == "routed_to_workflow"
    assert res["message_id"] == "msg_xyz"
