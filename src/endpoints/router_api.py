from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Union
from datetime import datetime

from src.services.retrieve import ask_agent

router = APIRouter()

class IncomingMessage(BaseModel):
    id: Optional[Any] = None
    tenant_id: Optional[str] = None
    message_id: Optional[str] = None
    session_id: Optional[str] = None
    channel: Optional[str] = None
    timestamp_unix: Optional[float] = None
    region: Optional[str] = None
    original_type: Optional[str] = None
    text: str = Field(..., description="The content of the message")
    media_url: Optional[str] = None
    media_id: Optional[str] = None
    media_mime_type: Optional[str] = None
    intent: Optional[str] = Field(None, description="The classified intent of the message")
    confidence: Optional[float] = None
    product_name: Optional[str] = None
    quantity_value: Optional[float] = None
    delivery_date: Optional[str] = None
    missing_entities: Optional[Union[List[str], str]] = None
    reply_text: Optional[str] = None
    replied_at: Optional[str] = None
    sender_name: Optional[str] = None
    sender_phone_number: Optional[str] = None
    trace_id: Optional[str] = None
    received_at: Optional[str] = None
    created_at: Optional[str] = None
    direction: Optional[str] = None
    invoice_number: Optional[str] = None
    payment_reference: Optional[str] = None
    quantity_unit: Optional[str] = None

@router.post("/route", tags=["Routing"])
def route_message(message: IncomingMessage):
    """
    Routes an incoming message based on its intent.
    If the intent is 'FAQ_KNOWLEDGE', the message text is routed directly to the
    Neo4j semantic search layer, scoped to the message's tenant_id namespace.
    Other intents bypass the semantic search and are forwarded to operational workflows.
    """
    try:
        if message.intent == "FAQ_KNOWLEDGE":
            # Normalize tenant_id to lowercase ("Inventaa" → "inventaa")
            tenant = message.tenant_id.lower().strip() if message.tenant_id else None
            
            # Temporary mapping: map frontend tenant aliases to the actual Neo4j tenant name
            from src.services.agent.config import AgentConfig
            aliases = AgentConfig.brain.get("tenant", {}).get("aliases", [])
            tenant_id = AgentConfig.brain.get("tenant", {}).get("id", "inventaa")
            if tenant in aliases:
                tenant = tenant_id

            # Combine text and reply_text if the user clicked an option button
            query_text = message.text or ""
            if message.reply_text:
                query_text = f"{query_text}\n(Selected option: {message.reply_text})" if query_text else f"(Selected option: {message.reply_text})"

            # Route to the LangChain Hybrid Agent, scoped to the tenant
            answer = ask_agent(query_text, tenant_id=tenant, session_id=message.session_id, message_id=message.message_id, user_id=message.session_id)

            return {
                "status": "routed_to_knowledge_base",
                "tenant_id": tenant,
                "message_id": message.message_id,
                "intent": message.intent,
                "response_text": answer,
            }
        else:
            # Bypass semantic search and route to operational workflows
            return {
                "status": "routed_to_workflow",
                "message_id": message.message_id,
                "intent": message.intent,
                "message": "Message bypassed semantic search and routed to operational workflows."
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Message routing failed: {str(e)}"
        )
