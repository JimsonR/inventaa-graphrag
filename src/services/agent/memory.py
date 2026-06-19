import os
import logging
from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from supabase import create_client, Client

logger = logging.getLogger(__name__)

_supabase: Optional[Client] = None

def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        _supabase = create_client(url, key)
    return _supabase

def get_recent_messages(session_id: str, exclude_message_id: Optional[str] = None, limit: int = 5) -> List[BaseMessage]:
    """
    Fetches the most recent inbound (user) messages for a given session from Supabase,
    and converts them to Langchain messages (chronological order).
    Consecutive messages from the same sender are grouped.
    """
    if not session_id:
        return []

    try:
        supabase = _get_supabase()
        # Only fetch inbound messages to avoid distracting the LLM with its own massive product dumps
        query = supabase.table("messages").select("message_id,text,direction,created_at,reply_text").eq("session_id", session_id).eq("direction", "inbound")
        
        # Order by created_at descending to get the most recent
        query = query.order("created_at", desc=True).limit(limit)
        
        response = query.execute()
        rows = response.data
        
        langchain_messages = []
        
        # rows are ordered newest to oldest, reverse to oldest to newest
        rows = list(reversed(rows))
        
        for row in rows:
            if exclude_message_id and row.get("message_id") == exclude_message_id:
                continue
                
            text = row.get("text") or ""
            reply_text = row.get("reply_text")
            
            if reply_text:
                text = f"{text}\n(Selected option: {reply_text})" if text else f"(Selected option: {reply_text})"
            
            if langchain_messages and isinstance(langchain_messages[-1], HumanMessage):
                langchain_messages[-1].content += f"\n\n{text}"
            else:
                langchain_messages.append(HumanMessage(content=text))
                
        return langchain_messages

    except Exception as e:
        logger.error(f"Failed to fetch conversational memory from Supabase: {e}", exc_info=True)
        return []
