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
    Fetches the most recent messages for a given session from Supabase,
    and converts them to Langchain messages (chronological order).
    """
    if not session_id:
        return []

    try:
        supabase = _get_supabase()
        query = supabase.table("messages").select("message_id,text,direction,created_at").eq("session_id", session_id)
        
        # Order by created_at descending to get the most recent
        query = query.order("created_at", desc=True).limit(limit)
        
        response = query.execute()
        rows = response.data
        
        langchain_messages = []
        for row in rows:
            # Skip the current message to avoid duplication if it's already in the DB
            if exclude_message_id and row.get("message_id") == exclude_message_id:
                continue
                
            text = row.get("text", "")
            direction = row.get("direction")
            
            if direction == "inbound":
                langchain_messages.append(HumanMessage(content=text))
            elif direction == "outbound":
                langchain_messages.append(AIMessage(content=text))
                
        # Reverse to chronological order (oldest first among the recent limit)
        return list(reversed(langchain_messages))

    except Exception as e:
        logger.error(f"Failed to fetch conversational memory from Supabase: {e}", exc_info=True)
        return []
