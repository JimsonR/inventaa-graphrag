import os
import logging
from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from supabase import create_client, Client

logger = logging.getLogger(__name__)

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

class BaseMemoryProvider(ABC):
    @abstractmethod
    def get_recent_messages(self, session_id: str, exclude_message_id: Optional[str] = None, limit: int = 5) -> List[BaseMessage]:
        pass

class SupabaseMemoryProvider(BaseMemoryProvider):
    def __init__(self, url: str, key: str):
        from supabase import create_client, Client
        self._supabase: Client = create_client(url, key)

    def get_recent_messages(self, session_id: str, exclude_message_id: Optional[str] = None, limit: int = 5) -> List[BaseMessage]:
        if not session_id:
            return []

        try:
            # Only fetch inbound messages to avoid distracting the LLM with its own massive product dumps
            query = self._supabase.table("messages").select("message_id,text,direction,created_at,reply_text").eq("session_id", session_id).eq("direction", "inbound")
            
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

class InMemoryProvider(BaseMemoryProvider):
    def __init__(self):
        self._store = {}

    def get_recent_messages(self, session_id: str, exclude_message_id: Optional[str] = None, limit: int = 5) -> List[BaseMessage]:
        if not session_id or session_id not in self._store:
            return []
            
        messages = self._store[session_id]
        if exclude_message_id:
            messages = [m for m in messages if getattr(m, "id", None) != exclude_message_id]
            
        # Very naive grouping
        langchain_messages = []
        for m in messages[-limit:]:
            if langchain_messages and type(langchain_messages[-1]) == type(m):
                langchain_messages[-1].content += f"\n\n{m.content}"
            else:
                langchain_messages.append(m)
        return langchain_messages
        
    def add_message(self, session_id: str, message: BaseMessage):
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append(message)


def get_recent_messages(session_id: str, exclude_message_id: Optional[str] = None, limit: int = 5) -> List[BaseMessage]:
    """
    Fetches the most recent inbound (user) messages for a given session from the active memory provider,
    and converts them to Langchain messages (chronological order).
    Consecutive messages from the same sender are grouped.
    """
    from src.services.agent.config import AgentConfig
    if AgentConfig.memory_provider is None:
        logger.warning("AgentConfig.memory_provider is not initialized, returning empty history.")
        return []
    return AgentConfig.memory_provider.get_recent_messages(session_id, exclude_message_id, limit)
