import contextvars
from typing import Optional

# Holds the current tenant ID during agent execution
tenant_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("tenant_id", default=None)
