"""
SQLite engine and session factory for GraphRAG.

Sync engine is used by the data migration and pipeline scripts.
Async engine is used by FastAPI endpoints.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session

from src.db.models import Base, Product, ProductSpec, ProductVariant


def _db_path() -> str:
    # Explicit override always wins.
    if "SQLITE_PATH" in os.environ:
        path = Path(os.environ["SQLITE_PATH"])
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    db_dir = Path(os.getcwd()) / "data" / "db"

    # Resolve the active tenant (request context > env > config) to pick a
    # tenant-scoped DB file. Kept import-local so this module stays lightweight
    # and free of hard config/context dependencies.
    tenant_id = None
    try:
        from src.services.agent.context import tenant_context
        tenant_id = tenant_context.get()
    except Exception:
        tenant_id = None
    tenant_id = tenant_id or os.getenv("TENANT_ID")

    candidates = []
    if tenant_id:
        candidates.append(db_dir / f"{tenant_id}_knowledge_base.db")
    candidates.append(db_dir / "knowledge_base.db")
    # Legacy single-tenant fallback — ONLY when no tenant is set or it is the
    # original 'inventaa' tenant. Never fall back to another tenant's DB file
    # (that would be a cross-tenant data leak).
    if not tenant_id or tenant_id == "inventaa":
        candidates.append(db_dir / "inventaa_knowledge_base.db")

    # Use the first candidate that already exists; otherwise create the most
    # specific one (tenant-scoped if known, else generic).
    for c in candidates:
        if c.exists():
            path = c
            break
    else:
        path = candidates[0]

    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


@lru_cache
def _engine_for(path: str):
    url = f"sqlite:///{path}"
    engine = create_engine(url, echo=False, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(conn, _):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

    _ensure_schema_migrated(engine)
    return engine


def get_engine():
    # Resolve the tenant-scoped path per call, but reuse a cached engine per path
    # so multi-tenant switching creates/reuses the correct DB (not a single
    # first-seen engine).
    return _engine_for(_db_path())


def _ensure_schema_migrated(engine):
    Base.metadata.create_all(engine)
    try:
        with engine.begin() as conn:
            cols = {row[1] for row in conn.execute(text("PRAGMA table_info(products)"))}
            if "primary_option_name" not in cols:
                conn.execute(text("ALTER TABLE products ADD COLUMN primary_option_name VARCHAR"))
            if "primary_options" not in cols:
                conn.execute(text("ALTER TABLE products ADD COLUMN primary_options TEXT"))
            v_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(product_variants)"))}
            if "option_1" not in v_cols:
                conn.execute(text("ALTER TABLE product_variants ADD COLUMN option_1 VARCHAR"))
            if "option_2" not in v_cols:
                conn.execute(text("ALTER TABLE product_variants ADD COLUMN option_2 VARCHAR"))
    except Exception:
        pass


@lru_cache
def _async_engine_for(path: str):
    url = f"sqlite+aiosqlite:///{path}"
    return create_async_engine(url, echo=False)


def get_async_engine():
    return _async_engine_for(_db_path())


def get_session() -> Session:
    factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return factory()


def get_async_session() -> AsyncSession:
    factory = async_sessionmaker(bind=get_async_engine(), expire_on_commit=False)
    return factory()


def init_db():
    """Create all tables and auto-migrate missing columns (idempotent — safe to call on every startup)."""
    _ensure_schema_migrated(get_engine())
