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

from src.db.models import Base


def _db_path() -> str:
    if "SQLITE_PATH" in os.environ:
        path = Path(os.environ["SQLITE_PATH"])
    else:
        db_dir = Path(os.getcwd()) / "data" / "db"
        legacy_path = db_dir / "inventaa_knowledge_base.db"
        generic_path = db_dir / "knowledge_base.db"
        path = legacy_path if legacy_path.exists() and not generic_path.exists() else generic_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


@lru_cache
def get_engine():
    url = f"sqlite:///{_db_path()}"
    engine = create_engine(url, echo=False, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(conn, _):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

    _ensure_schema_migrated(engine)
    return engine


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
def get_async_engine():
    url = f"sqlite+aiosqlite:///{_db_path()}"
    return create_async_engine(url, echo=False)


def get_session() -> Session:
    factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return factory()


def get_async_session() -> AsyncSession:
    factory = async_sessionmaker(bind=get_async_engine(), expire_on_commit=False)
    return factory()


def init_db():
    """Create all tables and auto-migrate missing columns (idempotent — safe to call on every startup)."""
    _ensure_schema_migrated(get_engine())
