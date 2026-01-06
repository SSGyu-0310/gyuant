import os
from typing import Optional

import psycopg
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from utils.env import load_env

load_env()

_PG_ENGINE: Optional[Engine] = None
_PG_CONNECTION: Optional[psycopg.Connection] = None


def get_database_url() -> str:
    """Build PostgreSQL SQLAlchemy URL from environment variables."""
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    database = os.getenv("PG_DATABASE", "gyuant_market")
    user = os.getenv("PG_USER", "postgres")
    password = os.getenv("PG_PASSWORD", "")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


def get_engine() -> Engine:
    """Get or create SQLAlchemy engine (singleton)."""
    global _PG_ENGINE
    if _PG_ENGINE is None:
        _PG_ENGINE = create_engine(
            get_database_url(),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            echo=os.getenv("SQL_ECHO", "").lower() == "true",
        )
    return _PG_ENGINE


def get_db_connection() -> psycopg.Connection:
    """Get or create psycopg connection (singleton)."""
    global _PG_CONNECTION
    if _PG_CONNECTION is None or _PG_CONNECTION.closed:
        _PG_CONNECTION = psycopg.connect(
            host=os.getenv("PG_HOST", "localhost"),
            port=os.getenv("PG_PORT", "5432"),
            dbname=os.getenv("PG_DATABASE", "gyuant_market"),
            user=os.getenv("PG_USER", "postgres"),
            password=os.getenv("PG_PASSWORD", ""),
        )
        _PG_CONNECTION.autocommit = False
    return _PG_CONNECTION


def close_db_connection() -> None:
    """Close the cached psycopg connection."""
    global _PG_CONNECTION
    if _PG_CONNECTION and not _PG_CONNECTION.closed:
        _PG_CONNECTION.close()
    _PG_CONNECTION = None
