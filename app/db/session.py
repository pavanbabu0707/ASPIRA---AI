# app/db/session.py
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings

DB_URL = settings.DATABASE_URL  # e.g., "sqlite+pysqlite:///./aspira_ai_db.db"

# SQLite-friendly connect args
is_sqlite = DB_URL.startswith("sqlite")
connect_args = {"check_same_thread": False, "timeout": 30} if is_sqlite else {}

engine = create_engine(
    DB_URL,
    future=True,
    pool_pre_ping=True,
    connect_args=connect_args,
)

# Enable WAL + sane pragmas for SQLite
if is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        try:
            cur = dbapi_connection.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA foreign_keys=ON;")
            cur.close()
        except Exception:
            pass

SessionLocal = sessionmaker(
    bind=engine,
    future=True,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a request-scoped Session.
    Use this SAME session for all writes within a request to avoid SQLite locks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
