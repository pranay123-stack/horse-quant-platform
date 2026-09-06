"""Engine and session management.

The platform uses the **synchronous** SQLAlchemy engine on purpose: the ETL,
feature-engineering, training and backtesting layers are all pandas/NumPy code
that is synchronous anyway, and FastAPI runs sync path operations in a worker
threadpool. One engine, one mental model, no colour-of-function problem.

The engine is created lazily so that importing any module -- in a unit test, in
Alembic, in a notebook -- never requires a live database.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from backend.utils.config import Settings, get_settings
from backend.utils.exceptions import DatabaseError
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="api")

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _engine_kwargs(settings: Settings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "echo": settings.db_echo,
        "pool_pre_ping": True,  # transparently recycle connections killed by the server
        "pool_recycle": settings.db_pool_recycle_seconds,
        "future": True,
    }
    url = settings.database_url
    if url.startswith("sqlite"):
        # SQLite (tests) has no server-side pooling knobs.
        kwargs["connect_args"] = {"check_same_thread": False}
        return kwargs

    kwargs["pool_size"] = settings.db_pool_size
    kwargs["max_overflow"] = settings.db_max_overflow
    kwargs["connect_args"] = {
        "application_name": "horse_quant_platform",
        "options": f"-c statement_timeout={settings.db_statement_timeout_ms}",
    }
    return kwargs


def get_engine(settings: Settings | None = None) -> Engine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        settings = settings or get_settings()
        _engine = create_engine(settings.database_url, **_engine_kwargs(settings))
        logger.info("database engine created", extra={"url": settings.safe_database_url})
    return _engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(settings),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,  # objects stay usable after commit
            class_=Session,
        )
    return _session_factory


def dispose_engine() -> None:
    """Tear down the engine and session factory (tests, worker shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        logger.info("database engine disposed")
    _engine = None
    _session_factory = None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts, ETL jobs and Celery tasks.

    Commits on success, rolls back on any exception, always closes::

        with session_scope() as session:
            session.add(race)
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session.

    The request handler owns the transaction boundary; this dependency only
    guarantees rollback-on-error and close-on-exit.
    """
    session = get_session_factory()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_connection(settings: Settings | None = None) -> bool:
    """Cheap liveness probe used by ``GET /health/db`` and startup checks."""
    try:
        with get_engine(settings).connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as exc:
        logger.warning("database health check failed", extra={"error": str(exc)})
        return False


def require_database(settings: Settings | None = None) -> None:
    """Raise :class:`DatabaseError` when the database is unreachable."""
    if not check_database_connection(settings):
        raise DatabaseError("database is unreachable")


__all__ = [
    "check_database_connection",
    "dispose_engine",
    "get_db",
    "get_engine",
    "get_session_factory",
    "require_database",
    "session_scope",
]
