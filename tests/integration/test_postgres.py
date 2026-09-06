"""Integration tests against a live PostgreSQL instance.

Run with::

    make db-up
    .venv/bin/pytest -m integration

They are skipped automatically when no database is reachable, so the default
``make test`` run stays green on a laptop with nothing running.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text

from backend.utils.config import Settings

pytestmark = [pytest.mark.integration]


def _live_settings() -> Settings:
    """Settings pointing at the docker-compose PostgreSQL, ignoring test overrides."""
    return Settings(
        postgres_host=os.environ.get("TEST_POSTGRES_HOST", "localhost"),
        postgres_port=int(os.environ.get("TEST_POSTGRES_PORT", "5432")),
        postgres_user=os.environ.get("TEST_POSTGRES_USER", "horse_quant"),
        postgres_password=os.environ.get("TEST_POSTGRES_PASSWORD", "horse_quant"),
        postgres_db=os.environ.get("TEST_POSTGRES_DB", "horse_quant"),
        DATABASE_URL=None,
    )


@pytest.fixture(scope="module")
def live_engine():
    settings = _live_settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL not available: {exc}")
    yield engine
    engine.dispose()


def test_server_is_postgres_16_or_newer(live_engine):
    with live_engine.connect() as connection:
        version = connection.execute(text("SHOW server_version")).scalar_one()
    major = int(str(version).split(".")[0])
    assert major >= 16, f"expected PostgreSQL >= 16, got {version}"


def test_round_trip_through_a_temporary_table(live_engine):
    with live_engine.begin() as connection:
        connection.execute(text("CREATE TEMPORARY TABLE _probe (id int primary key, note text)"))
        connection.execute(text("INSERT INTO _probe VALUES (1, 'hello')"))
        note = connection.execute(text("SELECT note FROM _probe WHERE id = 1")).scalar_one()
    assert note == "hello"


def test_statement_timeout_is_applied(live_engine):
    """Confirms the server accepts the timeout we set in connect_args."""
    with live_engine.connect() as connection:
        connection.execute(text("SET statement_timeout = 100"))
        assert connection.execute(text("SHOW statement_timeout")).scalar_one() == "100ms"
