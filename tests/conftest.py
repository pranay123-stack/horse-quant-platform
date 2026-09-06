"""Shared pytest fixtures.

The whole suite runs against an isolated, temporary configuration: a throwaway
log directory, a SQLite database file, and dummy Racing API credentials. No test
may touch the developer's real ``.env``, real database, or real API quota.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# Point pydantic-settings at a file that does not exist so the developer's real
# .env can never leak into a test run. Must happen before backend is imported.
os.environ["HQP_ENV_FILE"] = "/nonexistent/.env.test"

from datetime import UTC

from backend.utils.config import Settings, get_settings, reload_settings
from backend.utils.logging import configure_logging

TEST_ENV: dict[str, str] = {
    "ENVIRONMENT": "test",
    "DEBUG": "false",
    "LOG_LEVEL": "DEBUG",
    "LOG_JSON": "false",
    "LOG_TO_CONSOLE": "false",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_USER": "test_user",
    "POSTGRES_PASSWORD": "test_password",
    "POSTGRES_DB": "test_db",
    "RACING_API_USERNAME": "test-username",
    "RACING_API_PASSWORD": "test-password",
    "RACING_API_KEY": "test-key",
    "SECRET_KEY": "test-secret-key",
}


@pytest.fixture(scope="session", autouse=True)
def _test_environment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Install deterministic environment variables for the whole session."""
    workspace = tmp_path_factory.mktemp("hqp")
    overrides = {
        **TEST_ENV,
        "LOG_DIR": str(workspace / "logs"),
        "DATA_DIR": str(workspace / "data"),
        "MODELS_DIR": str(workspace / "models"),
        "REPORTS_DIR": str(workspace / "reports"),
        "DATABASE_URL": f"sqlite+pysqlite:///{workspace / 'test.db'}",
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    reload_settings()

    yield

    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    reload_settings()


@pytest.fixture(scope="session")
def settings(_test_environment: None) -> Settings:
    """The session-wide test settings object.

    Depends on ``_test_environment`` explicitly: autouse alone does not order a
    fixture before another session fixture that something else pulls in first.
    """
    return get_settings()


@pytest.fixture
def env_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the ambient test environment for a single test.

    Tests that construct ``Settings(...)`` directly to exercise defaults and
    validators must not inherit the session-wide overrides.
    """
    ambient = set(TEST_ENV) | {"DATABASE_URL", "LOG_DIR", "DATA_DIR", "MODELS_DIR", "REPORTS_DIR"}
    for key in ambient:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session", autouse=True)
def _logging(settings: Settings) -> None:
    settings.ensure_directories()
    configure_logging(settings, force=True)


@pytest.fixture
def log_dir(settings: Settings) -> Path:
    return settings.log_dir


@pytest.fixture
def app(settings: Settings):
    """A freshly built FastAPI application bound to the test settings."""
    from backend.main import create_app

    return create_app(settings)


@pytest.fixture
def client(app) -> Iterator:
    """Synchronous test client with startup/shutdown events executed."""
    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_engine() -> Iterator[None]:
    """Dispose the SQLAlchemy engine between tests so state never leaks."""
    yield
    from backend.database.session import dispose_engine

    dispose_engine()


# ---------------------------------------------------------------------------
# Database fixtures (Phase 2)
# ---------------------------------------------------------------------------
@pytest.fixture
def db_engine():
    """A private in-memory SQLite engine with the full schema.

    ``StaticPool`` keeps every connection pointed at the same in-memory database,
    and the ``PRAGMA`` turns on foreign-key enforcement -- off by default in
    SQLite, which would otherwise let broken references pass tests that
    PostgreSQL rejects in production.
    """
    from sqlalchemy import create_engine, event
    from sqlalchemy.pool import StaticPool

    from backend.models import Base  # imports every model, populating metadata

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Iterator:
    """A session bound to the private test database."""
    from sqlalchemy.orm import Session

    with Session(db_engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture
def api_client(app, db_session) -> Iterator:
    """TestClient whose request sessions hit the private test database."""
    from fastapi.testclient import TestClient

    from backend.database.session import get_db

    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def racing_credentials():
    from backend.services.racing_api.authentication import RacingAPICredentials

    return RacingAPICredentials(username="test-username", password="test-password")


def _build_race(session, *, race_date, race_id, with_results: bool):
    """A minimal but complete race: course, horses, runners, optional results.

    Phase 7 stores predictions against real foreign keys, so the product tests
    need genuine rows rather than bare ids.
    """
    from datetime import datetime

    from backend.models import Course, Horse, Race, RaceResult, RaceRunner

    course = session.get(Course, "crs_t01")
    if course is None:
        course = Course(course_id="crs_t01", name="Testfield Park", region_code="gb")
        session.add(course)

    # Six, so a card built from this fixture clears the minimum field size.
    horse_ids = [f"hrs_t{index:02d}" for index in range(6)]
    for index, horse_id in enumerate(horse_ids):
        if session.get(Horse, horse_id) is None:
            session.add(Horse(horse_id=horse_id, name=f"Test Horse {index}"))

    session.add(
        Race(
            race_id=race_id,
            race_date=race_date,
            off_time=datetime.combine(race_date, datetime.min.time()).replace(hour=15, minute=30, tzinfo=UTC),
            course_id=course.course_id,
            race_name="Test Handicap",
            field_size=len(horse_ids),
            has_result=with_results,
        )
    )
    session.flush()

    for index, horse_id in enumerate(horse_ids):
        session.add(RaceRunner(race_id=race_id, horse_id=horse_id, saddle_cloth=index + 1))
        if with_results:
            session.add(
                RaceResult(
                    race_id=race_id,
                    horse_id=horse_id,
                    finishing_position=index + 1,
                    is_winner=index == 0,
                )
            )
    session.commit()
    return {"race_id": race_id, "race_date": race_date, "horse_ids": horse_ids}


@pytest.fixture
def seeded_race(db_session):
    """A finished race, so predictions against it can be settled."""
    from datetime import date

    return _build_race(db_session, race_date=date(2026, 6, 15), race_id="rac_t01", with_results=True)


@pytest.fixture
def seeded_race_unsettled(db_session):
    """A race whose result has not arrived yet."""
    from datetime import date

    return _build_race(db_session, race_date=date(2026, 6, 16), race_id="rac_t02", with_results=False)
