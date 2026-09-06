"""End-to-end ingestion against a real PostgreSQL.

The unit suite runs on SQLite, which is fast but permissive: it ignores
``VARCHAR`` lengths and several constraint semantics. These tests exercise the
behaviour that only a real server enforces —  column limits, cascade deletes,
``NUMERIC`` precision and the unique constraints that make ingestion idempotent.

    make db-up
    .venv/bin/pytest -m integration

Skipped automatically when no database is reachable.
"""

from __future__ import annotations

import os
from decimal import Decimal

import httpx
import pytest
import respx
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from backend.data_pipeline import RaceDataImporter, import_racecards_for_date, import_results_for_range
from backend.database.base import Base
from backend.models import Horse, OddsHistory, Race, RaceResult, RaceRunner
from backend.services.racing_api.client import RacingAPIClient
from backend.services.racing_api.parsers import parse_racecard, parse_result
from backend.services.racing_api.schemas import RacecardPayload, ResultRacePayload
from backend.utils.config import Settings
from tests.fixtures import racing_api as fx

pytestmark = pytest.mark.integration

BASE = "https://api.theracingapi.com"


def _live_settings() -> Settings:
    return Settings(
        postgres_host=os.environ.get("TEST_POSTGRES_HOST", "localhost"),
        postgres_port=int(os.environ.get("TEST_POSTGRES_PORT", "5432")),
        postgres_user=os.environ.get("TEST_POSTGRES_USER", "horse_quant"),
        postgres_password=os.environ.get("TEST_POSTGRES_PASSWORD", "horse_quant"),
        postgres_db=os.environ.get("TEST_POSTGRES_DB", "horse_quant"),
        racing_api_username="test-username",
        racing_api_password="test-password",
        racing_api_rate_limit_per_second=10_000.0,
        racing_api_max_retries=1,
        DATABASE_URL=None,
    )


@pytest.fixture(scope="module")
def pg_engine():
    settings = _live_settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL not available: {exc}")

    # An isolated schema keeps these tests away from any real ingested data.
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS ingestion_test CASCADE"))
        connection.execute(text("CREATE SCHEMA ingestion_test"))

    scoped = create_engine(settings.database_url, connect_args={"options": "-csearch_path=ingestion_test"})
    Base.metadata.create_all(scoped)
    yield scoped

    scoped.dispose()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS ingestion_test CASCADE"))
    engine.dispose()


@pytest.fixture
def pg_session(pg_engine):
    with Session(pg_engine, expire_on_commit=False) as session:
        yield session
        session.rollback()
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()


# ---------------------------------------------------------------------------
# Real database semantics
# ---------------------------------------------------------------------------
def test_full_racecard_ingestion(pg_session):
    card = parse_racecard(
        RacecardPayload.model_validate(
            fx.racecard(runners=[fx.racecard_runner(odds=[fx.odds_entry(), fx.odds_entry("Sky Bet", "3.9")])])
        )
    )
    summary = RaceDataImporter(pg_session).import_races([card])

    assert summary.races_created == 1
    assert summary.runners_created == 1
    assert summary.odds_created == 2
    assert summary.failed == 0

    race = pg_session.get(Race, "rac_001")
    assert race.created_at is not None  # server_default fired
    assert race.updated_at is not None


def test_numeric_precision_is_preserved(pg_session):
    """Money and odds are NUMERIC, not float — no binary rounding drift."""
    card = parse_racecard(RacecardPayload.model_validate(fx.racecard(prize="£12,450.75")))
    RaceDataImporter(pg_session).import_races([card])

    race = pg_session.get(Race, "rac_001")
    assert race.prize_money == Decimal("12450.75")
    assert isinstance(race.prize_money, Decimal)


def test_oversized_identifier_is_rejected_by_the_column(pg_session):
    """PostgreSQL enforces VARCHAR length; SQLite does not. One race fails, alone."""
    good = parse_racecard(RacecardPayload.model_validate(fx.racecard(race_id="rac_ok")))
    broken = parse_racecard(RacecardPayload.model_validate(fx.racecard(race_id="x" * 200)))

    summary = RaceDataImporter(pg_session).import_races([good, broken])

    assert summary.races_created == 1
    assert summary.failed == 1
    assert pg_session.get(Race, "rac_ok") is not None


def test_unique_constraint_blocks_duplicate_runners(pg_session):
    card = parse_racecard(RacecardPayload.model_validate(fx.racecard()))
    RaceDataImporter(pg_session).import_races([card])
    RaceDataImporter(pg_session).import_races([card])

    runners = pg_session.execute(select(RaceRunner)).scalars().all()
    assert len(runners) == 1


def test_duplicate_odds_quote_is_skipped_not_inserted(pg_session):
    runner = fx.racecard_runner(odds=[fx.odds_entry()])
    card = parse_racecard(RacecardPayload.model_validate(fx.racecard(runners=[runner])))

    RaceDataImporter(pg_session).import_races([card])
    second = RaceDataImporter(pg_session).import_races([card])

    assert second.odds_skipped == 1
    assert len(pg_session.execute(select(OddsHistory)).scalars().all()) == 1


def test_cascade_delete_removes_dependent_rows(pg_session):
    card = parse_racecard(
        RacecardPayload.model_validate(fx.racecard(runners=[fx.racecard_runner(odds=[fx.odds_entry()])]))
    )
    RaceDataImporter(pg_session).import_races([card])
    result = parse_result(ResultRacePayload.model_validate(fx.result_race()))
    RaceDataImporter(pg_session).import_races([result])

    pg_session.delete(pg_session.get(Race, "rac_001"))
    pg_session.commit()

    assert pg_session.execute(select(RaceRunner)).scalars().all() == []
    assert pg_session.execute(select(RaceResult)).scalars().all() == []
    assert pg_session.execute(select(OddsHistory)).scalars().all() == []
    # Reference entities survive: a horse outlives any one race.
    assert pg_session.get(Horse, "hrs_001") is not None


# ---------------------------------------------------------------------------
# Full flow: mocked API -> client -> parser -> real database
# ---------------------------------------------------------------------------
@respx.mock
async def test_racecard_flow_from_api_to_postgres(pg_session):
    respx.get(f"{BASE}/v1/racecards/standard").mock(
        return_value=httpx.Response(
            200,
            json=fx.racecards_page(
                cards=[
                    fx.racecard(race_id="rac_a"),
                    fx.racecard(race_id="rac_b", runners=[fx.racecard_runner(horse_id="hrs_002")]),
                ]
            ),
        )
    )

    async with RacingAPIClient(_live_settings()) as client:
        summary = await import_racecards_for_date(pg_session, client, race_date="2026-08-10")

    assert summary.races_created == 2
    assert summary.api_requests >= 1
    assert summary.failed == 0

    stored = {race.race_id for race in pg_session.execute(select(Race)).scalars()}
    assert stored == {"rac_a", "rac_b"}


@respx.mock
async def test_results_flow_from_api_to_postgres(pg_session):
    respx.get(f"{BASE}/v1/results").mock(
        return_value=httpx.Response(200, json=fx.results_page(races=[fx.result_race(race_id="rac_r")]))
    )

    async with RacingAPIClient(_live_settings()) as client:
        summary = await import_results_for_range(
            pg_session, client, start_date="2026-08-10", end_date="2026-08-10"
        )

    assert summary.races_created == 1
    assert summary.results_created == 2

    race = pg_session.get(Race, "rac_r")
    assert race.has_result is True
    assert sum(1 for r in race.results if r.is_winner) == 1


@respx.mock
async def test_upstream_failure_leaves_the_database_untouched(pg_session):
    respx.get(f"{BASE}/v1/racecards/standard").mock(
        return_value=httpx.Response(401, json={"detail": "Subscription inactive"})
    )

    from backend.services.racing_api.exceptions import SubscriptionInactiveError

    async with RacingAPIClient(_live_settings()) as client:
        with pytest.raises(SubscriptionInactiveError):
            await import_racecards_for_date(pg_session, client, race_date="2026-08-10")

    assert pg_session.execute(select(Race)).scalars().all() == []


def test_indexes_exist_on_the_hot_query_paths(pg_engine):
    """Feature queries in Phase 5 scan by date and by horse; both must be indexed."""
    with pg_engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = 'ingestion_test' AND tablename IN "
                "('races', 'race_results', 'race_runners', 'odds_history')"
            )
        ).scalars()
        names = set(rows)

    assert any("race_date" in name for name in names)
    assert any("horse" in name for name in names)
    assert "uq_race_results_race_horse" in names
    assert "uq_odds_history_quote" in names
