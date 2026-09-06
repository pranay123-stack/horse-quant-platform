"""Ingestion tests.

The properties under test are the ones that decide whether a nightly job can be
re-run safely: idempotence, failure isolation, and not destroying known data
with a sparser later payload.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from backend.data_pipeline import ImportSummary, RaceDataImporter
from backend.models import Course, Horse, Jockey, OddsHistory, Race, RaceResult, RaceRunner, Trainer
from backend.services.racing_api.parsers import parse_racecard, parse_result
from backend.services.racing_api.schemas import (
    HorseSchema,
    OddsSchema,
    RacecardPayload,
    RaceSchema,
    ResultRacePayload,
)
from tests.fixtures import racing_api as fx

pytestmark = pytest.mark.unit


def card(**kwargs) -> RaceSchema:
    return parse_racecard(RacecardPayload.model_validate(fx.racecard(**kwargs)))


def result(**kwargs) -> RaceSchema:
    return parse_result(ResultRacePayload.model_validate(fx.result_race(**kwargs)))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_racecard_import_populates_every_table(db_session):
    summary = RaceDataImporter(db_session).import_races([card()])

    assert summary.races_created == 1
    assert summary.runners_created == 1
    assert summary.failed == 0

    race = db_session.get(Race, "rac_001")
    assert race.course_name == "Ascot"
    assert race.race_class == 3
    assert race.distance_yards == 4400
    assert race.prize_money == Decimal("12450.00")
    assert race.has_result is False

    assert db_session.get(Course, "crs_001").name == "Ascot"
    assert db_session.get(Horse, "hrs_001").name == "Thunder King"
    assert db_session.get(Jockey, "jky_001").name == "N de Boinville"
    assert db_session.get(Trainer, "trn_001").location == "Lambourn"


def test_result_import_sets_the_winner_flag(db_session):
    summary = RaceDataImporter(db_session).import_races([result()])

    assert summary.results_created == 2
    race = db_session.get(Race, "rac_001")
    assert race.has_result is True

    winners = [r for r in race.results if r.is_winner]
    assert len(winners) == 1
    assert winners[0].horse_id == "hrs_001"
    assert winners[0].starting_price == Decimal("3.500")


def test_non_finisher_has_no_position_but_keeps_a_status(db_session):
    payload = fx.result_race(runners=[fx.result_runner(position="PU")])
    RaceDataImporter(db_session).import_races([parse_result(ResultRacePayload.model_validate(payload))])

    stored = db_session.execute(select(RaceResult)).scalar_one()
    assert stored.finishing_position is None
    assert stored.finishing_status == "pulled_up"
    assert stored.is_winner is False


def test_messy_runner_imports_with_nulls(db_session):
    summary = RaceDataImporter(db_session).import_races([card(runners=[fx.MESSY_RUNNER])])

    assert summary.failed == 0
    runner = db_session.execute(select(RaceRunner)).scalar_one()
    assert runner.age is None
    assert runner.official_rating is None


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------
def test_reimport_creates_no_duplicates(db_session):
    RaceDataImporter(db_session).import_races([card()])
    second = RaceDataImporter(db_session).import_races([card()])

    assert second.races_created == 0
    assert second.races_updated == 1
    assert db_session.execute(select(Race)).scalars().all().__len__() == 1
    assert db_session.execute(select(RaceRunner)).scalars().all().__len__() == 1


def test_reimport_is_stable_over_many_runs(db_session):
    for _ in range(3):
        RaceDataImporter(db_session).import_races([card(), result()])

    assert len(db_session.execute(select(Race)).scalars().all()) == 1
    assert len(db_session.execute(select(RaceRunner)).scalars().all()) == 1
    assert len(db_session.execute(select(RaceResult)).scalars().all()) == 2


def test_racecard_then_result_upgrades_the_same_race(db_session):
    RaceDataImporter(db_session).import_races([card()])
    RaceDataImporter(db_session).import_races([result()])

    race = db_session.get(Race, "rac_001")
    assert race.has_result is True
    assert len(race.runners) == 1  # declaration preserved
    assert len(race.results) == 2  # outcome added


def test_sparse_update_does_not_erase_known_values(db_session):
    """A later, poorer payload must not blank a field an earlier one filled."""
    RaceDataImporter(db_session).import_races([card()])
    assert db_session.get(Horse, "hrs_001").colour == "bay"

    sparse = card(runners=[fx.racecard_runner(colour="", sex="", dob="")])
    RaceDataImporter(db_session).import_races([sparse])

    horse = db_session.get(Horse, "hrs_001")
    assert horse.colour == "bay"
    assert horse.sex == "gelding"


def test_changed_values_are_updated(db_session):
    RaceDataImporter(db_session).import_races([card()])
    RaceDataImporter(db_session).import_races([card(going="Soft", weather="Rain")])

    race = db_session.get(Race, "rac_001")
    assert race.going == "soft"
    assert race.weather == "Rain"


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------
def test_odds_are_imported_with_the_racecard(db_session):
    runner = fx.racecard_runner(odds=[fx.odds_entry(), fx.odds_entry("Sky Bet", "3.6")])
    summary = RaceDataImporter(db_session).import_races([card(runners=[runner])])

    assert summary.odds_created == 2
    quotes = db_session.execute(select(OddsHistory)).scalars().all()
    assert {q.bookmaker for q in quotes} == {"Bet365", "Sky Bet"}
    assert quotes[0].implied_probability == pytest.approx(1 / float(quotes[0].decimal_odds))


def test_identical_quotes_are_not_duplicated(db_session):
    runner = fx.racecard_runner(odds=[fx.odds_entry()])
    RaceDataImporter(db_session).import_races([card(runners=[runner])])
    second = RaceDataImporter(db_session).import_races([card(runners=[runner])])

    assert second.odds_created == 0
    assert second.odds_skipped == 1
    assert len(db_session.execute(select(OddsHistory)).scalars().all()) == 1


def test_a_price_move_creates_a_new_row(db_session):
    """Same bookmaker, new timestamp — that is a price history, not a duplicate."""
    first = fx.racecard_runner(odds=[fx.odds_entry(decimal="3.5")])
    RaceDataImporter(db_session).import_races([card(runners=[first])])

    moved = fx.racecard_runner(odds=[fx.odds_entry(decimal="3.0", updated="2026-08-10T13:00:00+01:00")])
    summary = RaceDataImporter(db_session).import_races([card(runners=[moved])])

    assert summary.odds_created == 1
    assert len(db_session.execute(select(OddsHistory)).scalars().all()) == 2


def test_import_odds_requires_the_race_to_exist(db_session):
    quote = OddsSchema(
        race_id="rac_unknown", horse_id="hrs_001", bookmaker="Bet365", decimal_odds=Decimal("3")
    )
    summary = RaceDataImporter(db_session).import_odds([quote])

    assert summary.odds_created == 0
    assert summary.failed == 1
    assert "import the racecard first" in summary.failures[0].message


def test_import_odds_for_a_known_race(db_session):
    RaceDataImporter(db_session).import_races([card()])
    quote = OddsSchema(
        race_id="rac_001", horse_id="hrs_001", bookmaker="Betfair", decimal_odds=Decimal("4.2")
    )
    summary = RaceDataImporter(db_session).import_odds([quote])
    assert summary.odds_created == 1


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------
def _failing_importer(db_session, monkeypatch, bad_race_id: str, error: Exception) -> RaceDataImporter:
    """An importer whose persistence blows up for exactly one race.

    Injection beats relying on a real constraint here: SQLite does not enforce
    ``VARCHAR`` length, so the obvious "oversized id" trick silently succeeds.
    The equivalent test against a real PostgreSQL constraint lives in
    ``tests/integration/test_ingestion_flow.py``.
    """
    importer = RaceDataImporter(db_session)
    original = importer._upsert_race

    def flaky(race):
        if race.race_id == bad_race_id:
            raise error
        return original(race)

    monkeypatch.setattr(importer, "_upsert_race", flaky)
    return importer


def test_a_bad_race_does_not_lose_the_rest_of_the_card(db_session, monkeypatch):
    good_one = card(race_id="rac_good_1")
    broken = card(race_id="rac_bad")
    good_two = card(race_id="rac_good_2", runners=[fx.racecard_runner(horse_id="hrs_002")])

    importer = _failing_importer(
        db_session, monkeypatch, "rac_bad", SQLAlchemyError("simulated constraint violation")
    )
    summary = importer.import_races([good_one, broken, good_two])

    assert summary.races_seen == 3
    assert summary.races_created == 2
    assert summary.failed == 1
    assert summary.failures[0].stage == "persist"

    stored = {race.race_id for race in db_session.execute(select(Race)).scalars()}
    assert stored == {"rac_good_1", "rac_good_2"}


def test_unexpected_errors_are_also_contained(db_session, monkeypatch):
    """A bug in our own code must degrade to one lost race, not a lost run."""
    importer = _failing_importer(db_session, monkeypatch, "rac_bad", RuntimeError("programmer error"))
    summary = importer.import_races([card(race_id="rac_good_1"), card(race_id="rac_bad")])

    assert summary.races_created == 1
    assert summary.failures[0].error_type == "RuntimeError"


def test_failure_details_are_captured_for_retry(db_session, monkeypatch):
    importer = _failing_importer(db_session, monkeypatch, "rac_bad", SQLAlchemyError("boom"))
    summary = importer.import_races([card(race_id="rac_bad")])

    failure = summary.failures[0]
    assert failure.record_id == "rac_bad"
    assert failure.stage == "persist"
    assert failure.error_type == "SQLAlchemyError"
    assert "boom" in failure.message


# ---------------------------------------------------------------------------
# Entity de-duplication within a run
# ---------------------------------------------------------------------------
def test_shared_trainer_is_upserted_once_per_run(db_session):
    runners = [fx.racecard_runner(horse_id=f"hrs_{i:03d}", number=str(i)) for i in range(1, 6)]
    summary = RaceDataImporter(db_session).import_races([card(runners=runners)])

    assert summary.horses_upserted == 5
    assert summary.trainers_upserted == 1  # same trainer on all five
    assert summary.jockeys_upserted == 1


# ---------------------------------------------------------------------------
# Summary object
# ---------------------------------------------------------------------------
def test_summary_render_and_dict():
    summary = ImportSummary()
    summary.races_seen = 520
    summary.races_created = 500
    summary.races_skipped = 20
    summary.record_failure("rac_x", "persist", ValueError("bad row"))

    rendered = summary.render()
    assert "Races seen      : 520" in rendered
    assert "rac_x" in rendered

    data = summary.as_dict()
    assert data["races"]["created"] == 500
    assert data["failed"] == 1
    assert data["failures"][0]["error_type"] == "ValueError"


def test_summaries_merge():
    first, second = ImportSummary(), ImportSummary()
    first.races_created = 10
    second.races_created = 5
    second.record_failure("r", "persist", ValueError("x"))

    first.merge(second)
    assert first.races_created == 15
    assert first.failed == 1


def test_is_clean_flag():
    summary = ImportSummary()
    assert summary.is_clean
    summary.record_failure("r", "persist", ValueError("x"))
    assert not summary.is_clean


def test_upsert_horse_is_cached_within_a_run(db_session):
    importer = RaceDataImporter(db_session)
    horse = HorseSchema(horse_id="hrs_cache", name="Cached")
    importer._upsert_horse(horse)
    importer._upsert_horse(horse)
    assert importer.summary.horses_upserted == 1
