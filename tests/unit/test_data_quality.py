"""Data quality rules and the database-wide validator."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from backend.data_quality import DataQualityReport, DataValidator, Severity, rules
from backend.models import OddsHistory, RaceResult
from backend.utils.timeutils import UK_TZ
from tests.fixtures.racing_builders import build_race_with_results

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Race rules
# ---------------------------------------------------------------------------
def test_missing_race_date_is_an_error():
    issues = rules.check_race_date("r1", None)
    assert issues[0].rule == "race_date_missing"
    assert issues[0].severity is Severity.ERROR


def test_absurdly_old_race_date_is_rejected():
    issues = rules.check_race_date("r1", date(1850, 1, 1))
    assert any(issue.rule == "race_date_implausible" for issue in issues)


def test_far_future_race_date_is_rejected():
    today = date(2026, 1, 1)
    assert rules.check_race_date("r1", date(2026, 6, 1), today=today) == []
    issues = rules.check_race_date("r1", date(2030, 1, 1), today=today)
    assert issues[0].rule == "race_date_too_far_ahead"


@pytest.mark.parametrize("distance", [None, 10, 999, 8001, 100_000])
def test_invalid_distances_are_errors(distance):
    issues = rules.check_race_distance("r1", distance)
    assert issues and issues[0].severity is Severity.ERROR


@pytest.mark.parametrize("distance", [1000, 2200, 8000])
def test_valid_distances_pass(distance):
    assert rules.check_race_distance("r1", distance) == []


def test_missing_class_is_only_a_warning():
    """Most UK jumps races have no class; rejecting them would skew the sample."""
    issues = rules.check_race_class("r1", None)
    assert issues[0].severity is Severity.WARNING


@pytest.mark.parametrize("value", [0, 8, -1, 99])
def test_out_of_range_class_is_an_error(value):
    issues = rules.check_race_class("r1", value)
    assert issues[0].severity is Severity.ERROR


def test_field_size_rules():
    assert rules.check_field_size("r1", 8, 8) == []
    assert rules.check_field_size("r1", 1, 1)[0].severity is Severity.ERROR
    assert rules.check_field_size("r1", 45, 45)[0].severity is Severity.WARNING
    mismatch = rules.check_field_size("r1", 12, 8)
    assert any(issue.rule == "field_size_mismatch" for issue in mismatch)


def test_duplicate_race_detection():
    seen: dict[tuple, str] = {}
    off = datetime(2026, 5, 1, 14, 0)

    assert rules.check_duplicate_race("r1", "c1", date(2026, 5, 1), off, seen) == []
    duplicate = rules.check_duplicate_race("r2", "c1", date(2026, 5, 1), off, seen)
    assert duplicate[0].rule == "duplicate_race"
    assert duplicate[0].value == "r1"

    # A different time at the same course is a different race.
    other = datetime(2026, 5, 1, 15, 0)
    assert rules.check_duplicate_race("r3", "c1", date(2026, 5, 1), other, seen) == []


def test_duplicate_check_is_idempotent_for_the_same_race():
    seen: dict[tuple, str] = {}
    off = datetime(2026, 5, 1, 14, 0)
    rules.check_duplicate_race("r1", "c1", date(2026, 5, 1), off, seen)
    assert rules.check_duplicate_race("r1", "c1", date(2026, 5, 1), off, seen) == []


# ---------------------------------------------------------------------------
# Runner rules
# ---------------------------------------------------------------------------
def test_missing_horse_id_is_an_error():
    assert rules.check_horse_id("r1", None)[0].severity is Severity.ERROR
    assert rules.check_horse_id("r1", "hrs_1") == []


@pytest.mark.parametrize("age", [0, 1, 16, 40, -3])
def test_impossible_ages_are_errors(age):
    assert rules.check_horse_age("x", age)[0].rule == "age_impossible"


@pytest.mark.parametrize("age", [2, 5, 15])
def test_plausible_ages_pass(age):
    assert rules.check_horse_age("x", age) == []


def test_missing_age_is_a_warning():
    assert rules.check_horse_age("x", None)[0].severity is Severity.WARNING


@pytest.mark.parametrize("weight", [50, 94, 191, 500])
def test_invalid_weights_are_errors(weight):
    assert rules.check_weight("x", weight)[0].rule == "weight_out_of_range"


def test_valid_weight_passes():
    assert rules.check_weight("x", 133) == []


def test_finishing_position_rules():
    assert rules.check_finishing_position("x", 1, "finished", 8) == []
    assert rules.check_finishing_position("x", None, "pulled_up", 8) == []
    assert rules.check_finishing_position("x", None, "finished", 8)[0].severity is Severity.WARNING
    assert rules.check_finishing_position("x", 0, "finished", 8)[0].severity is Severity.ERROR
    assert rules.check_finishing_position("x", 9, "finished", 8)[0].rule == "position_exceeds_field"


def test_winner_count_rules():
    assert rules.check_single_winner("r1", 1) == []
    assert rules.check_single_winner("r1", 0)[0].severity is Severity.ERROR
    # Dead heats are real.
    assert rules.check_single_winner("r1", 2)[0].severity is Severity.WARNING


# ---------------------------------------------------------------------------
# Odds rules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("odds", [None, Decimal("1"), Decimal("0.5"), Decimal("-2")])
def test_invalid_odds_are_errors(odds):
    assert rules.check_decimal_odds("q", odds)[0].severity is Severity.ERROR


def test_odds_of_exactly_one_are_rejected():
    """Decimal 1.0 means a stake returns only itself — never a real price."""
    assert rules.check_decimal_odds("q", Decimal("1.0"))[0].rule == "odds_not_above_one"


def test_valid_odds_pass():
    assert rules.check_decimal_odds("q", Decimal("3.5")) == []


def test_missing_timestamp_is_an_error():
    """Without a timestamp we cannot prove the quote predates the off."""
    assert rules.check_odds_timestamp("q", None)[0].severity is Severity.ERROR


def test_quote_after_the_off_is_rejected():
    off = datetime(2026, 5, 1, 14, 0, tzinfo=UK_TZ)
    assert rules.check_odds_before_off("q", off - timedelta(minutes=5), off) == []
    late = rules.check_odds_before_off("q", off + timedelta(minutes=1), off)
    assert late[0].rule == "odds_after_off"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def test_report_counts_and_pass_rate():
    report = DataQualityReport()
    report.total_races = 100
    report.add(rules.check_race_distance("r1", None))
    report.add(rules.check_race_class("r2", None))

    assert report.rejected_races == 1
    assert report.valid_races == 99
    assert report.error_count == 1
    assert report.warning_count == 1
    assert report.pass_rate == pytest.approx(0.99)
    assert not report.is_clean


def test_report_renders_the_headline_numbers():
    report = DataQualityReport()
    report.total_races = 50_000
    for index in range(250):
        report.add(rules.check_race_distance(f"r{index}", None))

    rendered = report.render()
    assert "Total races     : 50000" in rendered
    assert "Valid           : 49750" in rendered
    assert "Rejected        : 250" in rendered


def test_report_as_dict_shape():
    report = DataQualityReport()
    report.total_races = 10
    report.add(rules.check_horse_age("x", 99), race_id="r1")
    data = report.as_dict()
    assert data["races"]["rejected"] == 1
    assert data["issues"]["errors"] == 1
    assert "age_impossible" in data["issues"]["by_rule"]


# ---------------------------------------------------------------------------
# Database-wide validation
# ---------------------------------------------------------------------------
def test_clean_database_passes(db_session):
    build_race_with_results(
        db_session,
        race_id="rac_ok",
        off_time=datetime(2026, 5, 1, 14, 0, tzinfo=UK_TZ),
        runners=[("hrs_a", 1), ("hrs_b", 2), ("hrs_c", 3)],
    )
    db_session.commit()

    report = DataValidator(db_session, today=date(2026, 5, 2)).validate_database()
    assert report.total_races == 1
    assert report.rejected_races == 0
    assert report.is_clean


def test_bad_distance_rejects_the_race(db_session):
    build_race_with_results(
        db_session,
        race_id="rac_bad",
        off_time=datetime(2026, 5, 1, 14, 0, tzinfo=UK_TZ),
        runners=[("hrs_a", 1), ("hrs_b", 2)],
        distance_yards=50,
    )
    db_session.commit()

    report = DataValidator(db_session, today=date(2026, 5, 2)).validate_database()
    assert "rac_bad" in report.rejected_race_ids
    assert report.valid_races == 0


def test_impossible_age_rejects_the_race(db_session):
    build_race_with_results(
        db_session,
        race_id="rac_age",
        off_time=datetime(2026, 5, 1, 14, 0, tzinfo=UK_TZ),
        runners=[("hrs_a", 1), ("hrs_b", 2)],
    )
    db_session.commit()
    result = db_session.query(RaceResult).filter_by(race_id="rac_age", horse_id="hrs_a").one()
    result.age = 99
    db_session.commit()

    report = DataValidator(db_session, today=date(2026, 5, 2)).validate_database()
    assert "rac_age" in report.rejected_race_ids


def test_race_with_no_winner_is_rejected(db_session):
    build_race_with_results(
        db_session,
        race_id="rac_nowin",
        off_time=datetime(2026, 5, 1, 14, 0, tzinfo=UK_TZ),
        runners=[("hrs_a", 2), ("hrs_b", 3)],
    )
    db_session.commit()

    report = DataValidator(db_session, today=date(2026, 5, 2)).validate_database()
    assert any(issue.rule == "no_winner" for issue in report.issues)
    assert "rac_nowin" in report.rejected_race_ids


def test_bad_odds_reject_the_race(db_session):
    off = datetime(2026, 5, 1, 14, 0, tzinfo=UK_TZ)
    build_race_with_results(
        db_session,
        race_id="rac_odds",
        off_time=off,
        runners=[("hrs_a", 1), ("hrs_b", 2)],
        odds={"hrs_a": [(off - timedelta(minutes=10), 3.0)]},
    )
    db_session.commit()
    quote = db_session.query(OddsHistory).filter_by(race_id="rac_odds").one()
    quote.decimal_odds = Decimal("0.9")
    db_session.commit()

    report = DataValidator(db_session, today=date(2026, 5, 2)).validate_database()
    assert any(issue.rule == "odds_not_above_one" for issue in report.issues)
    assert "rac_odds" in report.rejected_race_ids


def test_duplicate_races_are_detected_across_the_database(db_session):
    off = datetime(2026, 5, 1, 14, 0, tzinfo=UK_TZ)
    for race_id in ("rac_dup_a", "rac_dup_b"):
        build_race_with_results(
            db_session, race_id=race_id, off_time=off, runners=[("hrs_a", 1), ("hrs_b", 2)]
        )
    db_session.commit()

    report = DataValidator(db_session, today=date(2026, 5, 2)).validate_database()
    assert any(issue.rule == "duplicate_race" for issue in report.issues)
    assert len(report.rejected_race_ids) == 1  # the first one is kept


def test_validation_window_limits_what_is_checked(db_session):
    for index, day in enumerate([date(2026, 4, 1), date(2026, 5, 1)]):
        build_race_with_results(
            db_session,
            race_id=f"rac_w{index}",
            off_time=datetime(day.year, day.month, day.day, 14, 0, tzinfo=UK_TZ),
            runners=[("hrs_a", 1), (f"hrs_b{index}", 2)],
        )
    db_session.commit()

    report = DataValidator(db_session, today=date(2026, 5, 2)).validate_database(date_from=date(2026, 4, 20))
    assert report.total_races == 1


def test_empty_database_produces_an_empty_report(db_session):
    report = DataValidator(db_session).validate_database()
    assert report.total_races == 0
    assert report.is_clean
    assert report.pass_rate == 1.0
