"""Parser tests.

These carry disproportionate weight: every downstream number in the platform —
every feature, every probability, every expected-value calculation — passes
through this layer first. A silent coercion bug here does not crash anything, it
just makes the model quietly wrong.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from backend.services.racing_api import parsers as p
from backend.services.racing_api.schemas import (
    OddsEntryPayload,
    RacecardPayload,
    ResultRacePayload,
    RunnerOddsPayload,
)
from tests.fixtures import racing_api as fx

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Null sentinels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sentinel", ["", " ", "-", "--", "–", "—", "N/A", "n/a", "None", "null", "nan", "?"])
def test_null_sentinels_become_none(sentinel):
    assert p.clean_str(sentinel) is None
    assert p.to_int(sentinel) is None
    assert p.to_float(sentinel) is None
    assert p.to_decimal(sentinel) is None


def test_clean_str_trims_and_preserves_real_values():
    assert p.clean_str("  Ascot  ") == "Ascot"
    assert p.clean_str("0") == "0"  # a real zero is not a null


# ---------------------------------------------------------------------------
# Numeric coercion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("7", 7), ("7th", 7), ("  12 ", 12), ("1,234", 1234), ("-3", -3), ("8.9", 8), ("abc", None)],
)
def test_to_int(raw, expected):
    assert p.to_int(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [("3.5", 3.5), ("2", 2.0), ("1,234.5", 1234.5), ("x", None)])
def test_to_float(raw, expected):
    assert p.to_float(raw) == expected


def test_to_decimal_is_exact():
    assert p.to_decimal("0.1") == Decimal("0.1")
    assert p.to_decimal("12.34") + p.to_decimal("0.66") == Decimal("13.00")


def test_to_bool():
    assert p.to_bool("true") is True
    assert p.to_bool("N") is False
    assert p.to_bool(True) is True
    assert p.to_bool("maybe") is None


# ---------------------------------------------------------------------------
# Racing quantities
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("9-7", 133), ("10-0", 140), ("8–13", 125), ("133", 133), ("", None), ("-", None)],
)
def test_parse_weight_lbs(raw, expected):
    assert p.parse_weight_lbs(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5/2", Decimal("3.5")),
        ("1/1", Decimal("2")),
        ("evens", Decimal("2")),
        ("EVS", Decimal("2")),
        ("100/30", Decimal("100") / Decimal("30") + 1),
        ("2/7", Decimal("2") / Decimal("7") + 1),
        ("SP", None),
        ("5/0", None),
        ("nonsense", None),
    ],
)
def test_parse_fractional_odds(raw, expected):
    assert p.parse_fractional_odds(raw) == expected


def test_decimal_odds_below_one_are_rejected():
    """A price under 1.0 implies >100% return with no stake — always bad data."""
    assert p.parse_decimal_odds("0.5") is None
    assert p.parse_decimal_odds("1.01") == Decimal("1.01")


def test_resolve_odds_prefers_decimal_then_falls_back():
    assert p.resolve_odds("3.5", "9/4") == Decimal("3.5")
    assert p.resolve_odds("", "9/4") == Decimal("3.25")
    assert p.resolve_odds("-", "-") is None


def test_implied_probability():
    assert p.implied_probability(Decimal("4")) == 0.25
    assert p.implied_probability(Decimal("2")) == 0.5
    assert p.implied_probability(None) is None
    assert p.implied_probability(Decimal("0.5")) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1m", 1760),
        ("2m4f", 4400),
        ("1m2f110y", 2310),
        ("6f", 1320),
        ("4390", 4390),  # explicit yardage
        ("10.5", 2310),  # bare furlongs
        ("", None),
    ],
)
def test_parse_distance_yards(raw, expected):
    assert p.parse_distance_yards(raw) == expected


def test_parse_distance_prefers_the_first_usable_value():
    assert p.parse_distance_yards("", "-", "2m4f") == 4400


def test_yards_to_furlongs():
    assert p.yards_to_furlongs(2200) == 10.0
    assert p.yards_to_furlongs(None) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", (1, "finished")),
        ("12", (12, "finished")),
        ("1=", (1, "finished")),  # dead heat still finished first
        ("PU", (None, "pulled_up")),
        ("F", (None, "fell")),
        ("UR", (None, "unseated_rider")),
        ("BD", (None, "brought_down")),
        ("DSQ", (None, "disqualified")),
        ("", (None, "unknown")),
        ("0", (None, "unknown")),
    ],
)
def test_parse_position(raw, expected):
    assert p.parse_position(raw) == expected


def test_parse_prize_strips_currency():
    assert p.parse_prize("£12,450") == Decimal("12450")
    assert p.parse_prize("12450.50") == Decimal("12450.50")
    assert p.parse_prize("-") is None


def test_parse_going_normalises():
    assert p.parse_going("  Good To   Soft ") == "good to soft"
    assert p.parse_going("") is None


def test_parse_form_string_drops_separators():
    assert p.parse_form_string("1-P23") == ["1", "P", "2", "3"]
    assert p.parse_form_string("12/3F") == ["1", "2", "3", "F"]
    assert p.parse_form_string("") == []


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
def test_to_date():
    assert p.to_date("2026-08-10") == date(2026, 8, 10)
    assert p.to_date("2026-08-10T14:35:00+01:00") == date(2026, 8, 10)
    assert p.to_date("nonsense") is None


def test_to_datetime_keeps_the_offset():
    parsed = p.to_datetime("2026-08-10T14:35:00+01:00")
    assert parsed == datetime.fromisoformat("2026-08-10T14:35:00+01:00")


def test_naive_datetime_is_assumed_uk_local():
    parsed = p.to_datetime("2026-08-10T14:35:00")
    assert parsed.utcoffset().total_seconds() == 3600  # BST in August


def test_combine_race_datetime_treats_early_hours_as_afternoon():
    """UK racing has no 2am cards — "2:35" is 14:35."""
    combined = p.combine_race_datetime(date(2026, 8, 10), "2:35")
    assert combined.hour == 14
    assert p.combine_race_datetime(date(2026, 8, 10), "14:35").hour == 14


def test_combine_race_datetime_rejects_garbage():
    assert p.combine_race_datetime(date(2026, 8, 10), "later") is None
    assert p.combine_race_datetime(None, "14:35") is None


# ---------------------------------------------------------------------------
# Field-name normalisation
# ---------------------------------------------------------------------------
def test_pick_returns_first_usable_alias():
    assert p.pick({"class": "", "race_class": "Class 3"}, "class", "race_class") == "Class 3"
    assert p.pick({"a": "-"}, "a", "b") is None


def test_normalise_race_fields_unifies_both_shapes():
    from_card = p.normalise_race_fields(fx.racecard())
    from_result = p.normalise_race_fields(fx.result_race())
    assert from_card["race_class"] == "Class 3"
    assert from_result["race_class"] == "Class 3"  # arrived as "class"
    assert from_card["off_time"] == from_result["off_time"]  # "off_time" vs "off"


# ---------------------------------------------------------------------------
# Payload -> domain
# ---------------------------------------------------------------------------
def test_parse_racecard_produces_typed_values():
    race = p.parse_racecard(RacecardPayload.model_validate(fx.racecard()))

    assert race.race_id == "rac_001"
    assert race.race_date == date(2026, 8, 10)
    assert race.race_class == 3
    assert race.distance_yards == 4400  # from "2m4f"
    assert race.distance_furlongs == 20.0
    assert race.prize_money == Decimal("12450")
    assert race.going == "good"
    assert race.is_result is False
    assert len(race.runners) == 1

    runner = race.runners[0]
    assert runner.age == 5
    assert runner.weight_lbs == 133
    assert runner.official_rating == 88
    assert runner.rpr == 95
    assert runner.topspeed == 82
    assert runner.days_since_last_run == 28
    assert runner.form_tokens == ["1", "P", "2", "3"]
    assert runner.horse.date_of_birth == date(2020, 3, 14)


def test_parse_racecard_survives_an_all_sentinel_runner():
    """The row that breaks naive parsing must import with NULLs, not explode."""
    race = p.parse_racecard(RacecardPayload.model_validate(fx.racecard(runners=[fx.MESSY_RUNNER])))
    runner = race.runners[0]

    assert runner.horse.horse_id == "hrs_messy"
    assert runner.age is None
    assert runner.draw is None
    assert runner.weight_lbs is None
    assert runner.official_rating is None
    assert runner.form_tokens == []


def test_parse_result_reads_the_alternate_field_names():
    race = p.parse_result(ResultRacePayload.model_validate(fx.result_race()))

    assert race.race_class == 3  # from "class"
    assert race.distance_yards == 4390  # from "dist_y"
    assert race.is_result is True
    assert race.off_time is not None  # from "off"
    assert len(race.results) == 2

    winner = race.results[0]
    assert winner.finishing_position == 1
    assert winner.finishing_status == "finished"
    assert winner.starting_price == Decimal("3.5")
    assert winner.weight_lbs == 133
    assert winner.prize_won == Decimal("8000")
    assert winner.official_rating == 88  # from "or"


def test_parse_result_handles_a_faller():
    payload = fx.result_race(runners=[fx.result_runner(position="F", sp_dec="", btn="")])
    race = p.parse_result(ResultRacePayload.model_validate(payload))
    runner = race.results[0]
    assert runner.finishing_position is None
    assert runner.finishing_status == "fell"


def test_build_odds_skips_unpriced_quotes():
    entry = OddsEntryPayload(bookmaker="Bet365", decimal="-", fractional="-")
    assert p.build_odds("rac_001", "hrs_001", entry) is None


def test_build_odds_computes_implied_probability():
    quote = p.build_odds("rac_001", "hrs_001", OddsEntryPayload(**fx.odds_entry(decimal="4.0")))
    assert quote.decimal_odds == Decimal("4.0")
    assert quote.implied_probability == 0.25
    assert quote.each_way_places == 3


def test_parse_runner_odds_returns_all_bookmakers():
    quotes = p.parse_runner_odds(RunnerOddsPayload.model_validate(fx.runner_odds()))
    assert len(quotes) == 3
    assert {q.bookmaker for q in quotes} == {"Bet365", "William Hill", "Paddy Power"}


def test_parse_runner_odds_needs_identifiers():
    assert p.parse_runner_odds(RunnerOddsPayload(race_id=None, horse_id=None, odds=[])) == []


def test_racecard_odds_are_attached_to_runners():
    runner = fx.racecard_runner(odds=[fx.odds_entry(), fx.odds_entry("Sky Bet", "3.6")])
    race = p.parse_racecard(RacecardPayload.model_validate(fx.racecard(runners=[runner])))
    assert len(race.runners[0].odds) == 2
    assert race.runners[0].odds[0].race_id == "rac_001"
