"""HTTP endpoint tests for races, horses and odds."""

from __future__ import annotations

from decimal import Decimal

import pytest

from backend.data_pipeline import RaceDataImporter
from backend.services.racing_api.parsers import parse_racecard, parse_result
from backend.services.racing_api.schemas import RacecardPayload, ResultRacePayload
from tests.fixtures import racing_api as fx

pytestmark = pytest.mark.unit

PREFIX = "/api/v1"


@pytest.fixture
def seeded(db_session):
    """A card, a result and a market — enough to exercise every endpoint."""
    runner_with_odds = fx.racecard_runner(
        odds=[
            fx.odds_entry("Bet365", "3.5"),
            fx.odds_entry("Sky Bet", "4.0"),
            fx.odds_entry("Paddy Power", "3.25"),
        ]
    )
    card = parse_racecard(RacecardPayload.model_validate(fx.racecard(runners=[runner_with_odds])))
    older = parse_racecard(
        RacecardPayload.model_validate(
            fx.racecard(race_id="rac_002", date="2026-08-09", course="Newbury", race_class="Class 5")
        )
    )
    # The result must agree with the card it belongs to: importing a result
    # legitimately updates the race row, so a mismatched date/class here would
    # silently overwrite the seeded values.
    result = parse_result(
        ResultRacePayload.model_validate(
            fx.result_race(race_id="rac_002", date="2026-08-09", course="Newbury", **{"class": "Class 5"})
        )
    )

    RaceDataImporter(db_session).import_races([card, older, result])
    return db_session


# ---------------------------------------------------------------------------
# Races
# ---------------------------------------------------------------------------
def test_list_races(api_client, seeded):
    response = api_client.get(f"{PREFIX}/races")
    assert response.status_code == 200

    body = response.json()
    assert body["total"] == 2
    assert body["limit"] == 50
    assert {race["race_id"] for race in body["items"]} == {"rac_001", "rac_002"}


def test_list_races_is_newest_first(api_client, seeded):
    items = api_client.get(f"{PREFIX}/races").json()["items"]
    assert [race["race_id"] for race in items] == ["rac_001", "rac_002"]


def test_list_races_filters_by_date(api_client, seeded):
    body = api_client.get(f"{PREFIX}/races", params={"race_date": "2026-08-09"}).json()
    assert body["total"] == 1
    assert body["items"][0]["race_id"] == "rac_002"


def test_list_races_filters_by_class_and_result_state(api_client, seeded):
    assert api_client.get(f"{PREFIX}/races", params={"race_class": 5}).json()["total"] == 1
    assert api_client.get(f"{PREFIX}/races", params={"has_result": True}).json()["total"] == 1
    assert api_client.get(f"{PREFIX}/races", params={"has_result": False}).json()["total"] == 1


def test_list_races_pagination(api_client, seeded):
    body = api_client.get(f"{PREFIX}/races", params={"limit": 1, "offset": 1}).json()
    assert len(body["items"]) == 1
    assert body["total"] == 2
    assert body["offset"] == 1


def test_list_races_rejects_an_absurd_limit(api_client, seeded):
    assert api_client.get(f"{PREFIX}/races", params={"limit": 5000}).status_code == 422


def test_race_detail_includes_the_card(api_client, seeded):
    body = api_client.get(f"{PREFIX}/races/rac_001").json()

    assert body["race_id"] == "rac_001"
    assert body["course_name"] == "Ascot"
    assert body["distance_yards"] == 4400
    assert len(body["runners"]) == 1
    assert body["runners"][0]["horse_name"] == "Thunder King"
    assert body["results"] == []


def test_race_detail_includes_results_once_run(api_client, seeded):
    body = api_client.get(f"{PREFIX}/races/rac_002").json()

    assert body["has_result"] is True
    assert len(body["results"]) == 2
    assert body["results"][0]["finishing_position"] == 1
    assert body["results"][0]["is_winner"] is True


def test_non_finishers_sort_after_finishers(api_client, db_session):
    payload = fx.result_race(
        runners=[
            fx.result_runner(horse_id="hrs_pu", position="PU"),
            fx.result_runner(horse_id="hrs_win", position="1"),
        ]
    )
    RaceDataImporter(db_session).import_races([parse_result(ResultRacePayload.model_validate(payload))])

    results = api_client.get(f"{PREFIX}/races/rac_001").json()["results"]
    assert results[0]["horse_id"] == "hrs_win"
    assert results[-1]["finishing_position"] is None


def test_unknown_race_returns_the_standard_error_envelope(api_client, seeded):
    response = api_client.get(f"{PREFIX}/races/rac_nope")
    assert response.status_code == 404

    body = response.json()
    assert body["error"] == "not_found"
    assert body["details"]["race_id"] == "rac_nope"
    assert "request_id" in body


# ---------------------------------------------------------------------------
# Horses
# ---------------------------------------------------------------------------
def test_search_horses_by_name(api_client, seeded):
    body = api_client.get(f"{PREFIX}/horses", params={"name": "thunder"}).json()
    assert body["total"] == 1
    assert body["items"][0]["name"] == "Thunder King"


def test_search_requires_a_usable_query(api_client, seeded):
    assert api_client.get(f"{PREFIX}/horses", params={"name": "a"}).status_code == 422


def test_horse_form(api_client, seeded):
    body = api_client.get(f"{PREFIX}/horses/hrs_001").json()

    assert body["horse"]["name"] == "Thunder King"
    assert body["total_runs"] == 1
    assert body["wins"] == 1
    assert body["places"] == 1
    assert len(body["runs"]) == 1


def test_career_totals_ignore_the_runs_window(api_client, seeded):
    """`runs` limits what is returned, never what the totals are computed over."""
    body = api_client.get(f"{PREFIX}/horses/hrs_001", params={"runs": 1}).json()
    assert body["total_runs"] == 1
    assert body["wins"] == 1


def test_unknown_horse_404s(api_client, seeded):
    assert api_client.get(f"{PREFIX}/horses/hrs_nope").status_code == 404


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------
def test_race_odds_picks_the_best_price(api_client, seeded):
    body = api_client.get(f"{PREFIX}/odds/rac_001").json()

    assert body["race_id"] == "rac_001"
    assert len(body["runners"]) == 1

    runner = body["runners"][0]
    assert Decimal(runner["best_decimal_odds"]) == Decimal("4.000")
    assert runner["best_bookmaker"] == "Sky Bet"
    assert runner["implied_probability"] == pytest.approx(0.25)
    assert len(runner["quotes"]) == 3


def test_quotes_are_sorted_best_price_first(api_client, seeded):
    quotes = api_client.get(f"{PREFIX}/odds/rac_001").json()["runners"][0]["quotes"]
    prices = [Decimal(q["decimal_odds"]) for q in quotes]
    assert prices == sorted(prices, reverse=True)


def test_overround_is_reported(api_client, seeded):
    body = api_client.get(f"{PREFIX}/odds/rac_001").json()
    # One runner at 4.0 -> 0.25. A real book sums above 1.0 across all runners.
    assert body["overround"] == pytest.approx(0.25)
    assert body["quote_count"] == 3


def test_odds_can_be_filtered_to_one_bookmaker(api_client, seeded):
    body = api_client.get(f"{PREFIX}/odds/rac_001", params={"bookmaker": "Bet365"}).json()
    assert body["quote_count"] == 1
    assert body["runners"][0]["best_bookmaker"] == "Bet365"


def test_race_with_no_market_returns_an_empty_book(api_client, seeded):
    body = api_client.get(f"{PREFIX}/odds/rac_002").json()
    assert body["runners"] == []
    assert body["overround"] is None


def test_odds_for_unknown_race_404s(api_client, seeded):
    assert api_client.get(f"{PREFIX}/odds/rac_nope").status_code == 404


def test_runner_price_history(api_client, seeded):
    quotes = api_client.get(f"{PREFIX}/odds/rac_001/hrs_001").json()
    assert len(quotes) == 3
    assert {q["bookmaker"] for q in quotes} == {"Bet365", "Sky Bet", "Paddy Power"}


def test_only_the_latest_quote_per_bookmaker_is_returned(api_client, db_session):
    """A moving price must not appear twice in the current market."""
    early = fx.racecard_runner(odds=[fx.odds_entry("Bet365", "5.0", updated="2026-08-10T10:00:00+01:00")])
    late = fx.racecard_runner(odds=[fx.odds_entry("Bet365", "3.0", updated="2026-08-10T13:00:00+01:00")])

    RaceDataImporter(db_session).import_races(
        [parse_racecard(RacecardPayload.model_validate(fx.racecard(runners=[early])))]
    )
    RaceDataImporter(db_session).import_races(
        [parse_racecard(RacecardPayload.model_validate(fx.racecard(runners=[late])))]
    )

    body = api_client.get(f"{PREFIX}/odds/rac_001").json()
    runner = body["runners"][0]
    assert body["quote_count"] == 1
    assert len(runner["quotes"]) == 1
    assert Decimal(runner["quotes"][0]["decimal_odds"]) == Decimal("3.000")


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
def test_new_endpoints_are_in_the_openapi_schema(api_client):
    paths = api_client.get("/openapi.json").json()["paths"]
    for path in (
        f"{PREFIX}/races",
        f"{PREFIX}/races/{{race_id}}",
        f"{PREFIX}/horses/{{horse_id}}",
        f"{PREFIX}/odds/{{race_id}}",
    ):
        assert path in paths
