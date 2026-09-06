"""Realistic Racing API payloads for tests.

Field names and the all-strings convention are taken from The Racing API
OpenAPI spec v1.4.3, including the awkward parts we must survive in production:

* numbers arrive as strings (``"9-7"``, ``"3.5"``, ``"133"``)
* missing values arrive as ``""`` or ``"-"``, never ``null``
* racecards and results name the same concept differently
  (``race_class``/``class``, ``distance``/``dist``, ``off_time``/``off``)
"""

from __future__ import annotations

from typing import Any


def racecard_runner(
    horse_id: str = "hrs_001",
    horse: str = "Thunder King",
    **overrides: Any,
) -> dict[str, Any]:
    runner = {
        "horse_id": horse_id,
        "horse": horse,
        "dob": "2020-03-14",
        "age": "5",
        "sex": "gelding",
        "sex_code": "G",
        "colour": "bay",
        "region": "GB",
        "breeder": "Someone Stud",
        "dam": "Storm Lady",
        "dam_id": "dam_001",
        "dam_region": "GB",
        "sire": "Thunder Bolt",
        "sire_id": "sir_001",
        "sire_region": "IRE",
        "damsire": "Old Storm",
        "damsire_id": "dsi_001",
        "trainer": "N Henderson",
        "trainer_id": "trn_001",
        "trainer_location": "Lambourn",
        "trainer_rtf": "42",
        "owner": "The Syndicate",
        "owner_id": "own_001",
        "number": "1",
        "draw": "3",
        "headgear": "b",
        "lbs": "133",
        "ofr": "88",
        "rpr": "95",
        "ts": "82",
        "jockey": "N de Boinville",
        "jockey_id": "jky_001",
        "silk_url": "https://example.test/silk.png",
        "last_run": "28",
        "form": "1-P23",
        "comment": "Travelled well last time.",
    }
    runner.update(overrides)
    return runner


def odds_entry(bookmaker: str = "Bet365", decimal: str = "3.5", **overrides: Any) -> dict[str, Any]:
    entry = {
        "bookmaker": bookmaker,
        "fractional": "5/2",
        "decimal": decimal,
        "ew_places": "3",
        "ew_denom": "5",
        "updated": "2026-08-10T12:30:00+01:00",
    }
    entry.update(overrides)
    return entry


def racecard(race_id: str = "rac_001", runners: list[dict] | None = None, **overrides: Any) -> dict[str, Any]:
    card = {
        "race_id": race_id,
        "course": "Ascot",
        "course_id": "crs_001",
        "date": "2026-08-10",
        "off_time": "14:35",
        "off_dt": "2026-08-10T14:35:00+01:00",
        "race_name": "The Example Handicap Chase",
        "distance_round": "2m4f",
        "distance": "2m3f210y",
        "distance_f": "20.0",
        "region": "GB",
        "pattern": "",
        "sex_restriction": "",
        "race_class": "Class 3",
        "type": "Chase",
        "age_band": "4yo+",
        "rating_band": "0-140",
        "prize": "£12,450",
        "field_size": "8",
        "going_detailed": "Good (Good to Soft in places)",
        "rail_movements": "",
        "stalls": "Inside",
        "weather": "Sunny",
        "going": "Good",
        "surface": "Turf",
        "jumps": "Chase",
        "runners": runners if runners is not None else [racecard_runner()],
        "big_race": False,
        "is_abandoned": False,
    }
    card.update(overrides)
    return card


def racecards_page(cards: list[dict] | None = None, **overrides: Any) -> dict[str, Any]:
    page = {
        "racecards": cards if cards is not None else [racecard()],
        "total": 1,
        "limit": 50,
        "skip": 0,
        "query": [],
    }
    page.update(overrides)
    return page


def result_runner(
    horse_id: str = "hrs_001",
    horse: str = "Thunder King",
    position: str = "1",
    **overrides: Any,
) -> dict[str, Any]:
    runner = {
        "horse_id": horse_id,
        "horse": horse,
        "sp": "5/2",
        "sp_dec": "3.5",
        "number": "1",
        "position": position,
        "draw": "3",
        "btn": "0",
        "ovr_btn": "0",
        "age": "5",
        "sex": "G",
        "weight": "9-7",
        "weight_lbs": "133",
        "headgear": "b",
        "time": "5m 12.30s",
        "or": "88",
        "rpr": "102",
        "tsr": "95",
        "prize": "£8,000",
        "jockey": "N de Boinville",
        "jockey_id": "jky_001",
        "trainer": "N Henderson",
        "trainer_id": "trn_001",
        "owner": "The Syndicate",
        "owner_id": "own_001",
        "sire": "Thunder Bolt",
        "sire_id": "sir_001",
        "dam": "Storm Lady",
        "dam_id": "dam_001",
        "damsire": "Old Storm",
        "damsire_id": "dsi_001",
        "comment": "Made all, kept on well.",
    }
    runner.update(overrides)
    return runner


def result_race(
    race_id: str = "rac_001", runners: list[dict] | None = None, **overrides: Any
) -> dict[str, Any]:
    """Note the *different* field names from :func:`racecard` -- this is real."""
    race = {
        "race_id": race_id,
        "date": "2026-08-10",
        "region": "GB",
        "course": "Ascot",
        "course_id": "crs_001",
        "off": "14:35",
        "off_dt": "2026-08-10T14:35:00+01:00",
        "race_name": "The Example Handicap Chase",
        "type": "Chase",
        "class": "Class 3",
        "pattern": "",
        "rating_band": "0-140",
        "age_band": "4yo+",
        "sex_rest": "",
        "dist": "2m3f210y",
        "dist_y": "4390",
        "dist_m": "4014",
        "dist_f": "19.95",
        "going": "Good",
        "surface": "Turf",
        "jumps": "Chase",
        "winning_time_detail": "5m 12.30s (slow by 2.10s)",
        "comments": "",
        "non_runners": "",
        "tote_win": "£3.50",
        "runners": runners
        if runners is not None
        else [
            result_runner(),
            result_runner(horse_id="hrs_002", horse="Silver Arrow", position="2", sp_dec="4.5", btn="2.5"),
        ],
    }
    race.update(overrides)
    return race


def results_page(races: list[dict] | None = None, **overrides: Any) -> dict[str, Any]:
    page = {
        "results": races if races is not None else [result_race()],
        "total": 1,
        "limit": 50,
        "skip": 0,
        "query": [],
    }
    page.update(overrides)
    return page


def runner_odds(race_id: str = "rac_001", horse_id: str = "hrs_001") -> dict[str, Any]:
    return {
        "race_id": race_id,
        "horse_id": horse_id,
        "horse": "Thunder King",
        "odds": [
            odds_entry("Bet365", "3.5"),
            odds_entry("William Hill", "3.75", fractional="11/4"),
            odds_entry("Paddy Power", "3.25", fractional="9/4"),
        ],
    }


def courses_page() -> dict[str, Any]:
    return {
        "courses": [
            {"id": "crs_001", "course": "Ascot", "region_code": "gb", "region": "Great Britain"},
            {"id": "crs_002", "course": "Leopardstown", "region_code": "ire", "region": "Ireland"},
        ]
    }


def horse_search_page() -> dict[str, Any]:
    return {
        "search_results": [
            {
                "id": "hrs_001",
                "name": "Thunder King",
                "sex": "gelding",
                "sex_code": "G",
                "dob": "2020-03-14",
                "colour": "bay",
                "sire": "Thunder Bolt",
                "sire_id": "sir_001",
                "dam": "Storm Lady",
                "dam_id": "dam_001",
                "damsire": "Old Storm",
                "damsire_id": "dsi_001",
            }
        ],
        "total": 1,
        "limit": 50,
        "skip": 0,
    }


def person_search_page(person_id: str = "jky_001", name: str = "N de Boinville") -> dict[str, Any]:
    return {"search_results": [{"id": person_id, "name": name}], "total": 1, "limit": 50, "skip": 0}


#: A runner where every optional field is a null sentinel -- the shape that
#: breaks naive ``int(...)`` parsing in production.
MESSY_RUNNER: dict[str, Any] = racecard_runner(
    horse_id="hrs_messy",
    horse="Awkward Customer",
    age="",
    draw="-",
    lbs="",
    ofr="-",
    rpr="",
    ts="-",
    last_run="",
    form="",
    dob="",
    headgear="",
    number="",
)


__all__ = [
    "MESSY_RUNNER",
    "courses_page",
    "horse_search_page",
    "odds_entry",
    "person_search_page",
    "racecard",
    "racecard_runner",
    "racecards_page",
    "result_race",
    "result_runner",
    "results_page",
    "runner_odds",
]
