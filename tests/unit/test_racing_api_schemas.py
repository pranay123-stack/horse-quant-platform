"""Wire-contract tests.

The wire models are deliberately permissive. Two properties are load-bearing and
easy to break by "tightening things up" later, so they are pinned here:

1. **Unknown upstream fields are ignored.** The Racing API adds fields without
   notice; a strict model would turn that into a production outage.
2. **Absent optional fields do not fail validation.** Only the identifiers are
   genuinely required.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.services.racing_api.schemas import (
    CoursesPagePayload,
    HorseSearchPagePayload,
    OddsEntryPayload,
    PersonSearchPagePayload,
    RacecardPayload,
    RacecardRunnerPayload,
    RacecardsPagePayload,
    RaceSchema,
    ResultRacePayload,
    ResultRunnerPayload,
    ResultsPagePayload,
    RunnerOddsPayload,
)
from tests.fixtures import racing_api as fx

pytestmark = pytest.mark.unit


def test_unknown_upstream_fields_are_ignored():
    payload = fx.racecard()
    payload["a_brand_new_field_v2"] = "surprise"
    payload["runners"][0]["another_new_field"] = {"nested": True}

    card = RacecardPayload.model_validate(payload)
    assert card.race_id == "rac_001"
    assert not hasattr(card, "a_brand_new_field_v2")


def test_only_the_identifier_is_required():
    card = RacecardPayload.model_validate({"race_id": "rac_x"})
    assert card.race_id == "rac_x"
    assert card.course is None
    assert card.runners == []


def test_missing_identifier_is_a_validation_error():
    with pytest.raises(ValidationError):
        RacecardPayload.model_validate({"course": "Ascot"})
    with pytest.raises(ValidationError):
        RacecardRunnerPayload.model_validate({"horse": "No Id"})


def test_racecard_reads_ofr_as_official_rating():
    runner = RacecardRunnerPayload.model_validate(fx.racecard_runner())
    assert runner.official_rating == "88"


def test_result_reads_or_as_official_rating():
    """``or`` is a Python keyword, so the alias is not optional."""
    runner = ResultRunnerPayload.model_validate(fx.result_runner())
    assert runner.official_rating == "88"


def test_result_race_maps_the_alternate_names():
    race = ResultRacePayload.model_validate(fx.result_race())
    assert race.race_class == "Class 3"  # from "class"
    assert race.off_time == "14:35"  # from "off"
    assert race.sex_restriction == ""  # from "sex_rest"


def test_whitespace_is_stripped_on_the_wire():
    runner = RacecardRunnerPayload.model_validate({"horse_id": "h1", "horse": "  Padded Name  "})
    assert runner.horse == "Padded Name"


def test_page_wrappers():
    assert len(RacecardsPagePayload.model_validate(fx.racecards_page()).racecards) == 1
    assert len(ResultsPagePayload.model_validate(fx.results_page()).results) == 1
    assert len(CoursesPagePayload.model_validate(fx.courses_page()).courses) == 2
    assert len(RunnerOddsPayload.model_validate(fx.runner_odds()).odds) == 3


def test_empty_page_is_valid():
    assert RacecardsPagePayload.model_validate({}).racecards == []


def test_search_pages_accept_either_key():
    """The API is inconsistent about the results key across search endpoints."""
    assert len(HorseSearchPagePayload.model_validate(fx.horse_search_page()).search_results) == 1
    assert len(HorseSearchPagePayload.model_validate({"horses": [{"id": "h1"}]}).search_results) == 1
    assert len(PersonSearchPagePayload.model_validate({"jockeys": [{"id": "j1"}]}).search_results) == 1
    assert len(PersonSearchPagePayload.model_validate({"trainers": [{"id": "t1"}]}).search_results) == 1


def test_odds_entry_tolerates_missing_history():
    entry = OddsEntryPayload.model_validate({"bookmaker": "Bet365", "decimal": "3.5"})
    assert entry.history is None


# ---------------------------------------------------------------------------
# Domain schemas are strict on purpose
# ---------------------------------------------------------------------------
def test_domain_schema_rejects_unknown_fields():
    """Wire models are lenient; domain models are not — typos must not pass."""
    with pytest.raises(ValidationError):
        RaceSchema(race_id="r1", nonexistent_field=1)


def test_runner_count_switches_on_result_flag():
    from backend.services.racing_api.parsers import parse_racecard, parse_result

    card = parse_racecard(RacecardPayload.model_validate(fx.racecard()))
    result = parse_result(ResultRacePayload.model_validate(fx.result_race()))

    assert card.runner_count == 1
    assert result.runner_count == 2
