"""Client tests against a mocked Racing API (respx).

Covers the parts that only misbehave in production: retry policy, the refusal to
retry unrecoverable auth failures, rate limiting, pagination and partial-failure
tolerance in concurrent fan-out.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from backend.services.racing_api.client import RacingAPIClient
from backend.services.racing_api.exceptions import (
    RacingAPIAuthenticationError,
    RacingAPINotFoundError,
    RacingAPIRateLimitError,
    RacingAPIResponseError,
    RacingAPIServerError,
    RacingAPITimeoutError,
    SubscriptionInactiveError,
)
from backend.utils.config import Settings
from tests.fixtures import racing_api as fx

pytestmark = pytest.mark.unit

BASE = "https://api.theracingapi.com"


@pytest.fixture
def fast_settings(settings) -> Settings:
    """Test settings with the rate limiter effectively disabled."""
    return settings.model_copy(
        update={
            "racing_api_rate_limit_per_second": 10_000.0,
            "racing_api_max_retries": 2,
            "racing_api_backoff_factor": 0.0,
        }
    )


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """Backoff delays must not make the suite slow."""

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant)


@pytest.fixture
async def client(fast_settings):
    async with RacingAPIClient(fast_settings) as api_client:
        yield api_client


# ---------------------------------------------------------------------------
# Authentication and headers
# ---------------------------------------------------------------------------
@respx.mock
async def test_requests_carry_basic_auth(client):
    route = respx.get(f"{BASE}/v1/courses").mock(return_value=httpx.Response(200, json=fx.courses_page()))
    await client.get_courses()

    assert route.called
    assert route.calls.last.request.headers["Authorization"].startswith("Basic ")


@respx.mock
async def test_user_agent_identifies_the_platform(client):
    route = respx.get(f"{BASE}/v1/courses").mock(return_value=httpx.Response(200, json=fx.courses_page()))
    await client.get_courses()
    assert "horse-quant-platform" in route.calls.last.request.headers["user-agent"]


def test_client_refuses_to_start_without_credentials():
    """Fail at construction, not on the first request in the middle of a backfill."""
    blank = Settings(racing_api_username="", racing_api_password="")
    with pytest.raises(RacingAPIAuthenticationError):
        RacingAPIClient(blank)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
@respx.mock
async def test_inactive_subscription_is_reported_precisely(client):
    respx.get(f"{BASE}/v1/courses").mock(
        return_value=httpx.Response(401, json={"detail": "Subscription inactive"})
    )
    with pytest.raises(SubscriptionInactiveError):
        await client.get_courses()


@respx.mock
async def test_auth_failures_are_never_retried(client):
    """Retrying bad credentials burns quota and can trip abuse protection."""
    route = respx.get(f"{BASE}/v1/courses").mock(
        return_value=httpx.Response(401, json={"detail": "Incorrect username"})
    )
    with pytest.raises(RacingAPIAuthenticationError):
        await client.get_courses()

    assert route.call_count == 1
    assert client.retry_count == 0


@respx.mock
async def test_404_is_not_retried(client):
    route = respx.get(f"{BASE}/v1/results/missing").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    with pytest.raises(RacingAPINotFoundError):
        await client.get_race_result("missing")
    assert route.call_count == 1


@respx.mock
async def test_server_errors_are_retried_then_surface(client):
    route = respx.get(f"{BASE}/v1/courses").mock(return_value=httpx.Response(503, json={"detail": "down"}))
    with pytest.raises(RacingAPIServerError):
        await client.get_courses()

    # max_retries=2 -> 1 initial attempt + 2 retries
    assert route.call_count == 3
    assert client.retry_count == 2


@respx.mock
async def test_transient_failure_then_success(client):
    route = respx.get(f"{BASE}/v1/courses").mock(
        side_effect=[
            httpx.Response(503, json={"detail": "down"}),
            httpx.Response(200, json=fx.courses_page()),
        ]
    )
    courses = await client.get_courses()

    assert route.call_count == 2
    assert len(courses) == 2
    assert courses[0].name == "Ascot"


@respx.mock
async def test_rate_limit_response_is_retried(client):
    respx.get(f"{BASE}/v1/courses").mock(
        side_effect=[
            httpx.Response(429, json={"detail": "slow down"}, headers={"Retry-After": "1"}),
            httpx.Response(200, json=fx.courses_page()),
        ]
    )
    assert len(await client.get_courses()) == 2


@respx.mock
async def test_exhausted_rate_limit_raises(client):
    respx.get(f"{BASE}/v1/courses").mock(
        return_value=httpx.Response(429, json={"detail": "slow down"}, headers={"Retry-After": "2"})
    )
    with pytest.raises(RacingAPIRateLimitError) as excinfo:
        await client.get_courses()
    assert excinfo.value.retry_after == 2.0


@respx.mock
async def test_timeouts_are_retried_and_reported(client):
    route = respx.get(f"{BASE}/v1/courses").mock(side_effect=httpx.ConnectTimeout("timed out"))
    with pytest.raises(RacingAPITimeoutError):
        await client.get_courses()
    assert route.call_count == 3


@respx.mock
async def test_transport_errors_are_retried(client):
    respx.get(f"{BASE}/v1/courses").mock(
        side_effect=[httpx.ConnectError("refused"), httpx.Response(200, json=fx.courses_page())]
    )
    assert len(await client.get_courses()) == 2


@respx.mock
async def test_non_json_response_is_rejected(client):
    respx.get(f"{BASE}/v1/courses").mock(
        return_value=httpx.Response(200, text="<html>oops</html>", headers={"content-type": "text/html"})
    )
    with pytest.raises(RacingAPIResponseError):
        await client.get_courses()


@respx.mock
async def test_schema_violation_is_reported_not_swallowed(client):
    """A racecard with no ``race_id`` is unusable; failing loudly beats importing junk."""
    respx.get(f"{BASE}/v1/racecards/standard").mock(
        return_value=httpx.Response(200, json={"racecards": [{"course": "Ascot"}]})
    )
    with pytest.raises(RacingAPIResponseError) as excinfo:
        await client.get_races(date="2026-08-10")
    assert excinfo.value.details["errors"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@respx.mock
async def test_get_races_defaults_to_today(client):
    route = respx.get(f"{BASE}/v1/racecards/standard").mock(
        return_value=httpx.Response(200, json=fx.racecards_page())
    )
    races = await client.get_races()

    assert len(races) == 1
    assert races[0].race_id == "rac_001"
    assert "date" in dict(route.calls.last.request.url.params)


@respx.mock
async def test_get_races_sends_region_filters(client):
    route = respx.get(f"{BASE}/v1/racecards/standard").mock(
        return_value=httpx.Response(200, json=fx.racecards_page())
    )
    await client.get_races(date="2026-08-10", region_codes=["gb", "ire"])

    params = route.calls.last.request.url.params
    assert params.get_list("region_codes") == ["gb", "ire"]


@respx.mock
async def test_none_params_are_not_sent(client):
    route = respx.get(f"{BASE}/v1/racecards/standard").mock(
        return_value=httpx.Response(200, json=fx.racecards_page())
    )
    await client.get_races(date="2026-08-10", region_codes=None, course_ids=None)

    query = str(route.calls.last.request.url)
    assert "region_codes" not in query
    assert "None" not in query


def test_invalid_tier_is_rejected_before_any_request():
    with pytest.raises(ValueError, match="unknown tier"):
        RacingAPIClient._validate_tier("platinum")


@respx.mock
async def test_race_details_requires_a_detailed_tier(client):
    with pytest.raises(ValueError, match="standard"):
        await client.get_race_details("rac_001", tier="free")


@respx.mock
async def test_get_race_details(client):
    respx.get(f"{BASE}/v1/racecards/rac_001/pro").mock(return_value=httpx.Response(200, json=fx.racecard()))
    race = await client.get_race_details("rac_001")
    assert race.race_id == "rac_001"
    assert race.race_class == 3


@respx.mock
async def test_get_results(client):
    respx.get(f"{BASE}/v1/results").mock(return_value=httpx.Response(200, json=fx.results_page()))
    results = await client.get_results(start_date="2026-08-01", end_date="2026-08-10")

    assert len(results) == 1
    assert results[0].is_result is True
    assert len(results[0].results) == 2


@respx.mock
async def test_get_odds_backfills_missing_identifiers(client):
    payload = fx.runner_odds()
    payload.pop("race_id")
    payload.pop("horse_id")
    respx.get(f"{BASE}/v1/odds/rac_001/hrs_001").mock(return_value=httpx.Response(200, json=payload))

    quotes = await client.get_odds("rac_001", "hrs_001")
    assert len(quotes) == 3
    assert all(q.race_id == "rac_001" and q.horse_id == "hrs_001" for q in quotes)


@respx.mock
async def test_get_race_odds_survives_one_runner_failing(client):
    """One dead runner must not lose the rest of the race's market."""
    respx.get(f"{BASE}/v1/odds/rac_001/hrs_001").mock(
        return_value=httpx.Response(200, json=fx.runner_odds(horse_id="hrs_001"))
    )
    respx.get(f"{BASE}/v1/odds/rac_001/hrs_002").mock(
        return_value=httpx.Response(404, json={"detail": "no odds"})
    )

    quotes = await client.get_race_odds("rac_001", ["hrs_001", "hrs_002"])
    assert len(quotes) == 3
    assert {q.horse_id for q in quotes} == {"hrs_001"}


@respx.mock
async def test_search_endpoints(client):
    respx.get(f"{BASE}/v1/horses/search").mock(return_value=httpx.Response(200, json=fx.horse_search_page()))
    respx.get(f"{BASE}/v1/jockeys/search").mock(
        return_value=httpx.Response(200, json=fx.person_search_page("jky_001", "N de Boinville"))
    )
    respx.get(f"{BASE}/v1/trainers/search").mock(
        return_value=httpx.Response(200, json=fx.person_search_page("trn_001", "N Henderson"))
    )

    horses = await client.get_horses("Thunder")
    jockeys = await client.get_jockeys("Boinville")
    trainers = await client.get_trainers("Henderson")

    assert horses[0].horse_id == "hrs_001"
    assert jockeys[0].jockey_id == "jky_001"
    assert trainers[0].trainer_id == "trn_001"


@respx.mock
async def test_get_horse_history(client):
    respx.get(f"{BASE}/v1/horses/hrs_001/results").mock(
        return_value=httpx.Response(200, json=fx.results_page())
    )
    history = await client.get_horse_history("hrs_001")
    assert len(history) == 1
    assert history[0].is_result is True


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
@respx.mock
async def test_iter_races_pages_until_a_short_page(client):
    full_page = fx.racecards_page(cards=[fx.racecard(race_id=f"rac_{i:03d}") for i in range(50)], total=60)
    short_page = fx.racecards_page(cards=[fx.racecard(race_id="rac_050")], total=60)
    route = respx.get(f"{BASE}/v1/racecards/standard").mock(
        side_effect=[httpx.Response(200, json=full_page), httpx.Response(200, json=short_page)]
    )

    races = [race async for race in client.iter_races(date="2026-08-10")]

    assert len(races) == 51
    assert route.call_count == 2
    assert dict(route.calls[1].request.url.params)["skip"] == "50"


@respx.mock
async def test_iter_races_stops_on_an_empty_page(client):
    respx.get(f"{BASE}/v1/racecards/standard").mock(
        return_value=httpx.Response(200, json=fx.racecards_page(cards=[], total=0))
    )
    assert [race async for race in client.iter_races(date="2026-08-10")] == []


@respx.mock
async def test_iter_results_pages(client):
    full = fx.results_page(races=[fx.result_race(race_id=f"rac_{i:03d}") for i in range(50)])
    short = fx.results_page(races=[fx.result_race(race_id="rac_050")])
    respx.get(f"{BASE}/v1/results").mock(
        side_effect=[httpx.Response(200, json=full), httpx.Response(200, json=short)]
    )

    results = [r async for r in client.iter_results(start_date="2026-08-01", end_date="2026-08-10")]
    assert len(results) == 51


# ---------------------------------------------------------------------------
# Connectivity probe
# ---------------------------------------------------------------------------
@respx.mock
async def test_check_connectivity_success(client):
    respx.get(f"{BASE}/v1/courses").mock(return_value=httpx.Response(200, json=fx.courses_page()))
    verdict = await client.check_connectivity()
    assert verdict == {
        "reachable": True,
        "authorised": True,
        "error_code": None,
        "message": "ok",
        "details": {},
    }


@respx.mock
async def test_check_connectivity_reports_inactive_subscription(client):
    respx.get(f"{BASE}/v1/courses").mock(
        return_value=httpx.Response(401, json={"detail": "Subscription inactive"})
    )
    verdict = await client.check_connectivity()

    assert verdict["reachable"] is True
    assert verdict["authorised"] is False
    assert verdict["error_code"] == "racing_api_subscription_inactive"
    assert "remedy" in verdict["details"]


@respx.mock
async def test_check_connectivity_never_raises(client):
    respx.get(f"{BASE}/v1/courses").mock(side_effect=httpx.ConnectError("dns failure"))
    verdict = await client.check_connectivity()
    assert verdict["authorised"] is False


# ---------------------------------------------------------------------------
# Counters and lifecycle
# ---------------------------------------------------------------------------
@respx.mock
async def test_request_counter_tracks_every_attempt(client):
    respx.get(f"{BASE}/v1/courses").mock(
        side_effect=[httpx.Response(503, json={}), httpx.Response(200, json=fx.courses_page())]
    )
    await client.get_courses()
    assert client.request_count == 2
    assert client.retry_count == 1


async def test_close_is_idempotent(fast_settings):
    api_client = RacingAPIClient(fast_settings)
    await api_client.aclose()
    await api_client.aclose()
