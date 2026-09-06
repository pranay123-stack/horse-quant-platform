"""What can this subscription actually do?

``check_connectivity`` answers one question — can we call the API at all — and
collapses two very different failures into a single ``authorised=False``. For
Phase 6 that is not enough. Before committing to an eight-year backfill we need
to know three separate things:

1. **Are the credentials right?** A rejected password and a lapsed plan are
   different problems with different owners. The API distinguishes them
   (``Incorrect password`` versus ``Subscription inactive``) and so must we.
2. **Which endpoints does the plan include?** Racecards, results and odds are
   sold separately. A plan that returns racecards but not results cannot label
   a training set.
3. **How far back does history go?** The pre-registered protocol trains from
   2018-01-01. If the plan only serves the last twelve months, the study cannot
   run as specified — and it is far better to learn that from one probe than
   from a backfill that quietly returns nothing for eighty months.

Probing is deliberately cheap and deliberately short-circuited: if the
subscription is inactive, every endpoint would fail identically, so we stop
after establishing that rather than making five more calls against an API that
has already said no.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from backend.services.racing_api.client import MAX_PAGE_SIZE, RacingAPIClient
from backend.services.racing_api.exceptions import (
    RacingAPIAuthenticationError,
    RacingAPIError,
    RacingAPINotFoundError,
    SubscriptionInactiveError,
)
from backend.utils.logging import get_logger, safe_extra
from backend.utils.timeutils import format_date, today_uk

logger = get_logger(__name__, channel="api")

#: Racecard tiers from cheapest to richest. ``pro`` carries bookmaker odds
#: inline, which is what makes an odds-per-runner backfill affordable.
TIERS: tuple[str, ...] = ("free", "basic", "standard", "pro")

#: The protocol's training start. History is probed here because that is the
#: date the study actually needs, not an arbitrary "old" date.
PROTOCOL_HISTORY_START = date(2018, 1, 1)

#: How far back to look when bracketing the earliest available result. Nothing
#: useful predates this for UK racing on this API, and a floor keeps the binary
#: search to a handful of calls.
EARLIEST_PLAUSIBLE_YEAR = 2000

#: Years of history the pre-registered protocol needs. Mirrors
#: ``research.audit.MIN_YEARS``, restated here rather than imported: the
#: services layer must not depend on the research layer.
MIN_YEARS = 5

#: Mid-June always has UK racing, so an empty window there means the plan
#: does not serve that year — not that the sport took a week off.
PROBE_MONTH = 6
PROBE_DAY = 10

#: Page size for range probes. Small: we want existence and a date, not data.
MAX_RANGE_PROBE = min(10, MAX_PAGE_SIZE)


@dataclass(slots=True)
class EndpointProbe:
    """One endpoint, and whether this plan may call it."""

    name: str
    available: bool = False
    detail: str = ""
    error_code: str | None = None
    skipped: bool = False

    @property
    def verdict(self) -> str:
        if self.skipped:
            return "SKIPPED"
        return "YES" if self.available else "NO"

    def as_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.name,
            "available": self.available,
            "verdict": self.verdict,
            "detail": self.detail,
            "error_code": self.error_code,
        }


@dataclass(slots=True)
class ApiCapabilities:
    """The full operational picture, safe to print."""

    masked_username: str = ""
    credentials_configured: bool = False
    reachable: bool = False
    authenticated: bool = False
    subscription_active: bool = False
    message: str = ""
    probes: list[EndpointProbe] = field(default_factory=list)
    #: Richest racecard tier this plan can read, if any.
    best_tier: str | None = None
    #: Whether results exist as far back as the protocol needs.
    history_reaches_protocol_start: bool = False
    history_probe_date: date = PROTOCOL_HISTORY_START
    #: The window the plan will actually serve, discovered by probing.
    earliest_result: date | None = None
    latest_result: date | None = None

    # ------------------------------------------------------------------
    def probe(self, name: str) -> EndpointProbe:
        for probe in self.probes:
            if probe.name == name:
                return probe
        return EndpointProbe(name=name, detail="not probed", skipped=True)

    @property
    def available_years(self) -> float:
        """Years of history the plan will serve."""
        if self.earliest_result is None or self.latest_result is None:
            return 0.0
        return (self.latest_result - self.earliest_result).days / 365.25

    @property
    def enough_for_protocol(self) -> bool:
        """Does the served window actually cover the pre-registered study?

        Not "is there a lot of history" — is there history *where the protocol
        needs it*. A plan serving 2021 onward has five years and still cannot
        run a study that trains from 2018.
        """
        if self.earliest_result is None or self.latest_result is None:
            return False
        return self.earliest_result <= self.history_probe_date and self.available_years >= MIN_YEARS

    @property
    def ready_for_backfill(self) -> bool:
        """Everything Phase 6 needs, in one boolean."""
        return (
            self.subscription_active
            and self.probe("results").available
            and self.probe("historical").available
            and self.enough_for_protocol
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "username": self.masked_username,
            "authentication": "PASS" if self.authenticated else "FAIL",
            "subscription": "ACTIVE" if self.subscription_active else "INACTIVE",
            "reachable": self.reachable,
            "best_racecard_tier": self.best_tier,
            "history_reaches_protocol_start": self.history_reaches_protocol_start,
            "earliest_result": str(self.earliest_result) if self.earliest_result else None,
            "latest_result": str(self.latest_result) if self.latest_result else None,
            "available_years": round(self.available_years, 2),
            "enough_for_protocol": self.enough_for_protocol,
            "ready_for_backfill": self.ready_for_backfill,
            "endpoints": [probe.as_dict() for probe in self.probes],
            "message": self.message,
        }

    def render(self) -> str:
        """A fixed-shape status block. Contains no credential material."""
        lines = [
            "Racing API status",
            "-----------------",
            f"  Username           {self.masked_username or '(not configured)'}",
            f"  Authentication     {'PASS' if self.authenticated else 'FAIL'}",
            f"  Subscription       {'ACTIVE' if self.subscription_active else 'INACTIVE'}",
            "",
            "  Available endpoints",
        ]
        for probe in self.probes:
            lines.append(f"    {probe.name.title():<12} {probe.verdict:<8} {probe.detail}")

        lines += [
            "",
            "  Data range",
            f"    Earliest result   {format_date(self.earliest_result) if self.earliest_result else '-'}",
            f"    Latest result     {format_date(self.latest_result) if self.latest_result else '-'}",
            f"    Span              {self.available_years:.1f} years",
            "",
            "  Coverage",
            f"    Best racecard tier   {self.best_tier or '-'}",
            f"    Protocol needs from  {format_date(self.history_probe_date)}",
            f"    Enough for protocol  {'YES' if self.enough_for_protocol else 'NO'}",
            "",
            f"  Ready for backfill {'YES' if self.ready_for_backfill else 'NO'}",
        ]
        if not self.ready_for_backfill:
            lines.append(f"  Blocked by         {self.blocking_reason}")
        return "\n".join(lines)

    @property
    def blocking_reason(self) -> str:
        """Why the backfill cannot start, in one line."""
        if not self.subscription_active:
            return self.message or "subscription inactive"
        if not self.probe("results").available:
            return "the plan does not serve results"
        # The discovered range explains the failure better than the probe does:
        # "history starts 2021" beats "no results at 2018-01-01".
        if self.earliest_result is not None and self.earliest_result > self.history_probe_date:
            return (
                f"history starts {format_date(self.earliest_result)}, but the protocol "
                f"trains from {format_date(self.history_probe_date)}"
            )
        if not self.probe("historical").available:
            return f"no results at {format_date(self.history_probe_date)}"
        if self.earliest_result is None:
            return "could not establish how far back history goes"
        if self.available_years < MIN_YEARS:
            return f"only {self.available_years:.1f} years available; {MIN_YEARS} required"
        return ""


async def probe_capabilities(
    client: RacingAPIClient,
    *,
    history_start: date = PROTOCOL_HISTORY_START,
) -> ApiCapabilities:
    """Establish auth, subscription and per-endpoint access, in that order.

    Never raises. Every failure becomes a field on the returned object, because
    the caller is a status command whose whole job is to explain a failure.
    """
    credentials = client.credentials
    capabilities = ApiCapabilities(
        masked_username=credentials.masked_username,
        credentials_configured=credentials.is_complete,
        history_probe_date=history_start,
    )

    # --- auth and subscription, from one call ------------------------------
    try:
        await client.request_json("/v1/courses", params={"limit": 1})
        capabilities.reachable = True
        capabilities.authenticated = True
        capabilities.subscription_active = True
        capabilities.message = "ok"
    except SubscriptionInactiveError as exc:
        # The credentials were accepted and then the plan was checked. That is
        # positive evidence the username and password are correct.
        capabilities.reachable = True
        capabilities.authenticated = True
        capabilities.message = exc.message
    except RacingAPIAuthenticationError as exc:
        capabilities.reachable = True
        capabilities.message = exc.message
    except RacingAPIError as exc:
        capabilities.reachable = True
        capabilities.message = exc.message
    except Exception as exc:
        capabilities.message = f"unreachable: {exc}"

    if not capabilities.subscription_active:
        # Do not probe five more endpoints against an API that has already
        # refused us. They would all fail for the same reason.
        reason = capabilities.message or "subscription inactive"
        capabilities.probes = [
            EndpointProbe(name=name, detail=reason, skipped=True)
            for name in ("racecards", "results", "odds", "historical")
        ]
        logger.warning("api probe short-circuited", extra=safe_extra({"reason": reason}))
        return capabilities

    # --- per-endpoint access ----------------------------------------------
    racecards, best_tier = await _probe_racecards(client)
    capabilities.best_tier = best_tier
    results = await _probe_results(client)
    odds = await _probe_odds(client, best_tier=best_tier)
    historical = await _probe_historical(client, history_start)

    capabilities.probes = [racecards, results, odds, historical]
    capabilities.history_reaches_protocol_start = historical.available

    earliest, latest = await _discover_range(client, history_start)
    capabilities.earliest_result = earliest
    capabilities.latest_result = latest

    logger.info("api capabilities probed", extra=safe_extra(capabilities.as_dict()))
    return capabilities


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------
async def _probe_racecards(client: RacingAPIClient) -> tuple[EndpointProbe, str | None]:
    """Find the richest tier the plan allows, stopping at the first that works."""
    best: str | None = None
    last_error = ""
    for tier in reversed(TIERS):
        try:
            await client.get_races(tier=tier, limit=1)
            best = tier
            break
        except RacingAPIError as exc:
            last_error = exc.message
        except Exception as exc:
            last_error = str(exc)

    if best is None:
        return EndpointProbe("racecards", False, last_error or "no tier accessible"), None
    detail = f"tiers up to '{best}'"
    if best == "pro":
        detail += " (odds included inline)"
    return EndpointProbe("racecards", True, detail), best


async def _probe_results(client: RacingAPIClient) -> EndpointProbe:
    """Results for a recent window. These are the training labels."""
    end = today_uk() - timedelta(days=1)
    start = end - timedelta(days=6)
    try:
        races = await client.get_results(start_date=format_date(start), end_date=format_date(end), limit=1)
    except RacingAPIError as exc:
        return EndpointProbe("results", False, exc.message, exc.error_code)
    except Exception as exc:
        return EndpointProbe("results", False, str(exc))
    return EndpointProbe("results", True, f"last 7 days returned {len(races)} race(s)")


async def _probe_odds(client: RacingAPIClient, *, best_tier: str | None) -> EndpointProbe:
    """Odds, either inline on pro racecards or from the dedicated endpoint.

    Inline is what matters for a backfill: one call per race rather than one per
    runner is the difference between hours and days.
    """
    if best_tier == "pro":
        return EndpointProbe("odds", True, "inline on pro racecards")

    try:
        races = await client.get_races(tier=best_tier or "standard", limit=1)
    except Exception as exc:
        return EndpointProbe("odds", False, f"could not find a race to probe: {exc}")

    if not races or not races[0].runners:
        return EndpointProbe("odds", False, "no runners available to probe today")

    race = races[0]
    try:
        quotes = await client.get_odds(race.race_id, race.runners[0].horse.horse_id)
    except RacingAPINotFoundError:
        # A 404 means the endpoint is reachable and this runner simply has no
        # price yet, which is access granted.
        return EndpointProbe("odds", True, "endpoint reachable (no price for the probed runner)")
    except RacingAPIError as exc:
        return EndpointProbe("odds", False, exc.message, exc.error_code)
    except Exception as exc:
        return EndpointProbe("odds", False, str(exc))
    return EndpointProbe("odds", True, f"dedicated endpoint returned {len(quotes)} quote(s)")


async def _probe_historical(client: RacingAPIClient, history_start: date) -> EndpointProbe:
    """Can we actually reach the protocol's training start?

    This is the probe that decides whether Phase 6 can run as pre-registered.
    """
    window_end = history_start + timedelta(days=6)
    try:
        races = await client.get_results(
            start_date=format_date(history_start), end_date=format_date(window_end), limit=1
        )
    except RacingAPIError as exc:
        return EndpointProbe("historical", False, exc.message, exc.error_code)
    except Exception as exc:
        return EndpointProbe("historical", False, str(exc))

    if not races:
        return EndpointProbe(
            "historical",
            False,
            f"no races returned for {format_date(history_start)} — the plan may not "
            "include history that far back",
        )
    return EndpointProbe("historical", True, f"results available from {format_date(history_start)}")


async def _has_results(client: RacingAPIClient, start: date, days: int = 13) -> list[Any]:
    """Does the plan serve any result in this window? Empty list if not."""
    try:
        return await client.get_results(
            start_date=format_date(start),
            end_date=format_date(start + timedelta(days=days)),
            limit=MAX_RANGE_PROBE,
        )
    except Exception:
        return []


def _min_race_date(races: Sequence[Any]) -> date | None:
    dates = [race.race_date for race in races if getattr(race, "race_date", None)]
    return min(dates) if dates else None


def _max_race_date(races: Sequence[Any]) -> date | None:
    dates = [race.race_date for race in races if getattr(race, "race_date", None)]
    return max(dates) if dates else None


async def _discover_range(client: RacingAPIClient, history_start: date) -> tuple[date | None, date | None]:
    """Find the window of results this plan will actually serve.

    Binary search on the year, which costs about five calls across a
    twenty-five year range instead of twenty-five. Precision is deliberately
    limited to the month: the question this answers is "does the served history
    cover the protocol?", and no decision changes on a fortnight.
    """
    today = today_uk()

    # --- latest: walk back from today until something answers --------------
    latest: date | None = None
    for weeks_back in (0, 1, 2, 4):
        window_start = today - timedelta(days=13 + weeks_back * 7)
        races = await _has_results(client, window_start)
        latest = _max_race_date(races)
        if latest:
            break

    # --- earliest: bisect the year -----------------------------------------
    low, high = EARLIEST_PLAUSIBLE_YEAR, today.year
    if not await _has_results(client, date(low, PROBE_MONTH, PROBE_DAY)):
        # Nothing at the floor, so the boundary is somewhere above it.
        while low < high:
            middle = (low + high) // 2
            if await _has_results(client, date(middle, PROBE_MONTH, PROBE_DAY)):
                high = middle
            else:
                low = middle + 1
        earliest_year = low
    else:
        earliest_year = low

    # Refine to the earliest observed date inside that year. January first,
    # because a plan that starts mid-year should not be reported as covering it.
    earliest: date | None = None
    for probe_start in (date(earliest_year, 1, 1), date(earliest_year, PROBE_MONTH, PROBE_DAY)):
        observed = _min_race_date(await _has_results(client, probe_start, days=30))
        if observed and (earliest is None or observed < earliest):
            earliest = observed
        if earliest:
            break

    logger.info(
        "history range discovered",
        extra=safe_extra(
            {
                "earliest": str(earliest),
                "latest": str(latest),
                "protocol_needs": str(history_start),
            }
        ),
    )
    return earliest, latest


__all__ = [
    "EARLIEST_PLAUSIBLE_YEAR",
    "MIN_YEARS",
    "PROTOCOL_HISTORY_START",
    "TIERS",
    "ApiCapabilities",
    "EndpointProbe",
    "probe_capabilities",
]
