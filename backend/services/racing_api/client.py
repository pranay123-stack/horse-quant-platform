"""Asynchronous client for The Racing API.

Design notes
------------
**Async.** Ingestion is IO-bound fan-out: a single race day is ~40 racecards,
each of which may need an odds call. Async lets those overlap on one connection
pool instead of serialising.

**Retry policy is deliberately narrow.** Transport errors, timeouts, 429 and 5xx
are retried with exponential backoff and full jitter. Everything else — and in
particular 401 — is *not*: repeatedly presenting bad credentials cannot succeed,
wastes quota, and risks tripping upstream abuse protection.

**Nothing sensitive is ever logged.** Request logs carry the endpoint, status and
duration; the ``Authorization`` header and response bodies are never emitted.

Usage::

    async with RacingAPIClient() as client:
        races = await client.get_races(date="2026-08-10", region_codes=["gb", "ire"])
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Mapping, Sequence
from types import TracebackType
from typing import Any, Final, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from backend.services.racing_api.authentication import RacingAPICredentials, credentials_from_settings
from backend.services.racing_api.exceptions import (
    RETRYABLE_STATUS_CODES,
    RacingAPIError,
    RacingAPIResponseError,
    RacingAPITimeoutError,
    classify_response_error,
)
from backend.services.racing_api.parsers import (
    parse_course,
    parse_horse,
    parse_jockey,
    parse_racecard,
    parse_result,
    parse_runner_odds,
    parse_trainer,
)
from backend.services.racing_api.rate_limit import AsyncTokenBucket
from backend.services.racing_api.schemas import (
    CourseSchema,
    CoursesPagePayload,
    HorseSchema,
    HorseSearchPagePayload,
    JockeySchema,
    OddsSchema,
    PersonSearchPagePayload,
    RacecardPayload,
    RacecardsPagePayload,
    RaceSchema,
    ResultRacePayload,
    ResultsPagePayload,
    RunnerOddsPayload,
    TrainerSchema,
)
from backend.utils.config import Settings, get_settings
from backend.utils.logging import get_logger
from backend.utils.timeutils import format_date, today_uk

logger = get_logger(__name__, channel="api")

TPayload = TypeVar("TPayload", bound=BaseModel)

#: Racecard/result detail tiers, cheapest first. ``pro`` includes odds.
Tier = str
DEFAULT_TIER: Final[Tier] = "standard"
VALID_TIERS: Final[frozenset[str]] = frozenset({"free", "basic", "standard", "pro"})

#: UK & Ireland. The platform is UK-focused; Irish form is included because a
#: large share of UK runners have their recent form there.
DEFAULT_REGIONS: Final[tuple[str, ...]] = ("gb", "ire")

#: The API caps page size at 50 on every paginated endpoint.
MAX_PAGE_SIZE: Final[int] = 50


class RacingAPIClient:
    """Authenticated, rate-limited, retrying async client for The Racing API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        credentials: RacingAPICredentials | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.credentials = (credentials or credentials_from_settings(self.settings)).require()
        self.base_url = self.settings.racing_api_base_url
        self.max_retries = self.settings.racing_api_max_retries
        self.backoff_factor = self.settings.racing_api_backoff_factor

        self._rate_limiter = AsyncTokenBucket(self.settings.racing_api_rate_limit_per_second)
        self._owns_client = client is None
        self._transport = transport
        self._client: httpx.AsyncClient | None = client

        # Observability counters -- surfaced in the import summary.
        self.request_count = 0
        self.retry_count = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            auth=self.credentials.as_httpx_auth(),
            timeout=httpx.Timeout(
                self.settings.racing_api_timeout_seconds,
                connect=min(10.0, self.settings.racing_api_timeout_seconds),
            ),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5, keepalive_expiry=30.0),
            headers={
                "Accept": "application/json",
                "User-Agent": f"horse-quant-platform/{self.settings.app_version}",
            },
            follow_redirects=True,
            transport=self._transport,
        )

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> RacingAPIClient:
        self.client  # noqa: B018 - force construction inside the running loop
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Core request path
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_params(params: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Drop ``None`` values so we never send ``?date=None``."""
        if not params:
            return None
        cleaned = {key: value for key, value in params.items() if value is not None}
        return cleaned or None

    def _backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        """Exponential backoff with full jitter, floored by any ``Retry-After``."""
        if retry_after is not None and retry_after > 0:
            return min(retry_after, 120.0)
        window = self.backoff_factor * (2**attempt)
        return min(random.uniform(0, window), 60.0)

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        raw = response.headers.get("retry-after")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    async def request_json(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        method: str = "GET",
    ) -> Any:
        """Perform a rate-limited, retrying request and return decoded JSON."""
        query = self._clean_params(params)
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            await self._rate_limiter.acquire()
            self.request_count += 1

            try:
                response = await self.client.request(method, path, params=query)
            except httpx.TimeoutException as exc:
                last_error = RacingAPITimeoutError(
                    f"Racing API request timed out after {self.settings.racing_api_timeout_seconds}s",
                    endpoint=path,
                )
                logger.warning(
                    "racing api timeout", extra={"endpoint": path, "attempt": attempt + 1, "error": str(exc)}
                )
            except httpx.HTTPError as exc:
                last_error = RacingAPIError(f"Racing API transport error: {exc}", endpoint=path)
                logger.warning(
                    "racing api transport error",
                    extra={"endpoint": path, "attempt": attempt + 1, "error": str(exc)},
                )
            else:
                if response.status_code < 400:
                    logger.debug(
                        "racing api ok",
                        extra={
                            "endpoint": path,
                            "status_code": response.status_code,
                            "elapsed_ms": round(response.elapsed.total_seconds() * 1000, 2)
                            if response.elapsed is not None
                            else None,
                        },
                    )
                    return self._decode(response, path)

                error = classify_response_error(
                    response.status_code,
                    self._safe_json(response),
                    endpoint=path,
                    retry_after=self._retry_after_seconds(response),
                )
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    # Not retryable: credentials, subscription, 404, bad request.
                    logger.error(
                        "racing api permanent failure",
                        extra={
                            "endpoint": path,
                            "status_code": response.status_code,
                            "error_code": error.error_code,
                        },
                    )
                    raise error
                last_error = error
                logger.warning(
                    "racing api retryable failure",
                    extra={
                        "endpoint": path,
                        "status_code": response.status_code,
                        "attempt": attempt + 1,
                    },
                )

            if attempt >= self.max_retries:
                break

            retry_after = getattr(last_error, "retry_after", None)
            delay = self._backoff_delay(attempt, retry_after)
            self.retry_count += 1
            logger.info(
                "retrying racing api request",
                extra={"endpoint": path, "attempt": attempt + 1, "delay_seconds": round(delay, 2)},
            )
            await asyncio.sleep(delay)

        assert last_error is not None
        logger.error(
            "racing api exhausted retries",
            extra={"endpoint": path, "attempts": self.max_retries + 1, "error": str(last_error)},
        )
        raise last_error

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text[:500]

    @staticmethod
    def _decode(response: httpx.Response, path: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise RacingAPIResponseError(
                f"Racing API returned non-JSON content ({response.headers.get('content-type')})",
                endpoint=path,
            ) from exc

    async def _get_model(
        self,
        path: str,
        model: type[TPayload],
        *,
        params: Mapping[str, Any] | None = None,
    ) -> TPayload:
        """Fetch and validate against a wire payload model."""
        data = await self.request_json(path, params=params)
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            logger.error(
                "racing api response failed validation",
                extra={"endpoint": path, "model": model.__name__, "error_count": exc.error_count()},
            )
            raise RacingAPIResponseError(
                f"Racing API response did not match {model.__name__}",
                endpoint=path,
                details={"errors": exc.errors()[:5]},
            ) from exc

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    async def check_connectivity(self) -> dict[str, Any]:
        """Probe the API and report exactly why it is or is not usable.

        Never raises: returns a structured verdict so callers (startup checks,
        the ``check-racing-api`` script, the readiness endpoint) can render it.
        """
        try:
            await self.request_json("/v1/courses", params={"limit": 1})
        except RacingAPIError as exc:
            return {
                "reachable": True,
                "authorised": False,
                "error_code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            }
        except Exception as exc:
            return {
                "reachable": False,
                "authorised": False,
                "error_code": "unreachable",
                "message": str(exc),
                "details": {},
            }
        return {"reachable": True, "authorised": True, "error_code": None, "message": "ok", "details": {}}

    # ------------------------------------------------------------------
    # Racecards
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_tier(tier: str) -> str:
        if tier not in VALID_TIERS:
            raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(VALID_TIERS)}")
        return tier

    async def get_races(
        self,
        *,
        date: str | None = None,
        region_codes: Sequence[str] | None = DEFAULT_REGIONS,
        course_ids: Sequence[str] | None = None,
        tier: Tier = DEFAULT_TIER,
        limit: int = MAX_PAGE_SIZE,
        skip: int = 0,
    ) -> list[RaceSchema]:
        """Racecards for a date (defaults to today, UK time).

        ``tier="pro"`` includes bookmaker odds inline, which avoids one odds
        call per runner — use it when the plan allows.
        """
        payload = await self._get_model(
            f"/v1/racecards/{self._validate_tier(tier)}",
            RacecardsPagePayload,
            params={
                "date": date or format_date(today_uk()),
                "region_codes": list(region_codes) if region_codes else None,
                "course_ids": list(course_ids) if course_ids else None,
                "limit": limit,
                "skip": skip,
            },
        )
        return [parse_racecard(card) for card in payload.racecards]

    async def iter_races(
        self,
        *,
        date: str | None = None,
        region_codes: Sequence[str] | None = DEFAULT_REGIONS,
        tier: Tier = DEFAULT_TIER,
        page_size: int = MAX_PAGE_SIZE,
        max_pages: int = 100,
    ) -> AsyncIterator[RaceSchema]:
        """Page through every racecard for a date."""
        skip = 0
        for _ in range(max_pages):
            races = await self.get_races(
                date=date, region_codes=region_codes, tier=tier, limit=page_size, skip=skip
            )
            if not races:
                return
            for race in races:
                yield race
            if len(races) < page_size:
                return
            skip += page_size

    async def get_race_details(self, race_id: str, *, tier: Tier = "pro") -> RaceSchema:
        """A single racecard in full detail."""
        if tier not in {"standard", "pro"}:
            raise ValueError("race details are available at the 'standard' and 'pro' tiers only")
        payload = await self._get_model(f"/v1/racecards/{race_id}/{tier}", RacecardPayload)
        return parse_racecard(payload)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    async def get_results(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        region: Sequence[str] | None = DEFAULT_REGIONS,
        course: Sequence[str] | None = None,
        limit: int = MAX_PAGE_SIZE,
        skip: int = 0,
    ) -> list[RaceSchema]:
        """Finished races — the supervised-learning label source for Phase 6."""
        payload = await self._get_model(
            "/v1/results",
            ResultsPagePayload,
            params={
                "start_date": start_date,
                "end_date": end_date,
                "region": list(region) if region else None,
                "course": list(course) if course else None,
                "limit": limit,
                "skip": skip,
            },
        )
        return [parse_result(result) for result in payload.results]

    async def iter_results(
        self,
        *,
        start_date: str,
        end_date: str,
        region: Sequence[str] | None = DEFAULT_REGIONS,
        page_size: int = MAX_PAGE_SIZE,
        max_pages: int = 1000,
    ) -> AsyncIterator[RaceSchema]:
        """Page through a date range of results — the historical backfill driver."""
        skip = 0
        for _ in range(max_pages):
            results = await self.get_results(
                start_date=start_date, end_date=end_date, region=region, limit=page_size, skip=skip
            )
            if not results:
                return
            for result in results:
                yield result
            if len(results) < page_size:
                return
            skip += page_size

    async def get_race_result(self, race_id: str) -> RaceSchema:
        """The result of one specific race."""
        payload = await self._get_model(f"/v1/results/{race_id}", ResultRacePayload)
        return parse_result(payload)

    # ------------------------------------------------------------------
    # Odds
    # ------------------------------------------------------------------
    async def get_odds(self, race_id: str, horse_id: str) -> list[OddsSchema]:
        """Every bookmaker's current price for one runner."""
        payload = await self._get_model(f"/v1/odds/{race_id}/{horse_id}", RunnerOddsPayload)
        # The endpoint omits the ids from some responses; restore them so the
        # caller always gets fully-identified quotes.
        payload.race_id = payload.race_id or race_id
        payload.horse_id = payload.horse_id or horse_id
        return parse_runner_odds(payload)

    async def get_race_odds(self, race_id: str, horse_ids: Sequence[str]) -> list[OddsSchema]:
        """Odds for every runner in a race, fetched concurrently.

        The shared rate limiter keeps the fan-out inside the plan's ceiling, and
        one runner's failure does not lose the rest of the race.
        """
        tasks = [self.get_odds(race_id, horse_id) for horse_id in horse_ids]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        quotes: list[OddsSchema] = []
        for horse_id, outcome in zip(horse_ids, gathered, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning(
                    "odds fetch failed for runner",
                    extra={"race_id": race_id, "horse_id": horse_id, "error": str(outcome)},
                )
                continue
            quotes.extend(outcome)
        return quotes

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------
    async def get_horses(self, name: str, *, limit: int = MAX_PAGE_SIZE, skip: int = 0) -> list[HorseSchema]:
        """Search horses by name."""
        payload = await self._get_model(
            "/v1/horses/search",
            HorseSearchPagePayload,
            params={"name": name, "limit": limit, "skip": skip},
        )
        return [parse_horse(horse) for horse in payload.search_results]

    async def get_horse_history(
        self, horse_id: str, *, limit: int = MAX_PAGE_SIZE, skip: int = 0
    ) -> list[RaceSchema]:
        """Every past race for one horse — the raw material for form features."""
        payload = await self._get_model(
            f"/v1/horses/{horse_id}/results",
            ResultsPagePayload,
            params={"limit": limit, "skip": skip},
        )
        return [parse_result(result) for result in payload.results]

    async def get_jockeys(
        self, name: str, *, limit: int = MAX_PAGE_SIZE, skip: int = 0
    ) -> list[JockeySchema]:
        payload = await self._get_model(
            "/v1/jockeys/search",
            PersonSearchPagePayload,
            params={"name": name, "limit": limit, "skip": skip},
        )
        return [parse_jockey(person) for person in payload.search_results]

    async def get_trainers(
        self, name: str, *, limit: int = MAX_PAGE_SIZE, skip: int = 0
    ) -> list[TrainerSchema]:
        payload = await self._get_model(
            "/v1/trainers/search",
            PersonSearchPagePayload,
            params={"name": name, "limit": limit, "skip": skip},
        )
        return [parse_trainer(person) for person in payload.search_results]

    async def get_courses(
        self, *, region_codes: Sequence[str] | None = DEFAULT_REGIONS
    ) -> list[CourseSchema]:
        payload = await self._get_model(
            "/v1/courses",
            CoursesPagePayload,
            params={"region_codes": list(region_codes) if region_codes else None},
        )
        return [parse_course(course) for course in payload.courses]


__all__ = [
    "DEFAULT_REGIONS",
    "DEFAULT_TIER",
    "MAX_PAGE_SIZE",
    "VALID_TIERS",
    "RacingAPIClient",
]
