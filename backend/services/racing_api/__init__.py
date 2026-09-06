"""The Racing API integration layer.

    from backend.services.racing_api import RacingAPIClient

    async with RacingAPIClient() as client:
        races = await client.get_races(date="2026-08-10")

Module map:

* :mod:`~backend.services.racing_api.authentication` — credentials, HTTP Basic
* :mod:`~backend.services.racing_api.client` — the async client
* :mod:`~backend.services.racing_api.exceptions` — error taxonomy
* :mod:`~backend.services.racing_api.parsers` — string payloads -> typed domain
* :mod:`~backend.services.racing_api.rate_limit` — token bucket
* :mod:`~backend.services.racing_api.schemas` — wire and domain contracts
"""

from backend.services.racing_api.authentication import (
    RacingAPICredentials,
    credentials_from_settings,
    warn_if_unconfigured,
)
from backend.services.racing_api.client import DEFAULT_REGIONS, RacingAPIClient
from backend.services.racing_api.exceptions import (
    RacingAPIAuthenticationError,
    RacingAPIError,
    RacingAPINotFoundError,
    RacingAPIRateLimitError,
    RacingAPIResponseError,
    RacingAPIServerError,
    RacingAPITimeoutError,
    SubscriptionInactiveError,
)
from backend.services.racing_api.schemas import (
    CourseSchema,
    HorseSchema,
    JockeySchema,
    OddsSchema,
    RaceSchema,
    ResultRunnerSchema,
    RunnerSchema,
    TrainerSchema,
)

__all__ = [
    "DEFAULT_REGIONS",
    "CourseSchema",
    "HorseSchema",
    "JockeySchema",
    "OddsSchema",
    "RaceSchema",
    "RacingAPIAuthenticationError",
    "RacingAPIClient",
    "RacingAPICredentials",
    "RacingAPIError",
    "RacingAPINotFoundError",
    "RacingAPIRateLimitError",
    "RacingAPIResponseError",
    "RacingAPIServerError",
    "RacingAPITimeoutError",
    "ResultRunnerSchema",
    "RunnerSchema",
    "SubscriptionInactiveError",
    "TrainerSchema",
    "credentials_from_settings",
    "warn_if_unconfigured",
]
