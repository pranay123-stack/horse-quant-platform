"""Racing API error taxonomy.

The API signals four *different* conditions with the same ``401`` status, and
they demand opposite responses from an automated system:

===========================  ==============================  ==================
Body                         Meaning                         Retry?
===========================  ==============================  ==================
``Not authenticated``        No credentials sent              No -- fix config
``Incorrect username``       Credentials wrong                No -- fix config
``Subscription inactive``    Account valid, plan not active   No -- billing
``Incorrect password``       Credentials wrong                No -- fix config
===========================  ==============================  ==================

Collapsing these into a generic "auth failed" would send an operator hunting for
a typo in a password that is perfectly correct. Each maps to its own exception.

Verified against the live API on 2026-08-10 (see ``scripts/check_racing_api.py``).
"""

from __future__ import annotations

from typing import Any

from backend.utils.exceptions import ExternalAPIError, RateLimitError

_SUBSCRIPTION_MARKERS = ("subscription inactive", "subscription expired", "no active subscription")
_CREDENTIAL_MARKERS = ("incorrect username", "incorrect password", "invalid credentials")
_MISSING_CREDENTIAL_MARKERS = ("not authenticated", "missing credentials")


class RacingAPIError(ExternalAPIError):
    """Base class for every Racing API failure."""

    error_code = "racing_api_error"


class RacingAPIAuthenticationError(RacingAPIError):
    """Credentials are missing or wrong. Not retryable -- fix the configuration."""

    status_code = 502
    error_code = "racing_api_authentication_error"


class SubscriptionInactiveError(RacingAPIError):
    """The account is valid but the plan is not active.

    Distinct from :class:`RacingAPIAuthenticationError` because the remedy is
    billing, not configuration. Retrying cannot help.
    """

    status_code = 502
    error_code = "racing_api_subscription_inactive"

    def __init__(self, message: str = "Racing API subscription is inactive", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.details.setdefault(
            "remedy",
            "Activate or renew the plan at https://www.theracingapi.com -- credentials are valid.",
        )


class RacingAPIRateLimitError(RateLimitError):
    """429 from the Racing API. Retryable after ``retry_after`` seconds."""

    error_code = "racing_api_rate_limit"


class RacingAPINotFoundError(RacingAPIError):
    """404 -- the race, horse or entity does not exist."""

    status_code = 404
    error_code = "racing_api_not_found"


class RacingAPIServerError(RacingAPIError):
    """5xx from upstream. Retryable."""

    status_code = 502
    error_code = "racing_api_server_error"


class RacingAPIResponseError(RacingAPIError):
    """The response was not valid JSON, or did not match the expected schema."""

    status_code = 502
    error_code = "racing_api_response_error"


class RacingAPITimeoutError(RacingAPIError):
    """The request timed out. Retryable."""

    status_code = 504
    error_code = "racing_api_timeout"


def _detail_text(body: Any) -> str:
    """Pull the human-readable message out of a Racing API error body."""
    if isinstance(body, dict):
        detail = body.get("detail", body.get("message", ""))
        if isinstance(detail, list):  # FastAPI validation errors
            return "; ".join(str(item.get("msg", item)) for item in detail)
        return str(detail)
    return str(body or "")


def classify_response_error(
    status: int,
    body: Any,
    *,
    endpoint: str | None = None,
    retry_after: float | None = None,
) -> RacingAPIError | RacingAPIRateLimitError:
    """Map an HTTP status and body onto the most specific exception available."""
    detail = _detail_text(body)
    lowered = detail.lower()

    if status == 401 or status == 403:
        if any(marker in lowered for marker in _SUBSCRIPTION_MARKERS):
            return SubscriptionInactiveError(
                f"Racing API subscription is inactive ({detail})",
                status=status,
                endpoint=endpoint,
            )
        if any(marker in lowered for marker in _MISSING_CREDENTIAL_MARKERS):
            return RacingAPIAuthenticationError(
                "Racing API credentials were not sent -- set RACING_API_USERNAME and RACING_API_PASSWORD",
                status=status,
                endpoint=endpoint,
            )
        if any(marker in lowered for marker in _CREDENTIAL_MARKERS):
            return RacingAPIAuthenticationError(
                f"Racing API rejected the credentials ({detail})",
                status=status,
                endpoint=endpoint,
            )
        return RacingAPIAuthenticationError(
            f"Racing API authorisation failed ({detail or status})",
            status=status,
            endpoint=endpoint,
        )

    if status == 404:
        return RacingAPINotFoundError(
            f"Racing API resource not found ({detail or endpoint})", status=status, endpoint=endpoint
        )

    if status == 429:
        return RacingAPIRateLimitError(
            f"Racing API rate limit exceeded ({detail})".strip(),
            retry_after=retry_after,
            status=status,
            endpoint=endpoint,
        )

    if status >= 500:
        return RacingAPIServerError(
            f"Racing API server error {status} ({detail})".strip(), status=status, endpoint=endpoint
        )

    return RacingAPIError(
        f"Racing API returned {status} ({detail})".strip(), status=status, endpoint=endpoint
    )


#: Statuses worth retrying. 4xx (other than 429) never becomes healthy by waiting.
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


__all__ = [
    "RETRYABLE_STATUS_CODES",
    "RacingAPIAuthenticationError",
    "RacingAPIError",
    "RacingAPINotFoundError",
    "RacingAPIRateLimitError",
    "RacingAPIResponseError",
    "RacingAPIServerError",
    "RacingAPITimeoutError",
    "SubscriptionInactiveError",
    "classify_response_error",
]
