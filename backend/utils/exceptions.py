"""Application exception hierarchy.

Every error raised deliberately by platform code derives from
:class:`PlatformError`. The FastAPI layer maps these onto HTTP responses in
``backend/api/errors.py``, so business code never has to import ``HTTPException``.
"""

from __future__ import annotations

from typing import Any


class PlatformError(Exception):
    """Base class for all deliberate platform errors."""

    status_code: int = 500
    error_code: str = "platform_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.error_code, "message": self.message, "details": self.details}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.message!r}, details={self.details!r})"


class ConfigurationError(PlatformError):
    """Missing or invalid configuration (bad credentials, absent env var)."""

    status_code = 500
    error_code = "configuration_error"


class ExternalAPIError(PlatformError):
    """The Racing API (or another upstream) returned an unusable response."""

    status_code = 502
    error_code = "external_api_error"

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        endpoint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged = {"upstream_status": status, "endpoint": endpoint, **(details or {})}
        super().__init__(message, details={k: v for k, v in merged.items() if v is not None})
        self.status = status
        self.endpoint = endpoint


class AuthenticationError(ExternalAPIError):
    """Upstream rejected our credentials (401/403)."""

    status_code = 502
    error_code = "upstream_authentication_error"


class RateLimitError(ExternalAPIError):
    """Upstream rate limit hit (429)."""

    status_code = 429
    error_code = "rate_limit_error"

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        if retry_after is not None:
            self.details["retry_after"] = retry_after


class DataValidationError(PlatformError):
    """A payload failed validation on the way into the warehouse."""

    status_code = 422
    error_code = "data_validation_error"


class NotFoundError(PlatformError):
    """A requested entity does not exist."""

    status_code = 404
    error_code = "not_found"


class DatabaseError(PlatformError):
    """Persistence layer failure."""

    status_code = 503
    error_code = "database_error"


class PipelineError(PlatformError):
    """An ETL stage failed."""

    status_code = 500
    error_code = "pipeline_error"


class ModelError(PlatformError):
    """Training / inference failure, or a missing model artefact."""

    status_code = 500
    error_code = "model_error"


class StrategyError(PlatformError):
    """Invalid strategy parameters or an unsatisfiable staking rule."""

    status_code = 400
    error_code = "strategy_error"


__all__ = [
    "AuthenticationError",
    "ConfigurationError",
    "DataValidationError",
    "DatabaseError",
    "ExternalAPIError",
    "ModelError",
    "NotFoundError",
    "PipelineError",
    "PlatformError",
    "RateLimitError",
    "StrategyError",
]
