"""Exception hierarchy tests."""

from __future__ import annotations

import pytest

from backend.utils.exceptions import (
    AuthenticationError,
    ExternalAPIError,
    NotFoundError,
    PlatformError,
    RateLimitError,
)

pytestmark = pytest.mark.unit


def test_every_error_is_a_platform_error():
    for cls in (ExternalAPIError, RateLimitError, NotFoundError, AuthenticationError):
        assert issubclass(cls, PlatformError)


def test_to_dict_shape():
    error = NotFoundError("no such race", details={"race_id": "rac_123"})
    assert error.to_dict() == {
        "error": "not_found",
        "message": "no such race",
        "details": {"race_id": "rac_123"},
    }
    assert error.status_code == 404


def test_external_api_error_records_context():
    error = ExternalAPIError("upstream failed", status=503, endpoint="/v1/racecards")
    assert error.details["upstream_status"] == 503
    assert error.details["endpoint"] == "/v1/racecards"
    assert error.status_code == 502


def test_external_api_error_omits_missing_context():
    error = ExternalAPIError("upstream failed")
    assert error.details == {}


def test_rate_limit_error_carries_retry_after():
    error = RateLimitError("slow down", retry_after=12.5, endpoint="/v1/results")
    assert error.retry_after == 12.5
    assert error.details["retry_after"] == 12.5
    assert error.status_code == 429


def test_errors_are_raisable_and_catchable_as_base():
    with pytest.raises(PlatformError) as excinfo:
        raise AuthenticationError("bad credentials", status=401)
    assert excinfo.value.error_code == "upstream_authentication_error"
