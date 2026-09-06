"""Error classification.

The live API's three distinct 401 bodies were captured on 2026-08-10:

    no credentials  -> {"detail": "Not authenticated"}
    wrong username  -> {"detail": "Incorrect username"}
    valid account,
    inactive plan   -> {"detail": "Subscription inactive"}

Conflating them would send an operator hunting for a typo in a password that is
demonstrably correct, so each is asserted here against the real payloads.
"""

from __future__ import annotations

import pytest

from backend.services.racing_api.exceptions import (
    RETRYABLE_STATUS_CODES,
    RacingAPIAuthenticationError,
    RacingAPIError,
    RacingAPINotFoundError,
    RacingAPIRateLimitError,
    RacingAPIServerError,
    SubscriptionInactiveError,
    classify_response_error,
)

pytestmark = pytest.mark.unit


def test_inactive_subscription_is_its_own_error():
    error = classify_response_error(401, {"detail": "Subscription inactive"}, endpoint="/v1/courses")
    assert isinstance(error, SubscriptionInactiveError)
    assert not isinstance(error, RacingAPIAuthenticationError)
    assert "theracingapi.com" in error.details["remedy"]


def test_wrong_username_is_an_authentication_error():
    error = classify_response_error(401, {"detail": "Incorrect username"})
    assert isinstance(error, RacingAPIAuthenticationError)
    assert not isinstance(error, SubscriptionInactiveError)


def test_missing_credentials_points_at_configuration():
    error = classify_response_error(401, {"detail": "Not authenticated"})
    assert isinstance(error, RacingAPIAuthenticationError)
    assert "RACING_API_USERNAME" in error.message


def test_unknown_401_body_still_classifies_as_auth():
    error = classify_response_error(401, {"detail": "Something new upstream"})
    assert isinstance(error, RacingAPIAuthenticationError)


def test_403_is_treated_like_401():
    assert isinstance(
        classify_response_error(403, {"detail": "Subscription inactive"}), SubscriptionInactiveError
    )


def test_404():
    error = classify_response_error(404, {"detail": "Race not found"}, endpoint="/v1/results/x")
    assert isinstance(error, RacingAPINotFoundError)
    assert error.status_code == 404


def test_429_carries_retry_after():
    error = classify_response_error(429, {"detail": "Too many requests"}, retry_after=30.0)
    assert isinstance(error, RacingAPIRateLimitError)
    assert error.retry_after == 30.0
    assert error.details["retry_after"] == 30.0


def test_5xx_is_a_server_error():
    assert isinstance(classify_response_error(503, {"detail": "upstream down"}), RacingAPIServerError)
    assert isinstance(classify_response_error(500, ""), RacingAPIServerError)


def test_other_4xx_falls_back_to_the_base_error():
    error = classify_response_error(400, {"detail": "bad query"})
    assert type(error) is RacingAPIError


def test_detail_can_be_a_validation_list():
    error = classify_response_error(422, {"detail": [{"msg": "field required"}, {"msg": "bad date"}]})
    assert "field required" in error.message
    assert "bad date" in error.message


def test_non_dict_body_is_tolerated():
    error = classify_response_error(500, "<html>Gateway Error</html>")
    assert isinstance(error, RacingAPIServerError)


@pytest.mark.parametrize("status", [401, 403, 404, 400, 422])
def test_permanent_failures_are_not_retryable(status):
    assert status not in RETRYABLE_STATUS_CODES


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 408])
def test_transient_failures_are_retryable(status):
    assert status in RETRYABLE_STATUS_CODES


def test_all_errors_share_a_common_base():
    for error in (
        SubscriptionInactiveError(),
        RacingAPIAuthenticationError("x"),
        RacingAPINotFoundError("x"),
        RacingAPIServerError("x"),
    ):
        assert isinstance(error, RacingAPIError)
        assert error.to_dict()["error"].startswith("racing_api")
