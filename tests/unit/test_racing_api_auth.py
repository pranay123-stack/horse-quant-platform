"""Credential handling — the tests that matter most for not leaking secrets."""

from __future__ import annotations

import base64

import httpx
import pytest

from backend.services.racing_api.authentication import (
    RacingAPICredentials,
    build_auth,
    credentials_from_settings,
    mask_secret,
    warn_if_unconfigured,
)
from backend.services.racing_api.exceptions import RacingAPIAuthenticationError
from backend.utils.config import Settings

pytestmark = pytest.mark.unit


def test_mask_secret_reveals_only_a_prefix():
    assert mask_secret("supersecretvalue") == "supe************"
    assert mask_secret("abc") == "***"
    assert mask_secret("") == "<empty>"


def test_credentials_never_render_the_password():
    credentials = RacingAPICredentials(username="myuser1234", password="mypassword")
    rendered = f"{credentials!r} {credentials!s}"
    assert "mypassword" not in rendered
    assert "<hidden>" in rendered
    assert "myus" in rendered  # enough to identify the account


def test_credentials_are_immutable(racing_credentials):
    with pytest.raises(Exception):  # noqa: B017 - dataclass raises FrozenInstanceError
        racing_credentials.password = "changed"  # type: ignore[misc]


def test_is_complete():
    assert RacingAPICredentials("u", "p").is_complete
    assert not RacingAPICredentials("", "p").is_complete
    assert not RacingAPICredentials("u", "").is_complete


def test_basic_auth_header_is_correct(racing_credentials):
    header = racing_credentials.basic_auth_header()
    assert header.startswith("Basic ")
    decoded = base64.b64decode(header.removeprefix("Basic ")).decode()
    assert decoded == "test-username:test-password"


def test_as_httpx_auth(racing_credentials):
    assert isinstance(racing_credentials.as_httpx_auth(), httpx.BasicAuth)


def test_require_passes_when_complete(racing_credentials):
    assert racing_credentials.require() is racing_credentials


@pytest.mark.parametrize(
    ("username", "password", "expected"),
    [
        ("", "p", ["RACING_API_USERNAME"]),
        ("u", "", ["RACING_API_PASSWORD"]),
        ("", "", ["RACING_API_USERNAME", "RACING_API_PASSWORD"]),
    ],
)
def test_require_names_the_missing_variables(username, password, expected):
    with pytest.raises(RacingAPIAuthenticationError) as excinfo:
        RacingAPICredentials(username, password).require()
    assert excinfo.value.details["missing"] == expected


def test_require_error_message_does_not_contain_the_password():
    with pytest.raises(RacingAPIAuthenticationError) as excinfo:
        RacingAPICredentials("", "hunter2hunter2").require()
    assert "hunter2" not in str(excinfo.value)


def test_credentials_from_settings_reads_secretstr():
    settings = Settings(racing_api_username="from-config", racing_api_password="secret-pw")
    credentials = credentials_from_settings(settings)
    assert credentials.username == "from-config"
    assert credentials.password == "secret-pw"


def test_build_auth_raises_when_unconfigured():
    settings = Settings(racing_api_username="", racing_api_password="")
    with pytest.raises(RacingAPIAuthenticationError):
        build_auth(settings)


def test_warn_if_unconfigured_reports_state(settings, caplog):
    assert warn_if_unconfigured(settings) is True

    blank = Settings(racing_api_username="", racing_api_password="")
    assert warn_if_unconfigured(blank) is False


def test_warning_log_does_not_leak_credentials(settings, caplog):
    with caplog.at_level("INFO"):
        warn_if_unconfigured(settings)
    assert settings.racing_api_password.get_secret_value() not in caplog.text
