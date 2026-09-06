"""Credential resolution and HTTP authentication for The Racing API.

The API uses **HTTP Basic** authentication over TLS. Credentials are read from
:class:`~backend.utils.config.Settings` (never from ``os.environ`` directly) and
are held as ``SecretStr`` all the way to the point of use, so they cannot be
printed by an accidental ``repr()``, log line or traceback.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import httpx

from backend.services.racing_api.exceptions import RacingAPIAuthenticationError
from backend.utils.config import Settings, get_settings
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="api")

#: How much of a credential to reveal when reporting configuration problems.
_VISIBLE_PREFIX = 4


def mask_secret(value: str, *, visible: int = _VISIBLE_PREFIX) -> str:
    """Render a credential as ``abcd...`` -- enough to identify, useless to steal."""
    if not value:
        return "<empty>"
    if len(value) <= visible:
        return "*" * len(value)
    return f"{value[:visible]}{'*' * (len(value) - visible)}"


@dataclass(frozen=True)
class RacingAPICredentials:
    """A validated username/password pair.

    ``frozen`` so credentials cannot be mutated in flight, and ``repr`` is
    overridden so the password never reaches a log or a debugger dump.
    """

    username: str
    password: str

    @property
    def is_complete(self) -> bool:
        return bool(self.username and self.password)

    @property
    def masked_username(self) -> str:
        return mask_secret(self.username)

    def basic_auth_header(self) -> str:
        """The ``Authorization`` header value, for clients that need it directly."""
        token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode("ascii")
        return f"Basic {token}"

    def as_httpx_auth(self) -> httpx.BasicAuth:
        return httpx.BasicAuth(self.username, self.password)

    def require(self) -> RacingAPICredentials:
        """Return self, or raise if either half is missing."""
        if not self.is_complete:
            missing = [
                name
                for name, value in (
                    ("RACING_API_USERNAME", self.username),
                    ("RACING_API_PASSWORD", self.password),
                )
                if not value
            ]
            raise RacingAPIAuthenticationError(
                f"Racing API credentials are not configured: {', '.join(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} empty. Set them in .env",
                details={"missing": missing},
            )
        return self

    def __repr__(self) -> str:
        return f"RacingAPICredentials(username={self.masked_username!r}, password=<hidden>)"

    __str__ = __repr__


def credentials_from_settings(settings: Settings | None = None) -> RacingAPICredentials:
    """Build credentials from configuration without validating them."""
    settings = settings or get_settings()
    return RacingAPICredentials(
        username=settings.racing_api_username.get_secret_value(),
        password=settings.racing_api_password.get_secret_value(),
    )


def build_auth(settings: Settings | None = None) -> httpx.BasicAuth:
    """Return an httpx auth object, raising if credentials are missing."""
    return credentials_from_settings(settings).require().as_httpx_auth()


def warn_if_unconfigured(settings: Settings | None = None) -> bool:
    """Log a startup warning when credentials are absent. Returns True if configured."""
    credentials = credentials_from_settings(settings)
    if credentials.is_complete:
        logger.info("Racing API credentials loaded", extra={"username": credentials.masked_username})
        return True
    logger.warning(
        "Racing API credentials are not configured; ingestion will fail",
        extra={
            "missing": [
                n
                for n, v in (("username", credentials.username), ("password", credentials.password))
                if not v
            ]
        },
    )
    return False


__all__ = [
    "RacingAPICredentials",
    "build_auth",
    "credentials_from_settings",
    "mask_secret",
    "warn_if_unconfigured",
]
