"""Centralised, validated application configuration.

Every runtime knob in the platform lives here. Nothing else in the codebase may
read ``os.environ`` directly -- that keeps configuration auditable, typed and
testable, and it guarantees that credentials are never hard-coded.

Settings are loaded from (in order of increasing precedence):

1. Defaults declared on the :class:`Settings` model.
2. The ``.env`` file at the project root.
3. Real process environment variables.

Usage::

    from backend.utils.config import get_settings

    settings = get_settings()
    print(settings.database_url)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# backend/utils/config.py -> backend/utils -> backend -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

Environment = Literal["local", "test", "dev", "staging", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Typed application settings.

    Field names map to upper-cased environment variables, e.g. ``api_port`` is
    populated from ``API_PORT``.
    """

    model_config = SettingsConfigDict(
        env_file=os.getenv("HQP_ENV_FILE", str(PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # We use ``models_dir`` rather than ``model_dir`` to stay clear of
        # pydantic's protected ``model_`` namespace; this keeps the guard on.
        protected_namespaces=("settings_",),
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = "Horse Quant Platform"
    app_version: str = "0.1.0"
    environment: Environment = "local"
    debug: bool = False
    timezone: str = "Europe/London"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_v1_prefix: str = "/api/v1"
    api_workers: int = 1
    # ``NoDecode`` is essential: without it pydantic-settings JSON-decodes complex
    # types straight from the dotenv source, which blows up on a plain
    # comma-separated list *before* the validator below ever runs.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8501"]
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: LogLevel = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"
    log_json: bool = False
    log_to_console: bool = True
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "horse_quant"
    postgres_password: SecretStr = SecretStr("horse_quant")
    postgres_db: str = "horse_quant"
    # Full override; when set it wins over the individual components above.
    # Held as a SecretStr because a DSN embeds the password.
    database_url_override: SecretStr | None = Field(default=None, alias="DATABASE_URL")

    db_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_recycle_seconds: int = 1800
    db_statement_timeout_ms: int = 30_000

    # ------------------------------------------------------------------
    # Redis / Celery (wired up properly in Phase 10)
    # ------------------------------------------------------------------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # ------------------------------------------------------------------
    # The Racing API (consumed in Phase 2)
    # ------------------------------------------------------------------
    racing_api_base_url: str = "https://api.theracingapi.com"
    racing_api_username: SecretStr = SecretStr("")
    racing_api_password: SecretStr = SecretStr("")
    racing_api_key: SecretStr = SecretStr("")
    racing_api_timeout_seconds: float = 30.0
    racing_api_max_retries: int = 5
    racing_api_backoff_factor: float = 0.5
    racing_api_rate_limit_per_second: float = 2.0

    # ------------------------------------------------------------------
    # Data / model artefacts
    # ------------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    active_model_version: str = "baseline_v1"

    # ------------------------------------------------------------------
    # Betting / risk defaults (used by Phases 7 & 8)
    # ------------------------------------------------------------------
    starting_bankroll: float = 10_000.0
    min_expected_value: float = 0.05  # 5% edge floor for a VALUE BET
    min_model_probability: float = 0.02
    max_odds: float = 51.0
    kelly_fraction: float = 0.25  # fractional Kelly
    max_stake_fraction: float = 0.05  # hard cap: 5% of bankroll on one bet
    flat_stake: float = 10.0

    # ------------------------------------------------------------------
    # Security (Phase 10)
    # ------------------------------------------------------------------
    secret_key: SecretStr = SecretStr("change-me-in-production")
    access_token_expire_minutes: int = 60 * 12

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv_origins(cls, value: object) -> object:
        """Allow ``CORS_ORIGINS=http://a,http://b`` as well as a JSON list."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("log_dir", "data_dir", "models_dir", "reports_dir", mode="before")
    @classmethod
    def _expand_path(cls, value: object) -> object:
        if isinstance(value, str):
            path = Path(value).expanduser()
            return path if path.is_absolute() else (PROJECT_ROOT / path)
        return value

    @field_validator("racing_api_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("kelly_fraction", "max_stake_fraction")
    @classmethod
    def _fraction_range(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        return value

    @model_validator(mode="after")
    def _guard_production_secrets(self) -> Settings:
        """Fail fast rather than run production with placeholder secrets."""
        if self.environment in ("staging", "prod"):
            if self.secret_key.get_secret_value() == "change-me-in-production":
                raise ValueError("SECRET_KEY must be set outside of local/dev environments")
            if self.debug:
                raise ValueError("DEBUG must be false in staging/prod")
        return self

    # ------------------------------------------------------------------
    # Derived values
    #
    # These are plain properties, NOT pydantic ``computed_field``s: a computed
    # field is included in ``repr()`` and ``model_dump()``, which would print
    # the database password every time a Settings object is logged.
    # ------------------------------------------------------------------
    @property
    def database_url(self) -> str:
        """SQLAlchemy connection URL (sync psycopg2 driver). Contains the password."""
        if self.database_url_override is not None:
            return self.database_url_override.get_secret_value()
        password = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def safe_database_url(self) -> str:
        """Connection URL with the password masked -- safe to log."""
        url = self.database_url
        if "@" not in url or "://" not in url:
            return url
        scheme, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    @property
    def is_production(self) -> bool:
        return self.environment in ("staging", "prod")

    @property
    def racing_api_configured(self) -> bool:
        """True when at least one credential pair is present."""
        return bool(
            self.racing_api_key.get_secret_value()
            or (self.racing_api_username.get_secret_value() and self.racing_api_password.get_secret_value())
        )

    def ensure_directories(self) -> None:
        """Create the writable directories the platform expects."""
        for path in (
            self.log_dir,
            self.data_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.models_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and re-read the environment (used by tests)."""
    get_settings.cache_clear()
    return get_settings()


__all__ = ["PROJECT_ROOT", "Settings", "get_settings", "reload_settings"]
