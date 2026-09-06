"""Configuration layer tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.utils.config import PROJECT_ROOT, Settings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate(env_free):
    """Every ``Settings(...)`` built in this module sees only its own arguments."""


def test_settings_load_from_environment(settings):
    assert settings.environment == "test"
    assert settings.app_name
    assert settings.api_v1_prefix == "/api/v1"


def test_database_url_is_composed_from_components():
    cfg = Settings(
        postgres_user="alice",
        postgres_password="s3cret",
        postgres_host="db.internal",
        postgres_port=6543,
        postgres_db="racing",
    )
    assert cfg.database_url == "postgresql+psycopg2://alice:s3cret@db.internal:6543/racing"


def test_database_url_override_wins():
    cfg = Settings(DATABASE_URL="postgresql+psycopg2://u:p@other:5432/db")
    assert cfg.database_url == "postgresql+psycopg2://u:p@other:5432/db"


def test_safe_database_url_masks_the_password():
    cfg = Settings(postgres_user="alice", postgres_password="s3cret")
    safe = cfg.safe_database_url
    assert "s3cret" not in safe
    assert "alice" in safe
    assert "***" in safe


def test_password_is_not_leaked_by_repr_or_dump():
    """A Settings object is logged during startup -- it must never carry secrets."""
    cfg = Settings(postgres_password="topsecret", racing_api_key="apikey123", secret_key="jwtsecret9")
    rendered = repr(cfg) + str(cfg.model_dump()) + cfg.model_dump_json()
    for secret in ("topsecret", "apikey123", "jwtsecret9"):
        assert secret not in rendered


def test_database_url_override_is_not_leaked_either():
    cfg = Settings(DATABASE_URL="postgresql+psycopg2://u:dsnpassword@host:5432/db")
    rendered = repr(cfg) + str(cfg.model_dump()) + cfg.model_dump_json()
    assert "dsnpassword" not in rendered
    # ...but the real DSN is still available to the engine factory.
    assert "dsnpassword" in cfg.database_url


def test_redis_url():
    cfg = Settings(redis_host="cache", redis_port=6380, redis_db=3)
    assert cfg.redis_url == "redis://cache:6380/3"
    # Celery falls back to Redis when no dedicated broker is configured.
    assert cfg.broker_url == cfg.redis_url
    assert cfg.result_backend == cfg.redis_url


def test_cors_origins_accepts_csv():
    cfg = Settings(cors_origins="http://a.test, http://b.test")
    assert cfg.cors_origins == ["http://a.test", "http://b.test"]


def test_relative_paths_resolve_against_project_root():
    cfg = Settings(log_dir="logs", models_dir="models")
    assert cfg.log_dir == PROJECT_ROOT / "logs"
    assert cfg.models_dir == PROJECT_ROOT / "models"


def test_base_url_trailing_slash_is_stripped():
    cfg = Settings(racing_api_base_url="https://api.theracingapi.com/")
    assert cfg.racing_api_base_url == "https://api.theracingapi.com"


def test_racing_api_configured_flag():
    assert not Settings(
        racing_api_username="", racing_api_password="", racing_api_key=""
    ).racing_api_configured
    assert Settings(racing_api_key="abc").racing_api_configured
    assert Settings(racing_api_username="u", racing_api_password="p").racing_api_configured


@pytest.mark.parametrize("value", [0, -0.1, 1.5])
def test_invalid_fractions_are_rejected(value):
    with pytest.raises(ValidationError):
        Settings(kelly_fraction=value)


def test_production_requires_a_real_secret_key():
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings(environment="prod", secret_key="change-me-in-production")


def test_production_forbids_debug():
    with pytest.raises(ValidationError, match="DEBUG"):
        Settings(environment="prod", secret_key="a-real-secret", debug=True)


def test_ensure_directories_creates_everything(tmp_path):
    cfg = Settings(
        log_dir=str(tmp_path / "logs"),
        data_dir=str(tmp_path / "data"),
        models_dir=str(tmp_path / "models"),
        reports_dir=str(tmp_path / "reports"),
    )
    cfg.ensure_directories()
    assert cfg.log_dir.is_dir()
    assert (cfg.data_dir / "raw").is_dir()
    assert (cfg.data_dir / "processed").is_dir()
    assert cfg.models_dir.is_dir()
    assert cfg.reports_dir.is_dir()
