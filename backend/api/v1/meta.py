"""Introspection endpoint exposing the *non-secret* effective configuration.

Extremely useful when debugging a container: it answers "which database is this
process actually pointing at, and is the Racing API wired up?" without shelling
in. Secrets are never returned -- only booleans indicating presence.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.api.deps import SettingsDep

router = APIRouter(prefix="/meta", tags=["meta"])


class BettingDefaults(BaseModel):
    starting_bankroll: float
    min_expected_value: float
    kelly_fraction: float
    max_stake_fraction: float
    flat_stake: float


class ConfigResponse(BaseModel):
    app_name: str
    app_version: str
    environment: str
    debug: bool
    timezone: str
    api_v1_prefix: str
    database: str
    redis: str
    log_level: str
    log_dir: str
    models_dir: str
    active_model_version: str
    racing_api_base_url: str
    racing_api_configured: bool
    betting_defaults: BettingDefaults


@router.get("/config", response_model=ConfigResponse, summary="Effective configuration (secrets masked)")
def effective_config(settings: SettingsDep) -> ConfigResponse:
    return ConfigResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        environment=settings.environment,
        debug=settings.debug,
        timezone=settings.timezone,
        api_v1_prefix=settings.api_v1_prefix,
        database=settings.safe_database_url,
        redis=settings.redis_url,
        log_level=settings.log_level,
        log_dir=str(settings.log_dir),
        models_dir=str(settings.models_dir),
        active_model_version=settings.active_model_version,
        racing_api_base_url=settings.racing_api_base_url,
        racing_api_configured=settings.racing_api_configured,
        betting_defaults=BettingDefaults(
            starting_bankroll=settings.starting_bankroll,
            min_expected_value=settings.min_expected_value,
            kelly_fraction=settings.kelly_fraction,
            max_stake_fraction=settings.max_stake_fraction,
            flat_stake=settings.flat_stake,
        ),
    )


__all__ = ["router"]
