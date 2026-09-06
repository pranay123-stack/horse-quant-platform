"""Liveness, readiness and build-info endpoints.

``/health/live``   -- process is up (used by Docker/K8s liveness).
``/health/ready``  -- dependencies reachable (used by readiness gates).
``/health/db``     -- explicit database probe.
``/health/version``-- build metadata.
"""

from __future__ import annotations

import platform
import time
from typing import Literal

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.api.deps import SettingsDep
from backend.database.session import check_database_connection
from backend.utils.config import Settings
from backend.utils.timeutils import utcnow

router = APIRouter(prefix="/health", tags=["health"])

_PROCESS_STARTED_AT = time.time()


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    environment: str
    version: str
    timestamp: str
    uptime_seconds: float = Field(..., description="Seconds since process start")


class ComponentStatus(BaseModel):
    name: str
    healthy: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    components: list[ComponentStatus]
    timestamp: str


class VersionResponse(BaseModel):
    app_name: str
    version: str
    environment: str
    python_version: str
    platform: str


def _base_health(settings: Settings) -> HealthResponse:
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        version=settings.app_version,
        timestamp=utcnow().isoformat(),
        uptime_seconds=round(time.time() - _PROCESS_STARTED_AT, 3),
    )


@router.get("", response_model=HealthResponse, summary="Basic health check")
@router.get("/live", response_model=HealthResponse, summary="Liveness probe")
def health(settings: SettingsDep) -> HealthResponse:
    return _base_health(settings)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe (checks dependencies)",
    responses={503: {"description": "One or more dependencies are unavailable"}},
)
def readiness(settings: SettingsDep) -> JSONResponse:
    components = [
        ComponentStatus(
            name="database",
            healthy=check_database_connection(settings),
            detail=settings.safe_database_url,
        ),
        ComponentStatus(
            name="racing_api_credentials",
            healthy=settings.racing_api_configured,
            detail="credentials present" if settings.racing_api_configured else "credentials missing",
        ),
    ]
    ready = all(component.healthy for component in components)
    body = ReadinessResponse(
        status="ready" if ready else "not_ready",
        components=components,
        timestamp=utcnow().isoformat(),
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=body.model_dump(),
    )


@router.get("/db", summary="Database connectivity probe")
def database_health(settings: SettingsDep) -> JSONResponse:
    healthy = check_database_connection(settings)
    return JSONResponse(
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "database": "up" if healthy else "down",
            "url": settings.safe_database_url,
            "timestamp": utcnow().isoformat(),
        },
    )


@router.get("/version", response_model=VersionResponse, summary="Build and runtime metadata")
def version(settings: SettingsDep) -> VersionResponse:
    return VersionResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        python_version=platform.python_version(),
        platform=platform.platform(),
    )


__all__ = ["router"]
