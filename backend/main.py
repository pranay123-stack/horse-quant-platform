"""FastAPI application factory and ASGI entrypoint.

Run locally::

    uvicorn backend.main:app --reload

or via the console entrypoint::

    python -m backend.main
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.errors import register_exception_handlers
from backend.api.middleware import RequestContextMiddleware
from backend.api.v1 import api_router
from backend.database.session import check_database_connection, dispose_engine
from backend.utils.config import PROJECT_ROOT, Settings, get_settings
from backend.utils.logging import configure_logging, get_logger

logger = get_logger(__name__, channel="api")

DESCRIPTION = """
Quantitative betting platform for UK horse racing.

* **Data** -- ingests racecards, results and odds from The Racing API
* **Features** -- form, speed, going/course suitability, jockey & trainer effects, market signals
* **Models** -- calibrated win-probability models (Logistic Regression, XGBoost, LightGBM)
* **Strategy** -- expected-value filter with flat and fractional-Kelly staking
* **Backtesting** -- ROI, drawdown, Sharpe and strike-rate reporting
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hooks."""
    settings: Settings = app.state.settings
    settings.ensure_directories()

    logger.info(
        "starting application",
        extra={
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "database": settings.safe_database_url,
        },
    )

    if check_database_connection(settings):
        logger.info("database connection verified")
    else:
        # Not fatal: the API must still serve /health so an operator can see why
        # the container is unhealthy instead of watching it crash-loop.
        logger.warning("database unreachable at startup; readiness will report not_ready")

    if not settings.racing_api_configured:
        logger.warning("Racing API credentials are not configured; ingestion endpoints will fail")

    yield

    logger.info("shutting down application")
    dispose_engine()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fully wired FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=settings.app_version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )

    register_exception_handlers(app)

    app.include_router(api_router, prefix=settings.api_v1_prefix)
    # Unprefixed aliases so container orchestrators can probe a stable path.
    app.include_router(api_router, include_in_schema=False)

    # The dashboard is a single static file that talks to the API above. No
    # build step, no bundler: one page that a user can open in the morning.
    dashboard = PROJECT_ROOT / "frontend" / "dashboard"
    if (dashboard / "index.html").exists():
        app.mount("/dashboard", StaticFiles(directory=dashboard, html=True), name="dashboard")

    @app.get("/", tags=["meta"], summary="Service banner")
    def root() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "docs": "/docs" if not settings.is_production else "disabled",
            "api": settings.api_v1_prefix,
            "dashboard": "/dashboard",
        }

    return app


app = create_app()


def main() -> None:  # pragma: no cover - manual entrypoint
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "local",
        workers=settings.api_workers if settings.environment != "local" else 1,
        log_config=None,  # our own dictConfig owns logging
    )


if __name__ == "__main__":  # pragma: no cover
    main()
