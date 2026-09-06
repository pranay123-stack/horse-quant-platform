"""Translation of platform exceptions into HTTP responses.

Handlers registered here are the *only* place the platform converts an
exception into a status code, so every error response has an identical shape::

    {
      "error": "external_api_error",
      "message": "Racing API returned 503",
      "details": {"endpoint": "/v1/racecards"},
      "request_id": "8f2c1a09b3de"
    }
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from backend.utils.exceptions import PlatformError
from backend.utils.logging import get_logger, request_id_var

logger = get_logger(__name__, channel="api")

#: Starlette's exception-handler signature.
ExceptionHandler = Callable[[Request, Exception], Response]

#: Spelled out rather than imported: Starlette renamed the 422 constant
#: (``..._UNPROCESSABLE_ENTITY`` -> ``..._UNPROCESSABLE_CONTENT``) and emits a
#: DeprecationWarning for the old name. The number is stable; the name is not.
HTTP_422_UNPROCESSABLE = 422


def _payload(error: str, message: str, details: dict | None = None) -> dict:
    return {
        "error": error,
        "message": message,
        "details": details or {},
        "request_id": request_id_var.get(),
    }


async def platform_error_handler(request: Request, exc: PlatformError) -> JSONResponse:
    log = logger.warning if exc.status_code < 500 else logger.error
    log(
        "platform error",
        extra={
            "error_code": exc.error_code,
            "path": request.url.path,
            "status_code": exc.status_code,
            "details": exc.details,
        },
        exc_info=exc.status_code >= 500,
    )
    return JSONResponse(
        status_code=exc.status_code, content=_payload(exc.error_code, exc.message, exc.details)
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info("request validation failed", extra={"path": request.url.path, "errors": exc.errors()})
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE,
        content=_payload("validation_error", "Request payload failed validation", {"errors": exc.errors()}),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_payload("http_error", str(exc.detail)),
        headers=getattr(exc, "headers", None),
    )


async def database_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.error("database error", extra={"path": request.url.path}, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=_payload("database_error", "Database operation failed"),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.critical("unhandled exception", extra={"path": request.url.path}, exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_payload("internal_error", "An unexpected error occurred"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire the handlers above onto the application.

    Starlette types its registry as ``Callable[[Request, Exception], ...]``; our
    handlers take the precise exception type they are registered for, which is
    correct at runtime but narrower than the declared signature, hence the cast.
    """
    handlers: list[tuple[type[Exception], Any]] = [
        (PlatformError, platform_error_handler),
        (RequestValidationError, validation_error_handler),
        (StarletteHTTPException, http_exception_handler),
        (SQLAlchemyError, database_error_handler),
        (Exception, unhandled_exception_handler),
    ]
    for exception_type, handler in handlers:
        app.add_exception_handler(exception_type, cast("ExceptionHandler", handler))


__all__ = ["register_exception_handlers"]
