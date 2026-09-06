"""HTTP middleware: correlation ids and access logging.

Every request is stamped with an ``X-Request-ID`` (accepted from the client when
supplied, generated otherwise). The id is stored in a context variable, so it
appears on *every* log line emitted while handling that request -- including
lines from the ETL or model layers -- which is what makes a production incident
traceable end to end.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from backend.utils.logging import bind_request_id, get_logger, reset_request_id

logger = get_logger(__name__, channel="api")

REQUEST_ID_HEADER = "X-Request-ID"
RESPONSE_TIME_HEADER = "X-Process-Time-Ms"

#: Endpoints that would otherwise flood the access log.
_QUIET_PATHS = frozenset({"/health", "/health/live", "/metrics", "/favicon.ico"})


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = bind_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        started = time.perf_counter()

        # Everything that logs must run *inside* the try, because the finally
        # clears the correlation id -- an access-log line emitted after it would
        # be stamped "-" and become untraceable.
        try:
            response = await call_next(request)

            elapsed_ms = (time.perf_counter() - started) * 1000
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers[RESPONSE_TIME_HEADER] = f"{elapsed_ms:.2f}"

            if request.url.path not in _QUIET_PATHS:
                logger.info(
                    "request completed",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration_ms": round(elapsed_ms, 2),
                        "client": request.client.host if request.client else None,
                    },
                )
            return response
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(elapsed_ms, 2),
                },
            )
            raise
        finally:
            reset_request_id()


__all__ = ["REQUEST_ID_HEADER", "RESPONSE_TIME_HEADER", "RequestContextMiddleware"]
