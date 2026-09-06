"""Version 1 of the public HTTP API.

Routers are registered here as each phase lands:

* Phase 1  -- ``health``, ``meta``                       (done)
* Phase 2  -- ``races``, ``horses``, ``odds``            (done)
* Phase 7  -- ``predictions`` (today's signals, model status, performance)
* Phase 8  -- ``backtests``
* Phase 10 -- ``auth``, ``users``
"""

from fastapi import APIRouter

from backend.api.v1 import health, horses, meta, odds, predictions, races

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(meta.router)
api_router.include_router(races.router)
api_router.include_router(horses.router)
api_router.include_router(odds.router)
api_router.include_router(predictions.router)

__all__ = ["api_router"]
