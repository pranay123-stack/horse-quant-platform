"""HTTP interface layer (FastAPI routers, middleware, error mapping)."""

from backend.api.v1 import api_router

__all__ = ["api_router"]
