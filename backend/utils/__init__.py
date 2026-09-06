"""Cross-cutting utilities: configuration, logging, exceptions, helpers."""

from backend.utils.config import PROJECT_ROOT, Settings, get_settings
from backend.utils.exceptions import PlatformError
from backend.utils.logging import configure_logging, get_logger

__all__ = [
    "PROJECT_ROOT",
    "PlatformError",
    "Settings",
    "configure_logging",
    "get_logger",
    "get_settings",
]
