"""Structured, channel-based logging for the whole platform.

Three log *channels* are defined, each with its own rotating file, exactly as
required by the architecture:

======================  ==================  ===============================
Channel                 File                Used by
======================  ==================  ===============================
``api``                 ``logs/api.log``    FastAPI, Racing API client
``pipeline``            ``logs/pipeline.log``  ETL / ingestion / features
``model``               ``logs/model.log``  Training, inference, backtests
======================  ==================  ===============================

Everything also flows to the console (and to the root logger), so a single
``docker logs`` gives the full picture while the per-channel files stay clean.

Usage::

    from backend.utils.logging import get_logger

    log = get_logger(__name__, channel="pipeline")
    log.info("ingested races", extra={"race_count": 42})
"""

from __future__ import annotations

import json
import logging
import logging.config
import sys
import uuid
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from backend.utils.config import Settings, get_settings

Channel = Literal["api", "pipeline", "model"]
CHANNELS: tuple[Channel, ...] = ("api", "pipeline", "model")

ROOT_LOGGER_NAME = "hqp"

#: Correlation id for the in-flight request / job. Set by middleware or by
#: :func:`bind_request_id`; automatically attached to every log record.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_configured = False

# Attributes present on every stdlib LogRecord -- anything else the caller
# passed via ``extra=`` is treated as structured context.
_RESERVED_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
        "request_id",
    }
)


class RequestIdFilter(logging.Filter):
    """Inject the current correlation id onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render records as one-line JSON -- ready for Loki/ELK/CloudWatch."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": getattr(record, "request_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console output with trailing structured context."""

    default_fmt = "%(asctime)s | %(levelname)-8s | %(request_id)s | %(name)s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.default_fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_")
        }
        if extras:
            rendered = " ".join(f"{k}={v!r}" for k, v in extras.items())
            base = f"{base} | {rendered}"
        return base


def _build_config(settings: Settings) -> dict[str, Any]:
    formatter = "json" if settings.log_json else "console"
    log_dir = settings.log_dir

    handlers: dict[str, Any] = {}
    if settings.log_to_console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": settings.log_level,
            "formatter": formatter,
            "filters": ["request_id"],
            "stream": sys.stdout,
        }

    for channel in CHANNELS:
        handlers[f"file_{channel}"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.log_level,
            "formatter": formatter,
            "filters": ["request_id"],
            "filename": str(log_dir / f"{channel}.log"),
            "maxBytes": settings.log_max_bytes,
            "backupCount": settings.log_backup_count,
            "encoding": "utf-8",
        }

    handlers["file_error"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "level": "ERROR",
        "formatter": formatter,
        "filters": ["request_id"],
        "filename": str(log_dir / "error.log"),
        "maxBytes": settings.log_max_bytes,
        "backupCount": settings.log_backup_count,
        "encoding": "utf-8",
    }

    console_handlers = ["console"] if settings.log_to_console else []

    loggers: dict[str, Any] = {
        ROOT_LOGGER_NAME: {
            "level": settings.log_level,
            "handlers": [*console_handlers, "file_error"],
            "propagate": False,
        },
    }
    for channel in CHANNELS:
        loggers[f"{ROOT_LOGGER_NAME}.{channel}"] = {
            "level": settings.log_level,
            "handlers": [f"file_{channel}"],
            # Propagate up to ``hqp`` so console + error file still see it.
            "propagate": True,
        }

    # Third-party noise control.
    loggers["uvicorn"] = {"level": settings.log_level, "handlers": console_handlers, "propagate": False}
    loggers["uvicorn.error"] = {"level": settings.log_level, "handlers": console_handlers, "propagate": False}
    loggers["uvicorn.access"] = {"level": "WARNING", "handlers": console_handlers, "propagate": False}
    loggers["sqlalchemy.engine"] = {"level": "INFO" if settings.db_echo else "WARNING", "propagate": True}
    loggers["httpx"] = {"level": "WARNING", "propagate": True}
    loggers["alembic"] = {"level": "INFO", "handlers": console_handlers, "propagate": False}

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"request_id": {"()": RequestIdFilter}},
        "formatters": {
            "console": {"()": ConsoleFormatter},
            "json": {"()": JsonFormatter},
        },
        "handlers": handlers,
        "loggers": loggers,
        "root": {"level": "WARNING", "handlers": console_handlers},
    }


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Install the logging configuration. Idempotent unless ``force=True``."""
    global _configured
    if _configured and not force:
        return

    settings = settings or get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(_build_config(settings))
    _configured = True

    get_logger(__name__).debug(
        "logging configured",
        extra={"level": settings.log_level, "json": settings.log_json, "log_dir": str(settings.log_dir)},
    )


def get_logger(name: str, channel: Channel = "api") -> logging.Logger:
    """Return a logger bound to ``channel``.

    ``name`` is normally ``__name__``; redundant prefixes are stripped so log
    lines read ``hqp.pipeline.collectors`` rather than
    ``hqp.pipeline.backend.data_pipeline.collectors``, and ``hqp.api.middleware``
    rather than ``hqp.api.api.middleware``.
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown log channel {channel!r}; expected one of {CHANNELS}")
    suffix = name.removeprefix("backend.").removeprefix("hqp.").removeprefix(f"{channel}.")
    logger_name = f"{ROOT_LOGGER_NAME}.{channel}"
    return logging.getLogger(f"{logger_name}.{suffix}" if suffix else logger_name)


def safe_extra(values: Mapping[str, Any], *, prefix: str = "ctx_") -> dict[str, Any]:
    """Make a dict safe to pass as ``logging`` ``extra=``.

    ``logging.Logger.makeRecord`` raises ``KeyError`` if an extra key collides
    with a built-in ``LogRecord`` attribute -- ``created``, ``name``, ``module``,
    ``message``, ``filename``… Those are ordinary words that appear naturally in
    domain dictionaries (an import summary really does have a ``created``
    count), so a raw ``extra=summary_dict`` is a crash waiting to happen *on the
    success path*, where it is least likely to be tested.

    Colliding keys are prefixed rather than dropped, so no context is lost.
    """
    cleaned: dict[str, Any] = {}
    for key, value in values.items():
        cleaned[f"{prefix}{key}" if key in _RESERVED_RECORD_ATTRS or key == "asctime" else key] = value
    return cleaned


def bind_request_id(request_id: str | None = None) -> str:
    """Set (or generate) the correlation id for the current context."""
    value = request_id or uuid.uuid4().hex[:12]
    request_id_var.set(value)
    return value


def reset_request_id() -> None:
    request_id_var.set("-")


def log_file_path(channel: Channel, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.log_dir / f"{channel}.log"


__all__ = [
    "CHANNELS",
    "Channel",
    "JsonFormatter",
    "bind_request_id",
    "configure_logging",
    "get_logger",
    "log_file_path",
    "request_id_var",
    "reset_request_id",
    "safe_extra",
]
