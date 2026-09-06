"""Logging layer tests: channel routing, correlation ids, JSON rendering."""

from __future__ import annotations

import json
import logging

import pytest

from backend.utils.logging import (
    CHANNELS,
    JsonFormatter,
    bind_request_id,
    configure_logging,
    get_logger,
    log_file_path,
    reset_request_id,
)

pytestmark = pytest.mark.unit


def _read(path):
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_every_channel_gets_its_own_file(settings):
    for channel in CHANNELS:
        get_logger("test", channel=channel).info(f"hello from {channel}")

    for channel in CHANNELS:
        path = log_file_path(channel, settings)
        assert path.exists(), f"{path} was not created"
        assert f"hello from {channel}" in _read(path)


def test_channels_do_not_bleed_into_each_other(settings):
    get_logger("isolation", channel="pipeline").info("pipeline-only-marker")

    assert "pipeline-only-marker" in _read(log_file_path("pipeline", settings))
    assert "pipeline-only-marker" not in _read(log_file_path("model", settings))
    assert "pipeline-only-marker" not in _read(log_file_path("api", settings))


def test_errors_are_mirrored_into_error_log(settings):
    get_logger("errors", channel="model").error("model-blew-up")

    assert "model-blew-up" in _read(settings.log_dir / "error.log")
    assert "model-blew-up" in _read(log_file_path("model", settings))


def test_unknown_channel_is_rejected():
    with pytest.raises(ValueError, match="unknown log channel"):
        get_logger("test", channel="nope")  # type: ignore[arg-type]


def test_logger_name_strips_the_backend_prefix():
    logger = get_logger("backend.data_pipeline.collectors", channel="pipeline")
    assert logger.name == "hqp.pipeline.data_pipeline.collectors"


def test_logger_name_does_not_repeat_the_channel():
    """``backend.api.middleware`` must not become ``hqp.api.api.middleware``."""
    assert get_logger("backend.api.middleware", channel="api").name == "hqp.api.middleware"
    assert get_logger("backend.ml.train", channel="model").name == "hqp.model.ml.train"


def test_request_id_is_attached_to_records(settings):
    request_id = bind_request_id("abc123def456")
    try:
        get_logger("corr", channel="api").info("correlated message")
    finally:
        reset_request_id()

    contents = _read(log_file_path("api", settings))
    assert request_id in contents


def test_bind_request_id_generates_one_when_absent():
    generated = bind_request_id()
    try:
        assert len(generated) == 12
    finally:
        reset_request_id()


def test_json_formatter_emits_valid_structured_output():
    record = logging.LogRecord(
        name="hqp.pipeline.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="ingested %d races",
        args=(7,),
        exc_info=None,
    )
    record.race_date = "2026-08-10"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "ingested 7 races"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "hqp.pipeline.test"
    assert payload["line"] == 42
    assert payload["race_date"] == "2026-08-10"
    assert "timestamp" in payload


def test_json_formatter_includes_the_traceback():
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            name="hqp.api.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="failed",
            args=(),
            exc_info=__import__("sys").exc_info(),
        )
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_is_idempotent(settings):
    configure_logging(settings)
    configure_logging(settings)  # must not raise or duplicate handlers

    handlers = logging.getLogger("hqp.api").handlers
    assert len(handlers) == len({id(h) for h in handlers})


# ---------------------------------------------------------------------------
# safe_extra
# ---------------------------------------------------------------------------
def test_reserved_extra_key_would_crash_stdlib_logging():
    """Documents the hazard that :func:`safe_extra` exists to prevent.

    ``logging`` raises if an ``extra`` key shadows a LogRecord attribute, and
    ``created`` is an entirely natural key for a domain dictionary (an import
    summary really does count things it created). The crash lands on the
    *success* path, which is exactly where it is least likely to be noticed.
    """
    from backend.utils.logging import safe_extra

    logger = get_logger("hazard", channel="pipeline")

    with pytest.raises(KeyError, match="created"):
        logger.info("boom", extra={"created": 5})

    logger.info("fine", extra=safe_extra({"created": 5}))


def test_safe_extra_renames_only_colliding_keys():
    from backend.utils.logging import safe_extra

    cleaned = safe_extra({"created": 1, "name": "x", "module": "m", "race_count": 42})

    assert cleaned == {"ctx_created": 1, "ctx_name": "x", "ctx_module": "m", "race_count": 42}


def test_safe_extra_survives_a_full_import_summary(settings):
    from backend.data_pipeline import ImportSummary
    from backend.utils.logging import safe_extra

    summary = ImportSummary()
    summary.races_created = 3
    logger = get_logger("summary", channel="pipeline")

    for section in summary.as_dict().values():
        if isinstance(section, dict):
            logger.info("section", extra=safe_extra(section))
