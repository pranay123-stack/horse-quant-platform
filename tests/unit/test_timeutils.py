"""Date/time helper tests -- these guard against look-ahead bias later."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from backend.utils.timeutils import (
    UK_TZ,
    date_range,
    days_between,
    format_date,
    parse_date,
    to_uk,
    to_utc,
    utcnow,
)

pytestmark = pytest.mark.unit


def test_utcnow_is_timezone_aware():
    now = utcnow()
    assert now.tzinfo is not None
    assert now.utcoffset().total_seconds() == 0


def test_naive_datetimes_are_interpreted_as_uk_local():
    # 2026-06-01 is British Summer Time (UTC+1).
    naive = datetime(2026, 6, 1, 14, 30)
    assert to_utc(naive) == datetime(2026, 6, 1, 13, 30, tzinfo=UTC)


def test_to_uk_converts_from_utc():
    instant = datetime(2026, 6, 1, 13, 30, tzinfo=UTC)
    local = to_uk(instant)
    assert local.hour == 14
    assert local.tzinfo == UK_TZ


def test_parse_and_format_round_trip():
    assert parse_date("2026-08-10") == date(2026, 8, 10)
    assert parse_date(datetime(2026, 8, 10, 15, 0)) == date(2026, 8, 10)
    assert format_date(date(2026, 8, 10)) == "2026-08-10"


def test_parse_date_rejects_garbage():
    with pytest.raises(ValueError):
        parse_date("10/08/2026")


def test_date_range_is_inclusive_by_default():
    days = list(date_range("2026-08-01", "2026-08-04"))
    assert days == [date(2026, 8, d) for d in (1, 2, 3, 4)]


def test_date_range_exclusive():
    days = list(date_range("2026-08-01", "2026-08-04", inclusive=False))
    assert days[-1] == date(2026, 8, 3)


def test_date_range_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="is after end"):
        list(date_range("2026-08-04", "2026-08-01"))


def test_days_between_handles_mixed_types():
    assert days_between(date(2026, 8, 1), datetime(2026, 8, 15, 12, 0)) == 14
