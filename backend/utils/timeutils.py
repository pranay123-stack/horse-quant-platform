"""Date/time helpers shared by the ingestion, feature and backtest layers.

UK racing operates on *race dates* in ``Europe/London`` local time, while every
timestamp we persist is timezone-aware UTC. Mixing the two silently is a classic
source of look-ahead bias in backtests, so the conversion lives in one place.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

UK_TZ = ZoneInfo("Europe/London")
UTC = UTC

ISO_DATE_FORMAT = "%Y-%m-%d"


def utcnow() -> datetime:
    """Timezone-aware current UTC time (never use ``datetime.utcnow()``)."""
    return datetime.now(tz=UTC)


def today_uk() -> date:
    """Today's *racing* date in UK local time."""
    return datetime.now(tz=UK_TZ).date()


def to_utc(value: datetime) -> datetime:
    """Normalise any datetime to aware UTC, assuming UK local when naive."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UK_TZ)
    return value.astimezone(UTC)


def to_uk(value: datetime) -> datetime:
    """Render an instant in UK local time (for display and race cards)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UK_TZ)


def parse_date(value: str | date | datetime) -> date:
    """Parse an ISO ``YYYY-MM-DD`` string (or pass through a date)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value.strip(), ISO_DATE_FORMAT).date()


def format_date(value: date | datetime) -> str:
    if isinstance(value, datetime):
        value = value.date()
    return value.strftime(ISO_DATE_FORMAT)


def date_range(start: str | date, end: str | date, *, inclusive: bool = True) -> Iterator[date]:
    """Yield each date from ``start`` to ``end``.

    Used by the historical backfill to page the Racing API one day at a time.
    """
    current = parse_date(start)
    last = parse_date(end)
    if current > last:
        raise ValueError(f"start {current} is after end {last}")
    step = timedelta(days=1)
    while current < last or (inclusive and current == last):
        yield current
        current += step


def days_between(earlier: date | datetime, later: date | datetime) -> int:
    """Whole days between two dates -- the basis of the ``rest_days`` feature."""
    a = earlier.date() if isinstance(earlier, datetime) else earlier
    b = later.date() if isinstance(later, datetime) else later
    return (b - a).days


__all__ = [
    "ISO_DATE_FORMAT",
    "UK_TZ",
    "UTC",
    "date_range",
    "days_between",
    "format_date",
    "parse_date",
    "to_uk",
    "to_utc",
    "today_uk",
    "utcnow",
]
