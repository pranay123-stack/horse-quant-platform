"""Time-based dataset splitting.

**Never split a racing dataset randomly.** Two reasons, and the second is the one
that ruins backtests quietly:

1. A random split puts races from the same day — often the *same race* — on both
   sides. The model then predicts a horse's result having seen its rivals'
   results in the same race.
2. Form features are built from history. A random split trains on features
   derived from races that sit in the test set, so information flows backwards
   across the boundary even when no row is literally duplicated.

Both inflate measured accuracy dramatically and neither shows up as an obvious
bug — the model simply looks good and then loses money. Every helper here splits
on the calendar, and :func:`assert_no_leakage_between` fails loudly if a
boundary is ever violated.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

DATE_COLUMN = "race_date"


class TemporalLeakageError(AssertionError):
    """Raised when a split would let future information into training."""


@dataclass(slots=True)
class DatasetSplit:
    """One train/test pair, with the boundary that produced it."""

    train: pd.DataFrame
    test: pd.DataFrame
    train_end: date
    test_start: date
    test_end: date
    label: str = ""

    @property
    def train_rows(self) -> int:
        return len(self.train)

    @property
    def test_rows(self) -> int:
        return len(self.test)

    def summary(self) -> dict[str, object]:
        return {
            "label": self.label,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def _dates(frame: pd.DataFrame, column: str = DATE_COLUMN) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"frame has no {column!r} column; a time split needs one")
    return pd.to_datetime(frame[column], errors="coerce").dt.date


def time_split(
    frame: pd.DataFrame,
    *,
    train_end: date,
    test_start: date | None = None,
    test_end: date | None = None,
    embargo_days: int = 0,
    column: str = DATE_COLUMN,
) -> DatasetSplit:
    """Split on a date. Training is ``<= train_end``, test is ``>= test_start``.

    ``embargo_days`` inserts a gap after ``train_end``. Rolling features span
    several weeks, so a race just after the boundary shares much of its history
    with the training period. The embargo removes that overlap; it costs a little
    data and buys an honest estimate.
    """
    dates = _dates(frame, column)
    if test_start is None:
        test_start = train_end + timedelta(days=1 + embargo_days)
    if test_start <= train_end:
        raise TemporalLeakageError(f"test_start {test_start} must be after train_end {train_end}")

    train = frame[dates <= train_end]
    mask = dates >= test_start
    if test_end is not None:
        mask &= dates <= test_end
    test = frame[mask]

    resolved_end = test_end or (dates[mask].max() if mask.any() else test_start)
    split = DatasetSplit(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        train_end=train_end,
        test_start=test_start,
        test_end=resolved_end,
        label=f"train<= {train_end} | test>= {test_start}",
    )
    assert_no_leakage_between(split, column=column)
    return split


def walk_forward_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    test_days: int = 30,
    min_train_days: int = 90,
    embargo_days: int = 0,
    column: str = DATE_COLUMN,
) -> list[DatasetSplit]:
    """Expanding-window walk-forward splits, oldest first.

    This is how a strategy would actually have been run: train on everything
    known up to a date, trade the next window, then roll forward. A single
    train/test split reports one draw from a noisy distribution; walk-forward
    shows whether an edge persists.
    """
    dates = _dates(frame, column).dropna()
    if dates.empty:
        return []

    first, last = dates.min(), dates.max()
    total_days = (last - first).days + 1
    if total_days < min_train_days + test_days:
        return []

    max_splits = (total_days - min_train_days) // test_days
    n_splits = max(0, min(n_splits, int(max_splits)))

    splits: list[DatasetSplit] = []
    for index in range(n_splits):
        # Newest window last: the final split always ends at the data's end.
        test_end = last - timedelta(days=test_days * (n_splits - index - 1))
        test_start = test_end - timedelta(days=test_days - 1)
        train_end = test_start - timedelta(days=1 + embargo_days)
        if (train_end - first).days + 1 < min_train_days:
            continue
        splits.append(
            time_split(
                frame,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                column=column,
            )
        )
    for position, split in enumerate(splits, start=1):
        split.label = f"fold {position}/{len(splits)}: test {split.test_start} → {split.test_end}"
    return splits


def assert_no_leakage_between(split: DatasetSplit, *, column: str = DATE_COLUMN) -> None:
    """Fail loudly if training data reaches into the test period, or races straddle it."""
    if split.train.empty or split.test.empty:
        return

    train_dates = _dates(split.train, column)
    test_dates = _dates(split.test, column)

    latest_train, earliest_test = train_dates.max(), test_dates.min()
    if latest_train >= earliest_test:
        raise TemporalLeakageError(f"training data reaches {latest_train}, test starts {earliest_test}")

    if "race_id" in split.train.columns and "race_id" in split.test.columns:
        shared = set(split.train["race_id"]) & set(split.test["race_id"])
        if shared:
            raise TemporalLeakageError(
                f"{len(shared)} race(s) appear in both train and test, e.g. {sorted(shared)[:3]}"
            )


def iter_race_groups(frame: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(race_id, runners)`` in chronological order.

    Used by the backtester, which must process races in the order they were run.
    """
    if frame.empty:
        return
    sort_columns = [column for column in (DATE_COLUMN, "off_time", "race_id") if column in frame.columns]
    ordered = frame.sort_values(sort_columns, kind="stable")
    for race_id, group in ordered.groupby("race_id", sort=False):
        yield str(race_id), group


__all__ = [
    "DATE_COLUMN",
    "DatasetSplit",
    "TemporalLeakageError",
    "assert_no_leakage_between",
    "iter_race_groups",
    "time_split",
    "walk_forward_splits",
]
