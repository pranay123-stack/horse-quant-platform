"""Trainer form features.

Two windows, because they measure different things:

* **Last 30 days** — a *calendar* window. Trainer form is famously streaky:
  yards run hot when the horses are well and cold when a virus goes through, and
  that is a property of the last few weeks, not the last N runners.
* **Last 100 runners** — a *sample-size* window, giving a stable baseline that a
  quiet fortnight cannot distort.

The trainer-jockey combination is tracked separately. Bookmakers price it, and
a yard's first-choice rider getting the call is genuine information about which
of a trainer's runners is expected to go well.

.. note::
   The calendar window is evaluated as of the trainer's **most recent prior
   runner**, not as of the target race. For an active yard running most days the
   two coincide; for a dormant one the window is stale. This is a documented
   limitation rather than a leak — staleness can only ever *withhold* recent
   information, never add future information.
"""

from __future__ import annotations

import pandas as pd

from backend.features.base import EVENT_TIME, FeatureConfig, as_of_join
from backend.features.horse_features import PRIOR_PLACE_RATE, PRIOR_WIN_RATE, _shrink
from backend.features.jockey_features import level_stake_profit

TRAINER_FEATURE_COLUMNS: tuple[str, ...] = (
    "trn_runners_30d",
    "trn_wins_30d",
    "trn_win_rate_30d",
    "trn_place_rate_30d",
    "trn_roi_30d",
    "trn_runners_window",
    "trn_win_rate_window",
    "trn_roi_window",
    "trn_career_runners",
    "trn_jky_runs",
    "trn_jky_win_rate",
)


def _roll(frame: pd.DataFrame, keys: list[str], column: str, window: int, how: str) -> pd.Series:
    grouped = frame.groupby(keys, sort=False, observed=True)[column]
    return grouped.transform(lambda s: getattr(s.rolling(window, min_periods=1), how)())


def _roll_time(frame: pd.DataFrame, keys: list[str], column: str, window: str, how: str) -> pd.Series:
    """Time-based rolling within each entity, inclusive of the current row.

    Written as an explicit per-group loop rather than
    ``groupby(...).rolling(...)``: the grouped form returns a frame indexed by
    ``(key, time)`` whose row order does not match the input, and silently
    misaligning a feature with its row is exactly the class of bug this module
    exists to prevent.
    """
    result = pd.Series(index=frame.index, dtype="float64")
    for _, group in frame.groupby(keys, sort=False, observed=True):
        series = pd.Series(
            pd.to_numeric(group[column], errors="coerce").to_numpy(),
            index=pd.DatetimeIndex(group[EVENT_TIME]),
        )
        rolled = getattr(series.rolling(window, min_periods=1), how)()
        result.loc[group.index] = rolled.to_numpy()
    return result


def build_trainer_state(history: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Per-runner cumulative trainer state, inclusive of each runner."""
    config = config or FeatureConfig()
    columns = ["trainer_id", EVENT_TIME, *TRAINER_FEATURE_COLUMNS]
    if history.empty or "trainer_id" not in history.columns:
        return pd.DataFrame(columns=columns)

    frame = history[history["trainer_id"].notna()].sort_values(EVENT_TIME, kind="stable").copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    keys = ["trainer_id"]
    days_window = f"{config.trainer_days_window}D"
    runner_window = config.trainer_runner_window

    frame["win_flag"] = frame["is_winner"].astype(float)
    frame["place_flag"] = frame["finishing_position"].le(config.place_threshold).fillna(False).astype(float)
    frame["profit"] = level_stake_profit(frame["is_winner"], frame["starting_price"])
    frame["one"] = 1.0

    runners_30d = _roll_time(frame, keys, "one", days_window, "sum")
    wins_30d = _roll_time(frame, keys, "win_flag", days_window, "sum")
    places_30d = _roll_time(frame, keys, "place_flag", days_window, "sum")

    career = frame.groupby(keys, sort=False, observed=True).cumcount() + 1
    window_runners = career.clip(upper=runner_window)
    window_wins = _roll(frame, keys, "win_flag", runner_window, "sum")

    return pd.DataFrame(
        {
            "trainer_id": frame["trainer_id"],
            EVENT_TIME: frame[EVENT_TIME],
            "trn_runners_30d": runners_30d,
            "trn_wins_30d": wins_30d,
            "trn_win_rate_30d": _shrink(wins_30d, runners_30d, PRIOR_WIN_RATE),
            "trn_place_rate_30d": _shrink(places_30d, runners_30d, PRIOR_PLACE_RATE),
            "trn_roi_30d": _roll_time(frame, keys, "profit", days_window, "mean"),
            "trn_runners_window": window_runners,
            "trn_win_rate_window": _shrink(window_wins, window_runners, PRIOR_WIN_RATE),
            "trn_roi_window": _roll(frame, keys, "profit", runner_window, "mean"),
            "trn_career_runners": career,
        }
    )


def build_trainer_jockey_state(history: pd.DataFrame) -> pd.DataFrame:
    """Cumulative record for a trainer-jockey pairing."""
    columns = ["trainer_id", "jockey_id", EVENT_TIME, "trn_jky_runs", "trn_jky_win_rate"]
    if history.empty or "trainer_id" not in history.columns:
        return pd.DataFrame(columns=columns)

    frame = history[history["trainer_id"].notna() & history["jockey_id"].notna()]
    frame = frame.sort_values(EVENT_TIME, kind="stable").copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    keys = ["trainer_id", "jockey_id"]
    frame["win_flag"] = frame["is_winner"].astype(float)
    runs = frame.groupby(keys, sort=False, observed=True).cumcount() + 1
    wins = frame.groupby(keys, sort=False, observed=True)["win_flag"].cumsum()

    return pd.DataFrame(
        {
            "trainer_id": frame["trainer_id"],
            "jockey_id": frame["jockey_id"],
            EVENT_TIME: frame[EVENT_TIME],
            "trn_jky_runs": runs,
            "trn_jky_win_rate": _shrink(wins, runs, PRIOR_WIN_RATE),
        }
    )


def build_trainer_features(
    targets: pd.DataFrame, history: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.DataFrame:
    config = config or FeatureConfig()

    frame = as_of_join(
        targets,
        build_trainer_state(history, config),
        by="trainer_id",
        columns=[
            "trn_runners_30d",
            "trn_wins_30d",
            "trn_win_rate_30d",
            "trn_place_rate_30d",
            "trn_roi_30d",
            "trn_runners_window",
            "trn_win_rate_window",
            "trn_roi_window",
            "trn_career_runners",
        ],
    )
    frame = as_of_join(
        frame,
        build_trainer_jockey_state(history),
        by=["trainer_id", "jockey_id"],
        columns=["trn_jky_runs", "trn_jky_win_rate"],
    )

    for column, default in (
        ("trn_runners_30d", 0.0),
        ("trn_wins_30d", 0.0),
        ("trn_runners_window", 0.0),
        ("trn_career_runners", 0.0),
        ("trn_jky_runs", 0.0),
        ("trn_win_rate_30d", PRIOR_WIN_RATE),
        ("trn_win_rate_window", PRIOR_WIN_RATE),
        ("trn_jky_win_rate", PRIOR_WIN_RATE),
        ("trn_place_rate_30d", PRIOR_PLACE_RATE),
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return frame


__all__ = [
    "TRAINER_FEATURE_COLUMNS",
    "build_trainer_features",
    "build_trainer_jockey_state",
    "build_trainer_state",
]
