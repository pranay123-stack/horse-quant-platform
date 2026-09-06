"""Jockey form features.

Rolling over the last *N* rides rather than a calendar window, because a jockey's
relevant form is measured in opportunities, not days: a rider with 30 rides in a
fortnight and one with 30 rides in three months have each given us the same
amount of evidence.

ROI is included alongside strike rate because they answer different questions. A
jockey who wins 20% of rides on odds-on favourites is not adding value; one who
wins 8% at an average of 20/1 is. Strike rate alone cannot tell them apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.features.base import EVENT_TIME, FeatureConfig, as_of_join
from backend.features.horse_features import PRIOR_PLACE_RATE, PRIOR_WIN_RATE, _shrink

JOCKEY_FEATURE_COLUMNS: tuple[str, ...] = (
    "jky_rides_window",
    "jky_wins_window",
    "jky_strike_rate",
    "jky_place_rate",
    "jky_roi",
    "jky_career_rides",
    "jky_career_strike_rate",
    "jky_course_rides",
    "jky_course_strike_rate",
)


def level_stake_profit(is_winner: pd.Series, starting_price: pd.Series) -> pd.Series:
    """Profit from a 1-unit win bet at starting price.

    A winner returns ``sp - 1`` (stake back plus profit, net of the stake); a
    loser returns ``-1``. Runs with no recorded price contribute nothing rather
    than a fabricated zero, so ROI is computed only over priced rides.
    """
    price = pd.to_numeric(starting_price, errors="coerce")
    profit = pd.Series(np.nan, index=is_winner.index, dtype="float64")
    priced = price.notna() & (price > 1)
    profit[priced & is_winner.astype(bool)] = (price - 1.0)[priced & is_winner.astype(bool)]
    profit[priced & ~is_winner.astype(bool)] = -1.0
    return profit


def _roll(frame: pd.DataFrame, keys: list[str], column: str, window: int, how: str) -> pd.Series:
    grouped = frame.groupby(keys, sort=False, observed=True)[column]
    return grouped.transform(lambda s: getattr(s.rolling(window, min_periods=1), how)())


def build_jockey_state(history: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Per-ride cumulative jockey state, inclusive of each ride."""
    config = config or FeatureConfig()
    columns = ["jockey_id", EVENT_TIME, *JOCKEY_FEATURE_COLUMNS]
    if history.empty or "jockey_id" not in history.columns:
        return pd.DataFrame(columns=columns)

    frame = history[history["jockey_id"].notna()].sort_values(EVENT_TIME, kind="stable").copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    keys = ["jockey_id"]
    window = config.jockey_rides_window

    frame["win_flag"] = frame["is_winner"].astype(float)
    frame["place_flag"] = frame["finishing_position"].le(config.place_threshold).fillna(False).astype(float)
    frame["profit"] = level_stake_profit(frame["is_winner"], frame["starting_price"])

    rides = frame.groupby(keys, sort=False, observed=True).cumcount() + 1
    window_rides = rides.clip(upper=window)
    window_wins = _roll(frame, keys, "win_flag", window, "sum")
    window_places = _roll(frame, keys, "place_flag", window, "sum")

    state = pd.DataFrame(
        {
            "jockey_id": frame["jockey_id"],
            EVENT_TIME: frame[EVENT_TIME],
            "jky_rides_window": window_rides,
            "jky_wins_window": window_wins,
            "jky_strike_rate": _shrink(window_wins, window_rides, PRIOR_WIN_RATE),
            "jky_place_rate": _shrink(window_places, window_rides, PRIOR_PLACE_RATE),
            "jky_roi": _roll(frame, keys, "profit", window, "mean"),
            "jky_career_rides": rides,
            "jky_career_strike_rate": _shrink(
                frame.groupby(keys, sort=False, observed=True)["win_flag"].cumsum(), rides, PRIOR_WIN_RATE
            ),
        }
    )
    return state


def build_jockey_course_state(history: pd.DataFrame) -> pd.DataFrame:
    """Cumulative record for a jockey at one course.

    Course form is a real effect in UK racing — undulating, tight or stiff tracks
    reward riders who know them — but the samples are thin, hence shrinkage.
    """
    columns = ["jockey_id", "course_id", EVENT_TIME, "jky_course_rides", "jky_course_strike_rate"]
    if history.empty or "jockey_id" not in history.columns:
        return pd.DataFrame(columns=columns)

    frame = history[history["jockey_id"].notna() & history["course_id"].notna()]
    frame = frame.sort_values(EVENT_TIME, kind="stable").copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    keys = ["jockey_id", "course_id"]
    frame["win_flag"] = frame["is_winner"].astype(float)
    rides = frame.groupby(keys, sort=False, observed=True).cumcount() + 1
    wins = frame.groupby(keys, sort=False, observed=True)["win_flag"].cumsum()

    return pd.DataFrame(
        {
            "jockey_id": frame["jockey_id"],
            "course_id": frame["course_id"],
            EVENT_TIME: frame[EVENT_TIME],
            "jky_course_rides": rides,
            "jky_course_strike_rate": _shrink(wins, rides, PRIOR_WIN_RATE),
        }
    )


def build_jockey_features(
    targets: pd.DataFrame, history: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.DataFrame:
    config = config or FeatureConfig()

    frame = as_of_join(
        targets,
        build_jockey_state(history, config),
        by="jockey_id",
        columns=[
            "jky_rides_window",
            "jky_wins_window",
            "jky_strike_rate",
            "jky_place_rate",
            "jky_roi",
            "jky_career_rides",
            "jky_career_strike_rate",
        ],
    )
    frame = as_of_join(
        frame,
        build_jockey_course_state(history),
        by=["jockey_id", "course_id"],
        columns=["jky_course_rides", "jky_course_strike_rate"],
    )

    for column, default in (
        ("jky_rides_window", 0.0),
        ("jky_wins_window", 0.0),
        ("jky_career_rides", 0.0),
        ("jky_course_rides", 0.0),
        ("jky_strike_rate", PRIOR_WIN_RATE),
        ("jky_career_strike_rate", PRIOR_WIN_RATE),
        ("jky_course_strike_rate", PRIOR_WIN_RATE),
        ("jky_place_rate", PRIOR_PLACE_RATE),
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return frame


__all__ = [
    "JOCKEY_FEATURE_COLUMNS",
    "build_jockey_course_state",
    "build_jockey_features",
    "build_jockey_state",
    "level_stake_profit",
]
