"""Race-context features and within-race relative measures.

Two kinds of feature live here, and the second matters more than it looks.

**Absolute context** — distance, class, going, prize money, field size. All are
published on the racecard, so all are safe.

**Relative position within the field** — where this runner's weight, rating and
draw sit *against the horses it is actually running against*. A rating of 85 is
meaningless in isolation: it is top-weight in one race and bottom of the handicap
in another. Racing is a ranking problem, so most of the signal is cross-sectional.

Every value used here comes from the declaration, so computing a rank across the
field uses no information from the race's outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.features.base import YARDS_PER_FURLONG, going_value, numeric_column

RACE_FEATURE_COLUMNS: tuple[str, ...] = (
    "race_field_size",
    "race_distance_furlongs",
    "race_going_value",
    "race_class_filled",
    "race_is_jumps",
    "race_is_handicap",
    "race_prize_money",
    "runner_age",
    "runner_weight_lbs",
    "runner_official_rating",
    "runner_draw",
    "runner_weight_rank",
    "runner_or_rank",
    "runner_rpr_rank",
    "runner_draw_pct",
    "runner_weight_vs_field",
    "runner_or_vs_field",
)

#: Class 5 is the modal UK race class; used when class is absent (common in jumps).
DEFAULT_RACE_CLASS = 5

_JUMPS_TYPES = ("chase", "hurdle", "nh flat", "bumper")


def _pct_rank(frame: pd.DataFrame, column: str, *, higher_is_better: bool = True) -> pd.Series:
    """Percentile rank within the race, where 1.0 is the top of the field.

    Expressed as ``higher_is_better`` rather than pandas' ``ascending``: the
    latter maps the *smallest* value to 1.0 when set to ``False``, which is the
    opposite of what "descending rank" suggests and is an easy way to ship a
    silently inverted feature.
    """
    if column not in frame:
        # np.nan, not pd.NA: constructing a float64 Series from pd.NA raises.
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return (
        frame.groupby("race_id", sort=False, observed=True)[column]
        .rank(pct=True, ascending=higher_is_better, na_option="keep")
        .astype("float64")
    )


def build_race_features(targets: pd.DataFrame) -> pd.DataFrame:
    """Attach race-context and within-race relative features."""
    frame = targets.copy()
    if frame.empty:
        for column in RACE_FEATURE_COLUMNS:
            frame[column] = pd.Series(dtype="float64")
        return frame

    # Actual runner count beats the declared field size, which can be stale once
    # non-runners are withdrawn.
    frame["race_field_size"] = frame.groupby("race_id", sort=False, observed=True)["horse_id"].transform(
        "size"
    )

    frame["race_distance_furlongs"] = numeric_column(frame, "distance_yards") / YARDS_PER_FURLONG
    frame["race_going_value"] = (
        frame["going"].map(going_value)
        if "going" in frame
        else pd.Series(np.nan, index=frame.index, dtype="float64")
    )
    frame["race_class_filled"] = numeric_column(frame, "race_class").fillna(DEFAULT_RACE_CLASS)

    race_type = frame.get("race_type", pd.Series("", index=frame.index)).fillna("").str.lower()
    frame["race_is_jumps"] = race_type.apply(lambda text: int(any(kind in text for kind in _JUMPS_TYPES)))
    race_name = frame.get("race_name", pd.Series("", index=frame.index))
    frame["race_is_handicap"] = (
        race_name.fillna("").str.lower().str.contains("handicap").astype(int) if "race_name" in frame else 0
    )
    frame["race_prize_money"] = numeric_column(frame, "prize_money")

    frame["runner_age"] = numeric_column(frame, "age")
    frame["runner_weight_lbs"] = numeric_column(frame, "weight_lbs")
    frame["runner_official_rating"] = numeric_column(frame, "official_rating")
    frame["runner_draw"] = numeric_column(frame, "draw")

    # 1.0 = top weight / highest rated / best RPR in the field.
    frame["runner_weight_rank"] = _pct_rank(frame, "runner_weight_lbs")
    frame["runner_or_rank"] = _pct_rank(frame, "runner_official_rating")
    frame["runner_rpr_rank"] = _pct_rank(frame, "rpr")

    # Draw as a fraction of the field: comparable across a 6-runner and a
    # 20-runner race, where a raw stall number is not.
    frame["runner_draw_pct"] = frame["runner_draw"] / frame["race_field_size"].where(
        frame["race_field_size"] > 0
    )

    for source, target in (
        ("runner_weight_lbs", "runner_weight_vs_field"),
        ("runner_official_rating", "runner_or_vs_field"),
    ):
        field_mean = frame.groupby("race_id", sort=False, observed=True)[source].transform("mean")
        frame[target] = frame[source] - field_mean

    return frame


__all__ = ["DEFAULT_RACE_CLASS", "RACE_FEATURE_COLUMNS", "build_race_features"]
