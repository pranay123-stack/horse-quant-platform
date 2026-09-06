"""Horse form, speed, class, distance, going and rest features.

Every aggregate here is computed **inclusive of each history row** and then
attached to a target race with :func:`~backend.features.base.as_of_join`, so a
horse's features for race *k* summarise runs 1…*k-1* and nothing else.

Conditional records ("what does this horse do on soft ground?") use the same
mechanism with a composite key, e.g. ``(horse_id, going_band)``. That keeps the
guarantee intact for slices as well as totals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.features.base import EVENT_TIME, FeatureConfig, as_of_join

#: Prior strength for shrinking a small-sample rate toward the population mean.
#: A horse that has won its only run over a trip is not a 100% distance
#: specialist; with k=6 that record is pulled most of the way back to average.
SHRINKAGE_K: float = 6.0

#: Base rates used as the shrinkage prior. Average field size in UK racing is
#: roughly 9, so a random runner wins about 11% and places about a third.
PRIOR_WIN_RATE: float = 0.11
PRIOR_PLACE_RATE: float = 0.33

HORSE_FEATURE_COLUMNS: tuple[str, ...] = (
    "horse_runs_before",
    "horse_wins_before",
    "horse_win_rate",
    "horse_place_rate",
    "horse_last1_position",
    "horse_last2_position",
    "horse_last3_position",
    "horse_form_last3_avg_pos",
    "horse_form_last5_avg_pos",
    "horse_career_avg_pos",
    "horse_consistency",
    "horse_avg_speed",
    "horse_best_speed",
    "horse_speed_trend",
    "horse_last_beaten_lengths",
    "horse_last_sp",
    "horse_prev_class",
    "horse_class_move",
    "horse_days_since_prev_run",
    "horse_distance_runs",
    "horse_distance_win_rate",
    "horse_distance_suitability",
    "horse_going_runs",
    "horse_going_win_rate",
    "horse_going_suitability",
    "horse_course_runs",
    "horse_course_win_rate",
)


def _shrink(successes: pd.Series, trials: pd.Series, prior: float, k: float = SHRINKAGE_K) -> pd.Series:
    """Empirical-Bayes shrinkage of a rate toward ``prior``.

    Without this, "1 win from 1 run" reads as a 100% strike rate and dominates
    any model that sees it. Shrinkage makes small samples say what they actually
    say: not much.
    """
    return (successes.fillna(0) + prior * k) / (trials.fillna(0) + k)


def _roll(frame: pd.DataFrame, keys: list[str], column: str, window: int, how: str) -> pd.Series:
    """Rolling aggregate within an entity, inclusive of the current row."""
    grouped = frame.groupby(keys, sort=False, observed=True)[column]
    return grouped.transform(lambda s: getattr(s.rolling(window, min_periods=1), how)())


def _expand(frame: pd.DataFrame, keys: list[str], column: str, how: str) -> pd.Series:
    grouped = frame.groupby(keys, sort=False, observed=True)[column]
    return grouped.transform(lambda s: getattr(s.expanding(min_periods=1), how)())


def build_horse_state(history: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Per-run cumulative horse state, inclusive of each run."""
    config = config or FeatureConfig()
    if history.empty:
        return pd.DataFrame(columns=["horse_id", EVENT_TIME, *HORSE_FEATURE_COLUMNS])

    frame = history.sort_values([EVENT_TIME], kind="stable").copy()
    keys = ["horse_id"]

    # A single "speed rating": Racing Post Rating first, Topspeed as fallback.
    # RPR is the more complete field in practice; both are on similar scales.
    frame["speed_rating"] = frame["rpr"].where(frame["rpr"].notna(), frame["topspeed"])
    frame["position"] = frame["finishing_position"]
    frame["win_flag"] = frame["is_winner"].astype(float)
    frame["place_flag"] = frame["position"].le(config.place_threshold).fillna(False).astype(float)

    state = pd.DataFrame({"horse_id": frame["horse_id"], EVENT_TIME: frame[EVENT_TIME]})

    runs = frame.groupby(keys, sort=False, observed=True).cumcount() + 1
    state["horse_runs_before"] = runs
    state["horse_wins_before"] = _expand(frame, keys, "win_flag", "sum")
    state["horse_win_rate"] = _shrink(state["horse_wins_before"], runs, PRIOR_WIN_RATE)
    state["horse_place_rate"] = _shrink(_expand(frame, keys, "place_flag", "sum"), runs, PRIOR_PLACE_RATE)

    # Most recent finishes: row k is the latest run, k-1 the one before it.
    grouped_position = frame.groupby(keys, sort=False, observed=True)["position"]
    state["horse_last1_position"] = frame["position"].to_numpy()
    state["horse_last2_position"] = grouped_position.shift(1).to_numpy()
    state["horse_last3_position"] = grouped_position.shift(2).to_numpy()

    state["horse_form_last3_avg_pos"] = _roll(frame, keys, "position", config.form_window_short, "mean")
    state["horse_form_last5_avg_pos"] = _roll(frame, keys, "position", config.form_window_long, "mean")
    state["horse_career_avg_pos"] = _expand(frame, keys, "position", "mean")

    # Consistency: low spread of recent finishing positions -> close to 1.
    #
    # Left NaN when a horse has fewer than two prior runs. An earlier version
    # filled it with the dataset-wide median spread, which is a genuine leak:
    # that median is computed over every row including races in the future, so
    # adding later races silently changed earlier feature values. Caught by
    # tests/unit/test_no_leakage.py. "Unknown" must be represented as unknown,
    # never as a statistic borrowed from data the row could not have seen.
    spread = _roll(frame, keys, "position", config.form_window_long, "std")
    state["horse_consistency"] = 1.0 / (1.0 + spread)

    state["horse_avg_speed"] = _roll(frame, keys, "speed_rating", config.speed_window, "mean")
    state["horse_best_speed"] = _expand(frame, keys, "speed_rating", "max")
    # Trend: recent form against the horse's own long-run level.
    state["horse_speed_trend"] = _roll(
        frame, keys, "speed_rating", config.form_window_short, "mean"
    ) - _expand(frame, keys, "speed_rating", "mean")

    state["horse_last_beaten_lengths"] = frame["beaten_lengths"].to_numpy()
    state["horse_last_sp"] = frame["starting_price"].to_numpy()
    state["horse_prev_class"] = frame["race_class"].to_numpy()
    #: Carried so the caller can compute rest days against the target's off-time.
    state["horse_last_run_time"] = frame[EVENT_TIME].to_numpy()

    return state


def build_horse_conditional_state(history: pd.DataFrame, band_column: str, prefix: str) -> pd.DataFrame:
    """Cumulative record within a slice — a going band, distance band or course."""
    if history.empty:
        return pd.DataFrame(
            columns=["horse_id", band_column, EVENT_TIME, f"{prefix}_runs", f"{prefix}_win_rate"]
        )

    frame = history.sort_values([EVENT_TIME], kind="stable").copy()
    frame["win_flag"] = frame["is_winner"].astype(float)
    keys = ["horse_id", band_column]

    runs = frame.groupby(keys, sort=False, observed=True).cumcount() + 1
    wins = _expand(frame, keys, "win_flag", "sum")

    return pd.DataFrame(
        {
            "horse_id": frame["horse_id"],
            band_column: frame[band_column],
            EVENT_TIME: frame[EVENT_TIME],
            f"{prefix}_runs": runs,
            f"{prefix}_wins": wins,
            f"{prefix}_win_rate": _shrink(wins, runs, PRIOR_WIN_RATE),
        }
    )


def build_horse_features(
    targets: pd.DataFrame, history: pd.DataFrame, config: FeatureConfig | None = None
) -> pd.DataFrame:
    """Attach every horse feature to ``targets``, point-in-time."""
    config = config or FeatureConfig()

    frame = as_of_join(
        targets,
        build_horse_state(history, config),
        by="horse_id",
        columns=[
            "horse_runs_before",
            "horse_wins_before",
            "horse_win_rate",
            "horse_place_rate",
            "horse_last1_position",
            "horse_last2_position",
            "horse_last3_position",
            "horse_form_last3_avg_pos",
            "horse_form_last5_avg_pos",
            "horse_career_avg_pos",
            "horse_consistency",
            "horse_avg_speed",
            "horse_best_speed",
            "horse_speed_trend",
            "horse_last_beaten_lengths",
            "horse_last_sp",
            "horse_prev_class",
            "horse_last_run_time",
        ],
    )

    # A horse with no prior runs is a debutant, not a horse with zero wins.
    frame["horse_runs_before"] = pd.to_numeric(frame["horse_runs_before"], errors="coerce").fillna(0)
    frame["horse_wins_before"] = pd.to_numeric(frame["horse_wins_before"], errors="coerce").fillna(0)
    frame["horse_is_debutant"] = (frame["horse_runs_before"] == 0).astype(int)

    # Class movement: positive means dropping in class, into easier company.
    frame["horse_class_move"] = frame["race_class"] - frame["horse_prev_class"]

    # Rest. Prefer our own computation over the API's ``days_since_last_run``,
    # which we cannot verify was correct at declaration time.
    if "horse_last_run_time" in frame.columns:
        last_run = pd.to_datetime(frame["horse_last_run_time"], utc=True, errors="coerce")
    else:
        last_run = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    frame["horse_days_since_prev_run"] = ((frame[EVENT_TIME] - last_run).dt.total_seconds() / 86400.0).round(
        2
    )
    frame = frame.drop(columns=["horse_last_run_time"], errors="ignore")

    conditionals = (
        ("distance_band", "horse_distance"),
        ("going_band", "horse_going"),
        ("course_id", "horse_course"),
    )
    for band_column, prefix in conditionals:
        if band_column not in frame.columns:
            continue
        state = build_horse_conditional_state(history, band_column, prefix)
        frame = as_of_join(
            frame,
            state,
            by=["horse_id", band_column],
            columns=[f"{prefix}_runs", f"{prefix}_wins", f"{prefix}_win_rate"],
        )
        # Coerce first: when the history is empty the as-of join leaves these as
        # object dtype, and ``fillna`` on object columns is deprecated.
        frame[f"{prefix}_runs"] = pd.to_numeric(frame[f"{prefix}_runs"], errors="coerce").fillna(0)
        frame[f"{prefix}_wins"] = pd.to_numeric(frame[f"{prefix}_wins"], errors="coerce").fillna(0)
        frame[f"{prefix}_win_rate"] = pd.to_numeric(frame[f"{prefix}_win_rate"], errors="coerce").fillna(
            PRIOR_WIN_RATE
        )

    # Suitability blends the shrunk strike rate with how much evidence backs it:
    # a strong record over many runs should outrank the same rate over two.
    for prefix in ("horse_distance", "horse_going"):
        evidence = np.log1p(frame[f"{prefix}_runs"]) / np.log1p(10)
        frame[f"{prefix}_suitability"] = frame[f"{prefix}_win_rate"] * evidence.clip(
            upper=1.0
        ) + PRIOR_WIN_RATE * (1 - evidence.clip(upper=1.0))

    return frame


__all__ = [
    "HORSE_FEATURE_COLUMNS",
    "PRIOR_PLACE_RATE",
    "PRIOR_WIN_RATE",
    "SHRINKAGE_K",
    "build_horse_conditional_state",
    "build_horse_features",
    "build_horse_state",
]
