"""Composite scores.

Raw features live on incompatible scales — a Racing Post Rating near 100, a
strike rate near 0.1, a finishing position near 4. Averaging them directly would
let whichever happens to have the biggest numbers dominate.

Every component is therefore converted to its **percentile rank within its own
race** before being combined. That does two useful things at once:

1. Puts everything on a common ``[0, 1]`` scale.
2. Makes each score *relative to the horses actually being beaten*. A speed
   figure of 95 is excellent in a Class 6 seller and moderate in a Group race;
   the percentile says which situation this is.

Ranking within a race uses only declaration-time values from the same race, so no
outcome information is involved.

These scores are an interpretable baseline, deliberately hand-weighted per the
Phase 3 specification. Phase 6 replaces them with a fitted model — at which point
they remain valuable as the benchmark that model has to beat.
"""

from __future__ import annotations

import pandas as pd

from backend.features.base import FeatureConfig, ScoreWeights

SCORE_COLUMNS: tuple[str, ...] = (
    "horse_form_score",
    "speed_score",
    "jockey_score",
    "trainer_score",
    "market_score",
    "distance_score",
    "going_score",
    "class_score",
)

#: Percentile assigned when a component is entirely missing for a runner.
#: Neutral rather than zero: absence of evidence must not read as evidence of
#: weakness, or every debutant would be scored as the worst horse in the race.
NEUTRAL = 0.5


def rank_within_race(frame: pd.DataFrame, column: str, *, higher_is_better: bool = True) -> pd.Series:
    """Percentile rank of ``column`` inside each race, where 1.0 is always best.

    The parameter is ``higher_is_better``, not pandas' ``ascending``, because the
    latter reads backwards here and an earlier version of this function got it
    exactly wrong: ``rank(pct=True, ascending=False)`` maps the *smallest* value
    to 1.0, so every "higher is better" feature was silently inverted and the
    composite score was anti-correlated with quality. Naming the intent rather
    than the mechanism makes that mistake impossible to repeat.

    Missing values become :data:`NEUTRAL` — absence of evidence is not evidence
    of weakness, and a debutant must not be scored as the worst horse in the race.
    """
    if column not in frame.columns:
        return pd.Series(NEUTRAL, index=frame.index, dtype="float64")
    ranked = (
        frame.groupby("race_id", sort=False, observed=True)[column]
        .rank(pct=True, ascending=higher_is_better, na_option="keep")
        .astype("float64")
    )
    return ranked.fillna(NEUTRAL)


def _blend(*parts: tuple[pd.Series, float]) -> pd.Series:
    total_weight = sum(weight for _, weight in parts)
    blended = parts[0][0] * parts[0][1]
    for series, weight in parts[1:]:
        blended = blended + series * weight
    return (blended / total_weight).clip(0.0, 1.0)


def build_scores(frame: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Add the composite score columns to a feature frame."""
    config = config or FeatureConfig()
    weights: ScoreWeights = config.weights
    result = frame.copy()

    if result.empty:
        for column in SCORE_COLUMNS:
            result[column] = pd.Series(dtype="float64")
        return result

    # --- components, each already a within-race percentile ----------------
    # Finishing position: a lower number is a better run.
    recent_form = _blend(
        (rank_within_race(result, "horse_form_last3_avg_pos", higher_is_better=False), 0.6),
        (rank_within_race(result, "horse_form_last5_avg_pos", higher_is_better=False), 0.4),
    )
    speed = _blend(
        (rank_within_race(result, "horse_avg_speed"), 0.6),
        (rank_within_race(result, "horse_best_speed"), 0.4),
    )
    distance_fit = rank_within_race(result, "horse_distance_suitability")
    going_fit = rank_within_race(result, "horse_going_suitability")
    # Class: where this horse's official rating sits in the field, nudged by
    # whether it is dropping into easier company than it last ran in.
    class_component = _blend(
        (rank_within_race(result, "runner_official_rating"), 0.75),
        (rank_within_race(result, "horse_class_move"), 0.25),
    )

    result["horse_form_score"] = (
        weights.recent_form * recent_form
        + weights.speed_rating * speed
        + weights.distance_fit * distance_fit
        + weights.going_fit * going_fit
        + weights.class_rating * class_component
    ).clip(0.0, 1.0)

    result["speed_score"] = speed
    result["distance_score"] = distance_fit
    result["going_score"] = going_fit
    result["class_score"] = class_component

    result["jockey_score"] = _blend(
        (rank_within_race(result, "jky_strike_rate"), 0.5),
        (rank_within_race(result, "jky_roi"), 0.25),
        (rank_within_race(result, "jky_course_strike_rate"), 0.25),
    )
    result["trainer_score"] = _blend(
        (rank_within_race(result, "trn_win_rate_30d"), 0.4),
        (rank_within_race(result, "trn_win_rate_window"), 0.35),
        (rank_within_race(result, "trn_jky_win_rate"), 0.25),
    )

    # The market's own normalised probability is already a calibrated [0,1]
    # number, so it is used directly rather than ranked. Where no market was
    # captured it falls back to a uniform prior over the field.
    if "mkt_implied_prob_norm" in result.columns:
        uniform = 1.0 / result.groupby("race_id", sort=False, observed=True)["horse_id"].transform("size")
        result["market_score"] = pd.to_numeric(result["mkt_implied_prob_norm"], errors="coerce").fillna(
            uniform
        )
    else:
        result["market_score"] = 1.0 / result.groupby("race_id", sort=False, observed=True)[
            "horse_id"
        ].transform("size")

    return result


__all__ = ["NEUTRAL", "SCORE_COLUMNS", "build_scores", "rank_within_race"]
