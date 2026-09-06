"""Race-level metrics — the ones that actually describe this problem.

A horse race is **not** a set of independent binary events. Exactly one of the
runners wins, so their win probabilities must sum to one. A per-runner binary
classifier knows nothing about that constraint: ask it about eight runners and
the answers might total 0.7 or 1.4.

That has two consequences, and both are handled here.

**Normalise within the race.** Dividing each runner's probability by the race
total imposes the constraint the sport actually has. It reliably improves log
loss, and — far more importantly for Phase 5 — it is what makes the numbers
comparable with bookmaker prices, which are themselves an over-round-inflated
probability distribution over the same field.

**Score the race, not the runner.** Per-runner log loss is dominated by the
easy negatives: in an eight-runner field, seven of eight labels are 0, and a
model that simply predicts "nobody wins" scores well. Race-level log loss
``-mean(log p_winner)`` cannot be gamed that way. Its reference point is the
uniform model, ``log(field_size)`` ≈ 2.08 for eight runners, and beating that is
the first real hurdle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.ml.metrics.classification import EPSILON

TOP_K_DEFAULT = 3


def normalise_within_race(
    probabilities: np.ndarray | pd.Series, race_ids: np.ndarray | pd.Series
) -> pd.Series:
    """Scale each race's probabilities so they sum to 1.

    Races whose probabilities sum to zero (or to something non-finite) fall back
    to a uniform distribution over the field — a defensible "no opinion" rather
    than a division by zero.
    """
    frame = pd.DataFrame(
        {
            "race_id": np.asarray(race_ids),
            "probability": np.asarray(probabilities, dtype=float),
        }
    )
    frame["probability"] = frame["probability"].clip(lower=0.0)

    totals = frame.groupby("race_id", sort=False)["probability"].transform("sum")
    sizes = frame.groupby("race_id", sort=False)["probability"].transform("size")

    usable = totals.gt(0) & np.isfinite(totals)
    normalised = np.where(usable, frame["probability"] / totals.where(usable, 1.0), 1.0 / sizes)
    return pd.Series(normalised, index=pd.Index(np.asarray(race_ids)).rename(None)).reset_index(drop=True)


@dataclass(slots=True)
class RacingMetrics:
    """How well the model picks winners, race by race."""

    races: int = 0
    runners: int = 0
    mean_field_size: float = 0.0

    #: -mean(log p_winner) after within-race normalisation. Lower is better.
    race_log_loss: float = 0.0
    #: The same for a uniform model — the number to beat.
    uniform_log_loss: float = 0.0
    #: 1 - race_log_loss/uniform_log_loss. 0 = no better than uniform.
    skill_score: float = 0.0

    top1_hit_rate: float = 0.0
    top3_hit_rate: float = 0.0
    mean_winner_rank: float = 0.0
    median_winner_rank: float = 0.0
    #: Mean normalised probability assigned to the actual winner.
    mean_winner_probability: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "races": self.races,
            "runners": self.runners,
            "mean_field_size": round(self.mean_field_size, 3),
            "race_log_loss": round(self.race_log_loss, 6),
            "uniform_log_loss": round(self.uniform_log_loss, 6),
            "skill_score": round(self.skill_score, 6),
            "top1_hit_rate": round(self.top1_hit_rate, 6),
            "top3_hit_rate": round(self.top3_hit_rate, 6),
            "mean_winner_rank": round(self.mean_winner_rank, 3),
            "median_winner_rank": round(self.median_winner_rank, 3),
            "mean_winner_probability": round(self.mean_winner_probability, 6),
        }


def compute_racing_metrics(
    race_ids: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    top_k: int = TOP_K_DEFAULT,
    already_normalised: bool = False,
) -> RacingMetrics:
    """Race-level evaluation.

    ``y_prob`` may be raw per-runner probabilities; they are normalised within
    each race unless ``already_normalised`` says otherwise.
    """
    frame = pd.DataFrame(
        {
            "race_id": np.asarray(race_ids),
            "won": np.asarray(y_true, dtype=int),
            "probability": np.asarray(y_prob, dtype=float),
        }
    )
    metrics = RacingMetrics()
    if frame.empty:
        return metrics

    if not already_normalised:
        frame["probability"] = normalise_within_race(frame["probability"], frame["race_id"]).to_numpy()

    # Rank 1 = the model's most likely winner.
    frame["rank"] = frame.groupby("race_id", sort=False)["probability"].rank(ascending=False, method="first")

    metrics.runners = len(frame)
    metrics.races = int(frame["race_id"].nunique())
    metrics.mean_field_size = float(frame.groupby("race_id", sort=False).size().mean())

    winners = frame[frame["won"] == 1]
    if winners.empty:
        return metrics

    # Races with no recorded winner cannot contribute to any of these.
    field_sizes = frame.groupby("race_id", sort=False).size()
    winner_probabilities = np.clip(winners["probability"].to_numpy(), EPSILON, 1.0)

    metrics.race_log_loss = float(-np.mean(np.log(winner_probabilities)))
    metrics.uniform_log_loss = float(np.mean(np.log(field_sizes.loc[winners["race_id"]].to_numpy())))
    metrics.skill_score = (
        1.0 - metrics.race_log_loss / metrics.uniform_log_loss if metrics.uniform_log_loss > 0 else 0.0
    )

    winner_ranks = winners["rank"].to_numpy()
    metrics.top1_hit_rate = float((winner_ranks == 1).mean())
    metrics.top3_hit_rate = float((winner_ranks <= top_k).mean())
    metrics.mean_winner_rank = float(winner_ranks.mean())
    metrics.median_winner_rank = float(np.median(winner_ranks))
    metrics.mean_winner_probability = float(winner_probabilities.mean())
    return metrics


def race_probability_frame(
    frame: pd.DataFrame, probabilities: np.ndarray | pd.Series, *, column: str = "probability"
) -> pd.DataFrame:
    """Attach normalised probabilities and within-race ranks to a frame."""
    result = frame.copy()
    result[column] = np.asarray(probabilities, dtype=float)
    result[f"{column}_normalised"] = normalise_within_race(result[column], result["race_id"]).to_numpy()
    result["model_rank"] = (
        result.groupby("race_id", sort=False)[f"{column}_normalised"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    return result


__all__ = [
    "TOP_K_DEFAULT",
    "RacingMetrics",
    "compute_racing_metrics",
    "normalise_within_race",
    "race_probability_frame",
]
