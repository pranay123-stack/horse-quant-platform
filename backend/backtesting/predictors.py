"""Baseline predictors.

A predictor maps one race's runners to win probabilities summing to 1. No models
are fitted in Phase 3 — these baselines exist so the backtester can be validated
against outcomes we can reason about in advance:

* :class:`MarketPredictor` follows the bookmakers. It **must** lose roughly the
  overround. If a backtest shows it profiting, the backtester is wrong.
* :class:`UniformPredictor` bets blind. It must lose more.
* :class:`FormScorePredictor` uses the Phase 3 composite score — the first thing
  a fitted model has to beat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Predictor(Protocol):
    """Maps a race's runners to win probabilities that sum to 1."""

    name: str

    def __call__(self, race: pd.DataFrame) -> pd.Series: ...


def _normalise(values: pd.Series, index: pd.Index) -> pd.Series:
    """Force a non-negative vector to sum to 1, falling back to uniform."""
    cleaned = pd.to_numeric(values, errors="coerce").astype("float64")
    cleaned = cleaned.where(cleaned > 0)
    total = cleaned.sum()
    if not np.isfinite(total) or total <= 0:
        return pd.Series(1.0 / len(index), index=index, dtype="float64")
    # An unpriced runner is not a zero-chance runner; give it the field average
    # before renormalising, so probabilities still sum to 1.
    filled = cleaned.fillna(cleaned.mean())
    return filled / filled.sum()


@dataclass(slots=True)
class UniformPredictor:
    """Every runner equally likely. The floor any model must clear."""

    name: str = "uniform"

    def __call__(self, race: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0 / len(race), index=race.index, dtype="float64")


@dataclass(slots=True)
class MarketPredictor:
    """The bookmakers' opinion, with the overround divided out.

    This is the benchmark. Backtesting it should produce a loss close to the
    market's margin — a negative control for the whole framework.
    """

    name: str = "market"
    column: str = "mkt_implied_prob_norm"
    fallback: str = "mkt_implied_prob"

    def __call__(self, race: pd.DataFrame) -> pd.Series:
        for column in (self.column, self.fallback):
            if column in race.columns and pd.to_numeric(race[column], errors="coerce").notna().any():
                return _normalise(race[column], race.index)
        return pd.Series(1.0 / len(race), index=race.index, dtype="float64")


@dataclass(slots=True)
class FormScorePredictor:
    """Softmax over the Phase 3 composite form score.

    ``temperature`` controls conviction: low values concentrate probability on
    the top-scoring runner, high values flatten towards uniform. It is a free
    parameter, and tuning it on the same data used to evaluate it is a subtle
    form of overfitting — set it from a training split, not from the test set.
    """

    name: str = "form_score"
    column: str = "horse_form_score"
    temperature: float = 0.15

    def __call__(self, race: pd.DataFrame) -> pd.Series:
        if self.column not in race.columns:
            return pd.Series(1.0 / len(race), index=race.index, dtype="float64")

        scores = pd.to_numeric(race[self.column], errors="coerce")
        if scores.notna().sum() == 0:
            return pd.Series(1.0 / len(race), index=race.index, dtype="float64")

        scores = scores.fillna(scores.mean())
        # Subtract the max before exponentiating: standard guard against overflow.
        exponent = (scores - scores.max()) / max(self.temperature, 1e-6)
        weights = np.exp(exponent).to_numpy()
        return pd.Series(weights / weights.sum(), index=race.index, dtype="float64")


@dataclass(slots=True)
class BlendPredictor:
    """Convex blend of two predictors.

    Blending a model with the market is the standard way to express partial
    confidence: ``weight=0.2`` says "mostly trust the market, tilt slightly
    towards the model", which is usually the honest position early on.
    """

    first: Predictor
    second: Predictor
    weight: float = 0.5
    name: str = "blend"

    def __call__(self, race: pd.DataFrame) -> pd.Series:
        left = self.first(race)
        right = self.second(race)
        blended = self.weight * left + (1.0 - self.weight) * right
        normalised: pd.Series = blended / blended.sum()
        return normalised


__all__ = [
    "BlendPredictor",
    "FormScorePredictor",
    "MarketPredictor",
    "Predictor",
    "UniformPredictor",
]
