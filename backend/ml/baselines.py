"""Baselines, and the test for information beyond the market price.

Phase 6.5 asks: *does the model contain signal beyond market price?* Comparing
log losses does not answer it. A model fed market features will always score
close to the market, and "slightly better" can come from noise, from a lucky
window, or from the model simply reproducing the price more smoothly.

The right instrument is a **forecast encompassing test**. Fit

    logit P(win) = a + b1·logit(p_market) + b2·logit(p_model)

on out-of-sample races. The market's forecast is already in the equation, so
``b2`` measures what the model adds *conditional on* the price already being
known. Three readings:

* ``b2 ≈ 0`` — the market encompasses the model. Whatever the model knows, the
  price knew first. This is the null, and on real racing it is the *expected*
  result.
* ``b2 > 0``, significant — the model carries genuine incremental information.
* ``b1 ≈ 0``, ``b2 > 0`` — the model encompasses the market, which for a
  bookmaker-priced sport should be treated as a bug until proven otherwise.

This is a far harder test to pass than "beats the market on log loss", and it is
the one worth reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from backend.ml.metrics.classification import EPSILON
from backend.ml.metrics.racing import compute_racing_metrics, normalise_within_race

MARKET_PROBABILITY_COLUMN = "market_probability"
MODEL_PROBABILITY_COLUMN = "model_probability"

#: Above this correlation between the two logit forecasts, the regression is
#: numerically hopeless and the answer is already known.
COLLINEARITY_THRESHOLD = 0.9999


class ProbabilityBaseline(Protocol):
    """Anything that assigns a win probability to each runner in a frame."""

    name: str

    def __call__(self, frame: pd.DataFrame) -> np.ndarray: ...


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1 - EPSILON)
    result: np.ndarray = np.log(clipped / (1 - clipped))
    return result


@dataclass(slots=True)
class RandomBaseline:
    """Uniform over the field — the floor. Seeded, so runs are comparable."""

    name: str = "random"
    seed: int = 42

    def __call__(self, frame: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        noise = pd.Series(rng.random(len(frame)), index=frame.index)
        return normalise_within_race(noise, frame["race_id"]).to_numpy()


@dataclass(slots=True)
class UniformBaseline:
    """Exactly ``1 / field_size``. The reference point for race log loss."""

    name: str = "uniform"

    def __call__(self, frame: pd.DataFrame) -> np.ndarray:
        sizes = frame.groupby("race_id", sort=False)["race_id"].transform("size")
        return (1.0 / sizes).to_numpy()


@dataclass(slots=True)
class MarketBaseline:
    """The bookmakers' margin-free probability — the benchmark that matters."""

    name: str = "market"
    column: str = MARKET_PROBABILITY_COLUMN

    def __call__(self, frame: pd.DataFrame) -> np.ndarray:
        if self.column not in frame.columns:
            return UniformBaseline()(frame)
        values = pd.to_numeric(frame[self.column], errors="coerce")
        return normalise_within_race(values.fillna(0.0), frame["race_id"]).to_numpy()


@dataclass(slots=True)
class FavouriteBaseline:
    """All probability on the shortest price.

    A crude strategy but a genuinely informative baseline: the favourite wins
    roughly a third of UK races, so top-1 hit rate has a hard floor that any
    model claiming skill must clear.
    """

    name: str = "favourite"
    column: str = "mkt_mean_odds"
    #: Probability given to the favourite; the rest is spread over the field so
    #: log loss stays finite when the favourite is beaten.
    confidence: float = 0.9

    def __call__(self, frame: pd.DataFrame) -> np.ndarray:
        if self.column not in frame.columns:
            return UniformBaseline()(frame)

        odds = pd.to_numeric(frame[self.column], errors="coerce")
        shortest = odds.groupby(frame["race_id"], sort=False).transform("min")
        is_favourite = (odds == shortest) & odds.notna()

        sizes = frame.groupby("race_id", sort=False)["race_id"].transform("size")
        spread = (1.0 - self.confidence) / np.maximum(sizes - 1, 1)
        probabilities = np.where(is_favourite, self.confidence, spread)
        return normalise_within_race(probabilities, frame["race_id"]).to_numpy()


@dataclass(slots=True)
class ModelBaseline:
    """The model's own stored probabilities, for uniform treatment."""

    name: str = "model"
    column: str = MODEL_PROBABILITY_COLUMN

    def __call__(self, frame: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(frame[self.column], errors="coerce").fillna(0.0)
        return normalise_within_race(values, frame["race_id"]).to_numpy()


DEFAULT_BASELINES: tuple[ProbabilityBaseline, ...] = (
    RandomBaseline(),
    UniformBaseline(),
    FavouriteBaseline(),
    MarketBaseline(),
    ModelBaseline(),
)


def compare_baselines(
    frame: pd.DataFrame,
    baselines: tuple[ProbabilityBaseline, ...] = DEFAULT_BASELINES,
    *,
    target: str = "won",
) -> pd.DataFrame:
    """Score every baseline and the model on the same races."""
    if frame.empty:
        return pd.DataFrame(columns=["forecaster", "race_log_loss", "top1_hit_rate", "skill_vs_uniform"])

    rows = []
    for baseline in baselines:
        probabilities = baseline(frame)
        metrics = compute_racing_metrics(
            frame["race_id"], frame[target], probabilities, already_normalised=True
        )
        rows.append(
            {
                "forecaster": baseline.name,
                "races": metrics.races,
                "race_log_loss": round(metrics.race_log_loss, 5),
                "top1_hit_rate": round(metrics.top1_hit_rate, 4),
                "top3_hit_rate": round(metrics.top3_hit_rate, 4),
                "skill_vs_uniform": round(metrics.skill_score, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("race_log_loss").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Forecast encompassing
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class EncompassingResult:
    """Does the model add anything once the price is known?"""

    rows: int = 0
    market_coefficient: float = 0.0
    market_z: float = 0.0
    model_coefficient: float = 0.0
    model_z: float = 0.0
    log_loss_market_only: float = 0.0
    log_loss_combined: float = 0.0
    likelihood_ratio: float = 0.0
    converged: bool = False
    #: The two forecasts are (near-)identical, so no regression can separate
    #: them. This is encompassing in its strongest form, not a failure.
    collinear: bool = False
    forecast_correlation: float = 0.0

    @property
    def model_adds_information(self) -> bool:
        """Two standard errors on the model's conditional coefficient."""
        return self.converged and self.model_coefficient > 0 and self.model_z >= 2.0

    @property
    def market_encompasses_model(self) -> bool:
        return self.collinear or (self.converged and not self.model_adds_information)

    @property
    def verdict(self) -> str:
        if self.collinear:
            return (
                "model is indistinguishable from the price "
                f"(forecast correlation {self.forecast_correlation:.4f}) — it carries no "
                "information the market does not already have"
            )
        if not self.converged:
            return "inconclusive — the encompassing regression did not converge"
        if self.model_adds_information:
            return (
                f"model adds information beyond the price "
                f"(b_model = {self.model_coefficient:+.3f}, z = {self.model_z:.2f})"
            )
        return (
            f"market encompasses the model — no incremental signal "
            f"(b_model = {self.model_coefficient:+.3f}, z = {self.model_z:.2f})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "market_coefficient": round(self.market_coefficient, 4),
            "market_z": round(self.market_z, 3),
            "model_coefficient": round(self.model_coefficient, 4),
            "model_z": round(self.model_z, 3),
            "log_loss_market_only": round(self.log_loss_market_only, 5),
            "log_loss_combined": round(self.log_loss_combined, 5),
            "likelihood_ratio": round(self.likelihood_ratio, 3),
            "collinear": self.collinear,
            "forecast_correlation": round(self.forecast_correlation, 5),
            "model_adds_information": self.model_adds_information,
            "verdict": self.verdict,
        }


def market_encompassing_test(
    frame: pd.DataFrame,
    *,
    target: str = "won",
    model_column: str = MODEL_PROBABILITY_COLUMN,
    market_column: str = MARKET_PROBABILITY_COLUMN,
) -> EncompassingResult:
    """Fit ``logit(win) ~ logit(p_market) + logit(p_model)`` and read ``b_model``.

    The z-statistic uses the observed information matrix, so it is a proper
    standard error rather than a bootstrap approximation.
    """
    result = EncompassingResult()
    if frame.empty or model_column not in frame or market_column not in frame:
        return result

    usable = frame[[target, model_column, market_column]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(usable) < 100 or usable[target].nunique() < 2:
        return result

    truth = usable[target].to_numpy(dtype=float)
    market_logit = _logit(usable[market_column].to_numpy())
    model_logit = _logit(usable[model_column].to_numpy())
    result.rows = len(usable)

    # Perfectly collinear forecasts make the Hessian singular, so the regression
    # cannot run at all. That is not an inconclusive result -- a model that
    # reproduces the price exactly is encompassed by definition, and saying so is
    # far more useful than reporting a convergence failure.
    if np.std(market_logit) > 0 and np.std(model_logit) > 0:
        result.forecast_correlation = float(np.corrcoef(market_logit, model_logit)[0, 1])
    if abs(result.forecast_correlation) >= COLLINEARITY_THRESHOLD:
        result.collinear = True
        return result

    design = np.column_stack([np.ones(len(usable)), market_logit, model_logit])

    beta, converged = _fit_logistic(design, truth)
    result.converged = converged
    if not converged:
        return result

    probabilities = 1.0 / (1.0 + np.exp(-design @ beta))
    result.log_loss_combined = float(_log_loss(truth, probabilities))

    market_only = design[:, :2]
    beta_market, converged_market = _fit_logistic(market_only, truth)
    if converged_market:
        market_probabilities = 1.0 / (1.0 + np.exp(-market_only @ beta_market))
        result.log_loss_market_only = float(_log_loss(truth, market_probabilities))
        # 2 * (LL_combined - LL_market) on one extra parameter.
        result.likelihood_ratio = float(
            2 * len(usable) * (result.log_loss_market_only - result.log_loss_combined)
        )

    standard_errors = _standard_errors(design, probabilities)
    result.market_coefficient = float(beta[1])
    result.model_coefficient = float(beta[2])
    if standard_errors is not None:
        result.market_z = float(beta[1] / standard_errors[1]) if standard_errors[1] > 0 else 0.0
        result.model_z = float(beta[2] / standard_errors[2]) if standard_errors[2] > 0 else 0.0
    return result


def _fit_logistic(
    design: np.ndarray, truth: np.ndarray, *, iterations: int = 100, tolerance: float = 1e-8
) -> tuple[np.ndarray, bool]:
    """Newton-Raphson logistic fit. Small problem, so no need for sklearn."""
    beta = np.zeros(design.shape[1])
    for _ in range(iterations):
        eta = np.clip(design @ beta, -35, 35)
        probabilities = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(probabilities * (1 - probabilities), 1e-10, None)

        gradient = design.T @ (truth - probabilities)
        hessian = design.T @ (design * weights[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:  # pragma: no cover - singular design
            return beta, False
        beta = beta + step
        if np.max(np.abs(step)) < tolerance:
            return beta, True
    return beta, False


def _standard_errors(design: np.ndarray, probabilities: np.ndarray) -> np.ndarray | None:
    weights = np.clip(probabilities * (1 - probabilities), 1e-10, None)
    hessian = design.T @ (design * weights[:, None])
    try:
        return np.sqrt(np.diag(np.linalg.inv(hessian)))
    except np.linalg.LinAlgError:  # pragma: no cover - singular design
        return None


def _log_loss(truth: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped)))


__all__ = [
    "COLLINEARITY_THRESHOLD",
    "DEFAULT_BASELINES",
    "EncompassingResult",
    "FavouriteBaseline",
    "MarketBaseline",
    "ModelBaseline",
    "ProbabilityBaseline",
    "RandomBaseline",
    "UniformBaseline",
    "compare_baselines",
    "market_encompassing_test",
]
