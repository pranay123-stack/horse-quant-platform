"""Probability-quality metrics.

For a betting model, **calibration matters more than discrimination**. A model
with a superb ROC-AUC but badly scaled probabilities will still lose money,
because expected value is computed from the probability itself:
``EV = p * odds - 1``. Get ``p`` systematically 20% too high and every stake is
placed on a false edge.

So this module leads with Brier score, log loss and expected calibration error,
and treats AUC as a secondary, ranking-only diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: Probabilities are clipped before taking logs. A single confident miss at
#: exactly 0.0 would otherwise send log loss to infinity.
EPSILON = 1e-15


def _clip(probabilities: np.ndarray) -> np.ndarray:
    clipped: np.ndarray = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return clipped


@dataclass(slots=True)
class ReliabilityCurve:
    """Predicted probability against observed frequency, by bin."""

    bin_centres: list[float] = field(default_factory=list)
    predicted: list[float] = field(default_factory=list)
    observed: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[Any]]:
        return {
            "bin_centres": self.bin_centres,
            "predicted": self.predicted,
            "observed": self.observed,
            "counts": self.counts,
        }

    def render(self, width: int = 34) -> str:
        """A text reliability plot.

        Deliberately not a PNG: a calibration curve is the thing to look at
        first, and one that renders in a terminal, a log file and a CI summary
        gets looked at.
        """
        lines = [
            "  predicted  observed      n  |0.0                              1.0|",
            "  ---------  --------  -----  +-----------------------------------+",
        ]
        for predicted, observed, count in zip(self.predicted, self.observed, self.counts, strict=True):
            marker = ["·"] * width
            p_index = min(width - 1, max(0, int(predicted * width)))
            o_index = min(width - 1, max(0, int(observed * width)))
            marker[p_index] = "p"
            marker[o_index] = "O" if o_index != p_index else "X"
            lines.append(f"  {predicted:>9.3f}  {observed:>8.3f}  {count:>5}  |{''.join(marker)}|")
        lines.append("  p = mean predicted, O = observed frequency, X = they coincide")
        return "\n".join(lines)


def reliability_curve(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, *, bins: int = 10
) -> ReliabilityCurve:
    """Bin predictions by probability and compare with the observed rate.

    Uses **quantile** bins rather than equal-width. Racing win probabilities pile
    up near zero, so equal-width bins leave the top half almost empty and the
    curve says nothing about exactly the region where bets get placed.
    """
    truth = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(y_prob, dtype=float)
    if truth.size == 0:
        return ReliabilityCurve()

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(probabilities, quantiles))
    if edges.size < 2:
        return ReliabilityCurve(
            bin_centres=[float(probabilities.mean())],
            predicted=[float(probabilities.mean())],
            observed=[float(truth.mean())],
            counts=[int(truth.size)],
        )

    indices = np.clip(np.digitize(probabilities, edges[1:-1], right=True), 0, len(edges) - 2)
    curve = ReliabilityCurve()
    for bin_index in range(len(edges) - 1):
        mask = indices == bin_index
        if not mask.any():
            continue
        curve.bin_centres.append(float((edges[bin_index] + edges[bin_index + 1]) / 2))
        curve.predicted.append(float(probabilities[mask].mean()))
        curve.observed.append(float(truth[mask].mean()))
        curve.counts.append(int(mask.sum()))
    return curve


def expected_calibration_error(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, *, bins: int = 10
) -> float:
    """Mean |predicted - observed| across bins, weighted by bin size.

    0 is perfect. Reported alongside Brier because Brier mixes calibration and
    discrimination, while ECE isolates calibration — the part that decides
    whether an expected-value calculation can be trusted.
    """
    curve = reliability_curve(y_true, y_prob, bins=bins)
    if not curve.counts:
        return 0.0
    total = sum(curve.counts)
    return float(
        sum(
            count * abs(predicted - observed)
            for predicted, observed, count in zip(curve.predicted, curve.observed, curve.counts, strict=True)
        )
        / total
    )


def maximum_calibration_error(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, *, bins: int = 10
) -> float:
    """Worst single-bin calibration gap — the blow-up risk ECE averages away."""
    curve = reliability_curve(y_true, y_prob, bins=bins)
    if not curve.counts:
        return 0.0
    return float(
        max(
            abs(predicted - observed)
            for predicted, observed in zip(curve.predicted, curve.observed, strict=True)
        )
    )


@dataclass(slots=True)
class ClassificationMetrics:
    """Per-runner probability quality."""

    rows: int = 0
    positives: int = 0
    base_rate: float = 0.0

    log_loss: float = 0.0
    brier: float = 0.0
    roc_auc: float = 0.0
    average_precision: float = 0.0

    precision: float = 0.0
    recall: float = 0.0
    threshold: float = 0.5

    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.0
    mean_predicted: float = 0.0
    #: Mean prediction / base rate. 1.0 means the model's average probability
    #: matches reality; >1 means it is systematically over-confident.
    calibration_ratio: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "positives": self.positives,
            "base_rate": round(self.base_rate, 6),
            "log_loss": round(self.log_loss, 6),
            "brier": round(self.brier, 6),
            "roc_auc": round(self.roc_auc, 6),
            "average_precision": round(self.average_precision, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "expected_calibration_error": round(self.expected_calibration_error, 6),
            "maximum_calibration_error": round(self.maximum_calibration_error, 6),
            "mean_predicted": round(self.mean_predicted, 6),
            "calibration_ratio": round(self.calibration_ratio, 6),
        }


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    threshold: float = 0.5,
    bins: int = 10,
) -> ClassificationMetrics:
    truth = np.asarray(y_true, dtype=int)
    probabilities = _clip(np.asarray(y_prob, dtype=float))

    metrics = ClassificationMetrics(rows=int(truth.size), threshold=threshold)
    if truth.size == 0:
        return metrics

    metrics.positives = int(truth.sum())
    metrics.base_rate = float(truth.mean())
    metrics.mean_predicted = float(probabilities.mean())
    metrics.calibration_ratio = metrics.mean_predicted / metrics.base_rate if metrics.base_rate > 0 else 0.0

    metrics.brier = float(brier_score_loss(truth, probabilities))
    metrics.log_loss = float(log_loss(truth, probabilities, labels=[0, 1]))

    # AUC and AP are undefined when only one class is present.
    if 0 < metrics.positives < truth.size:
        metrics.roc_auc = float(roc_auc_score(truth, probabilities))
        metrics.average_precision = float(average_precision_score(truth, probabilities))

    predicted_labels = (probabilities >= threshold).astype(int)
    metrics.precision = float(precision_score(truth, predicted_labels, zero_division=0))
    metrics.recall = float(recall_score(truth, predicted_labels, zero_division=0))

    metrics.expected_calibration_error = expected_calibration_error(truth, probabilities, bins=bins)
    metrics.maximum_calibration_error = maximum_calibration_error(truth, probabilities, bins=bins)
    return metrics


__all__ = [
    "EPSILON",
    "ClassificationMetrics",
    "ReliabilityCurve",
    "compute_classification_metrics",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_curve",
]
