"""Probability calibration.

Raw classifier outputs are scores, not probabilities. Gradient-boosted trees in
particular are systematically over-confident: trained to minimise log loss on an
imbalanced problem they push predictions towards the extremes, so a horse the
model calls "0.35" may win nearer one time in four.

For a classification report that hardly matters — the ranking is unchanged, and
AUC does not move at all. For this platform it is decisive, because every stake
is sized from ``EV = p * odds - 1``. A model that says 0.35 when the truth is
0.25 will find "value" at 3.5 that is really a 12% loss per bet, and will keep
finding it, race after race.

Two methods, and when each is right
===================================

**Platt scaling** (``method="sigmoid"``) fits a one-parameter logistic map. It
assumes the miscalibration is a smooth monotone squash, which is exactly what
over-confidence looks like. Two parameters means it is stable on a few thousand
rows.

**Isotonic regression** fits an arbitrary monotone step function. Strictly more
flexible, and it will fit any shape of distortion — including the noise. It
needs considerably more data, and it cannot extrapolate beyond the range it saw.

With a ~11% base rate, the positive class is thin: a 5,000-row calibration slice
holds only ~550 winners. Platt is the safer default; isotonic is offered and
compared, not assumed.

Fitting without leaking
=======================
``CalibratedClassifierCV`` defaults to cross-validated calibration, which
re-splits the data **at random**. On a time series that is a leak. Everything
here uses ``cv="prefit"`` against a chronologically later, otherwise-unused
slice — the ``valid_calib`` window from :mod:`backend.ml.dataset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from backend.ml.metrics.classification import (
    ReliabilityCurve,
    compute_classification_metrics,
    reliability_curve,
)
from backend.utils.exceptions import ModelError
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="model")

CALIBRATOR_FILENAME = "calibrator.joblib"
CalibrationMethod = Literal["sigmoid", "isotonic", "none"]

#: Below this many positives in the calibration slice, isotonic regression is
#: fitting noise and Platt is used instead.
MIN_POSITIVES_FOR_ISOTONIC = 200


class ProbabilityCalibrator:
    """Maps raw model scores onto calibrated probabilities."""

    def __init__(self, method: CalibrationMethod = "sigmoid") -> None:
        if method not in ("sigmoid", "isotonic", "none"):
            raise ValueError(f"unknown calibration method {method!r}")
        self.method: CalibrationMethod = method
        self.estimator: Any = None
        self.fitted_positives = 0
        self.fitted_rows = 0
        self.downgraded = False

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self.method == "none" or self.estimator is not None

    def fit(self, scores: np.ndarray | pd.Series, y_true: np.ndarray | pd.Series) -> ProbabilityCalibrator:
        raw = np.asarray(scores, dtype=float).reshape(-1, 1)
        truth = np.asarray(y_true, dtype=int)

        self.fitted_rows = int(truth.size)
        self.fitted_positives = int(truth.sum())

        if self.method == "none":
            return self
        if truth.size == 0 or self.fitted_positives == 0:
            raise ModelError("cannot calibrate on a slice with no positive examples")

        method = self.method
        if method == "isotonic" and self.fitted_positives < MIN_POSITIVES_FOR_ISOTONIC:
            logger.warning(
                "too few winners for isotonic calibration; falling back to Platt scaling",
                extra={"positives": self.fitted_positives, "minimum": MIN_POSITIVES_FOR_ISOTONIC},
            )
            method = "sigmoid"
            self.downgraded = True

        if method == "sigmoid":
            # A logistic fit on the single raw-score column: Platt scaling.
            self.estimator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            self.estimator.fit(raw, truth)
        else:
            self.estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.estimator.fit(raw.ravel(), truth)

        logger.info(
            "calibrator fitted",
            extra={"method": method, "rows": self.fitted_rows, "positives": self.fitted_positives},
        )
        return self

    def transform(self, scores: np.ndarray | pd.Series) -> np.ndarray:
        raw = np.asarray(scores, dtype=float)
        if self.method == "none":
            return raw
        if self.estimator is None:
            raise ModelError("calibrator is not fitted")

        if isinstance(self.estimator, IsotonicRegression):
            calibrated = self.estimator.predict(raw)
        else:
            calibrated = self.estimator.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)

    __call__ = transform

    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / CALIBRATOR_FILENAME
        joblib.dump(
            {
                "method": self.method,
                "estimator": self.estimator,
                "fitted_rows": self.fitted_rows,
                "fitted_positives": self.fitted_positives,
                "downgraded": self.downgraded,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, directory: str | Path) -> ProbabilityCalibrator:
        path = Path(directory) / CALIBRATOR_FILENAME
        if not path.exists():
            raise ModelError(f"no calibrator at {path}")
        payload = joblib.load(path)
        calibrator = cls(payload["method"])
        calibrator.estimator = payload["estimator"]
        calibrator.fitted_rows = payload["fitted_rows"]
        calibrator.fitted_positives = payload["fitted_positives"]
        calibrator.downgraded = payload.get("downgraded", False)
        return calibrator


@dataclass(slots=True)
class CalibrationComparison:
    """Before-and-after evidence that calibration did something useful."""

    method: str
    before_brier: float
    after_brier: float
    before_log_loss: float
    after_log_loss: float
    before_ece: float
    after_ece: float
    before_curve: ReliabilityCurve
    after_curve: ReliabilityCurve

    @property
    def ece_improvement(self) -> float:
        return self.before_ece - self.after_ece

    @property
    def improved(self) -> bool:
        """Calibration helped only if it sharpened reliability without costing score.

        Requiring log loss not to degrade matters: an already-calibrated model
        can have its ECE nudged down by a calibrator that is really just fitting
        noise on the calibration slice, and the damage shows up in log loss
        first. Calibration is not automatically beneficial and this property
        refuses to pretend otherwise.
        """
        return self.after_ece < self.before_ece and self.after_log_loss <= self.before_log_loss * 1.01

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "before": {
                "brier": round(self.before_brier, 6),
                "log_loss": round(self.before_log_loss, 6),
                "ece": round(self.before_ece, 6),
                "curve": self.before_curve.as_dict(),
            },
            "after": {
                "brier": round(self.after_brier, 6),
                "log_loss": round(self.after_log_loss, 6),
                "ece": round(self.after_ece, 6),
                "curve": self.after_curve.as_dict(),
            },
            "ece_improvement": round(self.ece_improvement, 6),
            "improved": self.improved,
        }

    def render(self) -> str:
        arrow = "improved" if self.improved else "NO IMPROVEMENT"
        return "\n".join(
            [
                f"Calibration: {self.method}  ({arrow})",
                f"  Brier    {self.before_brier:.5f} -> {self.after_brier:.5f}",
                f"  LogLoss  {self.before_log_loss:.5f} -> {self.after_log_loss:.5f}",
                f"  ECE      {self.before_ece:.5f} -> {self.after_ece:.5f}",
                "",
                "  Before:",
                self.before_curve.render(),
                "",
                "  After:",
                self.after_curve.render(),
            ]
        )


def compare_calibration(
    y_true: np.ndarray | pd.Series,
    raw_scores: np.ndarray | pd.Series,
    calibrated: np.ndarray | pd.Series,
    *,
    method: str = "sigmoid",
    bins: int = 10,
) -> CalibrationComparison:
    """Reliability before and after, on the same held-out rows."""
    before = compute_classification_metrics(y_true, raw_scores, bins=bins)
    after = compute_classification_metrics(y_true, calibrated, bins=bins)
    return CalibrationComparison(
        method=method,
        before_brier=before.brier,
        after_brier=after.brier,
        before_log_loss=before.log_loss,
        after_log_loss=after.log_loss,
        before_ece=before.expected_calibration_error,
        after_ece=after.expected_calibration_error,
        before_curve=reliability_curve(y_true, raw_scores, bins=bins),
        after_curve=reliability_curve(y_true, calibrated, bins=bins),
    )


__all__ = [
    "CALIBRATOR_FILENAME",
    "MIN_POSITIVES_FOR_ISOTONIC",
    "CalibrationComparison",
    "CalibrationMethod",
    "ProbabilityCalibrator",
    "compare_calibration",
]
