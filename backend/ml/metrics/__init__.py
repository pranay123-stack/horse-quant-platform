"""Evaluation metrics.

``classification`` scores each runner independently; ``racing`` scores the race,
which is the unit the sport — and the staking — actually operates on.
"""

from backend.ml.metrics.classification import (
    ClassificationMetrics,
    ReliabilityCurve,
    compute_classification_metrics,
    expected_calibration_error,
    reliability_curve,
)
from backend.ml.metrics.racing import (
    RacingMetrics,
    compute_racing_metrics,
    normalise_within_race,
    race_probability_frame,
)

__all__ = [
    "ClassificationMetrics",
    "RacingMetrics",
    "ReliabilityCurve",
    "compute_classification_metrics",
    "compute_racing_metrics",
    "expected_calibration_error",
    "normalise_within_race",
    "race_probability_frame",
    "reliability_curve",
]
