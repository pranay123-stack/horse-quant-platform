"""Probability calibration -- mandatory, not optional.

Raw classifier scores are not probabilities. Since every downstream decision is
``EV = p * odds - 1``, a systematically over-confident model does not merely
score badly, it manufactures betting opportunities that are not there.
"""

from backend.ml.calibration.calibrators import (
    CalibrationComparison,
    CalibrationMethod,
    ProbabilityCalibrator,
    compare_calibration,
)

__all__ = [
    "CalibrationComparison",
    "CalibrationMethod",
    "ProbabilityCalibrator",
    "compare_calibration",
]
