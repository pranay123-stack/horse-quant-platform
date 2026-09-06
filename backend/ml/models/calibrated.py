"""Model + calibrator as one deployable unit.

The order of operations is fixed and matters:

1. ``estimator.predict_proba`` — a raw score per runner
2. ``calibrator.transform`` — map that score onto observed frequency
3. ``normalise_within_race`` — impose the constraint that one horse wins

Calibration must come **before** normalisation. Calibration is a pointwise map
learned from (score, outcome) pairs; feeding it already-normalised numbers would
mean fitting it on values whose scale depends on how many horses happened to be
in the race. Normalisation then applies the race constraint to numbers that are
already on the right scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.ml.calibration.calibrators import ProbabilityCalibrator
from backend.ml.metrics.racing import normalise_within_race
from backend.ml.models.base import RacingModel
from backend.utils.exceptions import ModelError
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="model")

MODEL_CLASSES: dict[str, str] = {
    "logistic_regression": "backend.ml.models.logistic:LogisticRegressionModel",
    "xgboost": "backend.ml.models.gradient_boosting:XGBoostModel",
    "lightgbm": "backend.ml.models.gradient_boosting:LightGBMModel",
}


def model_class_for(name: str) -> type[RacingModel]:
    """Resolve a model name to its class, for loading from disk."""
    import importlib

    try:
        path = MODEL_CLASSES[name]
    except KeyError:
        raise ModelError(f"unknown model {name!r}; expected one of {sorted(MODEL_CLASSES)}") from None
    module_name, class_name = path.split(":")
    resolved: type[RacingModel] = getattr(importlib.import_module(module_name), class_name)
    return resolved


class CalibratedRacingModel:
    """A fitted model with its calibrator attached."""

    def __init__(self, model: RacingModel, calibrator: ProbabilityCalibrator | None = None) -> None:
        self.model = model
        self.calibrator = calibrator or ProbabilityCalibrator("none")

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        suffix = "" if self.calibrator.method == "none" else f"+{self.calibrator.method}"
        return f"{self.model.name}{suffix}"

    @property
    def is_calibrated(self) -> bool:
        return self.calibrator.method != "none"

    def raw_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Uncalibrated per-runner scores — for before/after comparisons."""
        return self.model.predict_proba(frame)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated per-runner P(win). Not yet normalised within the race."""
        return self.calibrator.transform(self.model.predict_proba(frame))

    def predict_race_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated probabilities normalised so each race sums to 1."""
        if "race_id" not in frame.columns:
            raise ModelError("race-normalised prediction needs a 'race_id' column")
        calibrated = self.predict_proba(frame)
        return normalise_within_race(calibrated, frame["race_id"]).to_numpy()

    def feature_importance(self, *, top: int | None = None) -> pd.DataFrame:
        return self.model.feature_importance(top=top)

    # ------------------------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        return {
            **self.model.metadata(),
            "calibration_method": self.calibrator.method,
            "calibration_rows": self.calibrator.fitted_rows,
            "calibration_positives": self.calibrator.fitted_positives,
            "calibration_downgraded": self.calibrator.downgraded,
        }

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        self.model.save(target)
        self.calibrator.save(target)

        import json

        (target / "model.json").write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, directory: str | Path) -> CalibratedRacingModel:
        import json

        target = Path(directory)
        metadata_path = target / "model.json"
        if not metadata_path.exists():
            raise ModelError(f"no model at {target}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = model_class_for(metadata["name"]).load(target)

        calibrator = (
            ProbabilityCalibrator.load(target)
            if (target / "calibrator.joblib").exists()
            else ProbabilityCalibrator("none")
        )
        logger.info("model loaded", extra={"model": metadata["name"], "path": str(target)})
        return cls(model, calibrator)


__all__ = ["MODEL_CLASSES", "CalibratedRacingModel", "model_class_for"]
