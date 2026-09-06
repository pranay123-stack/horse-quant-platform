"""Common interface for every win-probability model.

.. note::
   Three things in this repository are called "models". They are unrelated:

   * ``backend/models/`` — SQLAlchemy ORM entities (database tables)
   * ``backend/ml/models/`` — *this* package: estimator wrappers
   * ``models/`` at the repository root — serialised artefacts on disk

A model here bundles three things that must travel together or the predictions
are silently wrong: the fitted estimator, the preprocessor that produced its
input matrix, and the feature schema that pins the column contract.

A deliberate omission: none of these models rebalance the classes
=================================================================
Roughly one runner in nine wins, and the reflex is to reach for
``scale_pos_weight`` or ``class_weight="balanced"``. That is right when you want
a decision boundary and wrong here. Rebalancing changes the *base rate the model
is fitted to*, so its output stops being a probability of winning and becomes a
probability under a reweighted world that does not exist. Since every downstream
number is ``EV = p * odds - 1``, that distortion goes straight into the staking.

The imbalance is handled where it belongs: by calibration
(:mod:`backend.ml.calibration`), which maps scores onto observed frequencies
without touching what the model was fitted to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.ml.preprocessing.pipeline import FeaturePreprocessor
from backend.ml.preprocessing.schema import FeatureSchema
from backend.utils.exceptions import ModelError
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="model")

ESTIMATOR_FILENAME = "estimator.joblib"
METADATA_FILENAME = "model.json"

#: Fixed seed everywhere. A model whose metrics move between runs cannot be
#: compared with the model it is meant to replace.
DEFAULT_SEED = 42


class RacingModel(ABC):
    """Estimator + preprocessor + schema, saved and loaded as one unit."""

    #: Short identifier used in the registry and on disk.
    name: str = "base"
    #: Whether this estimator benefits from standardised inputs.
    needs_scaling: bool = False

    def __init__(
        self,
        schema: FeatureSchema,
        params: dict[str, Any] | None = None,
        *,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.schema = schema
        self.params = {**self.default_params(), **(params or {})}
        self.seed = seed
        self.preprocessor = FeaturePreprocessor(schema, scale=self.needs_scaling)
        self.estimator: Any = None
        self.fitted_at: str | None = None
        self.training_rows: int = 0
        self.best_iteration: int | None = None

    # ------------------------------------------------------------------
    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {}

    @abstractmethod
    def _build_estimator(self) -> Any:
        """Construct the underlying estimator from ``self.params``."""

    def _fit_estimator(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None,
        y_valid: pd.Series | None,
    ) -> None:
        """Fit, optionally using a validation set for early stopping."""
        self.estimator.fit(X, y)

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self.estimator is not None and self.fitted_at is not None

    def fit(
        self,
        train: pd.DataFrame,
        y_train: pd.Series,
        valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> RacingModel:
        """Fit the preprocessor on training data only, then the estimator.

        The preprocessor is fitted on ``train`` alone. Fitting imputation medians
        or one-hot vocabularies on train+valid would leak the validation
        distribution into the model, which is the quiet version of the same
        mistake the whole platform is built to avoid.
        """
        from backend.utils.timeutils import utcnow

        X = self.preprocessor.fit_transform(train)
        X_valid = self.preprocessor.transform(valid) if valid is not None and not valid.empty else None
        y_valid_clean = y_valid if X_valid is not None else None

        self.estimator = self._build_estimator()
        self._fit_estimator(X, y_train, X_valid, y_valid_clean)

        self.fitted_at = utcnow().isoformat()
        self.training_rows = len(train)
        logger.info(
            "model fitted",
            extra={
                "model": self.name,
                "rows": self.training_rows,
                "features": len(self.preprocessor.output_names),
                "best_iteration": self.best_iteration,
            },
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Per-runner P(win). Not normalised within the race — see
        :func:`backend.ml.metrics.racing.normalise_within_race`."""
        if not self.is_fitted:
            raise ModelError(f"{self.name} model is not fitted")
        if frame.empty:
            return np.zeros(0, dtype=float)
        matrix = self.preprocessor.transform(frame)
        return np.asarray(self.estimator.predict_proba(matrix))[:, 1]

    # ------------------------------------------------------------------
    def feature_importance(self, *, top: int | None = None) -> pd.DataFrame:
        """Importance per transformed feature, most important first."""
        if not self.is_fitted:
            raise ModelError(f"{self.name} model is not fitted")

        names = self.preprocessor.output_names
        values = self._raw_importance()
        if values is None:
            return pd.DataFrame(columns=["feature", "importance"])

        frame = pd.DataFrame({"feature": names, "importance": np.asarray(values, dtype=float)})
        total = frame["importance"].abs().sum()
        frame["importance_pct"] = frame["importance"].abs() / total if total > 0 else 0.0
        frame = frame.reindex(frame["importance"].abs().sort_values(ascending=False).index)
        frame = frame.reset_index(drop=True)
        return frame.head(top) if top else frame

    def _raw_importance(self) -> np.ndarray | None:
        if hasattr(self.estimator, "feature_importances_"):
            return np.asarray(self.estimator.feature_importances_)
        if hasattr(self.estimator, "coef_"):
            return np.asarray(self.estimator.coef_).ravel()
        return None

    # ------------------------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params,
            "seed": self.seed,
            "fitted_at": self.fitted_at,
            "training_rows": self.training_rows,
            "best_iteration": self.best_iteration,
            "feature_version": self.schema.feature_version,
            "schema_fingerprint": self.schema.fingerprint,
            "input_features": len(self.schema.input_columns),
            "matrix_features": len(self.preprocessor.output_names) if self.preprocessor.is_fitted else 0,
        }

    def save(self, directory: str | Path) -> Path:
        if not self.is_fitted:
            raise ModelError(f"cannot save an unfitted {self.name} model")

        import json

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, target / ESTIMATOR_FILENAME)
        self.preprocessor.save(target)
        (target / METADATA_FILENAME).write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")

        logger.info("model saved", extra={"model": self.name, "path": str(target)})
        return target

    @classmethod
    def load(cls, directory: str | Path) -> RacingModel:
        import json

        target = Path(directory)
        metadata_path = target / METADATA_FILENAME
        if not metadata_path.exists():
            raise ModelError(f"no model metadata at {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        preprocessor = FeaturePreprocessor.load(target)

        model = cls(preprocessor.schema, metadata.get("params"), seed=metadata.get("seed", DEFAULT_SEED))
        model.preprocessor = preprocessor
        model.estimator = joblib.load(target / ESTIMATOR_FILENAME)
        model.fitted_at = metadata.get("fitted_at")
        model.training_rows = metadata.get("training_rows", 0)
        model.best_iteration = metadata.get("best_iteration")

        if model.schema.fingerprint != metadata.get("schema_fingerprint"):
            raise ModelError(
                f"schema fingerprint mismatch for {model.name}: the saved feature contract "
                "does not match the stored schema file"
            )
        return model


__all__ = ["DEFAULT_SEED", "ESTIMATOR_FILENAME", "METADATA_FILENAME", "RacingModel"]
