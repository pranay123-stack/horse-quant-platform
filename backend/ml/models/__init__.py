"""Estimator wrappers.

.. note::
   Not to be confused with ``backend/models/`` (SQLAlchemy ORM entities) or the
   repository-root ``models/`` directory (serialised artefacts on disk).
"""

from backend.ml.models.base import RacingModel
from backend.ml.models.calibrated import CalibratedRacingModel, model_class_for
from backend.ml.models.gradient_boosting import LightGBMModel, XGBoostModel
from backend.ml.models.logistic import LogisticRegressionModel
from backend.ml.models.registry import ModelRecord, ModelRegistry

__all__ = [
    "CalibratedRacingModel",
    "LightGBMModel",
    "LogisticRegressionModel",
    "ModelRecord",
    "ModelRegistry",
    "RacingModel",
    "XGBoostModel",
    "model_class_for",
]
