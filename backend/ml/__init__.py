"""The probability engine: features in, calibrated win probabilities out.

    from backend.ml import TrainingDatasetBuilder, ModelTrainer, RacePredictor

    data = TrainingDatasetBuilder(session).build()
    run = ModelTrainer().train(data)
    print(run.render())

    prediction = RacePredictor(session).predict_race("rac_18f2a")

Pipeline::

    point-in-time features (Phase 3)
        |
    TrainingDatasetBuilder     train / valid_stop / valid_calib / test, by date
        |
    FeaturePreprocessor        impute, scale, one-hot -> model matrix
        |
    RacingModel                logistic regression | XGBoost | LightGBM
        |
    ProbabilityCalibrator      Platt or isotonic, fitted on a later slice
        |
    normalise_within_race      one horse wins, so probabilities sum to 1
        |
    ModelRegistry              versioned, reproducible, promotable

Phase 4 stops at the probability. Comparing it with a bookmaker's price and
sizing a stake is Phase 5.
"""

from backend.ml.dataset import SplitConfig, TrainingData, TrainingDatasetBuilder
from backend.ml.evaluate import EvaluationReport, comparison_table, evaluate_model, select_champion
from backend.ml.models.calibrated import CalibratedRacingModel
from backend.ml.models.registry import ModelRecord, ModelRegistry
from backend.ml.predict import RacePrediction, RacePredictor, RunnerPrediction
from backend.ml.train import ModelTrainer, TrainingConfig, TrainingRun, train_models

__all__ = [
    "CalibratedRacingModel",
    "EvaluationReport",
    "ModelRecord",
    "ModelRegistry",
    "ModelTrainer",
    "RacePrediction",
    "RacePredictor",
    "RunnerPrediction",
    "SplitConfig",
    "TrainingConfig",
    "TrainingData",
    "TrainingDatasetBuilder",
    "TrainingRun",
    "comparison_table",
    "evaluate_model",
    "select_champion",
    "train_models",
]
