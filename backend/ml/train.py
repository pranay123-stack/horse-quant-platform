"""Training orchestration.

One pass produces, for every model type, three calibration variants that share a
single fitted estimator:

* ``none`` — the raw output, as the baseline calibration has to beat
* ``sigmoid`` — Platt scaling
* ``isotonic`` — isotonic regression

Fitting the estimator once and attaching three calibrators is not just faster; it
makes the comparison *clean*. Any difference between the variants is attributable
to calibration alone, because the underlying scores are identical.

Data discipline
===============
``train`` fits the estimator and the preprocessor. ``valid_stop`` chooses the
stopping iteration. ``valid_calib`` fits the calibrators. ``test`` is scored once,
at the end, and is never seen by any fitting step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backend.ml.calibration.calibrators import CalibrationMethod, ProbabilityCalibrator
from backend.ml.dataset import TrainingData, detect_leaky_features
from backend.ml.evaluate import EvaluationReport, comparison_table, evaluate_model, select_champion
from backend.ml.models.base import DEFAULT_SEED, RacingModel
from backend.ml.models.calibrated import CalibratedRacingModel
from backend.ml.models.gradient_boosting import LightGBMModel, XGBoostModel
from backend.ml.models.logistic import LogisticRegressionModel
from backend.ml.models.registry import ModelRecord, ModelRegistry
from backend.ml.preprocessing.schema import FeatureSchema
from backend.utils.exceptions import ModelError
from backend.utils.logging import get_logger, safe_extra

logger = get_logger(__name__, channel="model")

MODEL_TYPES: dict[str, type[RacingModel]] = {
    "logistic_regression": LogisticRegressionModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
}

DEFAULT_CALIBRATIONS: tuple[CalibrationMethod, ...] = ("none", "sigmoid", "isotonic")


@dataclass(slots=True)
class TrainingConfig:
    """What to train, and how."""

    models: tuple[str, ...] = ("logistic_regression", "xgboost", "lightgbm")
    calibrations: tuple[CalibrationMethod, ...] = DEFAULT_CALIBRATIONS
    seed: int = DEFAULT_SEED
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    champion_criterion: str = "race_log_loss"
    register: bool = True
    #: Registered models are limited to the best calibration variant per model
    #: type; the rest exist only for the comparison table.
    register_best_variant_only: bool = True

    def __post_init__(self) -> None:
        unknown = [name for name in self.models if name not in MODEL_TYPES]
        if unknown:
            raise ModelError(f"unknown model type(s) {unknown}; expected {sorted(MODEL_TYPES)}")


@dataclass(slots=True)
class TrainingRun:
    """Everything one training pass produced."""

    models: dict[str, CalibratedRacingModel] = field(default_factory=dict)
    reports: list[EvaluationReport] = field(default_factory=list)
    validation_reports: list[EvaluationReport] = field(default_factory=list)
    records: list[ModelRecord] = field(default_factory=list)
    champion: EvaluationReport | None = None
    data_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def comparison(self) -> pd.DataFrame:
        return comparison_table(self.reports)

    def report_for(self, name: str) -> EvaluationReport | None:
        return next((report for report in self.reports if report.model == name), None)

    def render(self) -> str:
        lines = ["Model comparison (test split)", "=" * 100]
        table = self.comparison
        if table.empty:
            lines.append("  no models trained")
        else:
            lines.append(table.to_string(index=False))
        if self.champion is not None:
            lines += ["", f"Champion: {self.champion.model}"]
        return "\n".join(lines)


class ModelTrainer:
    """Trains, calibrates, evaluates and registers the model family."""

    def __init__(self, config: TrainingConfig | None = None, registry: ModelRegistry | None = None) -> None:
        self.config = config or TrainingConfig()
        self.registry = registry or ModelRegistry()

    # ------------------------------------------------------------------
    def train(self, data: TrainingData) -> TrainingRun:
        self._check_data(data)

        run = TrainingRun(data_summary=data.summary())
        schema = FeatureSchema.from_frame(
            data.train,
            data.numeric_features,
            data.categorical_features,
            feature_version=data.feature_version,
        )
        logger.info(
            "training started",
            extra=safe_extra(
                {
                    "models": list(self.config.models),
                    "train_rows": len(data.train),
                    "features": len(schema.input_columns),
                    "schema": schema.fingerprint,
                }
            ),
        )

        for model_name in self.config.models:
            variants = self._train_one(model_name, schema, data)
            best_variant: EvaluationReport | None = None

            for calibrated, report in variants:
                run.models[calibrated.name] = calibrated
                run.reports.append(report)
                if best_variant is None or report.racing.race_log_loss < best_variant.racing.race_log_loss:
                    best_variant = report

            if self.config.register:
                to_register = (
                    [(c, r) for c, r in variants if best_variant and c.name == best_variant.model]
                    if self.config.register_best_variant_only
                    else variants
                )
                for calibrated, report in to_register:
                    run.records.append(
                        self.registry.register(
                            calibrated,
                            metrics=report.as_dict(),
                            split=data.config.describe(),
                            notes=f"calibration={calibrated.calibrator.method}",
                        )
                    )

        run.champion = select_champion(run.reports, criterion=self.config.champion_criterion)
        if run.champion is not None and self.config.register:
            base_name = run.champion.model.split("+")[0]
            try:
                self.registry.promote(base_name)
            except ModelError:  # pragma: no cover - only if registration was skipped
                logger.warning("champion could not be promoted", extra={"model": base_name})

        logger.info(
            "training complete",
            extra=safe_extra(
                {"models": len(run.models), "champion": run.champion.model if run.champion else None}
            ),
        )
        return run

    # ------------------------------------------------------------------
    def _train_one(
        self, model_name: str, schema: FeatureSchema, data: TrainingData
    ) -> list[tuple[CalibratedRacingModel, EvaluationReport]]:
        model_class = MODEL_TYPES[model_name]
        model = model_class(schema, self.config.params.get(model_name), seed=self.config.seed)

        model.fit(
            data.train[schema.input_columns],
            data.train[data.target],
            data.valid_stop[schema.input_columns] if not data.valid_stop.empty else None,
            data.valid_stop[data.target] if not data.valid_stop.empty else None,
        )

        # One set of raw scores on the calibration slice; every calibrator is
        # fitted from these, so variants differ only by the mapping.
        calibration_scores = model.predict_proba(data.valid_calib) if not data.valid_calib.empty else None

        variants: list[tuple[CalibratedRacingModel, EvaluationReport]] = []
        for method in self.config.calibrations:
            calibrator = ProbabilityCalibrator(method)
            if method != "none":
                if calibration_scores is None or data.valid_calib[data.target].sum() == 0:
                    logger.warning(
                        "skipping calibration: the calibration slice has no winners",
                        extra={"model": model_name, "method": method},
                    )
                    continue
                calibrator.fit(calibration_scores, data.valid_calib[data.target])

            calibrated = CalibratedRacingModel(model, calibrator)
            variants.append((calibrated, evaluate_model(calibrated, data.test, split="test")))
        return variants

    # ------------------------------------------------------------------
    @staticmethod
    def _check_data(data: TrainingData) -> None:
        if data.train.empty:
            raise ModelError("training split is empty")
        if data.train[data.target].sum() == 0:
            raise ModelError("training split contains no winners")
        if data.test.empty:
            logger.warning("test split is empty; reported metrics will be meaningless")
        data.assert_no_leakage()

        # Fail before spending minutes training on a leaked target. A model that
        # scores perfectly is never good news.
        suspects = detect_leaky_features(data.train, data.feature_names, data.target)
        if suspects:
            raise ModelError(
                "target leakage detected: "
                + ", ".join(str(suspect) for suspect in suspects[:5])
                + " — these features separate winners perfectly and must not be model inputs"
            )


def train_models(data: TrainingData, config: TrainingConfig | None = None, **kwargs: Any) -> TrainingRun:
    """Convenience wrapper around :class:`ModelTrainer`."""
    return ModelTrainer(config, **kwargs).train(data)


__all__ = [
    "DEFAULT_CALIBRATIONS",
    "MODEL_TYPES",
    "ModelTrainer",
    "TrainingConfig",
    "TrainingRun",
    "train_models",
]
