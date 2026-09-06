"""Preprocessing, models, calibration, registry and inference."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backend.ml.calibration.calibrators import (
    MIN_POSITIVES_FOR_ISOTONIC,
    ProbabilityCalibrator,
    compare_calibration,
)
from backend.ml.evaluate import comparison_table, evaluate_model, select_champion
from backend.ml.models.calibrated import CalibratedRacingModel, model_class_for
from backend.ml.models.gradient_boosting import LightGBMModel, XGBoostModel
from backend.ml.models.logistic import LogisticRegressionModel
from backend.ml.models.registry import ModelRegistry, library_versions
from backend.ml.preprocessing.pipeline import FeaturePipeline, FeaturePreprocessor
from backend.ml.preprocessing.schema import FeatureSchema, FeatureSchemaError
from backend.ml.train import ModelTrainer, TrainingConfig
from backend.utils.exceptions import ModelError
from tests.fixtures.ml_builders import make_ml_session, make_training_data

pytestmark = pytest.mark.unit

MODEL_CLASSES = [LogisticRegressionModel, XGBoostModel, LightGBMModel]


@pytest.fixture(scope="module")
def ml_session():
    session = make_ml_session()
    yield session
    session.close()


@pytest.fixture(scope="module")
def training_data(ml_session):
    return make_training_data(ml_session)


@pytest.fixture(scope="module")
def schema(training_data) -> FeatureSchema:
    return FeatureSchema.from_frame(
        training_data.train, training_data.numeric_features, training_data.categorical_features
    )


@pytest.fixture(scope="module")
def fitted_lightgbm(training_data, schema):
    model = LightGBMModel(schema, {"n_estimators": 60})
    model.fit(
        training_data.train,
        training_data.train["won"],
        training_data.valid_stop,
        training_data.valid_stop["won"],
    )
    return model


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
def test_schema_records_column_order_and_hashes_it(schema):
    assert schema.input_columns == [*schema.numeric, *schema.categorical]
    assert len(schema.fingerprint) == 16


def test_fingerprint_changes_when_the_contract_changes(schema):
    altered = FeatureSchema(numeric=[*schema.numeric, "extra"], categorical=schema.categorical)
    assert altered.fingerprint != schema.fingerprint


def test_schema_rejects_a_frame_with_missing_features(schema, training_data):
    with pytest.raises(FeatureSchemaError, match="missing"):
        schema.validate(training_data.train.drop(columns=schema.numeric[:3]))


def test_schema_align_enforces_the_trained_order(schema, training_data):
    shuffled = training_data.train[list(reversed(training_data.train.columns))]
    assert list(schema.align(shuffled).columns) == schema.input_columns


def test_schema_round_trips_through_disk(schema, tmp_path):
    schema.save(tmp_path)
    reloaded = FeatureSchema.load(tmp_path)
    assert reloaded.fingerprint == schema.fingerprint
    assert reloaded.input_columns == schema.input_columns


def test_loading_a_missing_schema_raises(tmp_path):
    with pytest.raises(FeatureSchemaError, match="no feature schema"):
        FeatureSchema.load(tmp_path)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def test_preprocessor_produces_a_finite_numeric_matrix(schema, training_data):
    matrix = FeaturePreprocessor(schema).fit_transform(training_data.train)

    assert len(matrix) == len(training_data.train)
    assert np.isfinite(matrix.to_numpy()).all(), "matrix contains NaN or inf"
    assert matrix.shape[1] == len(FeaturePreprocessor(schema).fit(training_data.train).output_names)


def test_missing_values_produce_indicator_columns(schema, training_data):
    preprocessor = FeaturePreprocessor(schema, add_missing_indicators=True).fit(training_data.train)
    assert any("missing" in name for name in preprocessor.output_names)


def test_scaling_standardises_only_when_asked(schema, training_data):
    scaled = FeaturePreprocessor(schema, scale=True).fit_transform(training_data.train)
    unscaled = FeaturePreprocessor(schema, scale=False).fit_transform(training_data.train)

    numeric_scaled = scaled[[c for c in scaled.columns if c in schema.numeric]]
    numeric_unscaled = unscaled[[c for c in unscaled.columns if c in schema.numeric]]
    assert abs(numeric_scaled.to_numpy().mean()) < 0.2
    assert numeric_unscaled.to_numpy().std() != pytest.approx(numeric_scaled.to_numpy().std())


def test_unseen_categories_do_not_raise(schema, training_data):
    """A course that never appeared in training must still be scoreable."""
    preprocessor = FeaturePreprocessor(schema).fit(training_data.train)

    novel = training_data.test.head(5).copy()
    if schema.categorical:
        novel[schema.categorical[0]] = "a_course_that_never_existed"
    matrix = preprocessor.transform(novel)
    assert len(matrix) == 5
    assert np.isfinite(matrix.to_numpy()).all()


def test_transform_before_fit_raises(schema, training_data):
    with pytest.raises(FeatureSchemaError, match="must be fitted"):
        FeaturePreprocessor(schema).transform(training_data.train)


def test_preprocessor_round_trips(schema, training_data, tmp_path):
    original = FeaturePreprocessor(schema).fit(training_data.train)
    original.save(tmp_path)

    reloaded = FeaturePreprocessor.load(tmp_path)
    pd.testing.assert_frame_equal(
        original.transform(training_data.test.head(20)),
        reloaded.transform(training_data.test.head(20)),
    )


def test_specification_alias_is_the_same_class():
    assert FeaturePipeline is FeaturePreprocessor


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_every_model_fits_and_predicts_in_range(model_class, training_data, schema):
    model = model_class(schema, {"n_estimators": 40} if model_class is not LogisticRegressionModel else None)
    model.fit(
        training_data.train,
        training_data.train["won"],
        training_data.valid_stop,
        training_data.valid_stop["won"],
    )
    probabilities = model.predict_proba(training_data.test)

    assert len(probabilities) == len(training_data.test)
    assert ((probabilities >= 0) & (probabilities <= 1)).all(), "probabilities left [0, 1]"
    assert np.isfinite(probabilities).all()


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_predicting_before_fitting_raises(model_class, schema, training_data):
    with pytest.raises(ModelError, match="not fitted"):
        model_class(schema).predict_proba(training_data.test)


def test_predictions_are_deterministic(fitted_lightgbm, training_data):
    first = fitted_lightgbm.predict_proba(training_data.test)
    second = fitted_lightgbm.predict_proba(training_data.test)
    np.testing.assert_array_equal(first, second)


def test_same_seed_gives_the_same_model(training_data, schema):
    def train(seed: int) -> np.ndarray:
        model = LightGBMModel(schema, {"n_estimators": 40}, seed=seed)
        model.fit(training_data.train, training_data.train["won"])
        return model.predict_proba(training_data.test.head(50))

    np.testing.assert_allclose(train(7), train(7))


def test_predicting_an_empty_frame_returns_nothing(fitted_lightgbm):
    assert fitted_lightgbm.predict_proba(pd.DataFrame()).size == 0


def test_feature_importance_is_ranked(fitted_lightgbm):
    importance = fitted_lightgbm.feature_importance(top=10)

    assert len(importance) == 10
    assert list(importance.columns) == ["feature", "importance", "importance_pct"]
    assert importance["importance"].abs().is_monotonic_decreasing


def test_logistic_coefficients_carry_a_sign(training_data, schema):
    model = LogisticRegressionModel(schema)
    model.fit(training_data.train, training_data.train["won"])

    coefficients = model.coefficients(top=5)
    assert "coefficient" in coefficients.columns
    assert set(coefficients["direction"]) <= {"+", "-"}


def test_gradient_boosting_early_stops(training_data, schema):
    model = XGBoostModel(schema, {"n_estimators": 500})
    model.fit(
        training_data.train,
        training_data.train["won"],
        training_data.valid_stop,
        training_data.valid_stop["won"],
    )
    assert model.best_iteration is not None
    assert model.best_iteration < 500


def test_model_without_validation_still_fits(training_data, schema):
    model = XGBoostModel(schema, {"n_estimators": 30})
    model.fit(training_data.train, training_data.train["won"])
    assert model.is_fitted


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_models_round_trip_through_disk(model_class, training_data, schema, tmp_path):
    model = model_class(schema, {"n_estimators": 30} if model_class is not LogisticRegressionModel else None)
    model.fit(training_data.train, training_data.train["won"])
    model.save(tmp_path)

    reloaded = model_class.load(tmp_path)
    np.testing.assert_allclose(
        model.predict_proba(training_data.test.head(30)),
        reloaded.predict_proba(training_data.test.head(30)),
    )


def test_saving_an_unfitted_model_raises(schema, tmp_path):
    with pytest.raises(ModelError, match="unfitted"):
        LogisticRegressionModel(schema).save(tmp_path)


def test_loading_from_an_empty_directory_raises(tmp_path):
    with pytest.raises(ModelError, match="no model metadata"):
        LightGBMModel.load(tmp_path)


def test_model_class_lookup():
    assert model_class_for("xgboost") is XGBoostModel
    with pytest.raises(ModelError, match="unknown model"):
        model_class_for("neural_net")


def test_models_do_not_rebalance_classes():
    """Rebalancing changes the base rate the model is fitted to, so its output
    stops being a probability of winning."""
    for model_class in MODEL_CLASSES:
        params = model_class.default_params()
        assert "scale_pos_weight" not in params
        assert params.get("class_weight") is None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_platt_scaling_corrects_systematic_overconfidence():
    rng = np.random.default_rng(0)
    true_probability = rng.uniform(0.02, 0.4, 6000)
    truth = rng.binomial(1, true_probability)
    inflated = np.clip(true_probability * 2.2, 0, 0.999)

    calibrator = ProbabilityCalibrator("sigmoid").fit(inflated, truth)
    corrected = calibrator.transform(inflated)

    assert abs(corrected.mean() - truth.mean()) < abs(inflated.mean() - truth.mean())


def test_isotonic_calibration_also_corrects():
    rng = np.random.default_rng(1)
    true_probability = rng.uniform(0.02, 0.5, 8000)
    truth = rng.binomial(1, true_probability)
    inflated = np.clip(true_probability * 1.8, 0, 0.999)

    corrected = ProbabilityCalibrator("isotonic").fit(inflated, truth).transform(inflated)
    assert abs(corrected.mean() - truth.mean()) < abs(inflated.mean() - truth.mean())


def test_calibrated_output_stays_in_range():
    rng = np.random.default_rng(2)
    scores = rng.uniform(0, 1, 3000)
    truth = rng.binomial(1, scores)

    for method in ("sigmoid", "isotonic"):
        calibrated = ProbabilityCalibrator(method).fit(scores, truth).transform(scores)
        assert ((calibrated >= 0) & (calibrated <= 1)).all()


def test_none_calibration_is_the_identity():
    scores = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(ProbabilityCalibrator("none").transform(scores), scores)


def test_isotonic_downgrades_when_winners_are_scarce():
    rng = np.random.default_rng(3)
    scores = rng.uniform(0, 0.3, 500)
    truth = rng.binomial(1, 0.05, 500)

    calibrator = ProbabilityCalibrator("isotonic").fit(scores, truth)
    if truth.sum() < MIN_POSITIVES_FOR_ISOTONIC:
        assert calibrator.downgraded


def test_calibrating_without_positives_raises():
    with pytest.raises(ModelError, match="no positive"):
        ProbabilityCalibrator("sigmoid").fit(np.full(100, 0.2), np.zeros(100))


def test_unknown_calibration_method_rejected():
    with pytest.raises(ValueError, match="unknown calibration method"):
        ProbabilityCalibrator("magic")  # type: ignore[arg-type]


def test_calibrator_round_trips(tmp_path):
    rng = np.random.default_rng(4)
    scores = rng.uniform(0, 1, 2000)
    truth = rng.binomial(1, scores)

    original = ProbabilityCalibrator("sigmoid").fit(scores, truth)
    original.save(tmp_path)
    reloaded = ProbabilityCalibrator.load(tmp_path)

    np.testing.assert_allclose(original.transform(scores), reloaded.transform(scores))


def test_loading_a_missing_calibrator_raises(tmp_path):
    with pytest.raises(ModelError, match="no calibrator"):
        ProbabilityCalibrator.load(tmp_path)


def test_comparison_reports_both_sides():
    rng = np.random.default_rng(5)
    true_probability = rng.uniform(0.02, 0.4, 5000)
    truth = rng.binomial(1, true_probability)
    inflated = np.clip(true_probability * 2, 0, 0.999)
    calibrated = ProbabilityCalibrator("sigmoid").fit(inflated, truth).transform(inflated)

    comparison = compare_calibration(truth, inflated, calibrated)
    assert comparison.after_ece < comparison.before_ece
    assert comparison.improved
    assert "Calibration" in comparison.render()


def test_comparison_refuses_to_claim_improvement_when_log_loss_degrades():
    """An already-calibrated model can have its ECE nudged down by noise."""
    from backend.ml.calibration.calibrators import CalibrationComparison
    from backend.ml.metrics.classification import ReliabilityCurve

    comparison = CalibrationComparison(
        method="isotonic",
        before_brier=0.08,
        after_brier=0.08,
        before_log_loss=0.26,
        after_log_loss=0.30,  # materially worse
        before_ece=0.010,
        after_ece=0.008,  # nominally better
        before_curve=ReliabilityCurve(),
        after_curve=ReliabilityCurve(),
    )
    assert not comparison.improved


# ---------------------------------------------------------------------------
# Calibrated model wrapper
# ---------------------------------------------------------------------------
def test_race_probabilities_sum_to_one(fitted_lightgbm, training_data):
    calibrated = CalibratedRacingModel(fitted_lightgbm)
    probabilities = calibrated.predict_race_proba(training_data.test)

    totals = (
        pd.DataFrame({"race_id": training_data.test["race_id"], "p": probabilities})
        .groupby("race_id")["p"]
        .sum()
    )
    assert totals.round(6).eq(1.0).all()


def test_race_normalisation_needs_race_ids(fitted_lightgbm, training_data):
    with pytest.raises(ModelError, match="race_id"):
        CalibratedRacingModel(fitted_lightgbm).predict_race_proba(
            training_data.test.drop(columns=["race_id"])
        )


def test_calibrated_name_reflects_the_method(fitted_lightgbm, training_data):
    plain = CalibratedRacingModel(fitted_lightgbm)
    assert plain.name == "lightgbm"
    assert not plain.is_calibrated

    calibrator = ProbabilityCalibrator("sigmoid").fit(
        fitted_lightgbm.predict_proba(training_data.valid_calib), training_data.valid_calib["won"]
    )
    assert CalibratedRacingModel(fitted_lightgbm, calibrator).name == "lightgbm+sigmoid"


def test_calibrated_model_round_trips(fitted_lightgbm, training_data, tmp_path):
    calibrator = ProbabilityCalibrator("isotonic").fit(
        fitted_lightgbm.predict_proba(training_data.valid_calib), training_data.valid_calib["won"]
    )
    original = CalibratedRacingModel(fitted_lightgbm, calibrator)
    original.save(tmp_path)

    reloaded = CalibratedRacingModel.load(tmp_path)
    np.testing.assert_allclose(
        original.predict_proba(training_data.test.head(40)),
        reloaded.predict_proba(training_data.test.head(40)),
    )
    assert reloaded.name == original.name


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_versions_increment(fitted_lightgbm, tmp_path):
    registry = ModelRegistry(tmp_path)
    model = CalibratedRacingModel(fitted_lightgbm)

    first = registry.register(model, metrics={"classification": {"log_loss": 0.3}})
    second = registry.register(model, metrics={"classification": {"log_loss": 0.29}})

    assert first.version == "v1"
    assert second.version == "v2"


def test_registry_records_provenance(fitted_lightgbm, tmp_path):
    registry = ModelRegistry(tmp_path)
    record = registry.register(
        CalibratedRacingModel(fitted_lightgbm),
        metrics={"classification": {"log_loss": 0.3}},
        split={"train": "2018 → 2023"},
    )

    assert record.feature_version == "v1"
    assert record.schema_fingerprint
    assert record.training_rows > 0
    assert "sklearn" in record.libraries
    assert record.split["train"] == "2018 → 2023"


def test_registry_loads_what_it_saved(fitted_lightgbm, training_data, tmp_path):
    registry = ModelRegistry(tmp_path)
    original = CalibratedRacingModel(fitted_lightgbm)
    registry.register(original, metrics={})

    reloaded = registry.load("lightgbm")
    np.testing.assert_allclose(
        original.predict_proba(training_data.test.head(25)),
        reloaded.predict_proba(training_data.test.head(25)),
    )


def test_first_registration_becomes_champion(fitted_lightgbm, tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register(CalibratedRacingModel(fitted_lightgbm), metrics={})
    assert registry.champion().name == "lightgbm"


def test_promotion_is_reversible(fitted_lightgbm, tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register(CalibratedRacingModel(fitted_lightgbm), metrics={})
    registry.register(CalibratedRacingModel(fitted_lightgbm), metrics={})

    registry.promote("lightgbm", "v2")
    assert registry.champion().version == "v2"
    registry.promote("lightgbm", "v1")
    assert registry.champion().version == "v1"


def test_unknown_model_and_version_raise(fitted_lightgbm, tmp_path):
    registry = ModelRegistry(tmp_path)
    with pytest.raises(ModelError, match="no registered model"):
        registry.get("nonexistent")

    registry.register(CalibratedRacingModel(fitted_lightgbm), metrics={})
    with pytest.raises(ModelError, match="has no version"):
        registry.get("lightgbm", "v99")


def test_loading_a_champion_before_any_exists_raises(tmp_path):
    with pytest.raises(ModelError, match="no champion"):
        ModelRegistry(tmp_path).load_champion()


def test_registry_index_is_valid_json(fitted_lightgbm, tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register(CalibratedRacingModel(fitted_lightgbm), metrics={})
    payload = json.loads((tmp_path / "registry.json").read_text())
    assert "models" in payload and "champion" in payload


def test_comparison_table_is_sorted_by_log_loss(fitted_lightgbm, tmp_path):
    registry = ModelRegistry(tmp_path)
    registry.register(CalibratedRacingModel(fitted_lightgbm), metrics={"classification": {"log_loss": 0.4}})
    registry.register(CalibratedRacingModel(fitted_lightgbm), metrics={"classification": {"log_loss": 0.2}})

    table = registry.comparison_table()
    assert [row["log_loss"] for row in table] == [0.2, 0.4]


def test_library_versions_are_recorded():
    versions = library_versions()
    assert "python" in versions
    assert "sklearn" in versions


# ---------------------------------------------------------------------------
# Evaluation and champion selection
# ---------------------------------------------------------------------------
def test_evaluation_covers_all_three_axes(fitted_lightgbm, training_data):
    report = evaluate_model(CalibratedRacingModel(fitted_lightgbm), training_data.test)

    assert report.classification.rows == len(training_data.test)
    assert report.racing.races > 0
    assert 0 <= report.racing.top1_hit_rate <= 1
    assert "Probability quality" in report.render()
    assert "classification" in report.as_dict()


def test_evaluating_an_empty_split_is_safe(fitted_lightgbm):
    report = evaluate_model(CalibratedRacingModel(fitted_lightgbm), pd.DataFrame())
    assert report.classification.rows == 0


def test_champion_selection_uses_the_stated_criterion():
    from backend.ml.evaluate import EvaluationReport
    from backend.ml.metrics.classification import ClassificationMetrics
    from backend.ml.metrics.racing import RacingMetrics

    worse = EvaluationReport(
        model="worse",
        split="test",
        classification=ClassificationMetrics(log_loss=0.20, roc_auc=0.90),
        racing=RacingMetrics(race_log_loss=1.60),
    )
    better = EvaluationReport(
        model="better",
        split="test",
        classification=ClassificationMetrics(log_loss=0.25, roc_auc=0.85),
        racing=RacingMetrics(race_log_loss=1.40),
    )

    # Race log loss is the default and prefers the race-level winner.
    assert select_champion([worse, better]).model == "better"
    # Per-runner log loss prefers the other one -- the criterion matters.
    assert select_champion([worse, better], criterion="log_loss").model == "worse"
    assert select_champion([worse, better], criterion="roc_auc").model == "worse"


def test_champion_selection_rejects_an_unknown_criterion():
    from backend.ml.evaluate import EvaluationReport
    from backend.ml.metrics.classification import ClassificationMetrics
    from backend.ml.metrics.racing import RacingMetrics

    report = EvaluationReport("m", "test", ClassificationMetrics(), RacingMetrics())
    with pytest.raises(ValueError, match="unknown criterion"):
        select_champion([report], criterion="vibes")


def test_champion_of_nothing_is_none():
    assert select_champion([]) is None


def test_comparison_table_columns(fitted_lightgbm, training_data):
    report = evaluate_model(CalibratedRacingModel(fitted_lightgbm), training_data.test)
    table = comparison_table([report])
    assert {"model", "log_loss", "brier", "roc_auc", "race_log_loss"} <= set(table.columns)


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------
def test_training_run_produces_variants_and_a_champion(training_data, tmp_path):
    config = TrainingConfig(
        models=("logistic_regression", "lightgbm"),
        calibrations=("none", "sigmoid"),
        params={"lightgbm": {"n_estimators": 40}},
    )
    run = ModelTrainer(config, ModelRegistry(tmp_path)).train(training_data)

    assert len(run.reports) == 4  # 2 models x 2 calibrations
    assert run.champion is not None
    assert not run.comparison.empty
    assert "Model comparison" in run.render()


def test_training_registers_and_promotes(training_data, tmp_path):
    registry = ModelRegistry(tmp_path)
    config = TrainingConfig(
        models=("lightgbm",), calibrations=("none",), params={"lightgbm": {"n_estimators": 30}}
    )
    ModelTrainer(config, registry).train(training_data)

    assert registry.champion() is not None
    assert registry.load_champion().model.name == "lightgbm"


def test_training_rejects_an_unknown_model_type():
    with pytest.raises(ModelError, match="unknown model type"):
        TrainingConfig(models=("random_forest",))


def test_training_refuses_a_leaked_target(training_data, tmp_path):
    """The guardrail that would have caught the ``is_winner`` bug."""
    leaked = training_data.train.copy()
    leaked["sneaky_copy"] = leaked["won"].astype(float)

    from backend.ml.dataset import TrainingData

    broken = TrainingData(
        train=leaked,
        valid_stop=training_data.valid_stop,
        valid_calib=training_data.valid_calib,
        test=training_data.test,
        numeric_features=[*training_data.numeric_features, "sneaky_copy"],
        categorical_features=training_data.categorical_features,
    )
    with pytest.raises(ModelError, match="target leakage detected"):
        ModelTrainer(TrainingConfig(models=("lightgbm",)), ModelRegistry(tmp_path)).train(broken)


def test_training_on_an_empty_split_raises(training_data, tmp_path):
    from backend.ml.dataset import TrainingData

    empty = TrainingData(
        train=training_data.train.iloc[0:0],
        valid_stop=training_data.valid_stop,
        valid_calib=training_data.valid_calib,
        test=training_data.test,
        numeric_features=training_data.numeric_features,
        categorical_features=training_data.categorical_features,
    )
    with pytest.raises(ModelError, match="training split is empty"):
        ModelTrainer(TrainingConfig(models=("lightgbm",)), ModelRegistry(tmp_path)).train(empty)
