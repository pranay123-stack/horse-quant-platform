"""End-to-end probability engine against a real PostgreSQL.

Runs the whole Phase 4 chain — synthetic history → features → training splits →
three model families → calibration → registry → inference — on the database that
will actually hold it.

    make db-up
    .venv/bin/pytest -m integration
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.database.base import Base
from backend.ml import ModelRegistry, ModelTrainer, RacePredictor, SplitConfig, TrainingDatasetBuilder
from backend.ml.dataset import detect_leaky_features
from backend.ml.train import TrainingConfig
from backend.research.synthetic import SyntheticConfig, generate_synthetic_data
from backend.utils.config import Settings

pytestmark = pytest.mark.integration

SCHEMA = "ml_test"

SEASON = SyntheticConfig(
    n_horses=300,
    n_jockeys=30,
    n_trainers=25,
    n_courses=6,
    start_date=date(2023, 1, 1),
    n_days=540,
    races_per_day=2,
    runners_per_race=8,
    seed=31,
)

SPLIT = SplitConfig(
    train_start=date(2023, 1, 1),
    train_end=date(2023, 12, 31),
    valid_start=date(2024, 1, 1),
    valid_end=date(2024, 3, 31),
    test_start=date(2024, 4, 1),
    test_end=date(2024, 6, 30),
    embargo_days=7,
)


def _settings() -> Settings:
    return Settings(
        postgres_host=os.environ.get("TEST_POSTGRES_HOST", "localhost"),
        postgres_port=int(os.environ.get("TEST_POSTGRES_PORT", "5432")),
        postgres_user=os.environ.get("TEST_POSTGRES_USER", "horse_quant"),
        postgres_password=os.environ.get("TEST_POSTGRES_PASSWORD", "horse_quant"),
        postgres_db=os.environ.get("TEST_POSTGRES_DB", "horse_quant"),
        DATABASE_URL=None,
    )


@pytest.fixture(scope="module")
def pg_session():
    settings = _settings()
    admin = create_engine(settings.database_url, pool_pre_ping=True)
    try:
        with admin.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL not available: {exc}")

    with admin.begin() as connection:
        connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        connection.execute(text(f"CREATE SCHEMA {SCHEMA}"))

    engine = create_engine(settings.database_url, connect_args={"options": f"-csearch_path={SCHEMA}"})
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as session:
        generate_synthetic_data(session, SEASON)
        yield session

    engine.dispose()
    with admin.begin() as connection:
        connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    admin.dispose()


@pytest.fixture(scope="module")
def training_data(pg_session):
    return TrainingDatasetBuilder(pg_session).build(SPLIT)


@pytest.fixture(scope="module")
def training_run(training_data, tmp_path_factory):
    registry = ModelRegistry(tmp_path_factory.mktemp("ml_registry"))
    config = TrainingConfig(
        models=("logistic_regression", "xgboost", "lightgbm"),
        calibrations=("none", "sigmoid", "isotonic"),
        params={"xgboost": {"n_estimators": 200}, "lightgbm": {"n_estimators": 200}},
    )
    return ModelTrainer(config, registry).train(training_data), registry


# ---------------------------------------------------------------------------
def test_splits_cover_the_configured_windows(training_data):
    assert not training_data.train.empty
    assert not training_data.valid_stop.empty
    assert not training_data.valid_calib.empty
    assert not training_data.test.empty
    training_data.assert_no_leakage()


def test_dataset_has_no_leaky_features(training_data):
    suspects = detect_leaky_features(training_data.train, training_data.feature_names)
    assert suspects == [], f"leaky features: {[str(s) for s in suspects]}"


def test_all_three_model_families_train(training_run):
    run, _ = training_run
    trained = {name.split("+")[0] for name in run.models}
    assert trained == {"logistic_regression", "xgboost", "lightgbm"}


def test_every_model_beats_the_uniform_baseline(training_run):
    """A model with no skill is worse than useless — it would still place bets."""
    run, _ = training_run
    for report in run.reports:
        assert report.racing.skill_score > 0, f"{report.model} has no skill over uniform"
        assert report.racing.race_log_loss < report.racing.uniform_log_loss


def test_probabilities_are_valid_for_every_model(training_run, training_data):
    run, _ = training_run
    for name, model in run.models.items():
        probabilities = model.predict_proba(training_data.test)
        assert ((probabilities >= 0) & (probabilities <= 1)).all(), name

        normalised = model.predict_race_proba(training_data.test)
        totals = (
            pd.DataFrame({"race_id": training_data.test["race_id"], "p": normalised})
            .groupby("race_id")["p"]
            .sum()
        )
        assert totals.round(6).eq(1.0).all(), name


def test_comparison_table_ranks_every_variant(training_run):
    run, _ = training_run
    table = run.comparison
    assert len(table) == len(run.reports)
    assert table["log_loss"].is_monotonic_increasing


def test_a_champion_is_selected_and_promoted(training_run):
    run, registry = training_run
    assert run.champion is not None
    assert registry.champion() is not None


def test_registry_round_trips_the_champion(training_run, training_data):
    _, registry = training_run
    champion = registry.load_champion()
    probabilities = champion.predict_proba(training_data.test.head(50))
    assert len(probabilities) == 50


def test_calibration_is_measured_not_assumed(training_run):
    """Calibration is compared honestly; it is not always an improvement."""
    run, _ = training_run
    calibrated = [r for r in run.reports if r.calibration is not None]
    assert calibrated, "no calibration comparisons were produced"
    for report in calibrated:
        payload = report.calibration.as_dict()
        assert "before" in payload and "after" in payload
        assert isinstance(payload["improved"], bool)


def test_within_race_normalisation_helps(training_run):
    """Imposing "one horse wins" should not make the probabilities worse."""
    run, _ = training_run
    for report in run.reports:
        if report.normalised_classification is None:
            continue
        assert report.normalised_classification.log_loss <= report.classification.log_loss * 1.05


def test_end_to_end_race_prediction(pg_session, training_run, training_data):
    _, registry = training_run
    predictor = RacePredictor(pg_session, registry=registry)

    race_id = training_data.test["race_id"].iloc[0]
    prediction = predictor.predict_race(race_id)

    assert prediction.probability_sum == pytest.approx(1.0, abs=1e-6)
    assert prediction.favourite.probability >= prediction.runners[-1].probability
    assert all(0 <= runner.probability <= 1 for runner in prediction.runners)


def test_feature_importance_is_available(training_run):
    run, _ = training_run
    for report in run.reports:
        if report.importance is None:
            continue
        assert not report.importance.empty
        assert report.importance["importance_pct"].sum() == pytest.approx(1.0, abs=0.01)
