"""The product, end to end, against a real PostgreSQL.

Train a model on history, score a later day, apply the betting rules, store the
signals, settle them, and confirm the live path scores identically to the
backtest path. This is the whole business goal in one file:

    history → model → probabilities → EV → BET/NO_BET → dashboard → record

One expensive fixture trains once and every test reads from it.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.database.base import Base
from backend.models.predictions import Prediction, Recommendation
from backend.prediction_service import PredictionService, store_predictions
from backend.prediction_service.daily import run_daily_job, settle_predictions
from backend.prediction_service.performance import summarise_performance
from backend.prediction_service.replay import replay_predictions
from backend.prediction_service.training import load_manifest, train_production_model
from backend.research.synthetic import SyntheticConfig, generate_synthetic_data
from backend.utils.config import Settings

pytestmark = pytest.mark.integration

SCHEMA = "product_test"

SEASON = SyntheticConfig(
    n_horses=300,
    n_jockeys=25,
    n_trainers=20,
    n_courses=5,
    start_date=date(2021, 1, 1),
    n_days=1200,
    races_per_day=2,
    runners_per_race=8,
    seed=707,
)

TRAIN_START = date(2021, 1, 1)
TRAIN_END = date(2023, 6, 30)
TEST_START = date(2023, 7, 1)
TEST_END = date(2024, 4, 1)


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
def trained(pg_session, tmp_path_factory):
    """Train once. Every test below reads this model."""
    models_dir = tmp_path_factory.mktemp("models")
    model = train_production_model(
        pg_session,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        models_dir=models_dir,
        model_params={"lightgbm": {"n_estimators": 80}, "xgboost": {"n_estimators": 80}},
    )
    return model, models_dir


@pytest.fixture(scope="module")
def service(pg_session, trained):
    from backend.ml.models.registry import ModelRegistry

    _, models_dir = trained
    return PredictionService(pg_session, registry=ModelRegistry(models_dir))


def _first_scored_day(service) -> tuple[date, object]:
    """The first day in the test window that actually produces runners."""
    cursor = TEST_START
    while cursor < TEST_END:
        day = service.signals_for_date(cursor)
        if day.races:
            return cursor, day
        cursor += timedelta(days=1)
    pytest.skip("no scoreable day in the test window")


# ---------------------------------------------------------------------------
# Step 1 — training
# ---------------------------------------------------------------------------
def test_training_promotes_a_model_and_records_how(trained):
    model, _ = trained

    assert model.name
    assert model.version
    assert model.rows > 0
    assert model.feature_version
    assert model.dataset_hash, "a model without a dataset hash cannot be reproduced"
    assert model.train_start == str(TRAIN_START)


def test_all_three_candidates_are_compared(trained):
    """Logistic regression is the bar the boosters have to clear."""
    model, _ = trained
    names = {row.get("model") for row in model.comparison}

    assert len(model.comparison) >= 3
    assert any("logistic" in str(name) for name in names)
    assert all(row.get("race_log_loss") is not None for row in model.comparison)


def test_the_promoted_model_is_the_best_on_race_log_loss(trained):
    model, _ = trained
    losses = {row["model"]: row["race_log_loss"] for row in model.comparison}
    assert losses[model.name] == pytest.approx(min(losses.values()))


def test_the_manifest_is_written_where_production_looks_for_it(trained):
    model, models_dir = trained
    manifest = load_manifest(models_dir)

    assert manifest is not None
    assert manifest["model"] == model.name
    assert manifest["feature_version"] == model.feature_version


# ---------------------------------------------------------------------------
# Step 2 — scoring a day
# ---------------------------------------------------------------------------
def test_a_day_is_scored_into_races_and_runners(service):
    _, day = _first_scored_day(service)

    assert day.races
    assert day.total_runners > 0
    for race in day.races:
        assert race.race
        assert sum(runner.model_probability for runner in race.runners) == pytest.approx(1.0)


def test_every_runner_carries_a_recommendation(service):
    _, day = _first_scored_day(service)

    for race in day.races:
        for runner in race.runners:
            assert runner.recommendation in {Recommendation.BET, Recommendation.NO_BET}
            if runner.recommendation == Recommendation.NO_BET:
                assert runner.rejection_reason, "a refusal must say why"
            else:
                assert runner.rejection_reason is None


def test_recommended_bets_satisfy_the_production_rules(service):
    """The last thing between a model output and someone's money."""
    from backend.prediction_service.rules import RULES

    _, day = _first_scored_day(service)
    for _, runner in day.bets:
        assert runner.odds is not None
        assert RULES.min_odds <= runner.odds <= RULES.max_odds
        assert runner.model_probability > RULES.min_probability
        assert runner.expected_value is not None
        assert runner.expected_value > RULES.min_expected_value


# ---------------------------------------------------------------------------
# Step 6 — backtest / live consistency
# ---------------------------------------------------------------------------
def test_the_live_path_scores_exactly_as_the_model_does(service, pg_session):
    """If this diverges, the backtest stops being evidence about the product."""
    race_date, _ = _first_scored_day(service)
    result = replay_predictions(pg_session, race_date, service=service)

    assert result.runners_checked > 0
    assert result.max_absolute_difference == 0.0
    assert result.consistent, result.render()


# ---------------------------------------------------------------------------
# Step 3 — the daily job, storage and settlement
# ---------------------------------------------------------------------------
def test_the_daily_job_stores_a_scoreable_day(service, pg_session):
    race_date, _ = _first_scored_day(service)
    result, day = run_daily_job(pg_session, race_date=race_date, service=service, settle_previous=False)

    assert result.succeeded
    assert result.races_scored == len(day.races)
    stored = pg_session.query(Prediction).filter(Prediction.race_date == race_date).count()
    assert stored == day.total_runners


def test_re_running_the_job_replaces_rather_than_duplicating(service, pg_session):
    race_date, day = _first_scored_day(service)
    before = pg_session.query(Prediction).filter(Prediction.race_date == race_date).count()

    store_predictions(pg_session, day)

    after = pg_session.query(Prediction).filter(Prediction.race_date == race_date).count()
    assert after == before, "a second run duplicated instead of replacing"


def test_settlement_scores_the_predictions_against_results(service, pg_session):
    race_date, _ = _first_scored_day(service)
    settle_predictions(pg_session, race_date)

    rows = pg_session.query(Prediction).filter(Prediction.race_date == race_date).all()
    assert rows
    assert all(row.settled for row in rows), "results exist, so everything should settle"
    assert any(row.won for row in rows), "each race has a winner"


def test_performance_scores_only_settled_bets(service, pg_session):
    race_date, _ = _first_scored_day(service)
    settle_predictions(pg_session, race_date)

    summary = summarise_performance(pg_session)

    assert summary.bets >= 0
    assert summary.settled <= summary.bets
    assert summary.headline
    if summary.settled:
        assert summary.staked > 0
        assert -1.0 <= summary.roi <= 50.0
