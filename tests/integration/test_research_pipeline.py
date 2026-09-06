"""Full research pipeline against a real PostgreSQL.

Exercises the whole Phase 3 chain — synthetic data → quality audit → features →
feature store → dataset → split → backtest — on the database that will actually
hold it, where JSONB, NUMERIC precision and index behaviour are real.

    make db-up
    .venv/bin/pytest -m integration
"""

from __future__ import annotations

import os
from datetime import date

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from backend.backtesting import BacktestConfig, BacktestEngine, MarketPredictor, UniformPredictor
from backend.data_quality import DataValidator
from backend.database.base import Base
from backend.features import FeaturePipeline, load_features, save_features
from backend.models import RaceFeatures
from backend.research import ResearchDatasetBuilder
from backend.research.synthetic import SyntheticConfig, generate_synthetic_data
from backend.utils.config import Settings

pytestmark = pytest.mark.integration

SCHEMA = "research_test"


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
def pg_engine():
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

    scoped = create_engine(settings.database_url, connect_args={"options": f"-csearch_path={SCHEMA}"})
    Base.metadata.create_all(scoped)
    yield scoped

    scoped.dispose()
    with admin.begin() as connection:
        connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    admin.dispose()


@pytest.fixture(scope="module")
def seeded(pg_engine):
    """One synthetic season, generated once and shared by every test here."""
    with Session(pg_engine, expire_on_commit=False) as session:
        generate_synthetic_data(
            session, SyntheticConfig(n_days=90, races_per_day=3, runners_per_race=8, seed=17)
        )
        yield session


# ---------------------------------------------------------------------------
def test_data_quality_audit_runs_against_postgres(seeded):
    report = DataValidator(seeded, today=date(2025, 12, 31)).validate_database()

    assert report.total_races == 270
    assert report.total_runners == 270 * 8
    assert report.is_clean, report.render()
    assert report.pass_rate == 1.0


def test_feature_build_over_a_full_season(seeded):
    frame = FeaturePipeline(seeded).build()

    assert len(frame) == 270 * 8
    assert frame["horse_runs_before"].max() > 5, "form should accumulate over a season"
    assert frame["mkt_has_market"].sum() > 0


def test_features_persist_as_jsonb_and_reload(seeded):
    frame = FeaturePipeline(seeded).build().head(200)
    written = save_features(seeded, frame)
    assert written == 200

    reloaded = load_features(seeded)
    assert len(reloaded) == 200
    assert "horse_form_score" in reloaded.columns
    assert "jky_strike_rate" in reloaded.columns  # unpacked from JSONB

    # The JSON column really is JSONB, so PostgreSQL can index and query into it.
    value = seeded.execute(
        text(f"SELECT features->>'horse_runs_before' FROM {SCHEMA}.race_features LIMIT 1")
    ).scalar_one()
    assert value is not None


def test_feature_upsert_is_idempotent(seeded):
    frame = FeaturePipeline(seeded).build().head(50)
    save_features(seeded, frame)
    save_features(seeded, frame)

    count = seeded.execute(select(RaceFeatures).where(RaceFeatures.feature_version == "v1")).scalars().all()
    assert len({(row.race_id, row.horse_id) for row in count}) == len(count)


def test_dataset_build_and_time_split(seeded):
    dataset = ResearchDatasetBuilder(seeded).build(require_market=True)

    assert dataset.rows > 1000
    assert len(dataset.feature_names) > 40
    assert dataset.base_rate == pytest.approx(0.125, abs=0.02)

    split = dataset.split(train_end=date(2025, 2, 28), embargo_days=7)
    assert split.train_rows > 0
    assert split.test_rows > 0
    assert split.train["race_date"].max() < split.test["race_date"].min()


def test_walk_forward_folds_are_all_leak_free(seeded):
    dataset = ResearchDatasetBuilder(seeded).build()
    folds = dataset.walk_forward(n_splits=3, test_days=20, min_train_days=30)

    assert len(folds) == 3
    for fold in folds:
        assert not set(fold.train["race_id"]) & set(fold.test["race_id"])
        assert fold.train["race_date"].max() < fold.test["race_date"].min()


def test_market_baseline_finds_no_value_in_a_full_season(seeded):
    """The end-to-end negative control, on real infrastructure."""
    dataset = ResearchDatasetBuilder(seeded).build(require_market=True)
    result = BacktestEngine(BacktestConfig(min_expected_value=0.05)).run(dataset.frame, MarketPredictor())
    assert result.metrics.total_bets == 0


def test_blind_betting_loses_money(seeded):
    """A strategy with no information must lose. If it does not, something is wrong."""
    dataset = ResearchDatasetBuilder(seeded).build(require_market=True)
    result = BacktestEngine(BacktestConfig(min_expected_value=0.05, flat_stake=10)).run(
        dataset.frame, UniformPredictor()
    )

    assert result.metrics.total_bets > 100
    assert result.metrics.roi < 0
    assert result.metrics.final_bankroll < result.metrics.starting_bankroll


def test_dataset_written_to_parquet_round_trips(seeded, tmp_path):
    import pandas as pd

    dataset = ResearchDatasetBuilder(seeded).build()
    path = dataset.to_parquet(tmp_path / "research.parquet")

    reloaded = pd.read_parquet(path)
    assert reloaded.shape == dataset.frame.shape
    assert list(reloaded.columns) == list(dataset.frame.columns)


def test_numeric_columns_survive_the_round_trip(seeded):
    """NUMERIC -> float conversions must not silently lose the odds."""
    dataset = ResearchDatasetBuilder(seeded).build(require_market=True)
    odds = dataset.frame["mkt_latest_odds"]

    assert odds.notna().all()
    assert (odds > 1).all()
    assert odds.dtype.kind == "f"
