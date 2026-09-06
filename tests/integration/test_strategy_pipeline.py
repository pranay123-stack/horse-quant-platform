"""End-to-end value-betting pipeline against a real PostgreSQL.

Synthetic history → walk-forward retraining → market integration → strategy →
settled ledger. The controls are the point: a market-only bettor must find
nothing, and a strategy must never be able to bet at a price that did not exist
before the off.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.database.base import Base
from backend.research.synthetic import SyntheticConfig, generate_synthetic_data
from backend.strategy import (
    BacktestConfig,
    ExecutionConfig,
    StakingConfig,
    StrategyBacktester,
    WalkForwardPredictor,
    build_report,
    comparison_table,
)
from backend.strategy.filters import STRATEGY_A, STRATEGY_B
from backend.strategy.odds import build_market_frame
from backend.utils.config import Settings

pytestmark = pytest.mark.integration

SCHEMA = "strategy_test"

SEASON = SyntheticConfig(
    n_horses=400,
    n_jockeys=35,
    n_trainers=28,
    n_courses=6,
    start_date=date(2021, 1, 1),
    n_days=1400,
    races_per_day=2,
    runners_per_race=8,
    seed=77,
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
def predictions(pg_session):
    predictor = WalkForwardPredictor(pg_session, model_type="lightgbm", model_params={"n_estimators": 150})
    return predictor.run(
        history_start=date(2021, 1, 1),
        test_start=date(2023, 1, 1),
        test_end=date(2024, 10, 31),
        retrain_months=12,
    ), predictor


def _config(**overrides) -> BacktestConfig:
    defaults = {
        "strategy": STRATEGY_A,
        "staking": StakingConfig(method="flat", flat_stake=10.0),
        "execution": ExecutionConfig(slippage=0.02),
    }
    defaults.update(overrides)
    return BacktestConfig(**defaults)


# ---------------------------------------------------------------------------
def test_walk_forward_predicts_only_forward(predictions):
    frame, predictor = predictions

    assert len(predictor.windows) >= 2
    for window in predictor.windows:
        assert window.train_end < window.predict_start, "a window trained on its own test period"

    assert pd.to_datetime(frame["race_date"]).min().date() >= date(2023, 1, 1)


def test_predictions_are_race_normalised(predictions):
    frame, _ = predictions
    totals = frame.groupby("race_id")["model_probability"].sum()
    assert totals.round(6).eq(1.0).all()


def test_market_integration_produces_usable_probabilities(predictions):
    frame, _ = predictions
    market = build_market_frame(frame)

    priced = market[market["market_probability"].notna()]
    assert len(priced) > 0
    totals = priced.groupby("race_id")["market_probability"].sum()
    assert totals.round(4).between(0.99, 1.01).all()
    assert (market["book_overround"].dropna() > 1.0).all(), "generated market offers arbitrage"


def test_strategy_produces_a_settled_ledger(predictions):
    frame, _ = predictions
    run = StrategyBacktester(_config()).run(frame)

    assert run.bets > 0
    ledger = run.ledger
    assert set(ledger.columns) >= {"race_id", "horse_id", "race_date", "odds", "stake", "won", "profit_loss"}
    assert (ledger["stake"] > 0).all()
    assert ledger["won"].isin([0, 1]).all()


def test_ledger_settles_chronologically_and_compounds(predictions):
    frame, _ = predictions
    ledger = StrategyBacktester(_config()).run(frame).ledger

    assert pd.to_datetime(ledger["race_date"]).is_monotonic_increasing
    expected = 10_000.0 + ledger["profit_loss"].cumsum()
    pd.testing.assert_series_equal(
        ledger["bankroll_after"].astype(float).round(2),
        expected.round(2),
        check_names=False,
    )


def test_market_only_baseline_finds_nothing(predictions):
    """The negative control, end to end."""
    frame, _ = predictions
    market_only = build_market_frame(frame).copy()
    market_only["model_probability"] = market_only["market_probability"]

    assert StrategyBacktester(_config()).run(market_only).bets == 0


def test_no_bet_is_struck_at_a_price_that_did_not_exist(predictions):
    frame, _ = predictions
    ledger = StrategyBacktester(_config()).run(frame).ledger

    # Every struck price must be the observed price minus slippage, and every
    # observed price must come from the pre-race market columns.
    assert (ledger["odds"] <= ledger["observed_odds"]).all()
    merged = ledger.merge(
        frame[["race_id", "horse_id", "mkt_latest_odds", "starting_price"]],
        on=["race_id", "horse_id"],
        how="left",
    )
    assert (merged["observed_odds"].round(6) == merged["mkt_latest_odds"].round(6)).all()


def test_stricter_thresholds_place_fewer_bets(predictions):
    frame, _ = predictions
    loose = StrategyBacktester(_config(strategy=STRATEGY_A)).run(frame)
    strict = StrategyBacktester(_config(strategy=STRATEGY_B)).run(frame)
    assert strict.bets <= loose.bets


def test_costs_only_ever_reduce_profit(predictions):
    frame, _ = predictions
    free = build_report(
        StrategyBacktester(_config(execution=ExecutionConfig(slippage=0.0, commission=0.0))).run(frame)
    ).metrics
    costed = build_report(
        StrategyBacktester(_config(execution=ExecutionConfig(slippage=0.05, commission=0.05))).run(frame)
    ).metrics

    assert costed.profit <= free.profit
    assert costed.bets == free.bets


def test_full_report_renders_with_every_section(predictions):
    frame, _ = predictions
    report = build_report(StrategyBacktester(_config()).run(frame))

    assert not report.equity.empty
    assert not report.monthly.empty
    assert not report.by_odds.empty
    rendered = report.render()
    assert "VERDICT" in rendered
    assert "Equity curve" in rendered


def test_comparison_across_the_strategy_library(predictions):
    from backend.strategy import STRATEGY_LIBRARY

    frame, _ = predictions
    reports = [
        build_report(StrategyBacktester(_config(strategy=strategy)).run(frame))
        for strategy in STRATEGY_LIBRARY.values()
    ]
    table = comparison_table(reports)

    assert len(table) == len(STRATEGY_LIBRARY)
    assert table["roi"].is_monotonic_decreasing


def test_bets_persist_to_the_ledger_table(pg_session, predictions):
    """The ledger schema must actually accept what the backtester produces."""
    from decimal import Decimal

    from backend.models import Bet

    frame, _ = predictions
    ledger = StrategyBacktester(_config()).run(frame, run_id="itest").ledger.head(50)

    for record in ledger.to_dict(orient="records"):
        pg_session.add(
            Bet(
                run_id=record["run_id"],
                strategy=record["strategy"],
                race_id=record["race_id"],
                horse_id=record["horse_id"],
                race_date=record["race_date"],
                model_probability=record["model_probability"],
                market_probability=record["market_probability"],
                edge=record["edge"],
                expected_value=record["expected_value"],
                observed_odds=Decimal(str(round(record["observed_odds"], 3))),
                odds=Decimal(str(round(record["odds"], 3))),
                stake=Decimal(str(round(record["stake"], 2))),
                stake_constraint=record["stake_constraint"] or None,
                won=bool(record["won"]),
                profit_loss=Decimal(str(round(record["profit_loss"], 2))),
                bankroll_after=Decimal(str(round(record["bankroll_after"], 2))),
            )
        )
    pg_session.commit()

    stored = pg_session.query(Bet).filter_by(run_id="itest").count()
    assert stored == len(ledger)
