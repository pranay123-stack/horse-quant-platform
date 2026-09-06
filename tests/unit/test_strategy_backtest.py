"""Backtester and reporting.

The tests that matter most here are the ones that would catch a backtest lying:
no future odds, no SP, chronological settlement, and arithmetic that ties out to
hand-computed values.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.strategy.backtester import (
    BacktestConfig,
    ExecutionConfig,
    StrategyBacktester,
    WalkForwardPredictor,
)
from backend.strategy.filters import STRATEGY_A, STRATEGY_UNFILTERED, StrategyConfig
from backend.strategy.reports import (
    MIN_BETS_FOR_INFERENCE,
    build_report,
    calibration_on_bets,
    comparison_table,
    compute_metrics,
    drawdown_periods,
    equity_curve,
    monthly_returns,
    odds_distribution,
)
from backend.strategy.staking import StakingConfig
from backend.utils.exceptions import StrategyError

pytestmark = pytest.mark.unit


def make_predictions(
    *,
    races: int = 100,
    runners: int = 6,
    model_edge: float = 0.0,
    seed: int = 0,
    overround: float = 1.15,
    start: date = date(2025, 1, 1),
) -> pd.DataFrame:
    """Predictions with a known, controllable edge over the market.

    ``model_edge = 0`` means the model *is* the market, so no strategy should be
    able to profit. Positive values shift probability towards the true winner.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for race_index in range(races):
        strengths = rng.dirichlet(np.ones(runners) * 2)
        winner = rng.choice(runners, p=strengths)
        quoted = strengths * overround
        odds = 1.0 / quoted

        blended = (1 - model_edge) * strengths + model_edge * np.eye(runners)[winner]
        blended = blended / blended.sum()

        for runner_index in range(runners):
            rows.append(
                {
                    "race_id": f"rac_{race_index:05d}",
                    "horse_id": f"hrs_{race_index:05d}_{runner_index}",
                    "race_date": start + timedelta(days=race_index),
                    "model_probability": float(blended[runner_index]),
                    "mkt_mean_odds": float(odds[runner_index]),
                    "mkt_latest_odds": float(odds[runner_index]),
                    "starting_price": float(odds[runner_index]),
                    "mkt_bookmaker_count": 4,
                    "won": int(runner_index == winner),
                }
            )
    return pd.DataFrame(rows)


def flat_config(**overrides) -> BacktestConfig:
    defaults = {
        "strategy": STRATEGY_A,
        "staking": StakingConfig(
            method="flat", flat_stake=10.0, max_stake_fraction=1.0, max_daily_exposure=1.0
        ),
        "execution": ExecutionConfig(slippage=0.0, commission=0.0, max_stake_liquidity=None),
    }
    defaults.update(overrides)
    return BacktestConfig(**defaults)


# ---------------------------------------------------------------------------
# The controls
# ---------------------------------------------------------------------------
def test_a_model_that_equals_the_market_makes_no_money():
    """The negative control. Betting the market against itself must not profit."""
    predictions = make_predictions(races=400, model_edge=0.0, seed=1)
    predictions["model_probability"] = 1.0 / predictions["mkt_mean_odds"] / 1.15

    run = StrategyBacktester(flat_config()).run(predictions)
    assert run.bets == 0, "a model with no edge over the market found bets"


def test_a_model_with_a_real_edge_makes_money():
    """The positive control. If a known edge is not detected, the engine is broken."""
    predictions = make_predictions(races=600, model_edge=0.35, seed=2)
    run = StrategyBacktester(flat_config()).run(predictions)
    report = build_report(run)

    assert report.metrics.bets > 100
    assert report.metrics.roi > 0
    assert report.metrics.profit > 0


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------
def test_starting_price_is_refused_by_default():
    """SP is set at the off and arrives via the results feed — betting at it is
    look-ahead."""
    predictions = make_predictions(races=200, model_edge=0.35, seed=3)
    predictions["mkt_mean_odds"] = np.nan
    predictions["mkt_latest_odds"] = np.nan  # only SP survives

    run = StrategyBacktester(flat_config()).run(predictions)
    assert run.bets == 0
    assert run.rejections.get("no_odds", 0) > 0


def test_starting_price_can_be_opted_into():
    predictions = make_predictions(races=200, model_edge=0.35, seed=3)
    predictions["mkt_mean_odds"] = np.nan
    predictions["mkt_latest_odds"] = np.nan

    config = flat_config(
        execution=ExecutionConfig(slippage=0.0, allow_starting_price=True, max_stake_liquidity=None)
    )
    assert StrategyBacktester(config).run(predictions).bets > 0


def test_bets_are_settled_in_chronological_order():
    predictions = make_predictions(races=300, model_edge=0.35, seed=4)
    shuffled = predictions.sample(frac=1, random_state=7).reset_index(drop=True)

    ledger = StrategyBacktester(flat_config()).run(shuffled).ledger
    dates = pd.to_datetime(ledger["race_date"])
    assert dates.is_monotonic_increasing


def test_bankroll_is_carried_forward_across_bets():
    predictions = make_predictions(races=200, model_edge=0.35, seed=5)
    ledger = StrategyBacktester(flat_config()).run(predictions).ledger

    recomputed = 10_000.0 + ledger["profit_loss"].cumsum()
    np.testing.assert_allclose(ledger["bankroll_after"], recomputed, atol=0.02)


# ---------------------------------------------------------------------------
# Execution realism
# ---------------------------------------------------------------------------
def test_slippage_worsens_the_price_actually_struck():
    execution = ExecutionConfig(slippage=0.05)
    assert execution.struck_odds(6.0) == pytest.approx(5.7)
    assert execution.struck_odds(1.02) >= 1.01


def test_slippage_reduces_profit():
    predictions = make_predictions(races=500, model_edge=0.35, seed=6)

    clean = build_report(StrategyBacktester(flat_config()).run(predictions)).metrics
    slipped = build_report(
        StrategyBacktester(
            flat_config(execution=ExecutionConfig(slippage=0.05, max_stake_liquidity=None))
        ).run(predictions)
    ).metrics

    assert slipped.roi < clean.roi
    assert slipped.bets == clean.bets, "slippage must not change which bets are selected"


def test_commission_reduces_profit_but_not_bet_count():
    predictions = make_predictions(races=500, model_edge=0.35, seed=7)

    clean = build_report(StrategyBacktester(flat_config()).run(predictions)).metrics
    charged = build_report(
        StrategyBacktester(
            flat_config(execution=ExecutionConfig(slippage=0.0, commission=0.05, max_stake_liquidity=None))
        ).run(predictions)
    ).metrics

    assert charged.profit < clean.profit
    assert charged.bets == clean.bets


def test_liquidity_cap_limits_the_stake():
    predictions = make_predictions(races=200, model_edge=0.35, seed=8)
    config = flat_config(
        staking=StakingConfig(
            method="flat", flat_stake=1000.0, max_stake_fraction=1.0, max_daily_exposure=1.0
        ),
        execution=ExecutionConfig(slippage=0.0, max_stake_liquidity=50.0),
    )
    ledger = StrategyBacktester(config).run(predictions).ledger
    assert ledger["stake"].max() <= 50.0


def test_thin_markets_can_be_excluded():
    predictions = make_predictions(races=200, model_edge=0.35, seed=9)
    predictions["mkt_bookmaker_count"] = 1

    config = flat_config(execution=ExecutionConfig(slippage=0.0, min_bookmakers=3, max_stake_liquidity=None))
    assert StrategyBacktester(config).run(predictions).bets == 0


# ---------------------------------------------------------------------------
# Engine behaviour
# ---------------------------------------------------------------------------
def test_missing_target_is_rejected():
    predictions = make_predictions(races=10).drop(columns=["won"])
    with pytest.raises(StrategyError, match="'won' column"):
        StrategyBacktester(flat_config()).run(predictions)


def test_empty_predictions_produce_an_empty_run():
    run = StrategyBacktester(flat_config()).run(pd.DataFrame())
    assert run.bets == 0
    assert "no predictions to simulate" in run.notes


def test_ruin_stops_the_simulation():
    predictions = make_predictions(races=400, model_edge=0.0, seed=10)
    predictions["model_probability"] = 0.9  # bet everything, lose almost everything

    config = flat_config(
        strategy=STRATEGY_UNFILTERED,
        staking=StakingConfig(
            method="flat",
            flat_stake=2000.0,
            max_stake_fraction=1.0,
            max_daily_exposure=1.0,
            ruin_threshold=0.5,
        ),
    )
    run = StrategyBacktester(config).run(predictions)
    assert run.stopped_early


def test_rejection_reasons_are_aggregated():
    predictions = make_predictions(races=100, model_edge=0.0, seed=11)
    run = StrategyBacktester(flat_config()).run(predictions)
    assert sum(run.rejections.values()) > 0


def test_small_samples_are_flagged_as_inconclusive():
    predictions = make_predictions(races=20, model_edge=0.35, seed=12)
    run = StrategyBacktester(flat_config()).run(predictions)
    assert any("too few bets" in note or "far too few" in note for note in run.notes)


def test_run_ids_are_stable_when_supplied():
    predictions = make_predictions(races=50, model_edge=0.35, seed=13)
    run = StrategyBacktester(flat_config()).run(predictions, run_id="fixed_id")
    assert (run.ledger["run_id"] == "fixed_id").all()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _ledger(profits: list[float], stakes: list[float], odds: list[float], won: list[int]) -> pd.DataFrame:
    bankroll = 1000.0 + pd.Series(profits).cumsum()
    return pd.DataFrame(
        {
            "race_id": [f"r{i}" for i in range(len(profits))],
            "race_date": [date(2025, 1, 1) + timedelta(days=i) for i in range(len(profits))],
            "stake": stakes,
            "odds": odds,
            "won": won,
            "profit_loss": profits,
            "bankroll_after": bankroll,
            "model_probability": [0.3] * len(profits),
            "edge": [0.05] * len(profits),
            "expected_value": [0.1] * len(profits),
        }
    )


def test_metrics_arithmetic():
    ledger = _ledger([50.0, -10.0, -10.0, 30.0], [10.0] * 4, [6.0, 4.0, 3.0, 4.0], [1, 0, 0, 1])
    metrics = compute_metrics(ledger, starting_bankroll=1000, races_considered=10)

    assert metrics.bets == 4
    assert metrics.wins == 2
    assert metrics.strike_rate == 0.5
    assert metrics.staked == 40.0
    assert metrics.profit == 60.0
    assert metrics.roi == pytest.approx(1.5)
    assert metrics.bet_rate == 0.4


def test_profit_factor_is_gross_wins_over_gross_losses():
    ledger = _ledger([50.0, -10.0, -10.0, 30.0], [10.0] * 4, [6.0, 4.0, 3.0, 4.0], [1, 0, 0, 1])
    metrics = compute_metrics(ledger, starting_bankroll=1000)
    assert metrics.profit_factor == pytest.approx(80.0 / 20.0)


def test_profit_factor_is_infinite_without_losses():
    ledger = _ledger([50.0, 30.0], [10.0, 10.0], [6.0, 4.0], [1, 1])
    assert compute_metrics(ledger, starting_bankroll=1000).profit_factor == float("inf")


def test_max_drawdown_is_measured_from_the_peak():
    ledger = _ledger([100.0, -50.0, -50.0, -50.0], [10.0] * 4, [2.0] * 4, [1, 0, 0, 0])
    metrics = compute_metrics(ledger, starting_bankroll=1000)

    assert metrics.peak_bankroll == 1100.0
    assert metrics.max_drawdown == pytest.approx(150.0)
    assert metrics.max_drawdown_pct == pytest.approx(150 / 1100)


def test_calibration_on_bets_compares_confidence_with_outcome():
    ledger = _ledger([50.0, -10.0, -10.0, -10.0], [10.0] * 4, [6.0] * 4, [1, 0, 0, 0])
    metrics = compute_metrics(ledger, starting_bankroll=1000)
    # 25% strike against a 30% mean prediction.
    assert metrics.calibration_on_bets == pytest.approx(0.25 / 0.30)


def test_t_statistic_and_verdict_reflect_sample_size():
    small = compute_metrics(_ledger([50.0, -10.0], [10.0, 10.0], [6.0, 4.0], [1, 0]), starting_bankroll=1000)
    assert "inconclusive" in small.verdict
    assert not small.is_significant


def test_verdict_reports_negative_expectancy():
    profits = [-10.0] * (MIN_BETS_FOR_INFERENCE + 10)
    ledger = _ledger(profits, [10.0] * len(profits), [4.0] * len(profits), [0] * len(profits))
    assert compute_metrics(ledger, starting_bankroll=100_000).verdict == "negative expectancy"


def test_metrics_of_an_empty_ledger():
    metrics = compute_metrics(pd.DataFrame(), starting_bankroll=1000)
    assert metrics.bets == 0
    assert metrics.verdict == "no bets placed"


# ---------------------------------------------------------------------------
# Report components
# ---------------------------------------------------------------------------
def test_equity_curve_tracks_peak_and_drawdown():
    ledger = _ledger([100.0, -50.0, -50.0], [10.0] * 3, [2.0] * 3, [1, 0, 0])
    curve = equity_curve(ledger, 1000)

    assert len(curve) == 3
    assert curve["peak"].iloc[-1] == 1100.0
    assert curve["drawdown_pct"].iloc[-1] == pytest.approx(100 / 1100)


def test_monthly_returns_group_by_calendar_month():
    ledger = _ledger([10.0] * 40, [10.0] * 40, [2.0] * 40, [1] * 40)
    monthly = monthly_returns(ledger)

    assert len(monthly) >= 2
    assert set(monthly.columns) >= {"month", "bets", "staked", "profit", "roi"}


def test_odds_distribution_splits_by_price_band():
    ledger = _ledger([10.0, -10.0, -10.0], [10.0] * 3, [1.5, 4.0, 25.0], [1, 0, 0])
    bands = odds_distribution(ledger)

    assert len(bands) == 3
    assert set(bands.columns) >= {"band", "bets", "strike_rate", "roi"}


def test_calibration_table_bins_by_predicted_probability():
    rng = np.random.default_rng(0)
    count = 400
    probabilities = rng.uniform(0.1, 0.6, count)
    ledger = _ledger([1.0] * count, [10.0] * count, [3.0] * count, list(rng.binomial(1, probabilities)))
    ledger["model_probability"] = probabilities

    table = calibration_on_bets(ledger, bins=4)
    assert len(table) == 4
    assert (table["ratio"] > 0).all()


def test_drawdown_periods_are_ranked_by_depth():
    ledger = _ledger([100.0, -50.0, -50.0, 200.0, -30.0], [10.0] * 5, [2.0] * 5, [1, 0, 0, 1, 0])
    episodes = drawdown_periods(ledger, 1000)
    assert not episodes.empty
    assert episodes["depth_pct"].is_monotonic_decreasing


def test_report_renders_everything():
    predictions = make_predictions(races=400, model_edge=0.35, seed=14)
    report = build_report(StrategyBacktester(flat_config()).run(predictions))

    rendered = report.render()
    for expected in ("Strategy:", "ROI", "Max drawdown", "VERDICT", "By price band"):
        assert expected in rendered
    assert "metrics" in report.as_dict()


def test_comparison_table_sorts_by_roi():
    predictions = make_predictions(races=400, model_edge=0.35, seed=15)
    reports = []
    for name, strategy in (
        ("weak", StrategyConfig(name="weak", min_expected_value=0.0)),
        ("strong", STRATEGY_A),
    ):
        report = build_report(StrategyBacktester(flat_config(strategy=strategy)).run(predictions))
        report.metrics.strategy = name
        reports.append(report)

    table = comparison_table(reports)
    assert table["roi"].is_monotonic_decreasing
    assert set(table.columns) >= {"strategy", "bets", "roi", "profit_factor", "t_stat", "verdict"}


# ---------------------------------------------------------------------------
# Walk-forward configuration
# ---------------------------------------------------------------------------
def test_walk_forward_rejects_an_unknown_model(db_session):
    with pytest.raises(StrategyError, match="unknown model type"):
        WalkForwardPredictor(db_session, model_type="crystal_ball")


def test_walk_forward_rejects_a_test_period_before_the_history(db_session):
    predictor = WalkForwardPredictor(db_session)
    with pytest.raises(StrategyError, match="must start after"):
        predictor.run(history_start=date(2024, 1, 1), test_start=date(2023, 1, 1), test_end=date(2024, 6, 1))
