"""The value-betting strategy layer.

    from backend.strategy import WalkForwardPredictor, StrategyBacktester, BacktestConfig

    predictions = WalkForwardPredictor(session).run(
        history_start=date(2018, 1, 1), test_start=date(2020, 1, 1), test_end=date(2026, 6, 30)
    )
    run = StrategyBacktester(BacktestConfig()).run(predictions)
    print(build_report(run).render())

Pipeline::

    model probabilities (Phase 4)
        |
    odds.py             consensus price -> margin-free market probability
        |               best price      -> what a bet is struck at
    expected_value.py   EV = p * odds - 1,  edge = p_model - p_market
        |
    filters.py          where the model is allowed to be trusted
        |
    staking.py          flat | fractional Kelly, with hard limits
        |
    backtester.py       walk-forward: retrain, predict forward, settle
        |
    reports.py          ROI, drawdown, profit factor -- and a t-statistic,
                        because ROI without its uncertainty misleads

Two separations do most of the work. Market probability comes from the
**consensus** price while bets are struck at the **best** price -- using the best
price for both is circular and manufactures edge. And predictions are generated
once, then replayed through many strategies, so strategy differences are not
confounded with differences between independently retrained models.
"""

from backend.strategy.backtester import (
    BacktestConfig,
    ExecutionConfig,
    StrategyBacktester,
    StrategyRun,
    WalkForwardPredictor,
    WalkForwardWindow,
)
from backend.strategy.expected_value import ValueBetSignal, edge, expected_value, fair_odds
from backend.strategy.filters import STRATEGY_LIBRARY, StrategyConfig, select_bets
from backend.strategy.odds import build_market_frame, implied_probability, overround, remove_margin
from backend.strategy.reports import (
    StrategyMetrics,
    StrategyReport,
    build_report,
    comparison_table,
    compute_metrics,
)
from backend.strategy.robustness import (
    ConsistencySummary,
    SegmentResult,
    consistency_summary,
    segment_analysis,
)
from backend.strategy.staking import BankrollState, StakingConfig, StakingPlan, kelly_fraction, settle

__all__ = [
    "STRATEGY_LIBRARY",
    "BacktestConfig",
    "BankrollState",
    "ConsistencySummary",
    "ExecutionConfig",
    "SegmentResult",
    "StakingConfig",
    "StakingPlan",
    "StrategyBacktester",
    "StrategyConfig",
    "StrategyMetrics",
    "StrategyReport",
    "StrategyRun",
    "ValueBetSignal",
    "WalkForwardPredictor",
    "WalkForwardWindow",
    "build_market_frame",
    "build_report",
    "comparison_table",
    "compute_metrics",
    "consistency_summary",
    "edge",
    "expected_value",
    "fair_odds",
    "implied_probability",
    "kelly_fraction",
    "overround",
    "remove_margin",
    "segment_analysis",
    "select_bets",
    "settle",
]
