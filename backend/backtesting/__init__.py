"""Historical simulation.

    from backend.backtesting import BacktestEngine, BacktestConfig, MarketPredictor

    result = BacktestEngine(BacktestConfig(min_expected_value=0.05)).run(frame, MarketPredictor())
    print(result.render())

The framework is built to be **pessimistic**. Its assumptions are documented in
:mod:`backend.backtesting.engine`; all of them make live trading harder than the
simulation, never easier. A marginal backtest edge should be read as no edge.

Phase 8 extends this with reporting, parameter sweeps and walk-forward
aggregation. Phase 3 delivers the foundation and the baselines a model must beat.
"""

from backend.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    compare_predictors,
)
from backend.backtesting.metrics import (
    BacktestMetrics,
    compute_metrics,
    expected_value,
    kelly_fraction,
)
from backend.backtesting.predictors import (
    BlendPredictor,
    FormScorePredictor,
    MarketPredictor,
    Predictor,
    UniformPredictor,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestResult",
    "BlendPredictor",
    "FormScorePredictor",
    "MarketPredictor",
    "Predictor",
    "UniformPredictor",
    "compare_predictors",
    "compute_metrics",
    "expected_value",
    "kelly_fraction",
]
