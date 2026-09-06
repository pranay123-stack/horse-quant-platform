"""The product: today's races in, betting signals out.

    from backend.prediction_service import PredictionService

    service = PredictionService(session)
    today = service.signals_for_date()
    for race, runner in today.bets:
        print(race.race, runner.horse, runner.odds, runner.expected_value)

Module map:

* :mod:`~backend.prediction_service.rules`   the production BET/NO_BET guards
* :mod:`~backend.prediction_service.service` scoring and persistence
* :mod:`~backend.prediction_service.daily`   the scheduled morning job
* :mod:`~backend.prediction_service.replay`  proves live == backtest
* :mod:`~backend.prediction_service.performance` settled results, ROI, drawdown
"""

from backend.prediction_service.rules import RULES, BettingRules, apply_rules
from backend.prediction_service.service import (
    DayPredictions,
    PredictionService,
    RaceSignals,
    RunnerSignal,
    load_predictions,
    store_predictions,
)

__all__ = [
    "RULES",
    "BettingRules",
    "DayPredictions",
    "PredictionService",
    "RaceSignals",
    "RunnerSignal",
    "apply_rules",
    "load_predictions",
    "store_predictions",
]
