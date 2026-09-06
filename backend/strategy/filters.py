"""Bet selection rules.

Filters exist because the naive rule — "bet whenever EV > 0" — bets almost
exclusively on longshots, and loses.

The reason is arithmetic. ``EV = p * odds - 1`` is linear in the price, so a
fixed absolute error in ``p`` is multiplied by the odds. Being 1 percentage
point too generous about a 2.0 favourite costs 2% of EV; the same error on a
50.0 outsider costs 50%. Model error is largest exactly where the sample is
thinnest — the tail — so an unfiltered EV rule is a machine for converting
calibration noise into confident bets on 40/1 shots.

Every rule below is therefore a statement about *where the model is allowed to
be trusted*, not a performance tweak. They are applied in a fixed order and the
first failure is recorded, so a strategy's behaviour can be explained by
counting reasons rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backend.strategy.expected_value import ValueBetSignal


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """A named, reproducible set of betting rules.

    Defaults are Strategy V1 from the specification: EV > 5%, probability > 10%,
    odds ≤ 20.
    """

    name: str = "v1"
    description: str = "EV > 5%, probability > 10%, odds <= 20"

    min_expected_value: float = 0.05
    min_probability: float = 0.10
    #: A hard ceiling, not a preference. See the module docstring: EV error
    #: scales with the price, so the tail is where a mediocre model does its
    #: worst damage.
    max_odds: float = 20.0
    min_odds: float = 1.5
    #: Require the model to disagree with the market in probability terms too.
    #: EV alone can clear its threshold on a large price with a trivial edge.
    min_edge: float = 0.0

    min_field_size: int = 4
    #: A consensus overround far from ~1.0-1.30 means the price sample is broken
    #: (a missing runner, a stale quote), not that free money is available.
    min_overround: float = 1.0
    max_overround: float = 1.40

    #: Several "value" runners in one race usually means the race's probabilities
    #: are miscalibrated, not that it holds a basket of edges.
    max_bets_per_race: int = 1
    #: Rank bets within a race by EV or by edge when the cap binds.
    rank_by: str = "expected_value"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "min_expected_value": self.min_expected_value,
            "min_probability": self.min_probability,
            "min_edge": self.min_edge,
            "odds_range": [self.min_odds, self.max_odds],
            "min_field_size": self.min_field_size,
            "overround_range": [self.min_overround, self.max_overround],
            "max_bets_per_race": self.max_bets_per_race,
        }


# ---------------------------------------------------------------------------
# The specification's strategy family, plus the controls that make it meaningful
# ---------------------------------------------------------------------------
STRATEGY_A = StrategyConfig(name="A_ev5", description="EV > 5% (specification V1)")
STRATEGY_B = StrategyConfig(
    name="B_ev10", description="EV > 10% — fewer, stronger signals", min_expected_value=0.10
)
STRATEGY_C = StrategyConfig(
    name="C_top_pick",
    description="Model's top pick only, any positive EV",
    min_expected_value=0.0,
    min_probability=0.0,
    max_odds=50.0,
    max_bets_per_race=1,
    rank_by="model_probability",
)
STRATEGY_D = StrategyConfig(
    name="D_ev5_tight",
    description="EV > 5%, odds <= 10 — tail excluded entirely",
    min_expected_value=0.05,
    max_odds=10.0,
)
STRATEGY_EV15 = StrategyConfig(
    name="G_ev15",
    description="EV > 15% — only the strongest disagreements with the market",
    min_expected_value=0.15,
)
STRATEGY_F = StrategyConfig(
    name="F_edge5",
    description="EV > 5% AND edge > 5 points — the market must be beaten twice",
    min_expected_value=0.05,
    min_edge=0.05,
)
STRATEGY_UNFILTERED = StrategyConfig(
    name="E_unfiltered",
    description="Any positive EV, no guards — the cautionary control",
    min_expected_value=0.0,
    min_probability=0.0,
    min_odds=1.01,
    max_odds=1000.0,
    min_field_size=2,
    max_overround=99.0,
    max_bets_per_race=99,
)

STRATEGY_LIBRARY: dict[str, StrategyConfig] = {
    config.name: config
    for config in (
        STRATEGY_A,
        STRATEGY_B,
        STRATEGY_EV15,
        STRATEGY_C,
        STRATEGY_D,
        STRATEGY_F,
        STRATEGY_UNFILTERED,
    )
}


@dataclass(slots=True)
class SelectionResult:
    """Signals for one race, and why each runner was or was not backed."""

    signals: list[ValueBetSignal] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)

    @property
    def bets(self) -> list[ValueBetSignal]:
        return [signal for signal in self.signals if signal.is_bet]

    def record(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1


def _reject(signal_row: dict[str, Any], config: StrategyConfig) -> str:
    """First failing rule for a runner, or ``""`` if it passes.

    Ordered cheapest and most fundamental first, so the recorded reason is the
    most informative one rather than whichever check happened to run last.
    """
    odds = signal_row.get("bet_odds")
    probability = signal_row.get("model_probability")

    if odds is None or not np.isfinite(odds) or odds <= 1:
        return "no_odds"
    if probability is None or not np.isfinite(probability):
        return "no_probability"

    field_size = signal_row.get("field_size") or 0
    if field_size and field_size < config.min_field_size:
        return "field_too_small"

    book = signal_row.get("book_overround")
    if book is not None and np.isfinite(book) and not config.min_overround <= book <= config.max_overround:
        return "overround_implausible"

    if odds < config.min_odds:
        return "odds_too_low"
    if odds > config.max_odds:
        return "odds_too_high"
    if probability < config.min_probability:
        return "probability_too_low"

    runner_edge = signal_row.get("edge")
    if config.min_edge > 0 and (
        runner_edge is None or not np.isfinite(runner_edge) or runner_edge < config.min_edge
    ):
        return "edge_too_small"

    if signal_row.get("expected_value", -1.0) < config.min_expected_value:
        return "ev_below_threshold"
    return ""


def select_bets(race_frame: pd.DataFrame, config: StrategyConfig) -> SelectionResult:
    """Apply the rules to one race and return every runner's verdict."""
    result = SelectionResult()
    if race_frame.empty:
        return result

    field_size = len(race_frame)
    candidates: list[tuple[float, ValueBetSignal]] = []

    for raw in race_frame.to_dict(orient="records"):
        record: dict[str, Any] = {str(key): value for key, value in raw.items()}
        record.setdefault("field_size", field_size)
        reason = _reject(record, config)

        signal = ValueBetSignal(
            race_id=str(record["race_id"]),
            horse_id=str(record["horse_id"]),
            race_date=record.get("race_date"),
            horse_name=record.get("horse_name"),
            model_probability=float(record.get("model_probability", float("nan"))),
            market_probability=float(record.get("market_probability", float("nan"))),
            odds=float(record.get("bet_odds", float("nan"))),
            edge=float(record.get("edge", float("nan"))),
            expected_value=float(record.get("expected_value", -1.0)),
            decision="no_bet" if reason else "bet",
            reason=reason,
            field_size=field_size,
            book_overround=float(record.get("book_overround", float("nan"))),
        )
        if reason:
            result.record(reason)
            result.signals.append(signal)
        else:
            rank_value = (
                signal.expected_value
                if config.rank_by == "expected_value"
                else signal.model_probability
                if config.rank_by == "model_probability"
                else signal.edge
            )
            candidates.append((rank_value, signal))

    # Strongest first, then apply the per-race cap.
    candidates.sort(key=lambda item: item[0], reverse=True)
    for position, (_, signal) in enumerate(candidates):
        if position >= config.max_bets_per_race:
            signal.decision = "no_bet"
            signal.reason = "max_bets_per_race"
            result.record("max_bets_per_race")
        result.signals.append(signal)

    return result


__all__ = [
    "STRATEGY_A",
    "STRATEGY_B",
    "STRATEGY_C",
    "STRATEGY_D",
    "STRATEGY_EV15",
    "STRATEGY_F",
    "STRATEGY_LIBRARY",
    "STRATEGY_UNFILTERED",
    "SelectionResult",
    "StrategyConfig",
    "select_bets",
]
