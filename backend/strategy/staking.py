"""Bankroll management.

Sizing decides survival. A strategy with a genuine edge and reckless stakes goes
broke; a strategy with no edge and careful stakes merely bleeds. Only the second
is recoverable.

Kelly, and why it is fractional here
====================================
Full Kelly maximises long-run growth **given the true probability**. Ours is
estimated, and Kelly's sensitivity to that estimate is brutally asymmetric:
overstate an edge by half and the optimal-looking stake is roughly double what
it should be, which turns a positive-expectancy strategy into one that can still
lose everything. Halving the fraction costs about a quarter of the growth rate
and roughly halves the volatility — a trade worth taking every time.

Quarter-Kelly is the default. On top of that sit three hard limits, because a
single formula should never be the only thing between a model bug and the whole
bankroll:

* **per-bet cap** — no one race can matter that much
* **daily exposure cap** — a bad model on a busy Saturday cannot bet the bank
* **drawdown brake** — stakes shrink as the bankroll falls, so a losing run
  decays rather than compounds
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np

StakingMethod = Literal["flat", "kelly", "fixed_fraction"]


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    """Full-Kelly stake as a fraction of bankroll.

    ``f* = (p·b - q) / b`` with ``b = odds - 1``. Zero when there is no edge —
    Kelly never backs a negative-EV proposition.
    """
    if not np.isfinite(probability) or not np.isfinite(decimal_odds):
        return 0.0
    if decimal_odds <= 1 or not 0 < probability < 1:
        return 0.0
    b = decimal_odds - 1.0
    return max(0.0, (probability * b - (1.0 - probability)) / b)


@dataclass(frozen=True, slots=True)
class StakingConfig:
    """How much to stake, and the limits that override it."""

    method: StakingMethod = "flat"
    flat_stake: float = 10.0
    #: For ``fixed_fraction``: stake this share of the bankroll every time.
    fixed_fraction: float = 0.01
    kelly_multiplier: float = 0.25

    #: Hard ceiling per bet, as a fraction of the current bankroll.
    max_stake_fraction: float = 0.02
    max_stake_absolute: float | None = None
    #: Total stake allowed across one day, as a fraction of the bankroll.
    max_daily_exposure: float = 0.10

    #: Below this fraction of the peak bankroll, stakes are scaled down.
    drawdown_brake_at: float = 0.85
    #: Multiplier applied while the brake is engaged.
    drawdown_brake_factor: float = 0.5
    #: Stop betting entirely below this fraction of the starting bankroll.
    ruin_threshold: float = 0.25

    #: Bookmakers round to the penny and refuse trivial stakes.
    min_stake: float = 1.0
    round_to: float = 0.01

    def __post_init__(self) -> None:
        if self.method not in ("flat", "kelly", "fixed_fraction"):
            raise ValueError(f"unknown staking method {self.method!r}")
        if not 0 < self.kelly_multiplier <= 1:
            raise ValueError("kelly_multiplier must be in (0, 1]")
        if not 0 < self.max_stake_fraction <= 1:
            raise ValueError("max_stake_fraction must be in (0, 1]")
        if self.flat_stake < 0 or self.fixed_fraction < 0:
            raise ValueError("stake sizes must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "flat_stake": self.flat_stake,
            "kelly_multiplier": self.kelly_multiplier,
            "max_stake_fraction": self.max_stake_fraction,
            "max_daily_exposure": self.max_daily_exposure,
            "drawdown_brake_at": self.drawdown_brake_at,
            "ruin_threshold": self.ruin_threshold,
        }


@dataclass(slots=True)
class BankrollState:
    """Running bankroll, peak and today's exposure."""

    starting: float
    current: float
    peak: float
    day: date | None = None
    staked_today: float = 0.0
    is_ruined: bool = False

    @classmethod
    def start(cls, amount: float) -> BankrollState:
        return cls(starting=amount, current=amount, peak=amount)

    @property
    def drawdown(self) -> float:
        """Current shortfall from the peak, as a fraction."""
        return 0.0 if self.peak <= 0 else max(0.0, (self.peak - self.current) / self.peak)

    def begin_day(self, day: date | None) -> None:
        if day != self.day:
            self.day = day
            self.staked_today = 0.0

    def apply(self, stake: float, profit: float) -> None:
        self.staked_today += stake
        self.current += profit
        self.peak = max(self.peak, self.current)


class StakingPlan:
    """Turns a signal plus a bankroll into a stake."""

    def __init__(self, config: StakingConfig | None = None) -> None:
        self.config = config or StakingConfig()

    # ------------------------------------------------------------------
    def base_stake(self, bankroll: float, probability: float, odds: float) -> float:
        config = self.config
        if config.method == "flat":
            return config.flat_stake
        if config.method == "fixed_fraction":
            return bankroll * config.fixed_fraction
        return bankroll * kelly_fraction(probability, odds) * config.kelly_multiplier

    def stake_for(
        self, state: BankrollState, probability: float, odds: float, *, day: date | None = None
    ) -> tuple[float, str]:
        """Stake for one bet, and the limit that bound it (``""`` if none).

        Returning the binding constraint matters: "the strategy staked less than
        Kelly asked for" is a very different diagnosis from "the strategy found
        fewer bets", and a backtest that cannot tell them apart is hard to trust.
        """
        config = self.config
        state.begin_day(day)

        if state.is_ruined or state.current <= state.starting * config.ruin_threshold:
            state.is_ruined = True
            return 0.0, "ruined"

        stake = self.base_stake(state.current, probability, odds)
        binding = ""

        if stake <= 0:
            return 0.0, "no_edge"

        if state.drawdown >= (1.0 - config.drawdown_brake_at):
            stake *= config.drawdown_brake_factor
            binding = "drawdown_brake"

        cap = state.current * config.max_stake_fraction
        if stake > cap:
            stake, binding = cap, "max_stake_fraction"

        if config.max_stake_absolute is not None and stake > config.max_stake_absolute:
            stake, binding = config.max_stake_absolute, "max_stake_absolute"

        daily_room = state.current * config.max_daily_exposure - state.staked_today
        if daily_room <= 0:
            return 0.0, "daily_exposure"
        if stake > daily_room:
            stake, binding = daily_room, "daily_exposure"

        stake = min(stake, state.current)
        stake = round(stake / config.round_to) * config.round_to

        if stake < config.min_stake:
            return 0.0, "below_min_stake"
        return float(round(stake, 2)), binding


def settle(stake: float, decimal_odds: float, won: bool, *, commission: float = 0.0) -> float:
    """Profit or loss on a settled bet.

    A winner returns ``stake * (odds - 1)``, less commission on the winnings.
    A loser returns ``-stake``. Commission is charged on profit only, which is
    how betting exchanges actually work.
    """
    if not won:
        return -float(stake)
    gross = stake * (decimal_odds - 1.0)
    return float(gross * (1.0 - commission))


__all__ = [
    "BankrollState",
    "StakingConfig",
    "StakingMethod",
    "StakingPlan",
    "kelly_fraction",
    "settle",
]
