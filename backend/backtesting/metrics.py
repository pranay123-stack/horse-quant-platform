"""Backtest performance metrics.

Betting returns are not stock returns, and the differences matter:

* Outcomes are **discrete and highly skewed**. A 20/1 winner is a 21x return on
  that stake; most bets return nothing at all. Mean return alone is nearly
  useless without a sense of the spread.
* Bets are **not continuously compounded**. ROI is measured against total amount
  staked, not against the bankroll, because staking 1 unit a thousand times is a
  different exposure from staking 1000 units once.
* **Drawdown is what actually ends strategies.** A genuinely profitable staking
  plan with a 60% drawdown is one an operator abandons at the worst moment. It
  is reported here as both cash and percentage of peak.

The Sharpe ratio below is computed **per bet**, not annualised. Annualising
requires assuming a betting frequency, and quoting an annualised figure from a
few hundred bets implies a precision that is not there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class BacktestMetrics:
    """Everything needed to judge a strategy, and nothing that flatters it."""

    total_bets: int = 0
    wins: int = 0
    losses: int = 0

    total_staked: float = 0.0
    total_returned: float = 0.0
    profit: float = 0.0
    roi: float = 0.0

    strike_rate: float = 0.0
    average_odds: float = 0.0
    average_stake: float = 0.0
    average_ev: float = 0.0

    starting_bankroll: float = 0.0
    final_bankroll: float = 0.0
    peak_bankroll: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    sharpe_per_bet: float = 0.0
    longest_losing_streak: int = 0
    races_considered: int = 0
    races_with_bet: int = 0

    equity_curve: list[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def bet_rate(self) -> float:
        """Fraction of races that produced at least one bet."""
        return (self.races_with_bet / self.races_considered) if self.races_considered else 0.0

    @property
    def is_profitable(self) -> bool:
        return self.profit > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "bets": {
                "total": self.total_bets,
                "wins": self.wins,
                "losses": self.losses,
                "strike_rate": round(self.strike_rate, 6),
                "average_odds": round(self.average_odds, 4),
                "average_stake": round(self.average_stake, 4),
                "average_ev": round(self.average_ev, 6),
            },
            "money": {
                "staked": round(self.total_staked, 2),
                "returned": round(self.total_returned, 2),
                "profit": round(self.profit, 2),
                "roi": round(self.roi, 6),
            },
            "bankroll": {
                "starting": round(self.starting_bankroll, 2),
                "final": round(self.final_bankroll, 2),
                "peak": round(self.peak_bankroll, 2),
                "max_drawdown": round(self.max_drawdown, 2),
                "max_drawdown_pct": round(self.max_drawdown_pct, 6),
            },
            "risk": {
                "sharpe_per_bet": round(self.sharpe_per_bet, 4),
                "longest_losing_streak": self.longest_losing_streak,
            },
            "coverage": {
                "races_considered": self.races_considered,
                "races_with_bet": self.races_with_bet,
                "bet_rate": round(self.bet_rate, 6),
            },
        }

    def render(self) -> str:
        sign = "+" if self.profit >= 0 else ""
        return "\n".join(
            [
                "Backtest results",
                "----------------",
                f"  Races considered : {self.races_considered}",
                f"  Races bet        : {self.races_with_bet} ({self.bet_rate:.1%})",
                f"  Total bets       : {self.total_bets}",
                f"  Winners          : {self.wins}",
                f"  Losers           : {self.losses}",
                f"  Strike rate      : {self.strike_rate:.2%}",
                "",
                f"  Staked           : {self.total_staked:,.2f}",
                f"  Returned         : {self.total_returned:,.2f}",
                f"  Profit / loss    : {sign}{self.profit:,.2f}",
                f"  ROI              : {self.roi:+.2%}",
                "",
                f"  Bankroll start   : {self.starting_bankroll:,.2f}",
                f"  Bankroll end     : {self.final_bankroll:,.2f}",
                f"  Peak             : {self.peak_bankroll:,.2f}",
                f"  Max drawdown     : {self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2%})",
                "",
                f"  Avg odds taken   : {self.average_odds:.2f}",
                f"  Avg EV at bet    : {self.average_ev:+.2%}",
                f"  Sharpe (per bet) : {self.sharpe_per_bet:.3f}",
                f"  Longest losing   : {self.longest_losing_streak}",
            ]
        )


def compute_metrics(
    bets: pd.DataFrame,
    *,
    starting_bankroll: float,
    races_considered: int = 0,
) -> BacktestMetrics:
    """Compute metrics from a settled bet ledger.

    ``bets`` needs ``stake``, ``odds``, ``won``, ``profit`` and ``bankroll_after``.
    """
    metrics = BacktestMetrics(
        starting_bankroll=starting_bankroll,
        final_bankroll=starting_bankroll,
        peak_bankroll=starting_bankroll,
        races_considered=races_considered,
        equity_curve=[starting_bankroll],
    )
    if bets.empty:
        return metrics

    metrics.total_bets = len(bets)
    metrics.wins = int(bets["won"].sum())
    metrics.losses = metrics.total_bets - metrics.wins
    metrics.races_with_bet = int(bets["race_id"].nunique()) if "race_id" in bets else 0

    metrics.total_staked = float(bets["stake"].sum())
    metrics.profit = float(bets["profit"].sum())
    metrics.total_returned = metrics.total_staked + metrics.profit
    metrics.roi = metrics.profit / metrics.total_staked if metrics.total_staked else 0.0

    metrics.strike_rate = metrics.wins / metrics.total_bets
    metrics.average_odds = float(bets["odds"].mean())
    metrics.average_stake = float(bets["stake"].mean())
    if "expected_value" in bets:
        metrics.average_ev = float(bets["expected_value"].mean())

    equity = [starting_bankroll, *bets["bankroll_after"].tolist()]
    metrics.equity_curve = [float(value) for value in equity]
    metrics.final_bankroll = metrics.equity_curve[-1]

    running_peak = starting_bankroll
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    for value in metrics.equity_curve:
        running_peak = max(running_peak, value)
        drawdown = running_peak - value
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_pct = drawdown / running_peak if running_peak else 0.0
    metrics.peak_bankroll = running_peak
    metrics.max_drawdown = max_drawdown
    metrics.max_drawdown_pct = max_drawdown_pct

    # Per-bet Sharpe on return-on-stake, so bets of different sizes are comparable.
    returns = (bets["profit"] / bets["stake"]).replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(returns) > 1:
        spread = float(returns.std(ddof=1))
        metrics.sharpe_per_bet = float(returns.mean()) / spread if spread > 0 else 0.0

    metrics.longest_losing_streak = _longest_losing_streak(bets["won"].tolist())
    return metrics


def _longest_losing_streak(outcomes: list[Any]) -> int:
    longest = current = 0
    for outcome in outcomes:
        if outcome:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def kelly_fraction(probability: float, odds: float) -> float:
    """Full-Kelly stake as a fraction of bankroll.

    ``f* = (p·b - q) / b`` where ``b = odds - 1``. Returns 0 when there is no
    edge — Kelly never backs a negative-EV proposition.

    Full Kelly is the growth-optimal bet *given a correct probability*. Ours is
    estimated, and Kelly is brutally sensitive to overestimated edges, which is
    why the engine applies a fractional multiplier and a hard cap on top.
    """
    if odds <= 1 or not 0 < probability < 1:
        return 0.0
    b = odds - 1.0
    edge = probability * b - (1.0 - probability)
    return max(0.0, edge / b)


def expected_value(probability: float, odds: float) -> float:
    """``EV = p * odds - 1`` — profit per unit staked, at these odds."""
    if odds <= 0 or probability is None or math.isnan(probability):
        return -1.0
    return probability * odds - 1.0


__all__ = ["BacktestMetrics", "compute_metrics", "expected_value", "kelly_fraction"]
