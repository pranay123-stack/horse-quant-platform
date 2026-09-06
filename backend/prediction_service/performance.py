"""How the recommendations actually did.

Scored from settled `predictions` rows, at flat stakes, so the number on the
dashboard is the product's own record rather than a backtest replayed.

Two deliberate choices:

**Only settled BETs count.** An unsettled prediction is not a pending win, and
counting it would flatter every number on the page during a losing week.

**ROI carries its standard error and a t-statistic.** A headline ROI without its
uncertainty is how betting records mislead: 40 bets at +12% and 4,000 bets at
+12% are not the same claim, and the page should not present them as though they
were.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.predictions import Prediction, Recommendation
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="model")

#: Flat stake, so the record reflects selection rather than sizing.
FLAT_STAKE = 10.0
#: Below this, report the numbers but say they mean nothing yet.
MIN_BETS_FOR_INFERENCE = 200


@dataclass(slots=True)
class PerformanceSummary:
    """The product's live record."""

    bets: int = 0
    settled: int = 0
    pending: int = 0
    wins: int = 0
    staked: float = 0.0
    returned: float = 0.0
    profit: float = 0.0
    roi: float = 0.0
    strike_rate: float = 0.0
    average_odds: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    t_statistic: float = 0.0
    first_bet: date | None = None
    last_bet: date | None = None
    equity: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_significant(self) -> bool:
        """Enough bets, and far enough from zero, to mean anything."""
        return self.settled >= MIN_BETS_FOR_INFERENCE and abs(self.t_statistic) >= 2.0

    @property
    def headline(self) -> str:
        if self.settled == 0:
            return "No settled bets yet."
        if self.settled < MIN_BETS_FOR_INFERENCE:
            return (
                f"{self.roi:+.1%} ROI over {self.settled} settled bets — too few to "
                f"distinguish from chance ({MIN_BETS_FOR_INFERENCE} needed)."
            )
        if self.is_significant:
            return f"{self.roi:+.1%} ROI over {self.settled} bets (t = {self.t_statistic:+.2f})."
        return (
            f"{self.roi:+.1%} ROI over {self.settled} bets, but t = "
            f"{self.t_statistic:+.2f} — not distinguishable from chance."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "bets": self.bets,
            "settled": self.settled,
            "pending": self.pending,
            "wins": self.wins,
            "staked": round(self.staked, 2),
            "returned": round(self.returned, 2),
            "profit": round(self.profit, 2),
            "roi": round(self.roi, 4),
            "strike_rate": round(self.strike_rate, 4),
            "average_odds": round(self.average_odds, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "t_statistic": round(self.t_statistic, 3),
            "significant": self.is_significant,
            "headline": self.headline,
            "first_bet": str(self.first_bet) if self.first_bet else None,
            "last_bet": str(self.last_bet) if self.last_bet else None,
        }


def summarise_performance(
    session: Session,
    *,
    date_from: date | None = None,
    date_to: date | None = None,
    stake: float = FLAT_STAKE,
) -> PerformanceSummary:
    """Score every recommended bet that has been settled."""
    statement = select(Prediction).where(Prediction.recommendation == Recommendation.BET)
    if date_from is not None:
        statement = statement.where(Prediction.race_date >= date_from)
    if date_to is not None:
        statement = statement.where(Prediction.race_date <= date_to)

    rows = list(
        session.execute(statement.order_by(Prediction.race_date, Prediction.off_time)).scalars().all()
    )
    summary = PerformanceSummary(bets=len(rows))
    settled = [row for row in rows if row.settled and row.odds]
    summary.pending = len(rows) - len(settled)
    summary.settled = len(settled)

    if not settled:
        return summary

    summary.first_bet = settled[0].race_date
    summary.last_bet = settled[-1].race_date

    returns: list[float] = []
    bankroll = 0.0
    peak = 0.0
    odds_total = 0.0

    for row in settled:
        price = float(row.odds or 0.0)
        odds_total += price
        payout = stake * price if row.won else 0.0
        profit = payout - stake

        summary.staked += stake
        summary.returned += payout
        summary.wins += 1 if row.won else 0
        returns.append(profit / stake)

        bankroll += profit
        peak = max(peak, bankroll)
        drawdown = peak - bankroll
        if drawdown > summary.max_drawdown:
            summary.max_drawdown = drawdown
            summary.max_drawdown_pct = drawdown / peak if peak > 0 else 0.0

        summary.equity.append(
            {
                "date": str(row.race_date) if row.race_date else None,
                "bankroll": round(bankroll, 2),
                "bets": len(summary.equity) + 1,
            }
        )

    summary.profit = summary.returned - summary.staked
    summary.roi = summary.profit / summary.staked if summary.staked else 0.0
    summary.strike_rate = summary.wins / summary.settled
    summary.average_odds = odds_total / summary.settled

    # A betting record's uncertainty is dominated by the long tail of the price
    # distribution, so the t-statistic uses the spread of per-bet returns.
    if len(returns) > 1:
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        standard_error = math.sqrt(variance / len(returns))
        summary.t_statistic = mean / standard_error if standard_error > 0 else 0.0

    logger.info(
        "performance summarised",
        extra={"settled": summary.settled, "roi": round(summary.roi, 4)},
    )
    return summary


__all__ = ["FLAT_STAKE", "MIN_BETS_FOR_INFERENCE", "PerformanceSummary", "summarise_performance"]
