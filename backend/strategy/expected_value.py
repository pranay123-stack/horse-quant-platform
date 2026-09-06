"""Expected value and the value-bet signal.

    EV = model_probability * decimal_odds - 1

Profit per unit staked, at these odds, if the model probability is right. That
last clause carries all the risk: EV is a statement about the model, not about
the world. A model 20% too confident produces a fictitious +15% edge that will
be bet again and again.

Two numbers are therefore reported side by side:

**Edge** — ``model_probability - market_probability``. How much the model
disagrees with the market's *fair* estimate, in probability terms.

**Expected value** — ``p * odds - 1``. What that disagreement is worth at the
price actually available.

They are not interchangeable. A 2-point edge on a 1.5 favourite is worth little;
the same 2 points on a 25/1 shot is enormous EV — and is also far more likely to
be an artefact of a slightly miscalibrated tail. The filters in
:mod:`backend.strategy.filters` exist mainly to police that asymmetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from backend.features.base import numeric_column

Decision = Literal["bet", "no_bet"]

#: Reasons a runner was rejected. Counting them is how a strategy's behaviour
#: gets explained rather than guessed at.
REJECTION_REASONS = (
    "no_odds",
    "no_probability",
    "ev_below_threshold",
    "probability_too_low",
    "odds_too_low",
    "odds_too_high",
    "field_too_small",
    "overround_implausible",
    "edge_too_small",
    "max_bets_per_race",
)


def expected_value(model_probability: float, decimal_odds: float) -> float:
    """``p * odds - 1``. Returns -1 for an unusable price (stake lost)."""
    if not np.isfinite(model_probability) or not np.isfinite(decimal_odds) or decimal_odds <= 1:
        return -1.0
    return float(model_probability * decimal_odds - 1.0)


def edge(model_probability: float, market_probability: float) -> float:
    """Model probability minus the market's margin-free estimate."""
    if not np.isfinite(model_probability) or not np.isfinite(market_probability):
        return float("nan")
    return float(model_probability - market_probability)


def fair_odds(model_probability: float) -> float:
    """The price at which this probability would be a break-even bet."""
    if not np.isfinite(model_probability) or model_probability <= 0:
        return float("inf")
    return float(1.0 / model_probability)


@dataclass(slots=True)
class ValueBetSignal:
    """One runner assessed against the market."""

    race_id: str
    horse_id: str
    race_date: date | None
    model_probability: float
    market_probability: float
    odds: float
    edge: float
    expected_value: float
    decision: Decision
    reason: str = ""
    horse_name: str | None = None
    field_size: int = 0
    book_overround: float = float("nan")

    @property
    def is_bet(self) -> bool:
        return self.decision == "bet"

    @property
    def fair_odds(self) -> float:
        return fair_odds(self.model_probability)

    @property
    def odds_ratio(self) -> float:
        """Available price divided by fair price. >1 is value."""
        fair = self.fair_odds
        return self.odds / fair if np.isfinite(fair) and fair > 0 else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "race_id": self.race_id,
            "horse_id": self.horse_id,
            "race_date": self.race_date.isoformat() if self.race_date else None,
            "horse_name": self.horse_name,
            "model_probability": round(self.model_probability, 4),
            "market_probability": round(self.market_probability, 4)
            if np.isfinite(self.market_probability)
            else None,
            "odds": round(self.odds, 3) if np.isfinite(self.odds) else None,
            "fair_odds": round(self.fair_odds, 3) if np.isfinite(self.fair_odds) else None,
            "edge": round(self.edge, 4) if np.isfinite(self.edge) else None,
            "expected_value": round(self.expected_value, 4),
            "decision": self.decision,
            "reason": self.reason,
        }

    def render(self) -> str:
        marker = "BET " if self.is_bet else "    "
        name = self.horse_name or self.horse_id
        return (
            f"  {marker}{name:<26} model {self.model_probability:>6.1%}  "
            f"market {self.market_probability:>6.1%}  @ {self.odds:>6.2f}  "
            f"edge {self.edge:>+6.1%}  EV {self.expected_value:>+7.1%}"
            + (f"   [{self.reason}]" if self.reason and not self.is_bet else "")
        )


def compute_signal_frame(
    frame: pd.DataFrame,
    *,
    probability_column: str = "model_probability",
    odds_column: str = "bet_odds",
    market_column: str = "market_probability",
) -> pd.DataFrame:
    """Vectorised edge and EV for a whole frame."""
    result = frame.copy()
    if result.empty:
        result["edge"] = pd.Series(dtype="float64")
        result["expected_value"] = pd.Series(dtype="float64")
        result["fair_odds"] = pd.Series(dtype="float64")
        return result

    probability = pd.to_numeric(result[probability_column], errors="coerce")
    odds = pd.to_numeric(result[odds_column], errors="coerce")
    market = numeric_column(result, market_column)

    result["expected_value"] = np.where(odds > 1, probability * odds - 1.0, -1.0)
    result["edge"] = probability - market
    with np.errstate(divide="ignore", invalid="ignore"):
        result["fair_odds"] = np.where(probability > 0, 1.0 / probability, np.inf)
    return result


__all__ = [
    "REJECTION_REASONS",
    "Decision",
    "ValueBetSignal",
    "compute_signal_frame",
    "edge",
    "expected_value",
    "fair_odds",
]
