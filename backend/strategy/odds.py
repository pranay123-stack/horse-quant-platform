"""Odds handling and bookmaker-margin removal.

Turning a price into a probability is the step where value-betting systems most
often fool themselves, for two reasons.

Which price do you use?
=======================
A bettor takes the **best** price on offer, but the best price across many books
is not the market's opinion — it is the opinion of whichever book is currently
most wrong, or slowest. Its implied probabilities routinely sum to less than 1,
and dividing by that "overround" manufactures edge out of nothing.

So the two roles are kept separate here:

* **Consensus price** (mean across bookmakers) → the market's probability estimate
* **Best price** (max across bookmakers) → what the bet is actually struck at

Betting the best price against a consensus probability is what a real value
operation does. Computing both from the best price is circular and flattering.

How do you remove the margin?
=============================
Quoted probabilities sum to ~1.15. Getting from there to a fair distribution is
a modelling choice, not arithmetic, and the choice decides which runners look
like value:

``proportional``  divide by the total. Simple, standard, and biased: it assumes
                  the margin is loaded evenly, when bookmakers demonstrably load
                  more of it onto longshots. Systematically leaves longshots
                  looking better than they are.
``additive``      subtract the excess equally across runners. Over-corrects
                  favourites and can produce negative probabilities in big fields.
``power``         find ``k`` with ``Σ qᵢ^k = 1``. Shrinks longshots more than
                  favourites, which is the direction the favourite-longshot bias
                  actually runs.

``power`` is the default. The others exist because the difference between them
is not academic: on a 20/1 shot, proportional and power can disagree by enough
to flip a 5% "edge" into a negative one.

One caveat worth stating plainly. A strategy filtering on **expected value alone**
(``EV = p_model * odds - 1``) never consults the market probability, so the margin
method changes what is *reported* and nothing about which bets are placed. It
starts to bite only when a strategy also filters on ``edge`` — see
``STRATEGY_F`` in :mod:`backend.strategy.filters`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from backend.features.base import numeric_column

MarginMethod = Literal["proportional", "additive", "power", "none"]

#: Bisection bounds for the power method. k=1 is proportional-free (no change);
#: real books need k between about 1.0 and 1.5.
_POWER_LOW, _POWER_HIGH = 0.5, 3.0
_POWER_TOLERANCE = 1e-9
_POWER_MAX_ITERATIONS = 60

#: Below this, a "market" is one bookmaker's opinion rather than a market.
MIN_BOOKMAKERS_FOR_CONSENSUS = 2


def implied_probability(decimal_odds: float | np.ndarray) -> float | np.ndarray:
    """``1 / odds`` — the price expressed as a probability, margin included."""
    odds = np.asarray(decimal_odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        probabilities = np.where(odds > 1, 1.0 / odds, np.nan)
    return probabilities if probabilities.ndim else float(probabilities)


def _solve_power(quoted: np.ndarray) -> float:
    """Find ``k`` such that ``Σ qᵢ^k = 1`` by bisection.

    Monotone in ``k`` (every ``q < 1``, so raising the power lowers the sum),
    which makes bisection both safe and quick.
    """
    low, high = _POWER_LOW, _POWER_HIGH
    for _ in range(_POWER_MAX_ITERATIONS):
        mid = (low + high) / 2
        total = float(np.sum(quoted**mid))
        if abs(total - 1.0) < _POWER_TOLERANCE:
            return mid
        if total > 1.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def remove_margin(decimal_odds: np.ndarray | pd.Series, method: MarginMethod = "power") -> np.ndarray:
    """Fair probabilities for one race's runners, summing to 1.

    Runners with no usable price are excluded from the normalisation and come
    back as ``NaN`` — never as a fabricated probability, which would look to the
    strategy layer like a real opportunity.
    """
    odds = np.asarray(decimal_odds, dtype=float)
    quoted = np.where(odds > 1, 1.0 / odds, np.nan)
    usable = np.isfinite(quoted)

    result = np.full(odds.shape, np.nan, dtype=float)
    if not usable.any():
        return result

    values = quoted[usable]
    total = values.sum()
    if total <= 0:
        return result

    if method == "none":
        fair = values
    elif method == "proportional":
        fair = values / total
    elif method == "additive":
        excess = (total - 1.0) / values.size
        fair = np.clip(values - excess, 1e-9, None)
        fair = fair / fair.sum()
    elif method == "power":
        fair = values ** _solve_power(values)
        fair = fair / fair.sum()
    else:
        raise ValueError(f"unknown margin method {method!r}")

    result[usable] = fair
    return result


def overround(decimal_odds: np.ndarray | pd.Series) -> float:
    """Sum of quoted probabilities. 1.15 means a 15% book margin."""
    quoted = implied_probability(np.asarray(decimal_odds, dtype=float))
    finite = np.asarray(quoted)[np.isfinite(np.asarray(quoted))]
    return float(finite.sum()) if finite.size else float("nan")


@dataclass(slots=True)
class MarketView:
    """What the market thinks, and what we could actually bet at."""

    race_id: str
    horse_id: str
    #: Mean price across bookmakers — the consensus used for probability.
    consensus_odds: float
    #: Best price across bookmakers — what a bet is struck at.
    best_odds: float
    market_probability: float
    quoted_probability: float
    book_overround: float
    bookmaker_count: int

    @property
    def is_usable(self) -> bool:
        return (
            np.isfinite(self.best_odds)
            and self.best_odds > 1
            and np.isfinite(self.market_probability)
            and 0 < self.market_probability < 1
        )


CONSENSUS_COLUMN = "mkt_mean_odds"
BEST_PRICE_COLUMN = "mkt_latest_odds"


def build_market_frame(
    frame: pd.DataFrame,
    *,
    method: MarginMethod = "power",
    consensus_column: str = CONSENSUS_COLUMN,
    best_column: str = BEST_PRICE_COLUMN,
) -> pd.DataFrame:
    """Attach margin-free market probabilities and the bettable price.

    Adds:

    ``market_probability``   fair probability from the consensus price
    ``quoted_probability``   ``1 / consensus`` including the margin
    ``bet_odds``             best available price — what a stake is struck at
    ``book_overround``       consensus overround for the race
    """
    result = frame.copy()
    if result.empty:
        for column in ("market_probability", "quoted_probability", "bet_odds", "book_overround"):
            result[column] = pd.Series(dtype="float64")
        return result

    consensus = numeric_column(result, consensus_column)
    best = numeric_column(result, best_column)

    # Fall back only in the direction that cannot invent edge: if there is no
    # consensus price, use the best price for both roles and accept that the
    # measured edge is then conservative-at-best.
    consensus = consensus.where(consensus.notna(), best)
    best = best.where(best.notna(), consensus)

    result["bet_odds"] = best
    result["quoted_probability"] = implied_probability(consensus.to_numpy())

    fair = np.full(len(result), np.nan)
    overrounds = np.full(len(result), np.nan)
    for _, positions in result.groupby("race_id", sort=False).indices.items():
        race_odds = consensus.to_numpy()[positions]
        fair[positions] = remove_margin(race_odds, method)
        overrounds[positions] = overround(race_odds)

    result["market_probability"] = fair
    result["book_overround"] = overrounds
    return result


__all__ = [
    "BEST_PRICE_COLUMN",
    "CONSENSUS_COLUMN",
    "MIN_BOOKMAKERS_FOR_CONSENSUS",
    "MarginMethod",
    "MarketView",
    "build_market_frame",
    "implied_probability",
    "overround",
    "remove_margin",
]
