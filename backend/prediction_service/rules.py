"""The production betting rules.

Deliberately short and deliberately boring. Four conditions decide whether a
runner is backed, and every one of them exists because of something that goes
wrong without it:

* **EV above 5%** — below that, the edge is inside the noise of the price.
* **Probability above 10%** — a model is least reliable in the tail, and a 3%
  shout at 40/1 carries an EV estimate built almost entirely from extrapolation.
* **A real price** — no odds, no bet. There is nothing to strike a stake at.
* **A real field** — a three-runner race is a different game, and the model was
  fitted on ordinary ones.
* **Not an extreme price** — above 50.0 the market is thin, the price moves on
  contact, and Phase 5 measured profit turning negative above 5.0 anyway.

These are production *guards*, not a strategy to be tuned. Nothing here is
fitted to results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.models.predictions import Recommendation

#: Minimum expected value. `EV = p * odds - 1`.
MIN_EXPECTED_VALUE = 0.05
#: Below this the model is extrapolating rather than predicting.
MIN_PROBABILITY = 0.10
#: A field this small is not the game the model was fitted on.
MIN_FIELD_SIZE = 5
#: Above this the price is thin and moves on contact.
MAX_ODDS = 50.0
#: Below evens a bet cannot return a profit after the stake.
MIN_ODDS = 1.01


@dataclass(frozen=True, slots=True)
class BettingRules:
    """The thresholds, in one place so the API can report them."""

    min_expected_value: float = MIN_EXPECTED_VALUE
    min_probability: float = MIN_PROBABILITY
    min_field_size: int = MIN_FIELD_SIZE
    max_odds: float = MAX_ODDS
    min_odds: float = MIN_ODDS

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_expected_value": self.min_expected_value,
            "min_probability": self.min_probability,
            "min_field_size": self.min_field_size,
            "max_odds": self.max_odds,
            "min_odds": self.min_odds,
        }

    def describe(self) -> str:
        return (
            f"BET when EV > {self.min_expected_value:.0%} and probability > "
            f"{self.min_probability:.0%}, with a price between {self.min_odds} and "
            f"{self.max_odds} in a field of {self.min_field_size}+"
        )


RULES = BettingRules()


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """A numeric column, or an all-NaN one if it is absent."""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def apply_rules(frame: pd.DataFrame, rules: BettingRules = RULES) -> pd.DataFrame:
    """Label every runner BET or NO_BET, and say why when it is not.

    Vectorised, and the reason is recorded in the order the guards are checked
    so that "no price" never gets reported as "EV too low" — a missing price is
    a data problem and a low EV is a judgement, and confusing them wastes time.
    """
    result = frame.copy()
    if result.empty:
        result["recommendation"] = pd.Series(dtype="object")
        result["rejection_reason"] = pd.Series(dtype="object")
        return result

    probability = _numeric(result, "model_probability")
    odds = _numeric(result, "odds")
    expected_value = _numeric(result, "expected_value")
    field_size = (
        pd.to_numeric(result["field_size"], errors="coerce")
        if "field_size" in result.columns
        else result.groupby("race_id")["race_id"].transform("size")
    )

    reason = pd.Series("", index=result.index, dtype="object")

    def mark(mask: pd.Series, text: str) -> None:
        reason.loc[(reason == "") & mask.fillna(True)] = text

    # Order matters: data problems before judgements.
    mark(odds.isna(), "no price available")
    mark(odds < rules.min_odds, "price below the minimum")
    mark(odds > rules.max_odds, "price above the maximum")
    mark(field_size < rules.min_field_size, "field too small")
    mark(probability.isna(), "no model probability")
    mark(probability < rules.min_probability, "probability below the minimum")
    mark(expected_value.isna() | (expected_value <= rules.min_expected_value), "expected value too low")

    result["rejection_reason"] = reason.replace("", None)
    result["recommendation"] = np.where(reason == "", Recommendation.BET, Recommendation.NO_BET)
    return result


__all__ = [
    "MAX_ODDS",
    "MIN_EXPECTED_VALUE",
    "MIN_FIELD_SIZE",
    "MIN_ODDS",
    "MIN_PROBABILITY",
    "RULES",
    "BettingRules",
    "apply_rules",
]
