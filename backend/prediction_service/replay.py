"""Does the live path produce the same numbers as the backtest path?

This is the check that decides whether the backtest is evidence about the
product, or evidence about a different program that happens to share a
repository.

The failure it guards against is not dramatic. Nobody sets out to build two
models. It happens because the live service needs one small thing the backtest
did not — a column renamed, a NaN filled, a runner dropped — and each change is
individually reasonable. Six of them later the live model is scoring a different
feature matrix, the backtest still reports 6% ROI, and nothing has failed.

So this replays historical races through the *production* service and compares
the probabilities against the ones the model produces directly. They must agree
to floating-point tolerance, not approximately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.prediction_service.service import PredictionService
from backend.utils.logging import get_logger, safe_extra

logger = get_logger(__name__, channel="model")

#: Probabilities must match to this tolerance. Loose enough for float
#: arithmetic, far tighter than any difference a pipeline divergence would make.
TOLERANCE = 1e-9


@dataclass(slots=True)
class ReplayResult:
    """Whether the live path and the model path agree, and where they do not."""

    races_checked: int = 0
    runners_checked: int = 0
    max_absolute_difference: float = 0.0
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    missing_from_live: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return (
            self.runners_checked > 0
            and not self.mismatches
            and not self.missing_from_live
            and self.max_absolute_difference <= TOLERANCE
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "consistent": self.consistent,
            "races_checked": self.races_checked,
            "runners_checked": self.runners_checked,
            "max_absolute_difference": self.max_absolute_difference,
            "mismatches": self.mismatches[:10],
            "missing_from_live": self.missing_from_live[:10],
        }

    def render(self) -> str:
        status = "CONSISTENT" if self.consistent else "DIVERGED"
        lines = [
            "Backtest / live consistency",
            "---------------------------",
            f"  Status            {status}",
            f"  Races checked     {self.races_checked}",
            f"  Runners checked   {self.runners_checked}",
            f"  Largest gap       {self.max_absolute_difference:.2e}",
        ]
        if self.missing_from_live:
            lines.append(f"  Missing live      {len(self.missing_from_live)}")
        for mismatch in self.mismatches[:5]:
            lines.append(
                f"    {mismatch['race_id']}/{mismatch['horse_id']}: "
                f"live {mismatch['live']:.6f} vs model {mismatch['expected']:.6f}"
            )
        return "\n".join(lines)


def replay_predictions(
    session: Session,
    race_date: date,
    *,
    service: PredictionService | None = None,
) -> ReplayResult:
    """Score a past day both ways and compare, runner by runner.

    The reference side deliberately uses the same predictor the backtest used —
    features from :class:`FeaturePipeline`, probabilities from the champion
    model, normalised within the race. If the service ever grows its own
    handling, this is what notices.
    """
    service = service or PredictionService(session)
    result = ReplayResult()

    reference = service.pipeline.build(date_from=race_date, date_to=race_date)
    if reference.empty:
        logger.warning("nothing to replay", extra={"race_date": str(race_date)})
        return result

    scored = service.predictor.predict_frame(reference)
    expected = {
        (str(row["race_id"]), str(row["horse_id"])): float(row["probability_normalised"])
        for _, row in scored.iterrows()
    }

    live = service.signals_for_date(race_date)
    result.races_checked = len(live.races)

    seen: set[tuple[str, str]] = set()
    for race in live.races:
        for runner in race.runners:
            key = (race.race_id, runner.horse_id)
            seen.add(key)
            if key not in expected:
                result.missing_from_live.append(f"{key[0]}/{key[1]} not in the reference frame")
                continue

            difference = abs(runner.model_probability - expected[key])
            result.max_absolute_difference = max(result.max_absolute_difference, difference)
            result.runners_checked += 1
            if difference > TOLERANCE:
                result.mismatches.append(
                    {
                        "race_id": key[0],
                        "horse_id": key[1],
                        "live": runner.model_probability,
                        "expected": expected[key],
                        "difference": difference,
                    }
                )

    for key in expected:
        if key not in seen:
            result.missing_from_live.append(f"{key[0]}/{key[1]} was scored offline but not live")

    logger.info("replay complete", extra=safe_extra(result.as_dict()))
    return result


def compare_frames(live: pd.DataFrame, backtest: pd.DataFrame, *, column: str = "model_probability") -> float:
    """Largest absolute gap between two aligned scorings. A helper for tests."""
    if live.empty or backtest.empty:
        return float("inf")
    keys = ["race_id", "horse_id"]
    merged = live[[*keys, column]].merge(backtest[[*keys, column]], on=keys, suffixes=("_a", "_b"))
    if merged.empty:
        return float("inf")
    return float(np.abs(merged[f"{column}_a"] - merged[f"{column}_b"]).max())


__all__ = ["TOLERANCE", "ReplayResult", "compare_frames", "replay_predictions"]
