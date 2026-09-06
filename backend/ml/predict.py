"""Inference: a race in, win probabilities out.

    from backend.ml import RacePredictor

    predictor = RacePredictor(session)
    prediction = predictor.predict_race("rac_18f2a")
    print(prediction.render())

The output is the platform's deliverable for Phase 4:

    {"horse_id": "hrs_9c11", "probability": 0.285}

Probabilities are calibrated and normalised within the race, so they sum to 1.
No odds are consulted and no bet is suggested — comparing these numbers with a
price is Phase 5's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from backend.features import FeaturePipeline
from backend.features.base import FeatureConfig
from backend.ml.metrics.racing import race_probability_frame
from backend.ml.models.calibrated import CalibratedRacingModel
from backend.ml.models.registry import ModelRegistry
from backend.utils.exceptions import ModelError, NotFoundError
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="model")


@dataclass(slots=True)
class RunnerPrediction:
    """One runner's predicted chance."""

    horse_id: str
    probability: float
    rank: int
    horse_name: str | None = None
    raw_probability: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"horse_id": self.horse_id, "probability": round(self.probability, 4)}
        if self.horse_name:
            payload["horse_name"] = self.horse_name
        payload["rank"] = self.rank
        return payload


@dataclass(slots=True)
class RacePrediction:
    """Every runner in one race, most likely winner first."""

    race_id: str
    model: str
    runners: list[RunnerPrediction] = field(default_factory=list)
    race_date: str | None = None
    course: str | None = None

    @property
    def favourite(self) -> RunnerPrediction | None:
        return self.runners[0] if self.runners else None

    @property
    def probability_sum(self) -> float:
        return sum(runner.probability for runner in self.runners)

    def as_dict(self) -> dict[str, Any]:
        return {
            "race_id": self.race_id,
            "race_date": self.race_date,
            "course": self.course,
            "model": self.model,
            "runners": [runner.as_dict() for runner in self.runners],
        }

    def render(self) -> str:
        lines = [
            f"Race {self.race_id}  {self.course or ''}  {self.race_date or ''}".rstrip(),
            f"Model: {self.model}",
            "-" * 52,
            f"  {'rank':>4}  {'probability':>11}  horse",
        ]
        for runner in self.runners:
            name = runner.horse_name or runner.horse_id
            lines.append(f"  {runner.rank:>4}  {runner.probability:>10.1%}  {name}")
        lines.append("-" * 52)
        lines.append(f"  {'sum':>4}  {self.probability_sum:>10.1%}")
        return "\n".join(lines)


def _runner_predictions(frame: pd.DataFrame, names: dict[str, str | None]) -> list[RunnerPrediction]:
    """Build typed predictions from a scored frame."""
    predictions = []
    for record in frame.to_dict(orient="records"):
        horse_id = str(record["horse_id"])
        predictions.append(
            RunnerPrediction(
                horse_id=horse_id,
                probability=float(record["probability_normalised"]),
                rank=int(record["model_rank"]),
                horse_name=names.get(horse_id),
                raw_probability=float(record["probability"]),
            )
        )
    return predictions


class RacePredictor:
    """Loads a registered model and scores races."""

    def __init__(
        self,
        session: Session,
        model: CalibratedRacingModel | None = None,
        *,
        registry: ModelRegistry | None = None,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        self.session = session
        self.registry = registry or ModelRegistry()
        self._model = model
        self.pipeline = FeaturePipeline(session, feature_config)

    # ------------------------------------------------------------------
    @property
    def model(self) -> CalibratedRacingModel:
        """The champion model, loaded on first use."""
        if self._model is None:
            self._model = self.registry.load_champion()
        return self._model

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score a pre-built feature frame, returning it with probabilities."""
        if frame.empty:
            return frame.assign(probability=[], probability_normalised=[], model_rank=[])
        raw = self.model.predict_proba(frame)
        return race_probability_frame(frame, raw)

    def predict_race(self, race_id: str) -> RacePrediction:
        """Build features for one race and score it."""
        frame = self.pipeline.build_for_race(race_id)
        if frame.empty:
            raise NotFoundError(f"race {race_id!r} has no runners to score", details={"race_id": race_id})

        scored = self.predict_frame(frame).sort_values("model_rank")
        names = self._horse_names(scored["horse_id"].tolist())

        prediction = RacePrediction(
            race_id=race_id,
            model=self.model.name,
            race_date=str(scored["race_date"].iloc[0]) if "race_date" in scored else None,
            course=str(scored["course_name"].iloc[0]) if "course_name" in scored else None,
            runners=_runner_predictions(scored, names),
        )
        logger.info(
            "race scored",
            extra={"race_id": race_id, "runners": len(prediction.runners), "model": self.model.name},
        )
        return prediction

    def predict_date(self, race_date: str) -> list[RacePrediction]:
        """Score every race on a given day."""
        from datetime import date as date_type

        parsed = date_type.fromisoformat(race_date)
        frame = self.pipeline.build(date_from=parsed, date_to=parsed)
        if frame.empty:
            return []

        scored = self.predict_frame(frame)
        names = self._horse_names(scored["horse_id"].tolist())

        predictions = []
        for race_id, group in scored.sort_values("model_rank").groupby("race_id", sort=False):
            predictions.append(
                RacePrediction(
                    race_id=str(race_id),
                    model=self.model.name,
                    race_date=str(group["race_date"].iloc[0]),
                    course=str(group["course_name"].iloc[0]) if "course_name" in group else None,
                    runners=_runner_predictions(group, names),
                )
            )
        return predictions

    # ------------------------------------------------------------------
    def _horse_names(self, horse_ids: list[str]) -> dict[str, str | None]:
        from sqlalchemy import select

        from backend.models import Horse

        if not horse_ids:
            return {}
        rows = self.session.execute(
            select(Horse.horse_id, Horse.name).where(Horse.horse_id.in_(set(horse_ids)))
        ).all()
        names: dict[str, str | None] = {}
        for horse_id, name in rows:
            names[horse_id] = name
        return names


def load_champion(registry: ModelRegistry | None = None) -> CalibratedRacingModel:
    """Load the promoted champion, with a clear error when there is none."""
    registry = registry or ModelRegistry()
    record = registry.champion()
    if record is None:
        raise ModelError("no champion model has been promoted; run training first")
    return registry.load(record.name, record.version)


__all__ = ["RacePrediction", "RacePredictor", "RunnerPrediction", "load_champion"]
