"""Today's races in, betting signals out.

    racecards → features → production model → probabilities
              → market prices → EV → rules → recommendations

Every stage is the *same code the backtest used*. That is the point, and it is
the reason this module is thin: `FeaturePipeline`, `RacePredictor`,
`build_market_frame` and `compute_signal_frame` are all imported rather than
reimplemented, so a signal generated this morning is produced by the identical
path that generated the historical ones. A separate "live" implementation would
drift from the backtest within weeks, and the backtest would quietly stop being
evidence about the thing actually being run.

:mod:`backend.prediction_service.replay` exists to prove that claim rather than
assert it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.features.feature_pipeline import FeaturePipeline
from backend.ml.models.calibrated import CalibratedRacingModel
from backend.ml.models.registry import ModelRegistry
from backend.ml.predict import RacePredictor
from backend.models import Course, Horse, Race
from backend.models.features import FEATURE_VERSION
from backend.models.predictions import Prediction, Recommendation
from backend.prediction_service.rules import RULES, BettingRules, apply_rules
from backend.strategy.expected_value import compute_signal_frame
from backend.strategy.odds import build_market_frame
from backend.utils.exceptions import NotFoundError
from backend.utils.logging import get_logger, safe_extra
from backend.utils.timeutils import today_uk

logger = get_logger(__name__, channel="model")


@dataclass(slots=True)
class RunnerSignal:
    """One runner's recommendation, ready to render."""

    horse_id: str
    horse: str | None
    model_probability: float
    market_probability: float | None = None
    odds: float | None = None
    expected_value: float | None = None
    edge: float | None = None
    recommendation: str = Recommendation.NO_BET
    rejection_reason: str | None = None

    @property
    def is_bet(self) -> bool:
        return self.recommendation == Recommendation.BET

    def as_dict(self) -> dict[str, Any]:
        return {
            "horse": self.horse or self.horse_id,
            "horse_id": self.horse_id,
            "probability": round(self.model_probability, 4),
            "odds": self.odds,
            "expected_value": round(self.expected_value, 4) if self.expected_value is not None else None,
            "edge": round(self.edge, 4) if self.edge is not None else None,
            "recommendation": self.recommendation,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(slots=True)
class RaceSignals:
    """A race and every runner in it."""

    race_id: str
    race: str
    race_date: date | None = None
    off_time: str | None = None
    course: str | None = None
    runners: list[RunnerSignal] = field(default_factory=list)

    @property
    def bets(self) -> list[RunnerSignal]:
        return [runner for runner in self.runners if runner.is_bet]

    def as_dict(self) -> dict[str, Any]:
        return {
            "race": self.race,
            "race_id": self.race_id,
            "course": self.course,
            "race_date": str(self.race_date) if self.race_date else None,
            "off_time": self.off_time,
            "horses": [runner.as_dict() for runner in self.runners],
        }


@dataclass(slots=True)
class DayPredictions:
    """Everything the dashboard needs for one day."""

    race_date: date
    races: list[RaceSignals] = field(default_factory=list)
    model_name: str = ""
    model_version: str = ""
    generated_at: str = ""
    note: str = ""

    @property
    def total_runners(self) -> int:
        return sum(len(race.runners) for race in self.races)

    @property
    def bets(self) -> list[tuple[RaceSignals, RunnerSignal]]:
        return [(race, runner) for race in self.races for runner in race.bets]

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": str(self.race_date),
            "model": f"{self.model_name}:{self.model_version}" if self.model_name else None,
            "generated_at": self.generated_at,
            "races": len(self.races),
            "runners": self.total_runners,
            "bets": len(self.bets),
            "note": self.note,
            "predictions": [race.as_dict() for race in self.races],
        }


class PredictionService:
    """Scores upcoming races and turns the scores into betting signals."""

    def __init__(
        self,
        session: Session,
        *,
        model: CalibratedRacingModel | None = None,
        registry: ModelRegistry | None = None,
        rules: BettingRules = RULES,
    ) -> None:
        self.session = session
        self.registry = registry or ModelRegistry()
        self.rules = rules
        self.predictor = RacePredictor(session, model, registry=self.registry)
        self.pipeline: FeaturePipeline = self.predictor.pipeline

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self.predictor.model.name

    @property
    def model_version(self) -> str:
        champion = self.registry.champion()
        return champion.version if champion else "unregistered"

    # ------------------------------------------------------------------
    def signals_for_date(self, race_date: date | None = None) -> DayPredictions:
        """Score every race on a date and apply the production rules."""
        target = race_date or today_uk()
        result = DayPredictions(
            race_date=target,
            model_name=self.model_name,
            model_version=self.model_version,
            generated_at=pd.Timestamp.utcnow().isoformat(),
        )

        frame = self.pipeline.build(date_from=target, date_to=target)
        if frame.empty:
            result.note = f"no races found for {target} — has the racecard been imported?"
            logger.info("no races to score", extra={"race_date": str(target)})
            return result

        result.races = self._score(frame)
        logger.info(
            "day scored",
            extra=safe_extra(
                {
                    "race_date": str(target),
                    "races": len(result.races),
                    "bets": len(result.bets),
                    "model": self.model_name,
                }
            ),
        )
        return result

    def signals_for_race(self, race_id: str) -> RaceSignals:
        """Score one race."""
        frame = self.pipeline.build_for_race(race_id)
        if frame.empty:
            raise NotFoundError(f"race {race_id!r} has no runners to score", details={"race_id": race_id})
        races = self._score(frame)
        if not races:
            raise NotFoundError(f"race {race_id!r} could not be scored", details={"race_id": race_id})
        return races[0]

    # ------------------------------------------------------------------
    def _score(self, frame: pd.DataFrame) -> list[RaceSignals]:
        """features -> probabilities -> prices -> EV -> rules."""
        scored = self.predictor.predict_frame(frame)
        scored = scored.rename(columns={"probability_normalised": "model_probability"})

        # Market prices and EV, using exactly the backtest's helpers: the
        # consensus price for the probability, the best price for the bet.
        priced = build_market_frame(scored, method="power")
        priced["odds"] = priced["bet_odds"]
        signals = compute_signal_frame(priced, odds_column="odds")
        decided = apply_rules(signals, self.rules)

        names = self._horse_names(decided["horse_id"].tolist())
        courses = self._course_names(decided)
        races: list[RaceSignals] = []

        for race_id, group in decided.groupby("race_id", sort=False):
            ordered = group.sort_values("model_probability", ascending=False)
            head = ordered.iloc[0]
            course = courses.get(str(head.get("course_id"))) or _text(head.get("course_name"))
            off_time = self._format_off(head.get("off_time"))
            races.append(
                RaceSignals(
                    race_id=str(race_id),
                    race=f"{course} {off_time}".strip() if course else str(race_id),
                    race_date=_as_date(head.get("race_date")),
                    off_time=off_time,
                    course=course,
                    runners=[
                        RunnerSignal(
                            horse_id=str(row["horse_id"]),
                            horse=names.get(str(row["horse_id"])),
                            model_probability=float(row["model_probability"]),
                            market_probability=_as_float(row.get("market_probability")),
                            odds=_as_float(row.get("odds")),
                            expected_value=_as_float(row.get("expected_value")),
                            edge=_as_float(row.get("edge")),
                            recommendation=str(row["recommendation"]),
                            rejection_reason=(
                                None if pd.isna(row["rejection_reason"]) else str(row["rejection_reason"])
                            ),
                        )
                        for _, row in ordered.iterrows()
                    ],
                )
            )

        races.sort(key=lambda race: (race.off_time or "", race.race_id))
        return races

    # ------------------------------------------------------------------
    def _horse_names(self, horse_ids: list[str]) -> dict[str, str | None]:
        if not horse_ids:
            return {}
        rows = self.session.execute(
            select(Horse.horse_id, Horse.name).where(Horse.horse_id.in_(set(horse_ids)))
        ).all()
        return {str(horse_id): name for horse_id, name in rows}

    def _course_names(self, frame: pd.DataFrame) -> dict[str, str]:
        """Course names, which the feature frame carries only as ids.

        A card that says ``rac_s006175 12:30`` is not a product. The name is
        what a user recognises, so it is looked up rather than left out.
        """
        if "course_id" not in frame.columns:
            return {}
        ids = {str(value) for value in frame["course_id"].dropna().unique()}
        if not ids:
            return {}
        rows = self.session.execute(
            select(Course.course_id, Course.name).where(Course.course_id.in_(ids))
        ).all()
        return {str(course_id): name for course_id, name in rows if name}

    @staticmethod
    def _format_off(value: Any) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        stamp = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(stamp) else stamp.strftime("%H:%M")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def store_predictions(session: Session, day: DayPredictions) -> int:
    """Write a day's signals, replacing any earlier run for the same model.

    Re-running the job for the same day is expected — prices move, so a later
    run is a better answer, not a duplicate. Replacing rather than appending
    keeps "today's predictions" unambiguous.
    """
    race_ids = [race.race_id for race in day.races]
    if not race_ids:
        return 0

    existing = (
        session.query(Prediction)
        .filter(
            Prediction.race_id.in_(race_ids),
            Prediction.model_version == day.model_version,
        )
        .all()
    )
    for row in existing:
        session.delete(row)
    session.flush()

    off_times: dict[str, Any] = {
        str(race_id): off_time
        for race_id, off_time in session.execute(
            select(Race.race_id, Race.off_time).where(Race.race_id.in_(race_ids))
        ).all()
    }

    written = 0
    for race in day.races:
        for runner in race.runners:
            session.add(
                Prediction(
                    race_id=race.race_id,
                    horse_id=runner.horse_id,
                    race_date=race.race_date,
                    off_time=off_times.get(race.race_id),
                    course=race.course,
                    horse_name=runner.horse,
                    model_probability=runner.model_probability,
                    market_probability=runner.market_probability,
                    odds=runner.odds,
                    expected_value=runner.expected_value,
                    edge=runner.edge,
                    recommendation=runner.recommendation,
                    rejection_reason=runner.rejection_reason,
                    model_name=day.model_name,
                    model_version=day.model_version,
                    feature_version=FEATURE_VERSION,
                )
            )
            written += 1

    session.commit()
    logger.info(
        "predictions stored",
        extra=safe_extra({"race_date": str(day.race_date), "rows": written, "bets": len(day.bets)}),
    )
    return written


def load_predictions(session: Session, race_date: date, *, bets_only: bool = False) -> list[Prediction]:
    """Read back a stored day, newest model first."""
    statement = select(Prediction).where(Prediction.race_date == race_date)
    if bets_only:
        statement = statement.where(Prediction.recommendation == Recommendation.BET)
    statement = statement.order_by(Prediction.off_time, Prediction.model_probability.desc())
    return list(session.execute(statement).scalars().all())


def _text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _as_date(value: Any) -> date | None:
    stamp = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(stamp) else stamp.date()


__all__ = [
    "DayPredictions",
    "PredictionService",
    "RaceSignals",
    "RunnerSignal",
    "load_predictions",
    "store_predictions",
]
