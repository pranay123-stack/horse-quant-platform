"""The morning job: fetch today's card, score it, store the signals.

Runs before racing starts. Six steps, and it is deliberately safe to run more
than once — prices move through the morning, so a later run is a better answer
rather than a duplicate, and :func:`store_predictions` replaces rather than
appends.

The job also settles yesterday's predictions. Without that the performance page
would be a list of intentions; with it, the product scores itself against what
actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import RaceResult
from backend.models.predictions import Prediction
from backend.prediction_service.service import (
    DayPredictions,
    PredictionService,
    store_predictions,
)
from backend.utils.logging import get_logger, safe_extra
from backend.utils.timeutils import today_uk, utcnow

logger = get_logger(__name__, channel="pipeline")


@dataclass(slots=True)
class DailyJobResult:
    """What the morning run did."""

    race_date: date
    started_at: str = field(default_factory=lambda: utcnow().isoformat())
    finished_at: str | None = None
    races_imported: int = 0
    races_scored: int = 0
    runners_scored: int = 0
    bets_recommended: int = 0
    settled: int = 0
    errors: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def succeeded(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "race_date": str(self.race_date),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "races_imported": self.races_imported,
            "races_scored": self.races_scored,
            "runners_scored": self.runners_scored,
            "bets_recommended": self.bets_recommended,
            "settled": self.settled,
            "errors": self.errors,
            "note": self.note,
        }

    def render(self) -> str:
        lines = [
            f"Daily predictions — {self.race_date}",
            "-------------------------------",
            f"  Races imported     {self.races_imported}",
            f"  Races scored       {self.races_scored}",
            f"  Runners scored     {self.runners_scored}",
            f"  BET recommended    {self.bets_recommended}",
            f"  Yesterday settled  {self.settled}",
        ]
        if self.note:
            lines.append(f"  Note               {self.note}")
        for error in self.errors:
            lines.append(f"  ERROR              {error}")
        return "\n".join(lines)


def settle_predictions(session: Session, race_date: date) -> int:
    """Mark past predictions won/lost once results have arrived.

    A prediction nobody settles is an opinion. This is what turns the stored
    rows into a record the performance page can score.
    """
    pending = list(
        session.execute(
            select(Prediction).where(Prediction.race_date == race_date, Prediction.settled.is_(False))
        )
        .scalars()
        .all()
    )
    if not pending:
        return 0

    race_ids = {row.race_id for row in pending}
    winners = {
        (str(race_id), str(horse_id))
        for race_id, horse_id in session.execute(
            select(RaceResult.race_id, RaceResult.horse_id).where(
                RaceResult.race_id.in_(race_ids), RaceResult.is_winner.is_(True)
            )
        ).all()
    }
    settled_races = {
        str(race_id)
        for race_id in session.execute(
            select(RaceResult.race_id).where(RaceResult.race_id.in_(race_ids)).distinct()
        )
        .scalars()
        .all()
    }

    count = 0
    for row in pending:
        if row.race_id not in settled_races:
            continue  # the result has not arrived yet
        row.settled = True
        row.won = (row.race_id, row.horse_id) in winners
        count += 1

    session.commit()
    logger.info("predictions settled", extra={"race_date": str(race_date), "rows": count})
    return count


def run_daily_job(
    session: Session,
    *,
    race_date: date | None = None,
    service: PredictionService | None = None,
    import_racecards: Any | None = None,
    settle_previous: bool = True,
) -> tuple[DailyJobResult, DayPredictions]:
    """Fetch, score, store — and settle yesterday.

    ``import_racecards`` is injected because fetching is async and needs a live
    API client. Keeping it out means the rest of the job is testable, and means
    the job still works against already-imported data when the API is down.
    """
    target = race_date or today_uk()
    result = DailyJobResult(race_date=target)
    service = service or PredictionService(session)

    # --- 1 & 2. today's races and runners ----------------------------------
    if import_racecards is not None:
        try:
            result.races_imported = int(import_racecards())
        except Exception as exc:
            result.errors.append(f"racecard import failed: {exc}"[:300])
            logger.error("racecard import failed", extra={"error": str(exc)[:200]})

    # --- 3-6. features, model, probabilities, EV ---------------------------
    day = service.signals_for_date(target)
    result.races_scored = len(day.races)
    result.runners_scored = day.total_runners
    result.bets_recommended = len(day.bets)
    result.note = day.note

    # --- 7. store ----------------------------------------------------------
    if day.races:
        store_predictions(session, day)

    if settle_previous:
        try:
            result.settled = settle_predictions(session, target - timedelta(days=1))
        except Exception as exc:
            result.errors.append(f"settlement failed: {exc}"[:300])

    result.finished_at = utcnow().isoformat()
    logger.info("daily job complete", extra=safe_extra(result.as_dict()))
    return result, day


__all__ = ["DailyJobResult", "run_daily_job", "settle_predictions"]
