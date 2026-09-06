"""The product's endpoints: today's signals, model status, live performance.

    GET  /predictions/today          today's races and recommendations
    GET  /predictions/{race_id}      one race
    POST /predict/today              score now rather than reading stored rows
    GET  /model/status               which model is live, and on what
    GET  /performance                the settled record

``GET /predictions/today`` reads what the morning job stored; ``POST
/predict/today`` re-scores against current prices. The distinction matters:
prices move, so a stored recommendation is what was true at 08:00, and a caller
who needs the live position should ask for it explicitly rather than get it by
accident from a GET.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Query

from backend.api.deps import SessionDep
from backend.ml.models.registry import ModelRegistry
from backend.models.features import FEATURE_VERSION
from backend.prediction_service import PredictionService, load_predictions, store_predictions
from backend.prediction_service.performance import summarise_performance
from backend.prediction_service.rules import RULES
from backend.utils.exceptions import NotFoundError
from backend.utils.logging import get_logger
from backend.utils.timeutils import today_uk

logger = get_logger(__name__, channel="api")

router = APIRouter(tags=["predictions"])


def _stored_day(session: Any, race_date: date, *, bets_only: bool) -> dict[str, Any]:
    """Render stored rows into the same shape the live scorer produces."""
    rows = load_predictions(session, race_date, bets_only=bets_only)
    races: dict[str, dict[str, Any]] = {}
    for row in rows:
        race = races.setdefault(
            row.race_id,
            {
                "race": f"{row.course} {row.off_time:%H:%M}" if row.course and row.off_time else row.race_id,
                "race_id": row.race_id,
                "course": row.course,
                "race_date": str(row.race_date) if row.race_date else None,
                "off_time": row.off_time.strftime("%H:%M") if row.off_time else None,
                "horses": [],
            },
        )
        race["horses"].append(row.as_dict())

    bets = sum(1 for row in rows if row.is_bet)
    return {
        "date": str(race_date),
        "source": "stored",
        "races": len(races),
        "runners": len(rows),
        "bets": bets,
        "model": f"{rows[0].model_name}:{rows[0].model_version}" if rows else None,
        "predictions": list(races.values()),
    }


@router.get("/predictions/today", summary="Today's stored predictions")
def predictions_today(
    session: SessionDep,
    race_date: date | None = Query(None, description="Defaults to today (UK)"),
    bets_only: bool = Query(False, description="Only runners recommended as BET"),
) -> dict[str, Any]:
    """What the morning job stored for today.

    Returns an empty day rather than a 404 when nothing has been scored — "no
    races today" is a normal answer, not an error.
    """
    target = race_date or today_uk()
    payload = _stored_day(session, target, bets_only=bets_only)
    if payload["runners"] == 0:
        payload["note"] = (
            f"no stored predictions for {target} — run the daily job, or use POST /predict/today to score now"
        )
    return payload


@router.get("/predictions/{race_id}", summary="Stored predictions for one race")
def predictions_for_race(session: SessionDep, race_id: str) -> dict[str, Any]:
    from sqlalchemy import select

    from backend.models.predictions import Prediction

    rows = list(
        session.execute(
            select(Prediction)
            .where(Prediction.race_id == race_id)
            .order_by(Prediction.model_probability.desc())
        )
        .scalars()
        .all()
    )
    if not rows:
        raise NotFoundError(f"no predictions stored for race {race_id!r}", details={"race_id": race_id})

    head = rows[0]
    return {
        "race": f"{head.course} {head.off_time:%H:%M}" if head.course and head.off_time else race_id,
        "race_id": race_id,
        "course": head.course,
        "race_date": str(head.race_date) if head.race_date else None,
        "off_time": head.off_time.strftime("%H:%M") if head.off_time else None,
        "model": f"{head.model_name}:{head.model_version}",
        "horses": [row.as_dict() for row in rows],
    }


@router.post("/predict/today", summary="Score today's races now")
def predict_today(
    session: SessionDep,
    race_date: date | None = Query(None, description="Defaults to today (UK)"),
    store: bool = Query(False, description="Persist the result as well as returning it"),
) -> dict[str, Any]:
    """Run the pipeline against current prices.

    Slower than the GET because it rebuilds features and re-scores, but it is
    the honest answer when the price has moved since this morning.
    """
    service = PredictionService(session)
    day = service.signals_for_date(race_date)
    if store and day.races:
        store_predictions(session, day)

    payload = day.as_dict()
    payload["source"] = "live"
    return payload


@router.get("/model/status", summary="Which model is live")
def model_status(session: SessionDep) -> dict[str, Any]:
    registry = ModelRegistry()
    champion = registry.champion()

    if champion is None:
        return {
            "status": "no model promoted",
            "ready": False,
            "feature_version": FEATURE_VERSION,
            "rules": RULES.as_dict(),
            "note": "train and promote a model before predictions can be produced",
        }

    return {
        "status": "ready",
        "ready": True,
        "model": champion.name,
        "version": champion.version,
        "trained_at": getattr(champion, "trained_at", None) or getattr(champion, "created_at", None),
        "feature_version": FEATURE_VERSION,
        "metrics": champion.headline(),
        "rules": RULES.as_dict(),
        "rules_description": RULES.describe(),
    }


@router.get("/performance", summary="The settled betting record")
def performance(
    session: SessionDep,
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
) -> dict[str, Any]:
    """How the recommendations actually did.

    Only settled bets count. An unsettled prediction is not a pending win, and
    including it would flatter every number here during a losing week.
    """
    summary = summarise_performance(session, date_from=date_from, date_to=date_to)
    payload = summary.as_dict()
    payload["equity"] = summary.equity[-500:]
    return payload


__all__ = ["router"]
