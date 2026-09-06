"""Market/odds endpoints.

The returned ``overround`` is the sum of implied probabilities at the best
available price for each runner. It is the bookmaker's margin made explicit:

* ``> 1.0`` — the normal case; the book is balanced in the layer's favour.
* ``≈ 1.0`` — a very competitive market.
* ``< 1.0`` — every runner can be backed at a profit (arbitrage). Across a
  single bookmaker this effectively never happens; across the *best* prices from
  many books it occasionally does, and is worth flagging.

Phase 7 divides each runner's implied probability by the overround to recover a
normalised market probability, which is the benchmark the model must beat.
"""

from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, Query
from sqlalchemy import select
from sqlalchemy.sql.elements import ColumnElement

from backend.api.deps import SessionDep
from backend.models import Horse, OddsHistory, Race
from backend.schemas.racing import OddsRead, RaceOddsRead, RunnerOddsRead
from backend.utils.exceptions import NotFoundError
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="api")

router = APIRouter(prefix="/odds", tags=["odds"])

#: Safety valve: a race with a long polling history could hold many thousands of
#: quotes. We only need the newest per (runner, bookmaker), so cap the scan.
MAX_QUOTES_SCANNED = 5000


@router.get(
    "/{race_id}",
    response_model=RaceOddsRead,
    summary="Current market for a race",
    responses={404: {"description": "Race not found"}},
)
def get_race_odds(
    race_id: str,
    session: SessionDep,
    bookmaker: str | None = Query(None, description="Restrict to a single bookmaker"),
) -> RaceOddsRead:
    """Latest price per runner per bookmaker, plus the best available and overround."""
    race = session.get(Race, race_id)
    if race is None:
        raise NotFoundError(f"race {race_id!r} is not in the database", details={"race_id": race_id})

    filters: list[ColumnElement[bool]] = [OddsHistory.race_id == race_id]
    if bookmaker:
        filters.append(OddsHistory.bookmaker == bookmaker)

    quotes = (
        session.execute(
            select(OddsHistory)
            .where(*filters)
            # Newest first so the first row seen per (horse, bookmaker) wins.
            .order_by(OddsHistory.recorded_at.desc().nullslast(), OddsHistory.id.desc())
            .limit(MAX_QUOTES_SCANNED)
        )
        .scalars()
        .all()
    )

    latest: dict[tuple[str, str], OddsHistory] = {}
    for quote in quotes:
        latest.setdefault((quote.horse_id, quote.bookmaker), quote)

    by_horse: dict[str, list[OddsHistory]] = {}
    for (horse_id, _), quote in latest.items():
        by_horse.setdefault(horse_id, []).append(quote)

    names: dict[str, str | None] = {}
    for horse_id, horse_name in session.execute(
        select(Horse.horse_id, Horse.name).where(Horse.horse_id.in_(by_horse))
    ).all():
        names[horse_id] = horse_name

    runners: list[RunnerOddsRead] = []
    overround = 0.0
    for horse_id, horse_quotes in by_horse.items():
        priced = [q for q in horse_quotes if q.decimal_odds is not None]
        # ``priced`` is already filtered to non-None, but mypy cannot see that.
        best = max(priced, key=lambda q: q.decimal_odds or Decimal(0)) if priced else None
        implied = (1.0 / float(best.decimal_odds)) if best and best.decimal_odds else None
        if implied is not None:
            overround += implied

        runners.append(
            RunnerOddsRead(
                horse_id=horse_id,
                horse_name=names.get(horse_id),
                best_decimal_odds=best.decimal_odds if best else None,
                best_bookmaker=best.bookmaker if best else None,
                implied_probability=implied,
                quotes=sorted(
                    (OddsRead.model_validate(q) for q in horse_quotes),
                    key=lambda o: (o.decimal_odds is None, -(o.decimal_odds or 0)),
                ),
            )
        )

    runners.sort(key=lambda r: (r.best_decimal_odds is None, r.best_decimal_odds or 0))

    if overround > 0 and overround < 1.0:
        logger.info(
            "book overround below 1.0 -- possible arbitrage or stale prices",
            extra={"race_id": race_id, "overround": round(overround, 4)},
        )

    return RaceOddsRead(
        race_id=race_id,
        race_name=race.race_name,
        off_time=race.off_time,
        runners=runners,
        overround=round(overround, 6) if overround > 0 else None,
        quote_count=len(latest),
    )


@router.get(
    "/{race_id}/{horse_id}",
    response_model=list[OddsRead],
    summary="Full price history for one runner",
)
def get_runner_odds_history(
    race_id: str,
    horse_id: str,
    session: SessionDep,
    limit: int = Query(200, ge=1, le=2000),
) -> list[OddsRead]:
    """Every stored quote for a runner, newest first — the basis of drift features."""
    quotes = (
        session.execute(
            select(OddsHistory)
            .where(OddsHistory.race_id == race_id, OddsHistory.horse_id == horse_id)
            .order_by(OddsHistory.recorded_at.desc().nullslast(), OddsHistory.id.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return [OddsRead.model_validate(quote) for quote in quotes]


__all__ = ["router"]
