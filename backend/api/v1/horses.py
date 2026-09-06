"""Horse lookup and form endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import func, select
from sqlalchemy.sql.elements import ColumnElement

from backend.api.deps import SessionDep
from backend.models import Horse, Race, RaceResult
from backend.schemas.racing import HorseFormRead, HorseRead, Page, ResultRead
from backend.utils.exceptions import NotFoundError

router = APIRouter(prefix="/horses", tags=["horses"])

#: A "place" is a top-3 finish. Strictly this varies with field size and each-way
#: terms; top-3 is the convention used for form display.
PLACE_THRESHOLD = 3


@router.get("", response_model=Page[HorseRead], summary="Search horses by name")
def search_horses(
    session: SessionDep,
    name: str | None = Query(None, min_length=2, description="Case-insensitive substring match"),
    region: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Page[HorseRead]:
    filters: list[ColumnElement[bool]] = []
    if name:
        filters.append(Horse.name.ilike(f"%{name}%"))
    if region:
        filters.append(Horse.region == region)

    total = session.execute(select(func.count()).select_from(Horse).where(*filters)).scalar_one()
    rows = (
        session.execute(select(Horse).where(*filters).order_by(Horse.name).limit(limit).offset(offset))
        .scalars()
        .all()
    )
    return Page[HorseRead](
        items=[HorseRead.model_validate(row) for row in rows], total=total, limit=limit, offset=offset
    )


@router.get(
    "/{horse_id}",
    response_model=HorseFormRead,
    summary="Horse profile with recent form",
    responses={404: {"description": "Horse not found"}},
)
def get_horse(
    horse_id: str,
    session: SessionDep,
    runs: int = Query(10, ge=1, le=100, description="How many recent runs to return"),
) -> HorseFormRead:
    """A horse plus its most recent runs, newest first.

    Career totals are computed over *all* stored runs, not just the returned
    window, so the strike rate does not change with the ``runs`` parameter.
    """
    horse = session.get(Horse, horse_id)
    if horse is None:
        raise NotFoundError(f"horse {horse_id!r} is not in the database", details={"horse_id": horse_id})

    recent = (
        session.execute(
            select(RaceResult)
            .join(Race, Race.race_id == RaceResult.race_id)
            .where(RaceResult.horse_id == horse_id)
            .order_by(Race.race_date.desc().nullslast())
            .limit(runs)
        )
        .scalars()
        .all()
    )

    total_runs = session.execute(
        select(func.count()).select_from(RaceResult).where(RaceResult.horse_id == horse_id)
    ).scalar_one()
    wins = session.execute(
        select(func.count())
        .select_from(RaceResult)
        .where(RaceResult.horse_id == horse_id, RaceResult.is_winner.is_(True))
    ).scalar_one()
    places = session.execute(
        select(func.count())
        .select_from(RaceResult)
        .where(
            RaceResult.horse_id == horse_id,
            RaceResult.finishing_position.isnot(None),
            RaceResult.finishing_position <= PLACE_THRESHOLD,
        )
    ).scalar_one()

    return HorseFormRead(
        horse=HorseRead.model_validate(horse),
        runs=[ResultRead.model_validate(result) for result in recent],
        total_runs=total_runs,
        wins=wins,
        places=places,
    )


__all__ = ["router"]
