"""Race listing and detail endpoints."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Query
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from sqlalchemy.sql.elements import ColumnElement

from backend.api.deps import SessionDep
from backend.models import Race, RaceResult, RaceRunner
from backend.schemas.racing import (
    Page,
    RaceDetailRead,
    RaceSummaryRead,
    ResultRead,
    RunnerRead,
)
from backend.utils.exceptions import NotFoundError
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="api")

router = APIRouter(prefix="/races", tags=["races"])


def _runner_read(runner: RaceRunner) -> RunnerRead:
    return RunnerRead(
        horse_id=runner.horse_id,
        horse_name=runner.horse.name if runner.horse else None,
        jockey_id=runner.jockey_id,
        trainer_id=runner.trainer_id,
        saddle_cloth=runner.saddle_cloth,
        draw=runner.draw,
        age=runner.age,
        weight_lbs=runner.weight_lbs,
        headgear=runner.headgear,
        official_rating=runner.official_rating,
        rpr=runner.rpr,
        topspeed=runner.topspeed,
        days_since_last_run=runner.days_since_last_run,
        form=runner.form,
    )


def _result_read(result: RaceResult) -> ResultRead:
    return ResultRead(
        horse_id=result.horse_id,
        horse_name=result.horse.name if result.horse else None,
        jockey_id=result.jockey_id,
        trainer_id=result.trainer_id,
        finishing_position=result.finishing_position,
        finishing_status=result.finishing_status,
        is_winner=result.is_winner,
        starting_price=result.starting_price,
        beaten_lengths=result.beaten_lengths,
        official_rating=result.official_rating,
        rpr=result.rpr,
        topspeed=result.topspeed,
        prize_won=result.prize_won,
    )


@router.get("", response_model=Page[RaceSummaryRead], summary="List races")
def list_races(
    session: SessionDep,
    race_date: date | None = Query(None, description="Exact race date (YYYY-MM-DD)"),
    date_from: date | None = Query(None, description="Inclusive lower bound"),
    date_to: date | None = Query(None, description="Inclusive upper bound"),
    course_id: str | None = Query(None),
    region: str | None = Query(None, description="e.g. gb, ire"),
    race_class: int | None = Query(None, ge=1, le=7),
    has_result: bool | None = Query(None, description="Filter to run / not-yet-run races"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Page[RaceSummaryRead]:
    """Race listing, newest first, with the filters the dashboard needs."""
    filters: list[ColumnElement[bool]] = []
    if race_date is not None:
        filters.append(Race.race_date == race_date)
    if date_from is not None:
        filters.append(Race.race_date >= date_from)
    if date_to is not None:
        filters.append(Race.race_date <= date_to)
    if course_id is not None:
        filters.append(Race.course_id == course_id)
    if region is not None:
        filters.append(Race.region == region)
    if race_class is not None:
        filters.append(Race.race_class == race_class)
    if has_result is not None:
        filters.append(Race.has_result.is_(has_result))

    total = session.execute(select(func.count()).select_from(Race).where(*filters)).scalar_one()
    rows = (
        session.execute(
            select(Race)
            .where(*filters)
            .order_by(Race.race_date.desc().nullslast(), Race.off_time.desc().nullslast())
            .limit(limit)
            .offset(offset)
        )
        .scalars()
        .all()
    )

    return Page[RaceSummaryRead](
        items=[RaceSummaryRead.model_validate(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{race_id}",
    response_model=RaceDetailRead,
    summary="Race detail with card and result",
    responses={404: {"description": "Race not found"}},
)
def get_race(race_id: str, session: SessionDep) -> RaceDetailRead:
    race = session.execute(
        select(Race)
        .where(Race.race_id == race_id)
        .options(
            selectinload(Race.runners).selectinload(RaceRunner.horse),
            selectinload(Race.results).selectinload(RaceResult.horse),
        )
    ).scalar_one_or_none()

    if race is None:
        raise NotFoundError(f"race {race_id!r} is not in the database", details={"race_id": race_id})

    detail = RaceDetailRead.model_validate(race)
    detail.runners = [
        _runner_read(runner)
        for runner in sorted(race.runners, key=lambda r: (r.saddle_cloth is None, r.saddle_cloth or 0))
    ]
    # Non-finishers sort last: position None must not read as "won".
    detail.results = [
        _result_read(result)
        for result in sorted(
            race.results, key=lambda r: (r.finishing_position is None, r.finishing_position or 0)
        )
    ]
    return detail


__all__ = ["router"]
