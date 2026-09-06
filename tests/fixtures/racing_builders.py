"""Builders for hand-constructed racing scenarios.

Used where a test needs an exact, inspectable situation — "this horse ran three
times, finishing 1st, 2nd and 3rd" — rather than the statistical bulk the
synthetic generator produces.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

import pandas as pd
from sqlalchemy.orm import Session

from backend.features.base import EVENT_TIME, distance_band, going_band, going_value
from backend.models import Course, Horse, Jockey, OddsHistory, Race, RaceResult, RaceRunner, Trainer

DEFAULT_COURSE = "crs_test"
DEFAULT_JOCKEY = "jky_test"
DEFAULT_TRAINER = "trn_test"


def ensure_reference_rows(
    session: Session,
    *,
    course_id: str = DEFAULT_COURSE,
    jockey_id: str = DEFAULT_JOCKEY,
    trainer_id: str = DEFAULT_TRAINER,
) -> None:
    """Insert the parent rows a race's foreign keys require."""
    if session.get(Course, course_id) is None:
        session.add(Course(course_id=course_id, name="Test Park", region="Great Britain", region_code="gb"))
    if session.get(Jockey, jockey_id) is None:
        session.add(Jockey(jockey_id=jockey_id, name="Test Jockey"))
    if session.get(Trainer, trainer_id) is None:
        session.add(Trainer(trainer_id=trainer_id, name="Test Trainer"))
    session.flush()


def build_race_with_results(
    session: Session,
    *,
    race_id: str,
    off_time: datetime,
    runners: Sequence[tuple[str, int]],
    going: str = "Good",
    distance_yards: int = 2200,
    race_class: int = 4,
    course_id: str = DEFAULT_COURSE,
    jockey_ids: dict[str, str] | None = None,
    trainer_ids: dict[str, str] | None = None,
    odds: dict[str, list[tuple[datetime, float]]] | None = None,
    starting_prices: dict[str, float] | None = None,
) -> Race:
    """Create one finished race with declarations, results and optional odds.

    ``runners`` is ``[(horse_id, finishing_position), ...]``.
    """
    ensure_reference_rows(session, course_id=course_id)

    for horse_id, _ in runners:
        if session.get(Horse, horse_id) is None:
            session.add(Horse(horse_id=horse_id, name=horse_id.replace("_", " ").title(), region="GB"))
    for mapping, model, prefix in (
        (jockey_ids or {}, Jockey, "jockey"),
        (trainer_ids or {}, Trainer, "trainer"),
    ):
        for value in set(mapping.values()):
            if session.get(model, value) is None:
                session.add(model(**{f"{prefix}_id": value, "name": value}))
    session.flush()

    race = Race(
        race_id=race_id,
        race_date=off_time.date(),
        off_time=off_time,
        course_id=course_id,
        course_name="Test Park",
        race_name=f"Test Race {race_id}",
        race_class=race_class,
        race_type="Flat",
        distance_yards=distance_yards,
        distance_furlongs=round(distance_yards / 220, 2),
        going=going,
        region="GB",
        prize_money=Decimal("5000"),
        field_size=len(runners),
        has_result=True,
    )
    session.add(race)

    for slot, (horse_id, position) in enumerate(runners, start=1):
        jockey_id = (jockey_ids or {}).get(horse_id, DEFAULT_JOCKEY)
        trainer_id = (trainer_ids or {}).get(horse_id, DEFAULT_TRAINER)
        ensure_reference_rows(session, jockey_id=jockey_id, trainer_id=trainer_id)

        session.add(
            RaceRunner(
                race_id=race_id,
                horse_id=horse_id,
                jockey_id=jockey_id,
                trainer_id=trainer_id,
                saddle_cloth=slot,
                draw=slot,
                age=5,
                weight_lbs=130,
                official_rating=80,
                rpr=85,
                topspeed=80,
            )
        )
        session.add(
            RaceResult(
                race_id=race_id,
                horse_id=horse_id,
                jockey_id=jockey_id,
                trainer_id=trainer_id,
                finishing_position=position,
                finishing_status="finished",
                is_winner=position == 1,
                saddle_cloth=slot,
                draw=slot,
                age=5,
                weight_lbs=130,
                official_rating=80,
                rpr=85,
                topspeed=80,
                starting_price=Decimal(str((starting_prices or {}).get(horse_id, 4.0))),
            )
        )

    for horse_id, quotes in (odds or {}).items():
        for recorded_at, price in quotes:
            session.add(
                OddsHistory(
                    race_id=race_id,
                    horse_id=horse_id,
                    bookmaker="TestBook",
                    decimal_odds=Decimal(str(price)),
                    implied_probability=1.0 / price,
                    recorded_at=recorded_at,
                )
            )
    session.flush()
    return race


def build_history_frame(
    runs: Sequence[tuple[str, str, int]],
    *,
    going: str = "Good",
    distance_yards: int = 2200,
    race_class: int = 4,
    starting_price: float = 5.0,
) -> pd.DataFrame:
    """An in-memory history frame: ``[(horse_id, iso_date, position), ...]``.

    Lets the feature builders be tested without touching a database.
    """
    rows = []
    for index, (horse_id, iso_date, position) in enumerate(runs):
        rows.append(
            {
                "race_id": f"rac_h{index:04d}",
                "horse_id": horse_id,
                "jockey_id": DEFAULT_JOCKEY,
                "trainer_id": DEFAULT_TRAINER,
                "course_id": DEFAULT_COURSE,
                "finishing_position": position,
                "finishing_status": "finished",
                "is_winner": position == 1,
                "starting_price": starting_price,
                "rpr": 85,
                "topspeed": 80,
                "official_rating": 80,
                "weight_lbs": 130,
                "beaten_lengths": float(max(0, position - 1)),
                "prize_won": None,
                "race_date": pd.Timestamp(iso_date).date(),
                "off_time": pd.Timestamp(iso_date, tz="UTC"),
                "race_class": race_class,
                "race_type": "Flat",
                "distance_yards": distance_yards,
                "going": going,
                "region": "GB",
                EVENT_TIME: pd.Timestamp(iso_date, tz="UTC"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["is_placed"] = frame["finishing_position"].le(3)
    frame["going_band"] = frame["going"].map(going_band)
    frame["going_value"] = frame["going"].map(going_value)
    frame["distance_band"] = frame["distance_yards"].map(distance_band)
    return frame.sort_values(EVENT_TIME, kind="stable").reset_index(drop=True)


def build_target_frame(targets: Sequence[tuple[str, str]], **kwargs: object) -> pd.DataFrame:
    """A minimal target frame: ``[(horse_id, iso_date), ...]``."""
    rows = []
    for index, (horse_id, iso_date) in enumerate(targets):
        rows.append(
            {
                "race_id": kwargs.get("race_id", f"rac_t{index:04d}"),
                "horse_id": horse_id,
                "jockey_id": DEFAULT_JOCKEY,
                "trainer_id": DEFAULT_TRAINER,
                "course_id": DEFAULT_COURSE,
                "race_date": pd.Timestamp(iso_date).date(),
                "off_time": pd.Timestamp(iso_date, tz="UTC"),
                "race_class": kwargs.get("race_class", 4),
                "race_type": "Flat",
                "distance_yards": kwargs.get("distance_yards", 2200),
                "going": kwargs.get("going", "Good"),
                "region": "GB",
                "age": 5,
                "weight_lbs": 130,
                "official_rating": 80,
                "rpr": 85,
                "topspeed": 80,
                "draw": index + 1,
                "field_size": len(targets),
                EVENT_TIME: pd.Timestamp(iso_date, tz="UTC"),
            }
        )
    frame = pd.DataFrame(rows)
    frame["going_band"] = frame["going"].map(going_band)
    frame["going_value"] = frame["going"].map(going_value)
    frame["distance_band"] = frame["distance_yards"].map(distance_band)
    return frame.sort_values(EVENT_TIME, kind="stable").reset_index(drop=True)


__all__ = [
    "DEFAULT_COURSE",
    "DEFAULT_JOCKEY",
    "DEFAULT_TRAINER",
    "build_history_frame",
    "build_race_with_results",
    "build_target_frame",
    "ensure_reference_rows",
]
