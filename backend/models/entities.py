"""Reference entities: courses, horses, jockeys, trainers.

All primary keys are the **natural identifiers issued by The Racing API**
(``crs_123``, ``hrs_456``, …) rather than surrogate integers. Two reasons:

1. Ingestion becomes idempotent for free — re-importing a race day is an upsert
   on a key we already know, with no lookup round trip.
2. Every row is traceable back to its upstream record during debugging.

The trade-off is a wider key than an ``int``; at UK racing's data volume
(~10k races and ~100k runners a year) that is irrelevant.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Date, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.models.racing import Race, RaceResult, RaceRunner

#: Racing API identifiers are short opaque strings; 40 leaves generous headroom.
ID_LENGTH = 40


class Course(Base, TimestampMixin):
    """A racecourse."""

    __tablename__ = "courses"

    course_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(120), index=True)
    region: Mapped[str | None] = mapped_column(String(60))
    region_code: Mapped[str | None] = mapped_column(String(10), index=True)

    races: Mapped[list[Race]] = relationship(back_populates="course", lazy="selectin")


class Horse(Base, TimestampMixin):
    """A racehorse and its breeding."""

    __tablename__ = "horses"
    __table_args__ = (Index("ix_horses_name_lower", "name"),)

    horse_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(120), index=True)
    sex: Mapped[str | None] = mapped_column(String(30))
    sex_code: Mapped[str | None] = mapped_column(String(10))
    colour: Mapped[str | None] = mapped_column(String(40))
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    region: Mapped[str | None] = mapped_column(String(20), index=True)

    sire: Mapped[str | None] = mapped_column(String(120))
    sire_id: Mapped[str | None] = mapped_column(String(ID_LENGTH), index=True)
    dam: Mapped[str | None] = mapped_column(String(120))
    dam_id: Mapped[str | None] = mapped_column(String(ID_LENGTH), index=True)
    damsire: Mapped[str | None] = mapped_column(String(120))
    damsire_id: Mapped[str | None] = mapped_column(String(ID_LENGTH), index=True)

    runners: Mapped[list[RaceRunner]] = relationship(back_populates="horse")
    results: Mapped[list[RaceResult]] = relationship(back_populates="horse")


class Jockey(Base, TimestampMixin):
    """A jockey.

    ``win_rate`` and ``rides`` are rolling aggregates maintained by the feature
    pipeline in Phase 5, not values the API supplies directly. They are stored
    here so that inference does not have to recompute them per request.
    """

    __tablename__ = "jockeys"

    jockey_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(120), index=True)
    rides: Mapped[int | None] = mapped_column()
    wins: Mapped[int | None] = mapped_column()
    win_rate: Mapped[float | None] = mapped_column()

    results: Mapped[list[RaceResult]] = relationship(back_populates="jockey")


class Trainer(Base, TimestampMixin):
    """A trainer, with a rolling strike rate maintained by the feature pipeline."""

    __tablename__ = "trainers"

    trainer_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(120), index=True)
    location: Mapped[str | None] = mapped_column(String(120))
    runners_count: Mapped[int | None] = mapped_column()
    wins: Mapped[int | None] = mapped_column()
    strike_rate: Mapped[float | None] = mapped_column()

    results: Mapped[list[RaceResult]] = relationship(back_populates="trainer")


__all__ = ["ID_LENGTH", "Course", "Horse", "Jockey", "Trainer"]
