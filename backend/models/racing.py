"""Races, runners, results and odds.

Three tables carry the time-series core of the platform:

``race_runners``  — the *declaration*: who was entered, at what weight, with what
                    rating. Known **before** the off, so it is safe feature input.
``race_results``  — the *outcome*: finishing position, starting price, prize.
                    Known **after** the off; it is the label, never a feature.
``odds_history``  — bookmaker prices over time, the market's own probability.

Keeping declarations and results in separate tables is not tidiness — it is the
structural defence against look-ahead bias. A feature query that joins only
``race_runners`` cannot accidentally read a finishing position.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base, TimestampMixin
from backend.models.entities import ID_LENGTH

if TYPE_CHECKING:
    from backend.models.entities import Course, Horse, Jockey, Trainer

#: Money: up to 99,999,999.99 — comfortably above any UK prize fund.
MONEY = Numeric(12, 2)
#: Decimal odds: 1.01 to 1000.00, two decimal places, exact (never float).
ODDS = Numeric(10, 3)


class Race(Base, TimestampMixin):
    """A race, declared or finished."""

    __tablename__ = "races"
    __table_args__ = (
        Index("ix_races_date_course", "race_date", "course_id"),
        Index("ix_races_date_class", "race_date", "race_class"),
    )

    race_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    race_date: Mapped[date | None] = mapped_column(Date, index=True)
    off_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    course_id: Mapped[str | None] = mapped_column(
        String(ID_LENGTH), ForeignKey("courses.course_id", ondelete="SET NULL"), index=True
    )
    course_name: Mapped[str | None] = mapped_column(String(120))

    race_name: Mapped[str | None] = mapped_column(String(255))
    race_class: Mapped[int | None] = mapped_column(Integer, index=True)
    race_type: Mapped[str | None] = mapped_column(String(40), index=True)
    pattern: Mapped[str | None] = mapped_column(String(60))
    age_band: Mapped[str | None] = mapped_column(String(40))
    rating_band: Mapped[str | None] = mapped_column(String(40))
    sex_restriction: Mapped[str | None] = mapped_column(String(40))

    distance_yards: Mapped[int | None] = mapped_column(Integer, index=True)
    distance_furlongs: Mapped[float | None] = mapped_column()

    going: Mapped[str | None] = mapped_column(String(60), index=True)
    going_detailed: Mapped[str | None] = mapped_column(String(255))
    surface: Mapped[str | None] = mapped_column(String(40))
    jumps: Mapped[str | None] = mapped_column(String(40))
    weather: Mapped[str | None] = mapped_column(String(120))
    region: Mapped[str | None] = mapped_column(String(20), index=True)

    prize_money: Mapped[Decimal | None] = mapped_column(MONEY)
    field_size: Mapped[int | None] = mapped_column(Integer)
    winning_time_detail: Mapped[str | None] = mapped_column(String(120))

    is_abandoned: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    #: True once results have been ingested — drives incremental backfill.
    has_result: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", index=True)

    course: Mapped[Course | None] = relationship(back_populates="races")
    runners: Mapped[list[RaceRunner]] = relationship(
        back_populates="race", cascade="all, delete-orphan", passive_deletes=True
    )
    results: Mapped[list[RaceResult]] = relationship(
        back_populates="race", cascade="all, delete-orphan", passive_deletes=True
    )
    odds: Mapped[list[OddsHistory]] = relationship(
        back_populates="race", cascade="all, delete-orphan", passive_deletes=True
    )


class RaceRunner(Base, TimestampMixin):
    """A declared runner — everything knowable *before* the race is run."""

    __tablename__ = "race_runners"
    __table_args__ = (
        UniqueConstraint("race_id", "horse_id", name="uq_race_runners_race_horse"),
        Index("ix_race_runners_horse_race", "horse_id", "race_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(
        String(ID_LENGTH), ForeignKey("races.race_id", ondelete="CASCADE"), nullable=False, index=True
    )
    horse_id: Mapped[str] = mapped_column(
        String(ID_LENGTH), ForeignKey("horses.horse_id", ondelete="CASCADE"), nullable=False
    )
    jockey_id: Mapped[str | None] = mapped_column(
        String(ID_LENGTH), ForeignKey("jockeys.jockey_id", ondelete="SET NULL"), index=True
    )
    trainer_id: Mapped[str | None] = mapped_column(
        String(ID_LENGTH), ForeignKey("trainers.trainer_id", ondelete="SET NULL"), index=True
    )
    owner: Mapped[str | None] = mapped_column(String(160))
    owner_id: Mapped[str | None] = mapped_column(String(ID_LENGTH))

    saddle_cloth: Mapped[int | None] = mapped_column(Integer)
    draw: Mapped[int | None] = mapped_column(Integer)
    age: Mapped[int | None] = mapped_column(Integer)
    weight_lbs: Mapped[int | None] = mapped_column(Integer)
    headgear: Mapped[str | None] = mapped_column(String(40))

    official_rating: Mapped[int | None] = mapped_column(Integer)
    rpr: Mapped[int | None] = mapped_column(Integer)
    topspeed: Mapped[int | None] = mapped_column(Integer)
    days_since_last_run: Mapped[int | None] = mapped_column(Integer)
    form: Mapped[str | None] = mapped_column(String(60))
    comment: Mapped[str | None] = mapped_column(Text)

    race: Mapped[Race] = relationship(back_populates="runners")
    horse: Mapped[Horse] = relationship(back_populates="runners")


class RaceResult(Base, TimestampMixin):
    """A finished runner — the supervised-learning label.

    ``finishing_position`` is ``NULL`` for non-finishers; ``finishing_status``
    records *why* (``pulled_up``, ``fell``, ``unseated_rider``…). Treating a
    faller as "did not win" is correct; treating it as "finished last" is not,
    and that distinction is why the two columns are separate.
    """

    __tablename__ = "race_results"
    __table_args__ = (
        UniqueConstraint("race_id", "horse_id", name="uq_race_results_race_horse"),
        Index("ix_race_results_horse_position", "horse_id", "finishing_position"),
        Index("ix_race_results_jockey", "jockey_id", "finishing_position"),
        Index("ix_race_results_trainer", "trainer_id", "finishing_position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(
        String(ID_LENGTH), ForeignKey("races.race_id", ondelete="CASCADE"), nullable=False, index=True
    )
    horse_id: Mapped[str] = mapped_column(
        String(ID_LENGTH), ForeignKey("horses.horse_id", ondelete="CASCADE"), nullable=False
    )
    jockey_id: Mapped[str | None] = mapped_column(
        String(ID_LENGTH), ForeignKey("jockeys.jockey_id", ondelete="SET NULL")
    )
    trainer_id: Mapped[str | None] = mapped_column(
        String(ID_LENGTH), ForeignKey("trainers.trainer_id", ondelete="SET NULL")
    )
    owner: Mapped[str | None] = mapped_column(String(160))
    owner_id: Mapped[str | None] = mapped_column(String(ID_LENGTH))

    finishing_position: Mapped[int | None] = mapped_column(Integer, index=True)
    finishing_status: Mapped[str] = mapped_column(String(30), default="unknown", server_default="unknown")
    #: Denormalised winner flag — the classification target, and by far the most
    #: frequent filter in feature queries.
    is_winner: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False, index=True
    )

    saddle_cloth: Mapped[int | None] = mapped_column(Integer)
    draw: Mapped[int | None] = mapped_column(Integer)
    age: Mapped[int | None] = mapped_column(Integer)
    weight_lbs: Mapped[int | None] = mapped_column(Integer)
    headgear: Mapped[str | None] = mapped_column(String(40))

    starting_price: Mapped[Decimal | None] = mapped_column(ODDS)
    beaten_lengths: Mapped[float | None] = mapped_column()
    overall_beaten_lengths: Mapped[float | None] = mapped_column()
    official_rating: Mapped[int | None] = mapped_column(Integer)
    rpr: Mapped[int | None] = mapped_column(Integer)
    topspeed: Mapped[int | None] = mapped_column(Integer)
    finishing_time: Mapped[str | None] = mapped_column(String(40))
    prize_won: Mapped[Decimal | None] = mapped_column(MONEY)
    comment: Mapped[str | None] = mapped_column(Text)

    race: Mapped[Race] = relationship(back_populates="results")
    horse: Mapped[Horse] = relationship(back_populates="results")
    jockey: Mapped[Jockey | None] = relationship(back_populates="results")
    trainer: Mapped[Trainer | None] = relationship(back_populates="results")


class OddsHistory(Base, TimestampMixin):
    """A bookmaker's price for a runner at a point in time.

    The unique constraint on ``(race_id, horse_id, bookmaker, recorded_at)`` is
    what makes odds ingestion idempotent: polling the same race every five
    minutes inserts a new row only when the price actually moved, because
    unchanged quotes carry the same upstream ``updated`` timestamp.
    """

    __tablename__ = "odds_history"
    __table_args__ = (
        UniqueConstraint("race_id", "horse_id", "bookmaker", "recorded_at", name="uq_odds_history_quote"),
        Index("ix_odds_history_race_horse_time", "race_id", "horse_id", "recorded_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(
        String(ID_LENGTH), ForeignKey("races.race_id", ondelete="CASCADE"), nullable=False, index=True
    )
    horse_id: Mapped[str] = mapped_column(
        String(ID_LENGTH), ForeignKey("horses.horse_id", ondelete="CASCADE"), nullable=False
    )
    bookmaker: Mapped[str] = mapped_column(String(60), nullable=False, index=True)

    decimal_odds: Mapped[Decimal | None] = mapped_column(ODDS)
    fractional_odds: Mapped[str | None] = mapped_column(String(20))
    implied_probability: Mapped[float | None] = mapped_column()
    each_way_places: Mapped[int | None] = mapped_column(Integer)
    each_way_denominator: Mapped[int | None] = mapped_column(Integer)
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    race: Mapped[Race] = relationship(back_populates="odds")


__all__ = ["MONEY", "ODDS", "OddsHistory", "Race", "RaceResult", "RaceRunner"]
