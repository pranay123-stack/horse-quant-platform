"""Persisted feature vectors.

Why store features at all, when they can be recomputed on demand?

**Reproducibility.** A backtest run six months from now must see the features the
model actually saw, not what today's code would produce. Feature logic evolves;
the stored vector is the audit trail that makes a historical result checkable.

**Latency.** Scoring a live racecard cannot wait for a full-history pandas build.

The design is deliberately hybrid: the composite scores get real columns, because
they are stable, queryable and shown on the dashboard; the full evolving feature
vector goes in a JSON column, because adding a feature must not require a
migration. ``feature_version`` records which code produced the row.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import JSON, Date, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base, TimestampMixin
from backend.models.entities import ID_LENGTH

#: JSONB on PostgreSQL (indexable, binary); plain JSON elsewhere so the test
#: suite can run the same models on SQLite.
JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")

#: Bumped whenever feature semantics change. Rows from different versions must
#: never be mixed in one training set.
FEATURE_VERSION = "v1"


class RaceFeatures(Base, TimestampMixin):
    """One point-in-time feature vector per runner per race."""

    __tablename__ = "race_features"
    __table_args__ = (
        UniqueConstraint("race_id", "horse_id", "feature_version", name="uq_race_features_runner_version"),
        Index("ix_race_features_race", "race_id"),
        Index("ix_race_features_horse", "horse_id"),
        Index("ix_race_features_date", "race_date"),
        Index("ix_race_features_version_date", "feature_version", "race_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    race_id: Mapped[str] = mapped_column(
        String(ID_LENGTH), ForeignKey("races.race_id", ondelete="CASCADE"), nullable=False
    )
    horse_id: Mapped[str] = mapped_column(
        String(ID_LENGTH), ForeignKey("horses.horse_id", ondelete="CASCADE"), nullable=False
    )
    #: Denormalised from ``races`` so time-based splits never need a join.
    race_date: Mapped[date | None] = mapped_column(Date)
    feature_version: Mapped[str] = mapped_column(
        String(20), nullable=False, default=FEATURE_VERSION, server_default=FEATURE_VERSION
    )

    # --- composite scores (Phase 3 specification) ------------------------
    horse_form_score: Mapped[float | None] = mapped_column(Float)
    speed_score: Mapped[float | None] = mapped_column(Float)
    jockey_score: Mapped[float | None] = mapped_column(Float)
    trainer_score: Mapped[float | None] = mapped_column(Float)
    market_score: Mapped[float | None] = mapped_column(Float)
    distance_score: Mapped[float | None] = mapped_column(Float)
    going_score: Mapped[float | None] = mapped_column(Float)
    class_score: Mapped[float | None] = mapped_column(Float)

    #: Handy for filtering training sets to runners with real form behind them.
    horse_runs_before: Mapped[int | None] = mapped_column(Integer)

    #: The full feature vector as produced by the pipeline.
    features: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)


__all__ = ["FEATURE_VERSION", "JSON_TYPE", "RaceFeatures"]
