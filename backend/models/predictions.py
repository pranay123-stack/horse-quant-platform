"""Stored live predictions — the product's output.

One row per runner per race, written by the daily job and read by the API and
the dashboard. It records the recommendation *and the inputs that produced it*:
the probability, the price it was compared against, and the model version.

Storing only "BET" would make this a list of opinions. Storing the reasoning
makes it a record that can be settled and scored later — which is the only way
the performance page means anything.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base, TimestampMixin
from backend.models.entities import ID_LENGTH
from backend.models.racing import ODDS


class Recommendation(StrEnum):
    """What the production rules say to do about this runner."""

    BET = "BET"
    NO_BET = "NO_BET"


class Prediction(Base, TimestampMixin):
    """One runner, scored for an upcoming race."""

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_race_date", "race_date"),
        Index("ix_predictions_race", "race_id"),
        Index("uq_predictions_runner", "race_id", "horse_id", "model_version", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    race_id: Mapped[str] = mapped_column(
        String(ID_LENGTH), ForeignKey("races.race_id", ondelete="CASCADE"), nullable=False
    )
    horse_id: Mapped[str] = mapped_column(
        String(ID_LENGTH), ForeignKey("horses.horse_id", ondelete="CASCADE"), nullable=False
    )
    race_date: Mapped[date | None] = mapped_column(Date)
    off_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    course: Mapped[str | None] = mapped_column(String(120))
    horse_name: Mapped[str | None] = mapped_column(String(160))

    #: Calibrated, normalised so the race sums to one.
    model_probability: Mapped[float] = mapped_column(Float, nullable=False)
    #: Margin-free probability implied by the consensus price.
    market_probability: Mapped[float | None] = mapped_column(Float)
    #: Best price available at scoring time — what a stake would be struck at.
    odds: Mapped[float | None] = mapped_column(ODDS)
    expected_value: Mapped[float | None] = mapped_column(Float)
    edge: Mapped[float | None] = mapped_column(Float)

    recommendation: Mapped[str] = mapped_column(String(10), nullable=False, default=Recommendation.NO_BET)
    #: Why a runner was not backed. Empty for a BET.
    rejection_reason: Mapped[str | None] = mapped_column(String(60))

    model_name: Mapped[str] = mapped_column(String(60), nullable=False)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    feature_version: Mapped[str | None] = mapped_column(String(40))

    #: Filled in once the race has run, so the product can score itself.
    settled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    won: Mapped[bool | None] = mapped_column(Boolean)

    @property
    def is_bet(self) -> bool:
        return self.recommendation == Recommendation.BET

    def as_dict(self) -> dict[str, Any]:
        return {
            "race_id": self.race_id,
            "horse_id": self.horse_id,
            "horse": self.horse_name,
            "course": self.course,
            "race_date": str(self.race_date) if self.race_date else None,
            "off_time": self.off_time.isoformat() if self.off_time else None,
            "probability": round(self.model_probability, 4),
            "market_probability": round(self.market_probability, 4) if self.market_probability else None,
            "odds": float(self.odds) if self.odds is not None else None,
            "expected_value": round(self.expected_value, 4) if self.expected_value is not None else None,
            "edge": round(self.edge, 4) if self.edge is not None else None,
            "recommendation": self.recommendation,
            "rejection_reason": self.rejection_reason,
            "model": f"{self.model_name}:{self.model_version}",
            "settled": self.settled,
            "won": self.won,
        }

    def __repr__(self) -> str:
        return (
            f"Prediction({self.race_id}/{self.horse_id}, p={self.model_probability:.3f}, "
            f"{self.recommendation})"
        )


__all__ = ["Prediction", "Recommendation"]
