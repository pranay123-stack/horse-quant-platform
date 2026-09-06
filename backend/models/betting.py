"""The bet ledger.

Every simulated or placed bet is recorded here, with the *inputs that justified
it* alongside the outcome: the model probability, the market's probability, the
price taken and the stake. Storing only the result would make the ledger a
record of luck; storing the reasoning makes it auditable.

The distinction matters when a strategy underperforms. "We lost money" is not
actionable. "We lost money on bets whose average model probability was 0.22
against a realised strike rate of 0.14" identifies a calibration failure in the
tail, which is.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base, TimestampMixin
from backend.models.entities import ID_LENGTH
from backend.models.racing import MONEY, ODDS


class Bet(Base, TimestampMixin):
    """One bet: what was staked, why, and what came back."""

    __tablename__ = "bet_ledger"
    __table_args__ = (
        Index("ix_bet_ledger_run_date", "run_id", "race_date"),
        Index("ix_bet_ledger_race", "race_id"),
        Index("ix_bet_ledger_strategy", "strategy"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #: Groups every bet from one backtest or one live session, so results are
    #: comparable and a run can be deleted wholesale.
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(60), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(60))

    race_id: Mapped[str] = mapped_column(
        String(ID_LENGTH), ForeignKey("races.race_id", ondelete="CASCADE"), nullable=False
    )
    horse_id: Mapped[str] = mapped_column(
        String(ID_LENGTH), ForeignKey("horses.horse_id", ondelete="CASCADE"), nullable=False
    )
    race_date: Mapped[date | None] = mapped_column(Date, index=True)
    placed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # --- the reasoning ------------------------------------------------
    model_probability: Mapped[float | None] = mapped_column(Float)
    market_probability: Mapped[float | None] = mapped_column(Float)
    edge: Mapped[float | None] = mapped_column(Float)
    expected_value: Mapped[float | None] = mapped_column(Float)

    # --- the execution ------------------------------------------------
    #: The price seen when the signal fired.
    observed_odds: Mapped[Decimal | None] = mapped_column(ODDS)
    #: The price actually struck, after slippage.
    odds: Mapped[Decimal | None] = mapped_column(ODDS)
    stake: Mapped[Decimal | None] = mapped_column(MONEY)
    #: Which limit bound the stake, if any (``max_stake_fraction``, …).
    stake_constraint: Mapped[str | None] = mapped_column(String(40))

    # --- the outcome --------------------------------------------------
    won: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    finishing_position: Mapped[int | None] = mapped_column(Integer)
    profit_loss: Mapped[Decimal | None] = mapped_column(MONEY)
    bankroll_after: Mapped[Decimal | None] = mapped_column(MONEY)


__all__ = ["Bet"]
