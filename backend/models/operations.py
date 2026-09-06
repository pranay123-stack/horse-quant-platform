"""Operational records: what the pipeline did, and when.

The JSON checkpoint in :mod:`backend.data_pipeline.backfill` exists so a backfill
can *resume*. This table exists so it can be *audited*. They answer different
questions and neither replaces the other:

* the checkpoint answers "which month do I do next?" and must survive a killed
  process, so it lives on disk and is rewritten after every month;
* this table answers "what was imported, when, by which run, and what failed?"
  and must survive a dropped file, so it lives in the database next to the data
  it produced.

That second question is not bookkeeping. An eight-year backfill takes hours and
will be run more than once — after a rate-limit wall, after a schema fix, after
a subscription lapse. When the audit later reports a thin month, the first thing
worth knowing is whether that month was ever successfully imported at all.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import Date, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.base import Base, TimestampMixin


class BackfillStatus(StrEnum):
    """Lifecycle of one backfill invocation."""

    RUNNING = "running"
    COMPLETED = "completed"
    #: Finished, but at least one month failed. Deliberately distinct from
    #: ``completed`` — a partial import that reports success is how a dataset
    #: ends up with a silent hole in it.
    PARTIAL = "partial"
    FAILED = "failed"
    ABORTED = "aborted"


class BackfillRun(Base, TimestampMixin):
    """One invocation of the historical backfill."""

    __tablename__ = "backfill_runs"
    __table_args__ = (
        Index("ix_backfill_runs_status", "status"),
        Index("ix_backfill_runs_window", "start_date", "end_date"),
    )

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    regions: Mapped[str | None] = mapped_column(String(120))

    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=BackfillStatus.RUNNING, server_default="running"
    )

    #: Rows actually written — races and their runners/results.
    records_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Records the importer rejected or could not parse. Non-zero is a warning,
    #: not a failure: a handful of malformed historical races is normal.
    failed_records: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    months_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    months_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    months_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    api_requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    #: Why the run stopped, when it stopped badly. Truncated, and never contains
    #: credential material — the API error taxonomy carries codes, not payloads.
    error: Mapped[str | None] = mapped_column(Text)

    @property
    def is_finished(self) -> bool:
        return self.status != BackfillStatus.RUNNING

    @property
    def duration_seconds(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "status": self.status,
            "records_processed": self.records_processed,
            "failed_records": self.failed_records,
            "months": f"{self.months_completed}/{self.months_total}",
            "months_failed": self.months_failed,
            "api_requests": self.api_requests,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }

    def __repr__(self) -> str:
        return (
            f"BackfillRun(run_id={self.run_id!r}, status={self.status!r}, "
            f"{self.start_date}→{self.end_date}, records={self.records_processed})"
        )


class ResearchStatus(StrEnum):
    """Lifecycle of one validation run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    #: Stopped deliberately at a pre-flight gate. Not a failure — the system
    #: working — but distinct from a run that produced a verdict.
    BLOCKED = "blocked"


class ResearchRun(Base, TimestampMixin):
    """One execution of the Phase 6 validation workflow.

    Every research result must be reproducible, and reproducibility needs three
    hashes rather than one. The **protocol** hash says which analysis plan was
    applied; the **dataset** hash says which rows it was applied to; the
    **feature version** says how those rows were turned into inputs. A verdict
    with all three recorded can be re-derived. A verdict with none of them is an
    anecdote.
    """

    __tablename__ = "research_runs"
    __table_args__ = (
        Index("ix_research_runs_status", "status"),
        Index("ix_research_runs_started", "start_time"),
    )

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    #: Fingerprint of the frozen analysis plan. If this differs between two
    #: runs, they did not test the same hypothesis.
    protocol_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Content hash of the rows the study ran on — span, counts and source.
    dataset_hash: Mapped[str | None] = mapped_column(String(64))
    feature_version: Mapped[str | None] = mapped_column(String(40))
    model_version: Mapped[str | None] = mapped_column(String(60))

    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ResearchStatus.RUNNING, server_default="running"
    )
    #: One of the four pre-registered conclusions, or null if none was reached.
    final_verdict: Mapped[str | None] = mapped_column(String(60))

    #: Which workflow steps completed, and how long each took.
    steps_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    steps_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    blocked_by: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)

    @property
    def duration_seconds(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()

    @property
    def is_reproducible(self) -> bool:
        """Enough recorded to re-derive this result from scratch."""
        return bool(self.protocol_hash and self.dataset_hash and self.feature_version)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "protocol_hash": self.protocol_hash,
            "dataset_hash": self.dataset_hash,
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "final_verdict": self.final_verdict,
            "steps": f"{self.steps_completed}/{self.steps_total}",
            "reproducible": self.is_reproducible,
            "blocked_by": self.blocked_by,
            "error": self.error,
        }

    def __repr__(self) -> str:
        return f"ResearchRun(run_id={self.run_id!r}, status={self.status!r}, verdict={self.final_verdict!r})"


__all__ = ["BackfillRun", "BackfillStatus", "ResearchRun", "ResearchStatus"]
