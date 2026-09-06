"""Resumable historical backfill.

Eight years of UK racing is roughly 80,000 races. At the API's rate limit that is
hours of wall time, and any of the usual interruptions — a dropped connection, a
rate-limit wall, a laptop lid — will land somewhere in the middle. A backfill
that cannot resume is a backfill that gets abandoned.

So progress is checkpointed to disk after every month. Re-running the same
command picks up from the first unfinished month rather than starting again, and
because ingestion is idempotent (Phase 2), re-running a *completed* month is
harmless if the checkpoint is ever lost.

The checkpoint also records failures per month rather than aborting the run. One
bad month should not cost the other ninety-five.

Alongside the checkpoint, every invocation is recorded in ``backfill_runs``.
The two are not redundant: the checkpoint makes a run *resumable* and lives on
disk; the table makes it *auditable* and lives next to the data it produced. If
the audit later reports a thin month, the first question is whether that month
was ever successfully imported — and only a durable record answers it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.data_pipeline.importer import import_results_for_range
from backend.data_pipeline.report import ImportSummary
from backend.models.operations import BackfillRun, BackfillStatus
from backend.services.racing_api import RacingAPIClient
from backend.services.racing_api.exceptions import (
    RacingAPIAuthenticationError,
    SubscriptionInactiveError,
)
from backend.utils.exceptions import PipelineError
from backend.utils.logging import get_logger, safe_extra
from backend.utils.timeutils import utcnow

logger = get_logger(__name__, channel="pipeline")

CHECKPOINT_FILENAME = "backfill_progress.json"


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split a date range into whole months, clipped to the range."""
    windows: list[tuple[date, date]] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        next_month = (
            date(cursor.year + 1, 1, 1) if cursor.month == 12 else date(cursor.year, cursor.month + 1, 1)
        )
        windows.append((max(cursor, start), min(next_month - timedelta(days=1), end)))
        cursor = next_month
    return windows


@dataclass(slots=True)
class BackfillProgress:
    """Which months are done, which failed, and what they yielded."""

    started_at: str = field(default_factory=lambda: utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: utcnow().isoformat())
    regions: list[str] = field(default_factory=lambda: ["gb", "ire"])
    completed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    races_imported: int = 0
    runners_imported: int = 0
    api_requests: int = 0
    #: Records the importer rejected. Distinct from a failed *month*: a handful
    #: of unparseable historical races is normal and must not fail the window.
    failed_records: int = 0
    #: Links this checkpoint to its row in ``backfill_runs``.
    run_id: str = ""

    def is_done(self, window: tuple[date, date]) -> bool:
        return _key(window) in self.completed

    def mark_done(self, window: tuple[date, date], summary: ImportSummary) -> None:
        key = _key(window)
        if key not in self.completed:
            self.completed.append(key)
        self.failed.pop(key, None)
        self.races_imported += summary.races_created
        self.runners_imported += summary.results_created
        self.api_requests += summary.api_requests
        self.failed_records += summary.failed
        self.updated_at = utcnow().isoformat()

    def mark_failed(self, window: tuple[date, date], error: BaseException) -> None:
        self.failed[_key(window)] = f"{type(error).__name__}: {error}"[:300]
        self.updated_at = utcnow().isoformat()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self, total_windows: int) -> str:
        return "\n".join(
            [
                "Backfill progress",
                "-----------------",
                f"  Run id            {self.run_id or '-'}",
                f"  Months complete   {len(self.completed)} / {total_windows}",
                f"  Months failed     {len(self.failed)}",
                f"  Races imported    {self.races_imported:,}",
                f"  Results imported  {self.runners_imported:,}",
                f"  Records rejected  {self.failed_records:,}",
                f"  API requests      {self.api_requests:,}",
            ]
        )


def _key(window: tuple[date, date]) -> str:
    return window[0].strftime("%Y-%m")


def load_progress(path: str | Path) -> BackfillProgress:
    target = Path(path)
    if not target.exists():
        return BackfillProgress()
    payload = json.loads(target.read_text(encoding="utf-8"))
    return BackfillProgress(**payload)


def save_progress(progress: BackfillProgress, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(progress.as_dict(), indent=2), encoding="utf-8")
    return target


def new_run_id(start: date, end: date) -> str:
    """A readable, unique id for one backfill invocation."""
    stamp = utcnow().strftime("%Y%m%dT%H%M%S")
    return f"bf_{start:%Y%m}_{end:%Y%m}_{stamp}"


def open_run(
    session: Session,
    *,
    start: date,
    end: date,
    regions: tuple[str, ...],
    months_total: int,
    run_id: str | None = None,
) -> BackfillRun | None:
    """Record that a backfill has started.

    Returns ``None`` if the table is unavailable. The audit trail must never be
    the reason an import cannot run — a missing migration should degrade the
    bookkeeping, not block the data.
    """
    run = BackfillRun(
        run_id=run_id or new_run_id(start, end),
        start_date=start,
        end_date=end,
        regions=",".join(regions),
        status=BackfillStatus.RUNNING,
        months_total=months_total,
        started_at=utcnow(),
    )
    try:
        session.add(run)
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.warning("backfill run not recorded", extra={"error": str(exc)[:200]})
        return None
    return run


def _sync_run(session: Session, run: BackfillRun | None, progress: BackfillProgress) -> None:
    """Push progress onto the run row so a long import is observable live."""
    if run is None:
        return
    run.records_processed = progress.races_imported
    run.failed_records = progress.failed_records
    run.months_completed = len(progress.completed)
    run.months_failed = len(progress.failed)
    run.api_requests = progress.api_requests
    try:
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.warning("backfill run not updated", extra={"error": str(exc)[:200]})


def close_run(
    session: Session,
    run: BackfillRun | None,
    progress: BackfillProgress,
    *,
    status: BackfillStatus,
    error: str | None = None,
) -> None:
    """Finalise the run record."""
    if run is None:
        return
    run.status = status
    run.completed_at = utcnow()
    run.error = error[:2000] if error else None
    _sync_run(session, run, progress)


async def backfill_history(
    session: Session,
    client: RacingAPIClient,
    *,
    start: date,
    end: date,
    checkpoint: str | Path,
    regions: tuple[str, ...] = ("gb", "ire"),
    retry_failed: bool = True,
    run_id: str | None = None,
) -> BackfillProgress:
    """Import every month between ``start`` and ``end``, resuming where possible.

    Stops immediately on an authentication or subscription failure — those cannot
    be fixed by continuing, and hammering the API with credentials it has already
    rejected risks the account.

    Every invocation is recorded in ``backfill_runs``, updated after each month
    so a run that takes hours can be watched from another session.
    """
    verdict = await client.check_connectivity()
    if not verdict["authorised"]:
        raise PipelineError(
            f"cannot backfill: {verdict['message']}. " + str(verdict.get("details", {}).get("remedy", ""))
        )

    progress = load_progress(checkpoint)
    progress.regions = list(regions)
    windows = month_windows(start, end)
    run = open_run(
        session,
        start=start,
        end=end,
        regions=regions,
        months_total=len(windows),
        run_id=run_id,
    )
    progress.run_id = run.run_id if run else (run_id or new_run_id(start, end))

    logger.info(
        "backfill starting",
        extra=safe_extra(
            {
                "run_id": progress.run_id,
                "months": len(windows),
                "already_done": len(progress.completed),
                "start": str(start),
                "end": str(end),
            }
        ),
    )

    for window in windows:
        key = _key(window)
        if progress.is_done(window):
            continue
        if key in progress.failed and not retry_failed:
            continue

        try:
            summary = await import_results_for_range(
                session,
                client,
                start_date=window[0].isoformat(),
                end_date=window[1].isoformat(),
                regions=regions,
            )
            progress.mark_done(window, summary)
            logger.info(
                "backfill month complete",
                extra=safe_extra({"month": key, "races": summary.races_created, "failed": summary.failed}),
            )
        except (SubscriptionInactiveError, RacingAPIAuthenticationError) as exc:
            save_progress(progress, checkpoint)
            close_run(session, run, progress, status=BackfillStatus.ABORTED, error=str(exc))
            raise
        except Exception as exc:
            progress.mark_failed(window, exc)
            logger.error("backfill month failed", extra={"month": key, "error": str(exc)[:200]})

        save_progress(progress, checkpoint)
        _sync_run(session, run, progress)

    status = BackfillStatus.PARTIAL if progress.failed else BackfillStatus.COMPLETED
    close_run(session, run, progress, status=status)
    logger.info(
        "backfill finished",
        extra=safe_extra(
            {"run_id": progress.run_id, "completed": len(progress.completed), "status": str(status)}
        ),
    )
    return progress


def recent_runs(session: Session, *, limit: int = 10) -> list[BackfillRun]:
    """The most recent backfill invocations, newest first."""
    try:
        return list(
            session.execute(select(BackfillRun).order_by(BackfillRun.started_at.desc()).limit(limit))
            .scalars()
            .all()
        )
    except SQLAlchemyError:
        session.rollback()
        return []


__all__ = [
    "CHECKPOINT_FILENAME",
    "BackfillProgress",
    "backfill_history",
    "close_run",
    "load_progress",
    "month_windows",
    "new_run_id",
    "open_run",
    "recent_runs",
    "save_progress",
]
