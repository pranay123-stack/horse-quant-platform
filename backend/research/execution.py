"""The one controlled path from API activation to a validation report.

Seven steps: ingest, audit, features, train, predict, backtest, report. Each one
logs, records its runtime, and writes a checkpoint before the next begins, so an
interrupted run resumes at the step that failed rather than starting again.

That matters more here than in most pipelines. Step 1 is an eight-year backfill
measured in hours; steps 4 and 5 are a walk-forward retrain measured in tens of
minutes. A crash while rendering a markdown table must not cost either of them.

**Resume is not the same as reuse.** A completed step is skipped only if its
recorded protocol fingerprint still matches. Change the analysis plan and every
step invalidates, because a half-old, half-new run is not a study of anything —
it is two studies averaged together, and no one could say which.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.models.features import FEATURE_VERSION
from backend.models.operations import ResearchRun, ResearchStatus
from backend.research.audit import DatasetAudit, audit_dataset
from backend.research.audit import write_report as write_audit_report
from backend.research.preflight import PreflightResult, preflight
from backend.research.protocol import PROTOCOL, ValidationProtocol
from backend.research.validation_report import (
    REPORT_FILENAME,
    VERDICT_LABELS,
    ValidationRun,
    evaluate_predictions,
    generate_predictions,
)
from backend.research.validation_report import write_report as write_validation_report
from backend.utils.exceptions import PipelineError
from backend.utils.logging import get_logger, safe_extra
from backend.utils.timeutils import utcnow

logger = get_logger(__name__, channel="model")

CHECKPOINT_FILENAME = "phase6_execution.json"
PREDICTIONS_FILENAME = "phase6_predictions.parquet"
AUDIT_FILENAME = "REAL_DATA_AUDIT.md"

#: The seven steps, in order. Names are stable — they are checkpoint keys.
STEP_INGEST = "ingest"
STEP_AUDIT = "audit"
STEP_FEATURES = "features"
STEP_TRAIN = "train"
STEP_PREDICT = "predict"
STEP_BACKTEST = "backtest"
STEP_REPORT = "report"

STEP_ORDER: tuple[str, ...] = (
    STEP_INGEST,
    STEP_AUDIT,
    STEP_FEATURES,
    STEP_TRAIN,
    STEP_PREDICT,
    STEP_BACKTEST,
    STEP_REPORT,
)

STEP_DESCRIPTIONS: dict[str, str] = {
    STEP_INGEST: "Download historical data",
    STEP_AUDIT: "Run audit",
    STEP_FEATURES: "Build point-in-time features",
    STEP_TRAIN: "Train models",
    STEP_PREDICT: "Generate predictions",
    STEP_BACKTEST: "Run betting backtests",
    STEP_REPORT: "Generate final report",
}


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class StepRecord:
    """What happened when one step ran."""

    name: str
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    runtime_seconds: float = 0.0
    detail: str = ""

    @property
    def is_done(self) -> bool:
        return self.status == "done"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionState:
    """The resumable record of one workflow invocation."""

    run_id: str = ""
    protocol_fingerprint: str = ""
    started_at: str = field(default_factory=lambda: utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: utcnow().isoformat())
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    verdict: str | None = None
    dataset_hash: str | None = None

    # ------------------------------------------------------------------
    def record(self, name: str) -> StepRecord:
        payload = self.steps.get(name)
        return StepRecord(**payload) if payload else StepRecord(name=name)

    def is_done(self, name: str) -> bool:
        return self.record(name).is_done

    def save_step(self, record: StepRecord) -> None:
        self.steps[record.name] = record.as_dict()
        self.updated_at = utcnow().isoformat()

    @property
    def completed_steps(self) -> int:
        return sum(1 for name in STEP_ORDER if self.is_done(name))

    @property
    def total_runtime(self) -> float:
        return sum(self.record(name).runtime_seconds for name in STEP_ORDER)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = ["Phase 6 execution", "-----------------", f"  Run {self.run_id}", ""]
        for index, name in enumerate(STEP_ORDER, start=1):
            record = self.record(name)
            symbol = {"done": "OK  ", "failed": "FAIL", "running": "..  "}.get(record.status, "    ")
            runtime = f"{record.runtime_seconds:7.1f}s" if record.runtime_seconds else "        "
            lines.append(f"  {index}. [{symbol}] {STEP_DESCRIPTIONS[name]:<32} {runtime}  {record.detail}")
        lines += [
            "",
            f"  {self.completed_steps}/{len(STEP_ORDER)} steps · {self.total_runtime:.1f}s total",
        ]
        if self.verdict:
            lines.append(f"  Verdict: {self.verdict}")
        return "\n".join(lines)


def load_state(path: str | Path) -> ExecutionState:
    target = Path(path)
    if not target.exists():
        return ExecutionState()
    try:
        return ExecutionState(**json.loads(target.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("unreadable execution checkpoint, starting clean", extra={"error": str(exc)[:200]})
        return ExecutionState()


def save_state(state: ExecutionState, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state.as_dict(), indent=2), encoding="utf-8")
    return target


def new_run_id(protocol: ValidationProtocol) -> str:
    return f"phase6_{protocol.version}_{utcnow().strftime('%Y%m%dT%H%M%S')}"


def dataset_hash(audit: DatasetAudit) -> str:
    """A content hash of the rows a study ran on.

    Deliberately built from counts and span rather than the rows themselves:
    cheap to compute on eighty thousand races, and sensitive to exactly the
    changes that would invalidate a comparison between two runs.
    """
    payload = json.dumps(
        {
            "source": audit.source,
            "races": audit.races,
            "runners": audit.runners,
            "results": audit.results,
            "odds_quotes": audit.odds_quotes,
            "date_from": str(audit.date_from),
            "date_to": str(audit.date_to),
            "racing_days": audit.racing_days,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Experiment ledger
# ---------------------------------------------------------------------------
def open_research_run(session: Session, *, run_id: str, protocol: ValidationProtocol) -> ResearchRun | None:
    """Record that a validation run has started.

    Returns ``None`` if the ledger is unavailable — bookkeeping must never be
    the reason a study cannot run.
    """
    existing = session.get(ResearchRun, run_id)
    if existing is not None:
        return existing

    run = ResearchRun(
        run_id=run_id,
        protocol_hash=protocol.fingerprint,
        feature_version=FEATURE_VERSION,
        model_version=protocol.primary_model,
        start_time=utcnow(),
        status=ResearchStatus.RUNNING,
        steps_total=len(STEP_ORDER),
    )
    try:
        session.add(run)
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.warning("research run not recorded", extra={"error": str(exc)[:200]})
        return None
    return run


def close_research_run(
    session: Session,
    run: ResearchRun | None,
    state: ExecutionState,
    *,
    status: ResearchStatus,
    verdict: str | None = None,
    blocked_by: str | None = None,
    error: str | None = None,
) -> None:
    if run is None:
        return
    run.status = status
    run.end_time = utcnow()
    run.final_verdict = verdict
    run.blocked_by = blocked_by[:2000] if blocked_by else None
    run.error = error[:2000] if error else None
    run.steps_completed = state.completed_steps
    run.dataset_hash = state.dataset_hash
    try:
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.warning("research run not finalised", extra={"error": str(exc)[:200]})


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ExecutionResult:
    """Everything the caller needs to report on the run."""

    state: ExecutionState
    validation: ValidationRun | None = None
    audit: DatasetAudit | None = None
    preflight: PreflightResult | None = None
    blocked_by: str = ""
    report_path: Path | None = None
    audit_path: Path | None = None

    @property
    def completed(self) -> bool:
        return self.state.is_done(STEP_REPORT)

    @property
    def verdict_label(self) -> str:
        if self.validation is None or self.validation.verdict is None:
            return "INCONCLUSIVE"
        return VERDICT_LABELS.get(self.validation.verdict.verdict, "INCONCLUSIVE")

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.state.run_id,
            "completed": self.completed,
            "verdict": self.verdict_label if self.validation else None,
            "blocked_by": self.blocked_by,
            "steps": self.state.as_dict()["steps"],
            "report": str(self.report_path) if self.report_path else None,
        }


class StepRunner:
    """Runs one step, timing it and checkpointing either way."""

    def __init__(self, state: ExecutionState, checkpoint: Path):
        self.state = state
        self.checkpoint = checkpoint

    def run(self, name: str, action: Callable[[], str], *, force: bool = False) -> bool:
        """Execute ``action`` unless the step is already done.

        Returns ``True`` if the step is complete after this call. Raises nothing
        the caller has not asked for: a failing step is recorded, saved, and the
        exception re-raised so the workflow can stop at the right place.
        """
        if self.state.is_done(name) and not force:
            logger.info("step already complete, skipping", extra={"step": name})
            return True

        record = StepRecord(name=name, status="running", started_at=utcnow().isoformat())
        self.state.save_step(record)
        save_state(self.state, self.checkpoint)
        started = utcnow()
        logger.info("step starting", extra={"step": name, "description": STEP_DESCRIPTIONS[name]})

        try:
            detail = action()
        except Exception as exc:
            record.status = "failed"
            record.detail = f"{type(exc).__name__}: {exc}"[:300]
            record.finished_at = utcnow().isoformat()
            record.runtime_seconds = _elapsed(started)
            self.state.save_step(record)
            save_state(self.state, self.checkpoint)
            logger.error("step failed", extra={"step": name, "error": str(exc)[:200]})
            raise

        record.status = "done"
        record.detail = detail
        record.finished_at = utcnow().isoformat()
        record.runtime_seconds = _elapsed(started)
        self.state.save_step(record)
        save_state(self.state, self.checkpoint)
        logger.info(
            "step complete",
            extra=safe_extra({"step": name, "runtime": record.runtime_seconds, "detail": detail}),
        )
        return True


def _elapsed(since: datetime) -> float:
    return max(0.0, (utcnow() - since).total_seconds())


def run_workflow(
    session: Session,
    protocol: ValidationProtocol = PROTOCOL,
    *,
    workspace: Path,
    reports_dir: Path,
    ingest: Callable[[], str] | None = None,
    allow_unready: bool = False,
    expected_fingerprint: str | None = None,
    retrain_months: int = 12,
    model_params: dict[str, Any] | None = None,
    resume: bool = True,
    extra_reports: Callable[[ExecutionResult], str] | None = None,
) -> ExecutionResult:
    """Run every step, resuming where a previous attempt stopped.

    ``ingest`` is injected rather than called directly: downloading is async and
    needs a live API client, and keeping it out of here means the rest of the
    workflow is testable without one.
    """
    checkpoint = workspace / CHECKPOINT_FILENAME
    predictions_path = workspace / PREDICTIONS_FILENAME

    state = load_state(checkpoint) if resume else ExecutionState()
    if state.protocol_fingerprint and state.protocol_fingerprint != protocol.fingerprint:
        # A run half-executed under a different plan is not a study of anything.
        logger.warning(
            "protocol changed since the last run; discarding checkpoint",
            extra={"was": state.protocol_fingerprint, "now": protocol.fingerprint},
        )
        state = ExecutionState()
    state.protocol_fingerprint = protocol.fingerprint
    state.run_id = state.run_id or new_run_id(protocol)

    research_run = open_research_run(session, run_id=state.run_id, protocol=protocol)
    runner = StepRunner(state, checkpoint)
    result = ExecutionResult(state=state)

    try:
        # --- 1. download -------------------------------------------------
        runner.run(STEP_INGEST, ingest or (lambda: "skipped — no ingest callable supplied"))

        # --- 2. audit ----------------------------------------------------
        holder: dict[str, Any] = {}

        def do_audit() -> str:
            audit = audit_dataset(session)
            holder["audit"] = audit
            state.dataset_hash = dataset_hash(audit)
            result.audit_path = write_audit_report(audit, reports_dir / AUDIT_FILENAME)
            gates = preflight(audit, protocol, expected_fingerprint=expected_fingerprint)
            holder["preflight"] = gates
            if not gates.passed and not allow_unready:
                raise PipelineError(f"blocked at pre-flight gate '{gates.reason}'")
            return f"{audit.races:,} races, source '{audit.source}', hash {state.dataset_hash}"

        runner.run(STEP_AUDIT, do_audit, force=True)
        result.audit = holder.get("audit")
        result.preflight = holder.get("preflight")

        notes: list[str] = []
        gates = holder.get("preflight")
        if gates is not None and not gates.passed:
            notes.append(
                f"RUN ON A DATASET THAT FAILED A PRE-FLIGHT GATE — ({gates.reason}). "
                "Results are a harness check, not findings."
            )

        # --- 3. features -------------------------------------------------
        # Features are built inside the walk-forward predictor, per training
        # window, because that is what keeps them point-in-time. This step
        # records the version in force rather than pre-computing a table that
        # would then have to be trusted.
        runner.run(
            STEP_FEATURES,
            lambda: f"point-in-time features version {FEATURE_VERSION}, built per window",
        )

        # --- 4 & 5. train and predict ------------------------------------
        def do_predict() -> str:
            frame = generate_predictions(
                session, protocol, retrain_months=retrain_months, model_params=model_params
            )
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(predictions_path, index=False)
            return f"{len(frame):,} out-of-sample rows -> {predictions_path.name}"

        runner.run(
            STEP_TRAIN, lambda: f"walk-forward {protocol.primary_model}, retrain every {retrain_months}m"
        )
        runner.run(STEP_PREDICT, do_predict)

        if not predictions_path.exists():
            raise PipelineError(f"predictions cache missing at {predictions_path}; re-run without --resume")
        predictions = pd.read_parquet(predictions_path)

        # --- 6. backtests and analysis -----------------------------------
        def do_backtest() -> str:
            validation = evaluate_predictions(
                predictions,
                result.audit or DatasetAudit(),
                protocol,
                preflight_result=result.preflight,
                allow_unready=allow_unready,
                notes=notes,
            )
            holder["validation"] = validation
            state.verdict = (
                VERDICT_LABELS.get(validation.verdict.verdict, "INCONCLUSIVE") if validation.verdict else None
            )
            bets = validation.primary.metrics.bets if validation.primary else 0
            return f"{len(validation.strategy_reports)} strategies, {bets:,} primary bets"

        runner.run(STEP_BACKTEST, do_backtest, force=True)
        result.validation = holder.get("validation")

        # --- 7. reports ---------------------------------------------------
        def do_report() -> str:
            validation = result.validation
            if validation is None:
                raise PipelineError("no validation result to report")
            result.report_path = write_validation_report(validation, reports_dir / REPORT_FILENAME)
            written = [REPORT_FILENAME]
            if extra_reports is not None:
                written.append(extra_reports(result))
            return ", ".join(written)

        runner.run(STEP_REPORT, do_report, force=True)

    except PipelineError as exc:
        result.blocked_by = str(exc)
        close_research_run(
            session,
            research_run,
            state,
            status=ResearchStatus.BLOCKED,
            blocked_by=str(exc),
        )
        save_state(state, checkpoint)
        raise
    except Exception as exc:
        close_research_run(session, research_run, state, status=ResearchStatus.FAILED, error=str(exc))
        save_state(state, checkpoint)
        raise

    close_research_run(
        session,
        research_run,
        state,
        status=ResearchStatus.COMPLETED,
        verdict=state.verdict,
    )
    save_state(state, checkpoint)
    logger.info(
        "workflow complete",
        extra=safe_extra({"run_id": state.run_id, "verdict": state.verdict, "runtime": state.total_runtime}),
    )
    return result


def recent_research_runs(session: Session, *, limit: int = 10) -> list[ResearchRun]:
    """The experiment ledger, newest first."""
    from sqlalchemy import select

    try:
        return list(
            session.execute(select(ResearchRun).order_by(ResearchRun.start_time.desc()).limit(limit))
            .scalars()
            .all()
        )
    except SQLAlchemyError:
        session.rollback()
        return []


def protocol_window(protocol: ValidationProtocol = PROTOCOL) -> tuple[date, date]:
    """The full span the workflow needs to download."""
    return protocol.train_start, protocol.test_end


__all__ = [
    "AUDIT_FILENAME",
    "CHECKPOINT_FILENAME",
    "PREDICTIONS_FILENAME",
    "STEP_DESCRIPTIONS",
    "STEP_ORDER",
    "ExecutionResult",
    "ExecutionState",
    "StepRecord",
    "StepRunner",
    "close_research_run",
    "dataset_hash",
    "load_state",
    "new_run_id",
    "open_research_run",
    "protocol_window",
    "recent_research_runs",
    "run_workflow",
    "save_state",
]
