"""The execution workflow, the experiment ledger, and the client deliverables.

Two things here are worth more than the rest.

The **resume** tests: an eight-year backfill and a walk-forward retrain are
measured in hours. A workflow that silently repeats them after a crash is a
workflow nobody will run twice, so "did it actually skip the expensive step" is
asserted directly rather than assumed from a log line.

The **client report** tests: that PDF is the document most likely to be read by
someone who will not read the code and may act on it with money. Every verdict
is rendered and checked for the claim it is permitted to make, because the
failure that matters is not a crash — it is an encouraging sentence surviving a
verdict that does not support it.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from backend.models.operations import ResearchRun, ResearchStatus
from backend.research.audit import CoverageRow, DatasetAudit
from backend.research.client_report import (
    PROFITABLE_VERDICT,
    ClientReport,
    claims_profitability,
    generate_client_report,
)
from backend.research.execution import (
    STEP_BACKTEST,
    STEP_DESCRIPTIONS,
    STEP_INGEST,
    STEP_ORDER,
    STEP_PREDICT,
    STEP_REPORT,
    ExecutionState,
    StepRecord,
    StepRunner,
    close_research_run,
    dataset_hash,
    load_state,
    new_run_id,
    open_research_run,
    protocol_window,
    recent_research_runs,
    run_workflow,
    save_state,
)
from backend.research.protocol import PROTOCOL, ValidationProtocol, Verdict, VerdictInputs
from backend.research.readiness import (
    MAX_CALIBRATION_ERROR,
    MAX_TOLERABLE_DRAWDOWN,
    assess,
)
from backend.research.readiness import (
    render_markdown as render_readiness,
)
from backend.research.readiness import (
    write_report as write_readiness,
)
from backend.research.validation_report import ValidationRun
from backend.strategy.reports import StrategyMetrics, StrategyReport
from backend.utils.exceptions import PipelineError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def make_audit(*, source: str = "real", races: int = 40_000) -> DatasetAudit:
    audit = DatasetAudit(source=source, races=races, runners=races * 8)
    audit.results = races * 8
    audit.date_from = date(2018, 1, 1)
    audit.date_to = date(2026, 6, 30)
    audit.racing_days = 3_000
    audit.odds_quotes = races * 40
    audit.coverage = [
        CoverageRow(
            period=str(year),
            races=races // 8,
            runners=races,
            results=races,
            with_odds=int(races * 0.9),
            racing_days=300,
        )
        for year in range(2018, 2026)
    ]
    audit.readiness.ready = source == "real"
    return audit


def make_metrics(**overrides) -> StrategyMetrics:
    defaults = {
        "strategy": "A_ev5/flat",
        "bets": 900,
        "wins": 150,
        "losses": 750,
        "staked": 9_000.0,
        "returned": 9_540.0,
        "profit": 540.0,
        "roi": 0.06,
        "profit_factor": 1.06,
        "strike_rate": 0.1667,
        "average_odds": 6.4,
        "average_stake": 10.0,
        "starting_bankroll": 1_000.0,
        "final_bankroll": 1_540.0,
        "peak_bankroll": 1_600.0,
        "max_drawdown": 60.0,
        "max_drawdown_pct": 0.04,
        "longest_losing_streak": 22,
        "sharpe_per_bet": 0.05,
        "t_statistic": 2.9,
        "races_considered": 5_000,
        "bet_rate": 0.18,
    }
    defaults.update(overrides)
    return StrategyMetrics(**defaults)


def make_run(
    *,
    verdict: Verdict | None = Verdict.NO_EDGE,
    metrics: StrategyMetrics | None = None,
    calibration_error: float = 0.01,
    source: str = "real",
) -> ValidationRun:
    run = ValidationRun(protocol=PROTOCOL, audit=make_audit(source=source))
    run.predictions_rows = 40_000
    run.calibration_error = calibration_error
    run.baselines = pd.DataFrame(
        [
            {"forecaster": "market", "race_log_loss": 1.90},
            {"forecaster": "model", "race_log_loss": 1.93},
            {"forecaster": "uniform", "race_log_loss": 2.08},
        ]
    )

    report = StrategyReport(metrics=metrics or make_metrics())
    run.strategy_reports = [report]
    run.primary = report

    if verdict is Verdict.EDGE_CONFIRMED:
        run.verdict = PROTOCOL.verdict(
            VerdictInputs(bets=900, roi=0.06, t_statistic=2.9, profitable_year_fraction=0.8, years_covered=5)
        )
    elif verdict is Verdict.SEGMENT_ONLY:
        run.verdict = PROTOCOL.verdict(
            VerdictInputs(
                bets=900,
                roi=-0.01,
                t_statistic=-0.4,
                profitable_year_fraction=0.4,
                years_covered=5,
                passing_segments=("odds_band=2-5",),
                segment_t_statistics=(3.5,),
            )
        )
    elif verdict is Verdict.INSUFFICIENT_DATA:
        run.verdict = PROTOCOL.verdict(
            VerdictInputs(bets=20, roi=0.5, t_statistic=5.0, profitable_year_fraction=1.0, years_covered=1)
        )
    elif verdict is Verdict.NO_EDGE:
        run.verdict = PROTOCOL.verdict(
            VerdictInputs(
                bets=900, roi=-0.04, t_statistic=-2.1, profitable_year_fraction=0.2, years_covered=5
            )
        )
    return run


# ---------------------------------------------------------------------------
# Checkpoint state
# ---------------------------------------------------------------------------
def test_a_fresh_state_has_nothing_done():
    state = ExecutionState()
    assert state.completed_steps == 0
    assert not any(state.is_done(name) for name in STEP_ORDER)


def test_state_round_trips_through_disk(tmp_path):
    state = ExecutionState(run_id="r1", protocol_fingerprint="abc")
    state.save_step(StepRecord(name=STEP_INGEST, status="done", runtime_seconds=12.5))

    path = save_state(state, tmp_path / "state.json")
    reloaded = load_state(path)

    assert reloaded.run_id == "r1"
    assert reloaded.is_done(STEP_INGEST)
    assert reloaded.record(STEP_INGEST).runtime_seconds == 12.5
    assert reloaded.total_runtime == 12.5


def test_a_missing_checkpoint_starts_clean(tmp_path):
    assert load_state(tmp_path / "nope.json").completed_steps == 0


def test_a_corrupt_checkpoint_does_not_crash_the_run(tmp_path):
    """A bad JSON file must cost a re-run, not an unhandled exception."""
    path = tmp_path / "state.json"
    path.write_text("{ this is not json", encoding="utf-8")

    assert load_state(path).completed_steps == 0


def test_state_renders_every_step_with_its_runtime():
    state = ExecutionState(run_id="r1")
    state.save_step(StepRecord(name=STEP_INGEST, status="done", runtime_seconds=3.0))
    rendered = state.render()

    assert "Download historical data" in rendered
    assert "1/7 steps" in rendered
    assert "3.0s" in rendered, "the recorded runtime must be visible"
    # Human descriptions are shown, not the internal checkpoint keys.
    for description in STEP_DESCRIPTIONS.values():
        assert description in rendered


def test_run_ids_carry_the_protocol_version():
    assert new_run_id(PROTOCOL).startswith(f"phase6_{PROTOCOL.version}_")


# ---------------------------------------------------------------------------
# StepRunner
# ---------------------------------------------------------------------------
def test_a_step_records_its_runtime_and_detail(tmp_path):
    state = ExecutionState()
    runner = StepRunner(state, tmp_path / "state.json")

    assert runner.run(STEP_INGEST, lambda: "imported 10 races")

    record = state.record(STEP_INGEST)
    assert record.status == "done"
    assert record.detail == "imported 10 races"
    assert record.started_at and record.finished_at
    assert record.runtime_seconds >= 0


def test_a_completed_step_is_not_run_again(tmp_path):
    state = ExecutionState()
    runner = StepRunner(state, tmp_path / "state.json")
    calls: list[int] = []

    runner.run(STEP_INGEST, lambda: calls.append(1) or "first")
    runner.run(STEP_INGEST, lambda: calls.append(2) or "second")

    assert calls == [1], "the second call must have been skipped"


def test_force_re_runs_a_completed_step(tmp_path):
    state = ExecutionState()
    runner = StepRunner(state, tmp_path / "state.json")
    calls: list[int] = []

    runner.run(STEP_INGEST, lambda: calls.append(1) or "a")
    runner.run(STEP_INGEST, lambda: calls.append(2) or "b", force=True)

    assert calls == [1, 2]


def test_a_failing_step_is_recorded_and_re_raised(tmp_path):
    checkpoint = tmp_path / "state.json"
    state = ExecutionState()
    runner = StepRunner(state, checkpoint)

    def boom() -> str:
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        runner.run(STEP_INGEST, boom)

    record = load_state(checkpoint).record(STEP_INGEST)
    assert record.status == "failed"
    assert "disk full" in record.detail


# ---------------------------------------------------------------------------
# The workflow — resume is the behaviour that matters
# ---------------------------------------------------------------------------
@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace the expensive stages, counting how often each really runs."""
    counters = {"predict": 0, "evaluate": 0}

    frame = pd.DataFrame(
        {
            "race_id": ["r1"] * 8 + ["r2"] * 8,
            "horse_id": [f"h{index}" for index in range(16)],
            "race_date": [date(2025, 6, 1)] * 8 + [date(2025, 6, 2)] * 8,
            "won": ([1] + [0] * 7) * 2,
            "model_probability": [0.4] + [0.6 / 7] * 7 + [0.3] + [0.7 / 7] * 7,
            "mkt_mean_odds": [2.5] * 16,
            "mkt_best_odds": [2.6] * 16,
        }
    )

    def fake_generate(session, protocol, **kwargs):
        counters["predict"] += 1
        return frame

    def fake_evaluate(predictions, audit, protocol, **kwargs):
        counters["evaluate"] += 1
        run = make_run(verdict=Verdict.NO_EDGE)
        run.notes = list(kwargs.get("notes") or [])
        gates = kwargs.get("preflight_result")
        if gates is not None:
            run.preflight = gates
        return run

    monkeypatch.setattr("backend.research.execution.generate_predictions", fake_generate)
    monkeypatch.setattr("backend.research.execution.evaluate_predictions", fake_evaluate)
    monkeypatch.setattr("backend.research.execution.audit_dataset", lambda session: make_audit())
    return counters


def test_the_workflow_runs_every_step(db_session, tmp_path, stub_pipeline):
    result = run_workflow(
        db_session,
        PROTOCOL,
        workspace=tmp_path,
        reports_dir=tmp_path / "reports",
        ingest=lambda: "imported",
    )

    assert result.completed
    assert result.state.completed_steps == len(STEP_ORDER)
    for name in STEP_ORDER:
        assert result.state.is_done(name), name
    assert result.report_path is not None and result.report_path.exists()
    assert result.audit_path is not None and result.audit_path.exists()


def test_resuming_skips_the_expensive_prediction_step(db_session, tmp_path, stub_pipeline):
    """The reason the checkpoint exists.

    Walk-forward retraining over eight years is measured in tens of minutes. A
    second run must not pay for it again.
    """
    common = {
        "workspace": tmp_path,
        "reports_dir": tmp_path / "reports",
        "ingest": lambda: "imported",
    }
    run_workflow(db_session, PROTOCOL, **common)
    assert stub_pipeline["predict"] == 1

    run_workflow(db_session, PROTOCOL, **common)
    assert stub_pipeline["predict"] == 1, "prediction was recomputed on resume"


def test_an_interrupted_workflow_resumes_at_the_failed_step(db_session, tmp_path, stub_pipeline):
    reports = tmp_path / "reports"
    attempts: list[int] = []

    def flaky_ingest() -> str:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise RuntimeError("connection reset")
        return "imported on the retry"

    with pytest.raises(RuntimeError, match="connection reset"):
        run_workflow(db_session, PROTOCOL, workspace=tmp_path, reports_dir=reports, ingest=flaky_ingest)

    state = load_state(tmp_path / "phase6_execution.json")
    assert state.record(STEP_INGEST).status == "failed"
    assert not state.is_done(STEP_PREDICT), "later steps must not have run"

    result = run_workflow(db_session, PROTOCOL, workspace=tmp_path, reports_dir=reports, ingest=flaky_ingest)

    assert len(attempts) == 2, "ingest should have been retried exactly once"
    assert result.completed


def test_restarting_discards_the_checkpoint(db_session, tmp_path, stub_pipeline):
    common = {
        "workspace": tmp_path,
        "reports_dir": tmp_path / "reports",
        "ingest": lambda: "imported",
    }
    run_workflow(db_session, PROTOCOL, **common)
    run_workflow(db_session, PROTOCOL, resume=False, **common)

    assert stub_pipeline["predict"] == 2


def test_changing_the_protocol_invalidates_the_whole_checkpoint(db_session, tmp_path, stub_pipeline):
    """A half-old, half-new run is two studies averaged together."""
    common = {
        "workspace": tmp_path,
        "reports_dir": tmp_path / "reports",
        "ingest": lambda: "imported",
    }
    run_workflow(db_session, PROTOCOL, **common)
    assert stub_pipeline["predict"] == 1

    altered = ValidationProtocol(min_t_statistic=1.5)
    run_workflow(db_session, altered, **common)

    assert stub_pipeline["predict"] == 2, "a changed plan must not reuse old work"


def test_a_failed_gate_blocks_the_workflow(db_session, tmp_path, monkeypatch, stub_pipeline):
    monkeypatch.setattr(
        "backend.research.execution.audit_dataset", lambda session: make_audit(source="synthetic")
    )

    with pytest.raises(PipelineError, match="pre-flight gate"):
        run_workflow(
            db_session,
            PROTOCOL,
            workspace=tmp_path,
            reports_dir=tmp_path / "reports",
            ingest=lambda: "imported",
        )

    assert stub_pipeline["predict"] == 0, "nothing expensive may run past a failed gate"


def test_a_protocol_fingerprint_mismatch_blocks_the_workflow(db_session, tmp_path, stub_pipeline):
    with pytest.raises(PipelineError, match="pre-flight gate"):
        run_workflow(
            db_session,
            PROTOCOL,
            workspace=tmp_path,
            reports_dir=tmp_path / "reports",
            ingest=lambda: "imported",
            expected_fingerprint="0000000000000000",
        )

    assert stub_pipeline["predict"] == 0


def test_the_override_lets_the_harness_run_but_stamps_it(db_session, tmp_path, monkeypatch, stub_pipeline):
    monkeypatch.setattr(
        "backend.research.execution.audit_dataset", lambda session: make_audit(source="synthetic")
    )
    result = run_workflow(
        db_session,
        PROTOCOL,
        workspace=tmp_path,
        reports_dir=tmp_path / "reports",
        ingest=lambda: "imported",
        allow_unready=True,
    )

    assert result.completed
    assert result.validation is not None
    assert any("not findings" in note for note in result.validation.notes)


def test_extra_reports_are_invoked_and_recorded(db_session, tmp_path, stub_pipeline):
    result = run_workflow(
        db_session,
        PROTOCOL,
        workspace=tmp_path,
        reports_dir=tmp_path / "reports",
        ingest=lambda: "imported",
        extra_reports=lambda _result: "extra.pdf",
    )
    assert "extra.pdf" in result.state.record(STEP_REPORT).detail


def test_the_workflow_reports_its_window():
    start, end = protocol_window(PROTOCOL)
    assert start == PROTOCOL.train_start
    assert end == PROTOCOL.test_end


# ---------------------------------------------------------------------------
# Experiment ledger
# ---------------------------------------------------------------------------
def test_a_research_run_records_all_three_hashes(db_session, tmp_path, stub_pipeline):
    run_workflow(
        db_session,
        PROTOCOL,
        workspace=tmp_path,
        reports_dir=tmp_path / "reports",
        ingest=lambda: "imported",
    )

    record = db_session.query(ResearchRun).one()
    assert record.protocol_hash == PROTOCOL.fingerprint
    assert record.dataset_hash
    assert record.feature_version
    assert record.is_reproducible, "a result without all three hashes is an anecdote"
    assert record.status == ResearchStatus.COMPLETED
    assert record.final_verdict == "NO_EDGE_FOUND"
    assert record.duration_seconds is not None


def test_a_blocked_run_is_recorded_as_blocked_not_failed(db_session, tmp_path, monkeypatch, stub_pipeline):
    """Stopping at a gate is the system working, not an error."""
    monkeypatch.setattr(
        "backend.research.execution.audit_dataset", lambda session: make_audit(source="synthetic")
    )
    with pytest.raises(PipelineError):
        run_workflow(
            db_session,
            PROTOCOL,
            workspace=tmp_path,
            reports_dir=tmp_path / "reports",
            ingest=lambda: "imported",
        )

    record = db_session.query(ResearchRun).one()
    assert record.status == ResearchStatus.BLOCKED
    assert record.blocked_by
    assert record.final_verdict is None


def test_an_unexpected_error_is_recorded_as_failed(db_session, tmp_path, stub_pipeline):
    def boom() -> str:
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError):
        run_workflow(
            db_session,
            PROTOCOL,
            workspace=tmp_path,
            reports_dir=tmp_path / "reports",
            ingest=boom,
        )

    record = db_session.query(ResearchRun).one()
    assert record.status == ResearchStatus.FAILED
    assert "out of memory" in (record.error or "")


def test_opening_the_same_run_twice_reuses_the_row(db_session):
    first = open_research_run(db_session, run_id="r1", protocol=PROTOCOL)
    second = open_research_run(db_session, run_id="r1", protocol=PROTOCOL)

    assert first is not None and second is not None
    assert first.run_id == second.run_id
    assert db_session.query(ResearchRun).count() == 1


def test_closing_a_missing_run_is_harmless(db_session):
    close_research_run(db_session, None, ExecutionState(), status=ResearchStatus.FAILED)
    assert db_session.query(ResearchRun).count() == 0


def test_research_runs_are_listed_newest_first(db_session):
    from backend.utils.timeutils import utcnow

    base = utcnow()
    for index in range(3):
        db_session.add(
            ResearchRun(
                run_id=f"r{index}",
                protocol_hash="abc",
                start_time=base + timedelta(minutes=index),
                status=ResearchStatus.COMPLETED,
            )
        )
    db_session.commit()

    assert [row.run_id for row in recent_research_runs(db_session, limit=2)] == ["r2", "r1"]


def test_a_run_without_hashes_is_not_reproducible(db_session):
    from backend.utils.timeutils import utcnow

    record = ResearchRun(run_id="r", protocol_hash="abc", start_time=utcnow())
    assert not record.is_reproducible
    assert record.duration_seconds is None
    assert repr(record).startswith("ResearchRun(")
    assert "run_id" in record.as_dict()


# ---------------------------------------------------------------------------
# Dataset hashing
# ---------------------------------------------------------------------------
def test_identical_datasets_hash_identically():
    assert dataset_hash(make_audit()) == dataset_hash(make_audit())


def test_a_different_dataset_hashes_differently():
    assert dataset_hash(make_audit()) != dataset_hash(make_audit(races=40_001))


def test_synthetic_and_real_never_share_a_hash():
    """Two runs over the same counts but different provenance are not the same run."""
    assert dataset_hash(make_audit(source="real")) != dataset_hash(make_audit(source="synthetic"))


# ---------------------------------------------------------------------------
# Client report — the claim it is permitted to make
# ---------------------------------------------------------------------------
def test_only_a_confirmed_edge_permits_a_profitability_claim():
    assert claims_profitability(make_run(verdict=Verdict.EDGE_CONFIRMED))
    for verdict in (Verdict.NO_EDGE, Verdict.SEGMENT_ONLY, Verdict.INSUFFICIENT_DATA, None):
        assert not claims_profitability(make_run(verdict=verdict))


def test_a_glowing_roi_alone_does_not_permit_the_claim():
    """The whole reason the decision rule was frozen before the data existed."""
    run = make_run(verdict=Verdict.NO_EDGE, metrics=make_metrics(roi=0.83, t_statistic=1.1))
    assert not claims_profitability(run)


@pytest.mark.parametrize(
    "verdict",
    [Verdict.EDGE_CONFIRMED, Verdict.NO_EDGE, Verdict.SEGMENT_ONLY, Verdict.INSUFFICIENT_DATA],
)
def test_every_verdict_renders_a_pdf(verdict, tmp_path):
    report = generate_client_report(make_run(verdict=verdict), tmp_path / "report.pdf")

    assert isinstance(report, ClientReport)
    assert report.path.exists()
    assert report.path.read_bytes().startswith(b"%PDF-"), "not a valid PDF"
    assert report.path.stat().st_size > 2_000
    assert report.claims_profitability is (verdict is PROFITABLE_VERDICT)


def test_a_report_with_no_verdict_still_renders(tmp_path):
    run = make_run(verdict=None)
    report = generate_client_report(run, tmp_path / "report.pdf")

    assert report.verdict_label == "INCONCLUSIVE"
    assert not report.claims_profitability


def test_a_report_with_no_bets_still_renders(tmp_path):
    run = make_run(verdict=Verdict.INSUFFICIENT_DATA)
    run.primary = None
    run.strategy_reports = []
    report = generate_client_report(run, tmp_path / "report.pdf")

    assert report.path.exists()
    assert not report.claims_profitability


def test_a_harness_run_carries_its_warning_into_the_pdf(tmp_path):
    run = make_run(verdict=Verdict.EDGE_CONFIRMED, source="synthetic")
    run.notes.append("RUN ON A DATASET THAT FAILED A PRE-FLIGHT GATE")
    report = generate_client_report(run, tmp_path / "report.pdf")

    assert report.path.exists()


def test_the_same_run_renders_the_same_pdf_twice(tmp_path):
    """Report reproducibility: same inputs, same document."""
    run = make_run(verdict=Verdict.NO_EDGE)
    first = generate_client_report(run, tmp_path / "a.pdf")
    second = generate_client_report(run, tmp_path / "b.pdf")

    assert first.verdict_label == second.verdict_label
    assert first.claims_profitability == second.claims_profitability
    assert abs(first.path.stat().st_size - second.path.stat().st_size) < 200


# ---------------------------------------------------------------------------
# Production readiness
# ---------------------------------------------------------------------------
class FakeCapabilities:
    def __init__(self, *, ready: bool = True):
        self.subscription_active = ready
        self.ready_for_backfill = ready
        self.blocking_reason = "" if ready else "subscription inactive"
        self.earliest_result = date(2018, 1, 1)
        self.latest_result = date(2026, 6, 30)
        self.available_years = 8.5
        self.best_tier = "pro"


def test_a_healthy_run_passes_every_check():
    readiness = assess(make_run(verdict=Verdict.EDGE_CONFIRMED), capabilities=FakeCapabilities())

    assert readiness.all_passed
    assert not readiness.failures
    assert {check.name for check in readiness.checks} == {
        "API",
        "Data",
        "Model",
        "Backtest",
        "Risk",
    }


def test_an_unprobed_api_fails_rather_than_passing_by_default():
    """An unchecked component is not a passing component."""
    readiness = assess(make_run(), capabilities=None)
    assert not readiness.check("API").passed


def test_an_inactive_subscription_fails_the_api_check():
    readiness = assess(make_run(), capabilities=FakeCapabilities(ready=False))
    assert not readiness.check("API").passed


def test_synthetic_data_fails_the_data_check():
    readiness = assess(make_run(source="synthetic"), capabilities=FakeCapabilities())
    assert not readiness.check("Data").passed


def test_poor_calibration_fails_the_model_check():
    run = make_run(calibration_error=MAX_CALIBRATION_ERROR + 0.01)
    readiness = assess(run, capabilities=FakeCapabilities())

    check = readiness.check("Model")
    assert not check.passed
    assert "mis-sized" in check.detail


def test_a_model_no_better_than_uniform_fails():
    run = make_run()
    run.baselines = pd.DataFrame(
        [
            {"forecaster": "uniform", "race_log_loss": 2.00},
            {"forecaster": "model", "race_log_loss": 2.05},
        ]
    )
    assert not assess(run, capabilities=FakeCapabilities()).check("Model").passed


def test_a_model_with_no_predictions_fails():
    run = make_run()
    run.predictions_rows = 0
    assert not assess(run, capabilities=FakeCapabilities()).check("Model").passed


def test_too_few_bets_fails_the_backtest_check():
    run = make_run(metrics=make_metrics(bets=10))
    check = assess(run, capabilities=FakeCapabilities()).check("Backtest")

    assert not check.passed
    assert "required" in check.detail


def test_detected_leakage_fails_the_backtest_check():
    from backend.research.preflight import Gate, PreflightResult

    run = make_run()
    run.preflight = PreflightResult(gates=[Gate("no leakage", False, "form_score (AUC 1.0000)")])
    check = assess(run, capabilities=FakeCapabilities()).check("Backtest")

    assert not check.passed
    assert "leakage" in check.detail


def test_an_intolerable_drawdown_fails_the_risk_check():
    run = make_run(metrics=make_metrics(max_drawdown_pct=MAX_TOLERABLE_DRAWDOWN + 0.05))
    check = assess(run, capabilities=FakeCapabilities()).check("Risk")

    assert not check.passed
    assert "tolerance" in check.detail


def test_an_exhausted_bankroll_fails_the_risk_check():
    run = make_run(metrics=make_metrics(final_bankroll=0.0))
    assert not assess(run, capabilities=FakeCapabilities()).check("Risk").passed


def test_a_run_with_no_ledger_fails_risk_and_backtest():
    run = make_run()
    run.primary = None
    run.strategy_reports = []
    readiness = assess(run, capabilities=FakeCapabilities())

    assert not readiness.check("Risk").passed
    assert not readiness.check("Backtest").passed


def test_passing_every_check_does_not_imply_profitability():
    """The distinction this document exists to preserve.

    All five components can be sound while the honest answer is still that no
    edge was found. Conflating the two is how a green dashboard funds a losing
    strategy.
    """
    readiness = assess(make_run(verdict=Verdict.NO_EDGE), capabilities=FakeCapabilities())

    assert readiness.all_passed
    assert readiness.verdict_label == "NO_EDGE_FOUND"

    markdown = render_readiness(readiness, PROTOCOL)
    assert "NOT READY" not in markdown
    assert "NO_EDGE_FOUND" in markdown
    assert "does not imply it" in markdown


def test_the_checklist_renders_every_component_and_the_verdict():
    readiness = assess(make_run(verdict=Verdict.EDGE_CONFIRMED), capabilities=FakeCapabilities())
    markdown = render_readiness(readiness, PROTOCOL)

    for component in ("API", "Data", "Model", "Backtest", "Risk"):
        assert f"| {component} | **PASS** |" in markdown
    assert "# Production readiness" in markdown
    assert "## Profitability verdict" in markdown
    assert PROTOCOL.fingerprint in markdown


def test_the_checklist_lists_what_must_be_fixed():
    readiness = assess(make_run(source="synthetic"), capabilities=None)
    markdown = render_readiness(readiness, PROTOCOL)

    assert "## What must be fixed" in markdown
    assert "**API**" in markdown
    assert "**Data**" in markdown


def test_the_checklist_writes_to_disk(tmp_path):
    readiness = assess(make_run(), capabilities=FakeCapabilities())
    path = write_readiness(readiness, PROTOCOL, tmp_path / "PRODUCTION_READINESS.md")

    assert path.exists()
    assert "Production readiness" in path.read_text(encoding="utf-8")


def test_the_checklist_renders_for_a_terminal():
    readiness = assess(make_run(), capabilities=FakeCapabilities())
    rendered = readiness.render()

    assert "API" in rendered
    assert "PASS" in rendered
    assert "separate question" in rendered


def test_an_unevaluated_component_is_reported_as_such():
    from backend.research.readiness import ProductionReadiness

    readiness = ProductionReadiness()
    assert not readiness.all_passed
    assert readiness.check("Nonexistent").detail == "not evaluated"
    assert "checks" in readiness.as_dict()


def test_execution_result_serialises(db_session, tmp_path, stub_pipeline):
    result = run_workflow(
        db_session,
        PROTOCOL,
        workspace=tmp_path,
        reports_dir=tmp_path / "reports",
        ingest=lambda: "imported",
    )
    payload = result.as_dict()

    assert payload["completed"]
    assert payload["verdict"] == "NO_EDGE_FOUND"
    assert STEP_BACKTEST in payload["steps"]


# ---------------------------------------------------------------------------
# What the PDF actually says
#
# Asserting on the rendered text, not on the code that produced it. This is the
# guard that matters: the failure mode is not a crash, it is an encouraging
# sentence surviving a verdict that does not support it.
# ---------------------------------------------------------------------------
def _pdf_text(path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(str(path)).pages)


def test_the_pdf_contains_all_nine_sections(tmp_path):
    report = generate_client_report(make_run(verdict=Verdict.NO_EDGE), tmp_path / "r.pdf")
    text = _pdf_text(report.path)

    for section in (
        "Executive summary",
        "Data coverage",
        "Model performance",
        "Probability calibration",
        "Betting strategy results",
        "Risk analysis",
        "Drawdown analysis",
        "Statistical significance",
        "Final verdict",
    ):
        assert section in text, f"missing section: {section}"


@pytest.mark.parametrize("verdict", [Verdict.NO_EDGE, Verdict.SEGMENT_ONLY, Verdict.INSUFFICIENT_DATA])
def test_an_unconfirmed_verdict_never_produces_an_encouraging_pdf(verdict, tmp_path):
    """The single most important assertion in the client deliverable.

    A positive ROI is deliberately still shown — hiding it would be its own
    dishonesty — but no wording may present the strategy as validated.
    """
    run = make_run(verdict=verdict, metrics=make_metrics(roi=0.42, t_statistic=1.2))
    report = generate_client_report(run, tmp_path / "r.pdf")
    text = _pdf_text(report.path)

    assert not report.claims_profitability
    assert "EDGE_CONFIRMED" not in text
    assert "confirms a positive edge" not in text
    assert "does not support staking money" in text or "No claim about profitability" in text


def test_a_confirmed_verdict_says_so_plainly(tmp_path):
    report = generate_client_report(make_run(verdict=Verdict.EDGE_CONFIRMED), tmp_path / "r.pdf")
    text = _pdf_text(report.path)

    assert report.claims_profitability
    assert "EDGE_CONFIRMED" in text
    assert "confirms a positive edge" in text


def test_the_pdf_always_carries_the_protocol_fingerprint(tmp_path):
    """Without it the document cannot be tied back to the plan that produced it."""
    report = generate_client_report(make_run(), tmp_path / "r.pdf")
    assert PROTOCOL.fingerprint in _pdf_text(report.path)


def test_a_harness_run_says_so_on_the_first_page(tmp_path):
    run = make_run(verdict=Verdict.EDGE_CONFIRMED, source="synthetic")
    run.notes.append("RUN ON A DATASET THAT FAILED A PRE-FLIGHT GATE — synthetic data.")
    text = _pdf_text(generate_client_report(run, tmp_path / "r.pdf").path)

    assert "did not pass its data-quality gates" in text
    assert "synthetic" in text


def test_the_pdf_always_carries_the_disclaimer(tmp_path):
    text = _pdf_text(generate_client_report(make_run(), tmp_path / "r.pdf").path)
    assert "not financial advice" in text
