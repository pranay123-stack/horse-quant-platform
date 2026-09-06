"""The pre-registered protocol, the readiness gate and the backfill checkpoint.

The protocol tests matter more than they look. Its whole value is that it cannot
quietly change, so the tests pin the fingerprint, exercise every branch of the
verdict rule, and check that the multiplicity correction actually bites.
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import pytest

from backend.data_pipeline.backfill import (
    BackfillProgress,
    backfill_history,
    load_progress,
    month_windows,
    save_progress,
)
from backend.data_pipeline.report import ImportSummary
from backend.models.operations import BackfillRun, BackfillStatus
from backend.research.audit import (
    MIN_ODDS_COVERAGE,
    MIN_YEARS,
    DatasetAudit,
    audit_dataset,
    render_markdown,
    write_report,
)
from backend.research.protocol import (
    PROTOCOL,
    SECONDARY_TEST_COUNT,
    ProtocolViolationError,
    ValidationProtocol,
    Verdict,
    VerdictInputs,
    assert_protocol_unchanged,
    bonferroni_t_threshold,
)
from backend.research.synthetic import SyntheticConfig, generate_synthetic_data

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
def test_protocol_matches_the_specified_windows():
    assert PROTOCOL.train_start == date(2018, 1, 1)
    assert PROTOCOL.train_end == date(2022, 12, 31)
    assert PROTOCOL.valid_start == date(2023, 1, 1)
    assert PROTOCOL.valid_end == date(2024, 12, 31)
    assert PROTOCOL.test_start == date(2025, 1, 1)


def test_fingerprint_is_stable():
    """Two identical protocols hash the same; a changed one does not."""
    assert ValidationProtocol().fingerprint == PROTOCOL.fingerprint
    assert ValidationProtocol(min_t_statistic=1.5).fingerprint != PROTOCOL.fingerprint


def test_editing_the_protocol_is_detected():
    assert_protocol_unchanged(PROTOCOL.fingerprint)
    with pytest.raises(ProtocolViolationError, match="has changed"):
        assert_protocol_unchanged("0000000000000000")


def test_split_and_execution_configs_follow_the_protocol():
    split = PROTOCOL.split_config()
    assert split.train_end == PROTOCOL.train_end
    assert split.embargo_days == PROTOCOL.embargo_days

    execution = PROTOCOL.execution_config()
    assert execution.slippage == PROTOCOL.slippage
    assert execution.commission == PROTOCOL.commission
    assert execution.allow_starting_price is False


def test_bonferroni_makes_secondary_tests_harder():
    corrected = bonferroni_t_threshold(2.0, SECONDARY_TEST_COUNT)
    assert corrected > 2.0
    assert 2.8 < corrected < 3.1
    # More tests -> a higher bar.
    assert bonferroni_t_threshold(2.0, 50) > corrected
    # One test -> no correction.
    assert bonferroni_t_threshold(2.0, 1) == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# The verdict rule — every branch
# ---------------------------------------------------------------------------
def _inputs(**overrides) -> VerdictInputs:
    defaults = {
        "bets": 1000,
        "roi": 0.06,
        "t_statistic": 3.0,
        "profitable_year_fraction": 0.75,
        "years_covered": 4,
    }
    defaults.update(overrides)
    return VerdictInputs(**defaults)


def test_too_few_bets_is_inconclusive_however_good_the_roi():
    result = PROTOCOL.verdict(_inputs(bets=50, roi=0.90, t_statistic=9.0))
    assert result.verdict is Verdict.INSUFFICIENT_DATA


def test_a_strong_consistent_result_confirms_an_edge():
    assert PROTOCOL.verdict(_inputs()).verdict is Verdict.EDGE_CONFIRMED


def test_a_positive_roi_without_significance_is_not_an_edge():
    result = PROTOCOL.verdict(_inputs(t_statistic=1.2))
    assert result.verdict is Verdict.NO_EDGE


def test_profit_concentrated_in_one_year_is_not_an_edge():
    result = PROTOCOL.verdict(_inputs(profitable_year_fraction=0.25, years_covered=4))
    assert result.verdict is Verdict.NO_EDGE


def test_a_losing_strategy_is_not_an_edge():
    result = PROTOCOL.verdict(_inputs(roi=-0.05, t_statistic=-3.0))
    assert result.verdict is Verdict.NO_EDGE


def test_a_segment_surviving_correction_gives_the_segment_verdict():
    corrected = bonferroni_t_threshold()
    result = PROTOCOL.verdict(
        _inputs(
            roi=-0.01,
            t_statistic=-0.4,
            passing_segments=("odds_band=2-5",),
            segment_t_statistics=(corrected + 0.5,),
        )
    )
    assert result.verdict is Verdict.SEGMENT_ONLY
    assert "odds_band=2-5" in result.headline


def test_a_segment_that_only_clears_the_nominal_bar_does_not_count():
    """This is the multiplicity guard doing its job."""
    result = PROTOCOL.verdict(
        _inputs(
            roi=-0.01,
            t_statistic=-0.4,
            passing_segments=("year=2021",),
            segment_t_statistics=(2.3,),  # clears 2.0, fails the corrected 2.89
        )
    )
    assert result.verdict is Verdict.NO_EDGE


def test_verdict_records_its_reasoning():
    result = PROTOCOL.verdict(_inputs(model_vs_market_log_loss_delta=-0.05))
    assert any("ROI" in reason for reason in result.reasons)
    assert any("beats the market" in reason for reason in result.reasons)
    assert result.as_dict()["verdict"] == str(Verdict.EDGE_CONFIRMED)


def test_protocol_renders_the_plan():
    rendered = PROTOCOL.render()
    for expected in ("PRIMARY HYPOTHESIS", "DECISION RULE", "Bonferroni", PROTOCOL.fingerprint):
        assert expected in rendered


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------
def test_empty_database_is_blocked(db_session):
    audit = audit_dataset(db_session)
    assert not audit.readiness.ready
    assert "no races" in audit.readiness.blocking[0]


def test_synthetic_data_is_always_blocked(db_session):
    """The most important guard in Phase 6."""
    generate_synthetic_data(db_session, SyntheticConfig(n_days=60, races_per_day=2, seed=1))
    audit = audit_dataset(db_session)

    assert audit.source == "synthetic"
    assert not audit.readiness.ready
    assert any("synthetic" in reason for reason in audit.readiness.blocking)


def test_short_history_is_blocked(db_session):
    generate_synthetic_data(
        db_session, SyntheticConfig(n_days=400, races_per_day=2, seed=2, start_date=date(2024, 1, 1))
    )
    audit = audit_dataset(db_session)
    assert any(f"{MIN_YEARS}+" in reason for reason in audit.readiness.blocking)


def test_audit_counts_entities(db_session):
    generate_synthetic_data(db_session, SyntheticConfig(n_days=30, races_per_day=2, seed=3))
    audit = audit_dataset(db_session)

    assert audit.races == 60
    assert audit.runners == 60 * 8
    assert audit.horses > 0
    assert audit.odds_quotes > 0
    assert audit.date_from is not None


def test_audit_reports_coverage_by_year(db_session):
    generate_synthetic_data(
        db_session, SyntheticConfig(n_days=400, races_per_day=2, seed=4, start_date=date(2023, 6, 1))
    )
    audit = audit_dataset(db_session)

    assert len(audit.coverage) >= 2
    for row in audit.coverage:
        assert 0 <= row.result_coverage <= 1
        assert 0 <= row.odds_coverage <= 1


def test_low_odds_coverage_blocks(db_session):
    from backend.models import OddsHistory

    generate_synthetic_data(db_session, SyntheticConfig(n_days=30, races_per_day=2, seed=5))
    db_session.query(OddsHistory).delete()
    db_session.commit()

    audit = audit_dataset(db_session)
    assert audit.odds_coverage == 0.0
    assert any(f"{MIN_ODDS_COVERAGE:.0%}" in reason for reason in audit.readiness.blocking)


def test_post_off_quotes_block_the_study(db_session):
    from datetime import timedelta

    from backend.models import OddsHistory, Race

    generate_synthetic_data(db_session, SyntheticConfig(n_days=20, races_per_day=2, seed=6))
    race = db_session.query(Race).first()
    quote = db_session.query(OddsHistory).filter_by(race_id=race.race_id).first()
    quote.recorded_at = race.off_time + timedelta(minutes=5)
    db_session.commit()

    audit = audit_dataset(db_session)
    assert audit.odds_after_off >= 1
    assert any("after the off" in reason for reason in audit.readiness.blocking)


def test_audit_report_renders_and_writes(db_session, tmp_path):
    generate_synthetic_data(db_session, SyntheticConfig(n_days=30, races_per_day=2, seed=7))
    audit = audit_dataset(db_session)

    markdown = render_markdown(audit)
    for expected in ("# Real data audit", "## Totals", "## Coverage by year", "NOT READY"):
        assert expected in markdown

    path = write_report(audit, tmp_path / "report.md")
    assert path.exists()
    assert "Real data audit" in path.read_text()


def test_audit_dict_shape():
    payload = DatasetAudit().as_dict()
    assert {"totals", "span", "completeness", "integrity", "readiness"} <= set(payload)


# ---------------------------------------------------------------------------
# Backfill checkpointing
# ---------------------------------------------------------------------------
def test_month_windows_cover_the_range_without_gaps():
    windows = month_windows(date(2018, 1, 15), date(2018, 4, 10))

    assert len(windows) == 4
    assert windows[0] == (date(2018, 1, 15), date(2018, 1, 31))
    assert windows[-1] == (date(2018, 4, 1), date(2018, 4, 10))
    for earlier, later in pairwise(windows):
        assert (later[0] - earlier[1]).days == 1


def test_month_windows_across_a_year_boundary():
    windows = month_windows(date(2018, 12, 1), date(2019, 2, 28))
    assert [window[0].strftime("%Y-%m") for window in windows] == ["2018-12", "2019-01", "2019-02"]


def test_eight_years_is_about_a_hundred_months():
    assert len(month_windows(date(2018, 1, 1), date(2026, 6, 30))) == 102


def test_progress_round_trips(tmp_path):
    progress = BackfillProgress()
    window = (date(2020, 3, 1), date(2020, 3, 31))

    summary = ImportSummary()
    summary.races_created = 120
    summary.results_created = 1000
    summary.api_requests = 40
    progress.mark_done(window, summary)

    path = save_progress(progress, tmp_path / "progress.json")
    reloaded = load_progress(path)

    assert reloaded.is_done(window)
    assert reloaded.races_imported == 120
    assert reloaded.api_requests == 40


def test_a_completed_month_is_skipped_on_resume():
    progress = BackfillProgress()
    window = (date(2020, 3, 1), date(2020, 3, 31))
    assert not progress.is_done(window)

    progress.mark_done(window, ImportSummary())
    assert progress.is_done(window)


def test_failures_are_recorded_and_cleared_on_retry():
    progress = BackfillProgress()
    window = (date(2020, 3, 1), date(2020, 3, 31))

    progress.mark_failed(window, RuntimeError("rate limited"))
    assert "2020-03" in progress.failed

    progress.mark_done(window, ImportSummary())
    assert "2020-03" not in progress.failed


def test_missing_checkpoint_starts_clean(tmp_path):
    progress = load_progress(tmp_path / "does_not_exist.json")
    assert progress.completed == []


def test_progress_renders():
    progress = BackfillProgress()
    progress.mark_done((date(2020, 1, 1), date(2020, 1, 31)), ImportSummary())
    assert "Months complete" in progress.render(10)


# ---------------------------------------------------------------------------
# The backfill loop itself
#
# This is the code that will run unattended for hours against a paid API, so its
# failure behaviour matters more than its happy path. Three things must hold: it
# must not start against a dead subscription, one bad month must not cost the
# rest, and a credential failure mid-run must stop immediately rather than
# hammer an API that has already said no.
# ---------------------------------------------------------------------------
class FakeClient:
    """A Racing API client that records what was asked of it."""

    def __init__(self, *, authorised: bool = True):
        self.authorised = authorised
        self.requested: list[tuple[str, str]] = []

    async def check_connectivity(self) -> dict:
        if self.authorised:
            return {"authorised": True, "message": "ok"}
        return {
            "authorised": False,
            "message": "Racing API subscription is inactive",
            "error_code": "racing_api_subscription_inactive",
            "details": {"remedy": "enable a plan"},
        }


async def test_a_dead_subscription_stops_the_backfill_before_it_starts(db_session, tmp_path):
    from backend.utils.exceptions import PipelineError

    client = FakeClient(authorised=False)
    with pytest.raises(PipelineError, match="subscription is inactive"):
        await backfill_history(
            db_session,
            client,
            start=date(2020, 1, 1),
            end=date(2020, 3, 31),
            checkpoint=tmp_path / "p.json",
        )
    assert client.requested == [], "no month should be requested against a dead subscription"
    assert db_session.query(BackfillRun).count() == 0, (
        "a run that never started must not leave a row claiming it did"
    )


async def test_every_month_is_imported_once(db_session, monkeypatch, tmp_path):
    client = FakeClient()

    async def fake_import(session, api, *, start_date, end_date, regions):
        client.requested.append((start_date, end_date))
        summary = ImportSummary()
        summary.races_created = 10
        summary.api_requests = 3
        return summary

    monkeypatch.setattr("backend.data_pipeline.backfill.import_results_for_range", fake_import)
    progress = await backfill_history(
        db_session, client, start=date(2020, 1, 1), end=date(2020, 3, 31), checkpoint=tmp_path / "p.json"
    )

    assert len(client.requested) == 3
    assert client.requested[0] == ("2020-01-01", "2020-01-31")
    assert progress.races_imported == 30
    assert progress.api_requests == 9
    assert not progress.failed

    run = db_session.query(BackfillRun).one()
    assert run.status == BackfillStatus.COMPLETED
    assert run.records_processed == 30
    assert run.months_completed == 3
    assert run.months_total == 3
    assert run.months_failed == 0
    assert run.completed_at is not None
    assert run.run_id == progress.run_id, "checkpoint and table must name the same run"


async def test_resuming_skips_the_months_already_done(db_session, monkeypatch, tmp_path):
    """The reason the checkpoint exists: a re-run must not re-pay for old months."""
    checkpoint = tmp_path / "p.json"
    done = BackfillProgress()
    done.mark_done((date(2020, 1, 1), date(2020, 1, 31)), ImportSummary())
    done.mark_done((date(2020, 2, 1), date(2020, 2, 29)), ImportSummary())
    save_progress(done, checkpoint)

    client = FakeClient()

    async def fake_import(session, api, *, start_date, end_date, regions):
        client.requested.append((start_date, end_date))
        return ImportSummary()

    monkeypatch.setattr("backend.data_pipeline.backfill.import_results_for_range", fake_import)
    await backfill_history(
        db_session, client, start=date(2020, 1, 1), end=date(2020, 3, 31), checkpoint=checkpoint
    )

    assert client.requested == [("2020-03-01", "2020-03-31")]


async def test_one_bad_month_does_not_end_the_run(db_session, monkeypatch, tmp_path):
    client = FakeClient()

    async def fake_import(session, api, *, start_date, end_date, regions):
        client.requested.append((start_date, end_date))
        if start_date.startswith("2020-02"):
            raise RuntimeError("gateway timeout")
        return ImportSummary()

    monkeypatch.setattr("backend.data_pipeline.backfill.import_results_for_range", fake_import)
    progress = await backfill_history(
        db_session, client, start=date(2020, 1, 1), end=date(2020, 3, 31), checkpoint=tmp_path / "p.json"
    )

    assert len(client.requested) == 3, "the run must continue past the failure"
    assert "2020-02" in progress.failed
    assert "gateway timeout" in progress.failed["2020-02"]
    assert len(progress.completed) == 2

    run = db_session.query(BackfillRun).one()
    assert run.status == BackfillStatus.PARTIAL, (
        "a run with a missing month must not report plain 'completed' — that is "
        "how a dataset ends up with a silent hole in it"
    )
    assert run.months_failed == 1


async def test_a_failed_month_can_be_left_alone_on_retry(db_session, monkeypatch, tmp_path):
    checkpoint = tmp_path / "p.json"
    state = BackfillProgress()
    state.mark_failed((date(2020, 2, 1), date(2020, 2, 29)), RuntimeError("boom"))
    save_progress(state, checkpoint)

    client = FakeClient()

    async def fake_import(session, api, *, start_date, end_date, regions):
        client.requested.append((start_date, end_date))
        return ImportSummary()

    monkeypatch.setattr("backend.data_pipeline.backfill.import_results_for_range", fake_import)
    await backfill_history(
        db_session,
        client,
        start=date(2020, 1, 1),
        end=date(2020, 3, 31),
        checkpoint=checkpoint,
        retry_failed=False,
    )

    assert ("2020-02-01", "2020-02-29") not in client.requested


async def test_credentials_failing_mid_run_stops_immediately_and_saves(db_session, monkeypatch, tmp_path):
    """Never keep calling an API that has just rejected the credentials."""
    from backend.services.racing_api.exceptions import SubscriptionInactiveError

    checkpoint = tmp_path / "p.json"
    client = FakeClient()

    async def fake_import(session, api, *, start_date, end_date, regions):
        client.requested.append((start_date, end_date))
        if start_date.startswith("2020-02"):
            raise SubscriptionInactiveError("subscription cancelled mid-run")
        return ImportSummary()

    monkeypatch.setattr("backend.data_pipeline.backfill.import_results_for_range", fake_import)
    with pytest.raises(SubscriptionInactiveError):
        await backfill_history(
            db_session, client, start=date(2020, 1, 1), end=date(2020, 6, 30), checkpoint=checkpoint
        )

    assert len(client.requested) == 2, "it must stop at the failure, not carry on"
    # January's work survives the abort.
    assert load_progress(checkpoint).is_done((date(2020, 1, 1), date(2020, 1, 31)))

    run = db_session.query(BackfillRun).one()
    assert run.status == BackfillStatus.ABORTED
    assert run.error and "cancelled mid-run" in run.error
    assert run.completed_at is not None
