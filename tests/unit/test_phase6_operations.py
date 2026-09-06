"""Operational readiness: capability probing, pre-flight gates, run records.

Phase 6's promise is "press one button after the subscription activates". These
tests cover the parts that decide whether that button is safe to press — what
the API will actually serve, which gate stops a bad run, and whether the report
that comes out says the same thing twice.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import ClassVar

import pandas as pd
import pytest

from backend.data_pipeline.backfill import (
    BackfillProgress,
    close_run,
    new_run_id,
    open_run,
    recent_runs,
)
from backend.models.operations import BackfillRun, BackfillStatus
from backend.research.audit import CoverageRow, DatasetAudit
from backend.research.preflight import (
    MIN_ROWS_FOR_STUDY,
    Gate,
    PreflightResult,
    gate_data_source,
    gate_no_leakage,
    gate_odds_coverage,
    gate_protocol_integrity,
    gate_sufficient_data,
    gate_timestamp_integrity,
    preflight,
)
from backend.research.protocol import PROTOCOL, ValidationProtocol
from backend.research.validation_report import (
    VERDICT_LABELS,
    ValidationRun,
    render_markdown,
)
from backend.services.racing_api.capabilities import (
    PROTOCOL_HISTORY_START,
    ApiCapabilities,
    EndpointProbe,
    probe_capabilities,
)
from backend.services.racing_api.exceptions import (
    RacingAPIAuthenticationError,
    RacingAPINotFoundError,
    SubscriptionInactiveError,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeCredentials:
    masked_username = "abcd********"
    is_complete = True


class FakeHorse:
    horse_id = "hrs_1"


class FakeRunner:
    horse = FakeHorse()


class FakeRace:
    race_id = "rac_1"
    runners: ClassVar[list[FakeRunner]] = [FakeRunner()]

    def __init__(self, race_date: date | None = None):
        self.race_date = race_date


class FakeClient:
    """A Racing API client with switchable failure modes."""

    def __init__(
        self,
        *,
        connect_error: Exception | None = None,
        allowed_tiers: tuple[str, ...] = ("free", "basic", "standard", "pro"),
        results_error: Exception | None = None,
        historical_empty: bool = False,
        odds_error: Exception | None = None,
        history_from: date = date(2015, 1, 1),
        history_to: date | None = None,
    ):
        self.credentials = FakeCredentials()
        self.connect_error = connect_error
        self.allowed_tiers = allowed_tiers
        self.results_error = results_error
        self.historical_empty = historical_empty
        self.odds_error = odds_error
        #: The window this fake plan actually serves — everything outside it
        #: comes back empty, exactly as a limited plan behaves.
        self.history_from = history_from
        self.history_to = history_to or date.today()
        self.calls: list[str] = []

    async def request_json(self, path, **kwargs):
        self.calls.append(f"request:{path}")
        if self.connect_error:
            raise self.connect_error
        return {}

    async def get_races(self, *, tier="standard", limit=1, **kwargs):
        self.calls.append(f"racecards:{tier}")
        if tier not in self.allowed_tiers:
            raise SubscriptionInactiveError(f"tier {tier} not included in this plan")
        return [FakeRace()]

    async def get_results(self, *, start_date, end_date, limit=1, **kwargs):
        self.calls.append(f"results:{start_date}")
        if self.results_error:
            raise self.results_error
        if self.historical_empty and start_date.startswith("2018"):
            return []

        window_start = date.fromisoformat(start_date)
        window_end = date.fromisoformat(end_date)
        served_start = max(window_start, self.history_from)
        served_end = min(window_end, self.history_to)
        if served_start > served_end:
            return []
        return [FakeRace(served_start), FakeRace(served_end)]

    async def get_odds(self, race_id, horse_id):
        self.calls.append("odds")
        if self.odds_error:
            raise self.odds_error
        return [object(), object()]


# ---------------------------------------------------------------------------
# 6.1 — capability probing
# ---------------------------------------------------------------------------
async def test_a_healthy_plan_reports_every_endpoint_available():
    capabilities = await probe_capabilities(FakeClient())

    assert capabilities.authenticated
    assert capabilities.subscription_active
    assert capabilities.best_tier == "pro"
    for name in ("racecards", "results", "odds", "historical"):
        assert capabilities.probe(name).available, name
    assert capabilities.ready_for_backfill
    assert capabilities.history_reaches_protocol_start


async def test_an_inactive_subscription_still_proves_the_credentials_are_right():
    """The distinction the whole command exists to make.

    ``Subscription inactive`` is returned *after* the credentials are accepted,
    so it is positive evidence the username and password are correct. Reporting
    it as an authentication failure would send someone to reset a password that
    was never wrong.
    """
    client = FakeClient(connect_error=SubscriptionInactiveError("Subscription inactive"))
    capabilities = await probe_capabilities(client)

    assert capabilities.authenticated, "an inactive plan is not a credential failure"
    assert not capabilities.subscription_active
    assert not capabilities.ready_for_backfill


async def test_bad_credentials_fail_authentication():
    client = FakeClient(connect_error=RacingAPIAuthenticationError("Incorrect password"))
    capabilities = await probe_capabilities(client)

    assert not capabilities.authenticated
    assert not capabilities.subscription_active


async def test_a_dead_subscription_does_not_get_probed_five_more_times():
    """Never keep calling an API that has already refused us."""
    client = FakeClient(connect_error=SubscriptionInactiveError("Subscription inactive"))
    capabilities = await probe_capabilities(client)

    assert len(client.calls) == 1, f"probed anyway: {client.calls}"
    assert all(probe.skipped for probe in capabilities.probes)
    assert all(probe.verdict == "SKIPPED" for probe in capabilities.probes)


async def test_the_richest_available_tier_is_found_and_probing_stops_there():
    client = FakeClient(allowed_tiers=("free", "basic"))
    capabilities = await probe_capabilities(client)

    assert capabilities.best_tier == "basic"
    # pro and standard are refused, basic succeeds — and free is never tried,
    # because the answer is already known.
    assert "racecards:free" not in client.calls


async def test_a_plan_without_pro_falls_back_to_the_odds_endpoint():
    capabilities = await probe_capabilities(FakeClient(allowed_tiers=("free", "basic", "standard")))

    assert capabilities.best_tier == "standard"
    odds = capabilities.probe("odds")
    assert odds.available
    assert "dedicated endpoint" in odds.detail


async def test_pro_tier_means_odds_arrive_inline():
    capabilities = await probe_capabilities(FakeClient())
    assert "inline" in capabilities.probe("odds").detail


async def test_a_missing_price_is_access_granted_not_access_denied():
    """A 404 on one runner means the endpoint works and that runner has no price."""
    client = FakeClient(
        allowed_tiers=("free", "basic", "standard"),
        odds_error=RacingAPINotFoundError("no odds for this runner"),
    )
    capabilities = await probe_capabilities(client)

    assert capabilities.probe("odds").available


async def test_history_that_stops_short_of_the_protocol_is_reported():
    """The probe that decides whether the pre-registered study can run at all."""
    capabilities = await probe_capabilities(FakeClient(historical_empty=True))

    assert capabilities.subscription_active
    assert capabilities.probe("results").available
    assert not capabilities.probe("historical").available
    assert not capabilities.history_reaches_protocol_start
    assert not capabilities.ready_for_backfill, (
        "a plan that cannot reach 2018 cannot run the pre-registered protocol"
    )


async def test_results_failure_blocks_the_backfill():
    client = FakeClient(results_error=SubscriptionInactiveError("results not in this plan"))
    capabilities = await probe_capabilities(client)

    assert not capabilities.probe("results").available
    assert not capabilities.ready_for_backfill


async def test_an_unreachable_host_never_raises():
    capabilities = await probe_capabilities(FakeClient(connect_error=OSError("connection refused")))

    assert not capabilities.reachable
    assert not capabilities.authenticated
    assert "unreachable" in capabilities.message


async def test_capabilities_render_and_serialise_without_credentials():
    capabilities = await probe_capabilities(FakeClient())
    rendered = capabilities.render()
    payload = capabilities.as_dict()

    assert "Authentication     PASS" in rendered
    assert "Subscription       ACTIVE" in rendered
    assert payload["authentication"] == "PASS"
    assert payload["subscription"] == "ACTIVE"
    # The masked form is the only username that may ever appear.
    assert "abcd********" in rendered
    assert "password" not in rendered.lower()


def test_probe_lookup_of_an_unknown_endpoint_is_safe():
    capabilities = ApiCapabilities(probes=[EndpointProbe("results", True)])
    missing = capabilities.probe("nonexistent")
    assert missing.skipped
    assert not missing.available


def test_history_probe_defaults_to_the_protocol_start():
    assert PROTOCOL.train_start == PROTOCOL_HISTORY_START


# ---------------------------------------------------------------------------
# 6.4 — pre-flight gates
# ---------------------------------------------------------------------------
def healthy_audit(*, source: str = "real") -> DatasetAudit:
    audit = DatasetAudit(source=source, races=40_000, runners=320_000)
    audit.results = 320_000
    audit.runners_without_odds = 0
    audit.races_without_results = 0
    audit.coverage = [
        CoverageRow(
            period=str(year), races=5_000, runners=40_000, results=40_000, with_odds=38_000, racing_days=300
        )
        for year in range(2018, 2026)
    ]
    return audit


def test_a_healthy_dataset_passes_every_gate():
    result = preflight(healthy_audit(), PROTOCOL, expected_fingerprint=PROTOCOL.fingerprint)

    assert result.passed
    assert result.failure is None
    assert len(result.gates) == 5


def test_an_edited_protocol_stops_the_run():
    """The enforcement behind 'do not optimize after seeing results'."""
    gate = gate_protocol_integrity(ValidationProtocol(min_t_statistic=1.5), PROTOCOL.fingerprint)

    assert not gate.passed
    assert "has been edited" in gate.detail


def test_an_unpinned_protocol_says_so_rather_than_pretending_to_check():
    gate = gate_protocol_integrity(PROTOCOL, None)
    assert gate.passed
    assert "not pinned" in gate.detail


def test_the_protocol_gate_runs_before_anything_expensive():
    """A void run must not spend minutes auditing eight years of odds first."""
    result = preflight(healthy_audit(), ValidationProtocol(min_bets=1), expected_fingerprint="deadbeef")

    assert len(result.gates) == 1, "evaluation must stop at the first failure"
    assert result.failure is not None
    assert result.failure.name == "protocol integrity"


@pytest.mark.parametrize("source", ["synthetic", "mixed", "empty"])
def test_non_real_data_stops_the_run(source):
    gate = gate_data_source(DatasetAudit(source=source))
    assert not gate.passed


def test_real_data_passes_the_source_gate():
    assert gate_data_source(DatasetAudit(source="real")).passed


def test_short_history_stops_the_run():
    audit = healthy_audit()
    audit.coverage = audit.coverage[:3]
    gate = gate_sufficient_data(audit)

    assert not gate.passed
    assert "3 year(s)" in gate.detail


def test_a_year_with_a_thin_calendar_does_not_count_as_a_year():
    audit = healthy_audit()
    for row in audit.coverage[:4]:
        row.racing_days = 20
    assert not gate_sufficient_data(audit).passed


def test_too_few_rows_stops_the_run():
    audit = healthy_audit()
    audit.runners = MIN_ROWS_FOR_STUDY - 1
    gate = gate_sufficient_data(audit)

    assert not gate.passed
    assert "runner rows" in gate.detail


def test_thin_odds_coverage_stops_the_run():
    audit = healthy_audit()
    audit.runners_without_odds = int(audit.runners * 0.8)
    gate = gate_odds_coverage(audit)

    assert not gate.passed
    assert "not a random sample" in gate.detail


def test_thin_result_coverage_stops_the_run():
    audit = healthy_audit()
    audit.races_without_results = int(audit.races * 0.5)
    assert not gate_odds_coverage(audit).passed


def test_post_off_quotes_stop_the_run():
    audit = healthy_audit()
    audit.odds_after_off = 12
    gate = gate_timestamp_integrity(audit)

    assert not gate.passed
    assert "not something anyone could have bet at" in gate.detail


def test_clean_timestamps_pass():
    assert gate_timestamp_integrity(healthy_audit()).passed


def test_leakage_is_detected_by_behaviour_not_by_name():
    """A column that is the target in disguise must be caught whatever it is called."""
    frame = pd.DataFrame(
        {
            "won": [1, 0, 0, 0, 1, 0, 0, 0] * 40,
            "innocuous_metric": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] * 40,
            "noise": list(range(320)),
        }
    )
    gate = gate_no_leakage(frame, ["innocuous_metric", "noise"])

    assert not gate.passed
    assert "innocuous_metric" in gate.detail


def test_honest_features_pass_the_leakage_gate():
    rng = pd.Series(range(400))
    frame = pd.DataFrame({"won": [1, 0, 0, 0] * 100, "form": rng % 7, "weight": rng % 11})
    assert gate_no_leakage(frame, ["form", "weight"]).passed


def test_the_leakage_gate_is_a_no_op_with_nothing_to_check():
    assert gate_no_leakage(pd.DataFrame(), []).passed
    assert gate_no_leakage(pd.DataFrame({"won": [1, 0]}), []).passed


def test_preflight_renders_and_serialises():
    result = preflight(healthy_audit(), PROTOCOL, expected_fingerprint=PROTOCOL.fingerprint)
    rendered = result.render()
    payload = result.as_dict()

    assert "[PASS]" in rendered
    assert payload["passed"]
    assert payload["failed_gate"] is None
    assert len(payload["gates"]) == 5


def test_a_stopped_preflight_names_the_gate_in_its_output():
    result = PreflightResult(gates=[Gate("data source", False, "synthetic")])
    assert "STOPPED at 'data source'" in result.render()
    assert result.reason.startswith("data source:")


def test_an_empty_preflight_has_not_passed():
    """Vacuous truth is the wrong default for a safety gate."""
    assert not PreflightResult().passed


# ---------------------------------------------------------------------------
# 6.2 — the backfill audit trail
# ---------------------------------------------------------------------------
def test_run_ids_are_unique_and_readable():
    first = new_run_id(date(2018, 1, 1), date(2026, 8, 1))
    assert first.startswith("bf_201801_202608_")


def test_a_run_is_recorded_and_closed(db_session):
    progress = BackfillProgress()
    run = open_run(db_session, start=date(2020, 1, 1), end=date(2020, 3, 31), regions=("gb",), months_total=3)

    assert run is not None
    assert run.status == BackfillStatus.RUNNING
    assert run.completed_at is None
    assert not run.is_finished

    progress.races_imported = 500
    close_run(db_session, run, progress, status=BackfillStatus.COMPLETED)

    stored = db_session.query(BackfillRun).one()
    assert stored.status == BackfillStatus.COMPLETED
    assert stored.records_processed == 500
    assert stored.is_finished
    assert stored.duration_seconds is not None


def test_closing_a_run_that_was_never_opened_is_harmless(db_session):
    """The audit trail must never be the reason an import cannot run."""
    close_run(db_session, None, BackfillProgress(), status=BackfillStatus.FAILED)
    assert db_session.query(BackfillRun).count() == 0


def test_recent_runs_are_newest_first(db_session):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(3):
        db_session.add(
            BackfillRun(
                run_id=f"bf_{index}",
                start_date=date(2020, 1, 1),
                end_date=date(2020, 2, 1),
                status=BackfillStatus.COMPLETED,
                started_at=base + timedelta(days=index),
            )
        )
    db_session.commit()

    rows = recent_runs(db_session, limit=2)
    assert [row.run_id for row in rows] == ["bf_2", "bf_1"]


def test_a_run_serialises_for_the_cli(db_session):
    run = open_run(
        db_session, start=date(2020, 1, 1), end=date(2020, 2, 29), regions=("gb", "ire"), months_total=2
    )
    payload = run.as_dict()

    assert payload["run_id"] == run.run_id
    assert payload["months"] == "0/2"
    assert payload["status"] == BackfillStatus.RUNNING
    assert "2020-01-01" in payload["start_date"]
    assert repr(run).startswith("BackfillRun(")


# ---------------------------------------------------------------------------
# 6.5 — report reproducibility
# ---------------------------------------------------------------------------
def sample_run() -> ValidationRun:
    run = ValidationRun(protocol=PROTOCOL, audit=healthy_audit())
    run.preflight = preflight(run.audit, PROTOCOL, expected_fingerprint=PROTOCOL.fingerprint)
    run.baselines = pd.DataFrame(
        [
            {"forecaster": "market", "race_log_loss": 1.90},
            {"forecaster": "model", "race_log_loss": 1.92},
        ]
    )
    run.predictions_rows = 12_345
    return run


def test_the_same_run_renders_identically_twice():
    """A report that differs between renders cannot be cited."""
    run = sample_run()
    assert render_markdown(run) == render_markdown(run)


def test_two_runs_holding_the_same_numbers_render_identically():
    first, second = sample_run(), sample_run()
    second.generated_at = first.generated_at

    assert render_markdown(first) == render_markdown(second)


def test_the_report_carries_the_protocol_fingerprint():
    assert PROTOCOL.fingerprint in render_markdown(sample_run())


def test_the_report_has_all_eight_sections():
    markdown = render_markdown(sample_run())
    for section in (
        "## 1. Dataset",
        "## 2. Model performance",
        "## 3. Probability calibration",
        "## 4. Betting strategy performance",
        "## 5. Year-by-year results",
        "## 6. Segment analysis",
        "## 7. Statistical significance",
        "## 8. Final verdict",
    ):
        assert section in markdown, f"missing {section}"


def test_a_run_with_no_verdict_still_renders():
    """Rendering must not depend on the pipeline having completed."""
    markdown = render_markdown(sample_run())
    assert "INCONCLUSIVE" in markdown
    assert "not computed" in markdown


def test_every_verdict_has_exactly_one_label():
    from backend.research.protocol import Verdict

    assert set(VERDICT_LABELS) == set(Verdict)
    assert sorted(VERDICT_LABELS.values()) == [
        "EDGE_CONFIRMED",
        "INCONCLUSIVE",
        "NO_EDGE_FOUND",
        "SEGMENT_EDGE",
    ]


def test_labels_do_not_touch_the_decision_function():
    """The labels are presentation. The rule that produces them is frozen."""
    assert PROTOCOL.fingerprint == "76e18ddb6f11f6ad"


# ---------------------------------------------------------------------------
# Probe failure paths
#
# A status command exists to explain failures, so its own failure handling is
# the part that must not fail. Every branch below is a way the API can misbehave
# in production; none of them may raise.
# ---------------------------------------------------------------------------
class BrokenClient(FakeClient):
    """Raises a chosen error from a chosen probe."""

    def __init__(self, *, failing: str, error: Exception, **kwargs):
        super().__init__(**kwargs)
        self.failing = failing
        self.error = error

    async def get_races(self, *, tier="standard", limit=1, **kwargs):
        if self.failing == "racecards":
            raise self.error
        # Tier access is decided first, exactly as the real API would: a plan
        # that excludes 'pro' must not be reported as having inline odds.
        result = await super().get_races(tier=tier, limit=limit, **kwargs)
        if self.failing == "no_races":
            return []
        if self.failing == "no_runners":
            race = FakeRace()
            race.runners = []
            return [race]
        return result

    async def get_results(self, *, start_date, end_date, limit=1, **kwargs):
        if self.failing == "results":
            raise self.error
        if self.failing == "historical" and start_date.startswith("2018"):
            raise self.error
        return await super().get_results(start_date=start_date, end_date=end_date, limit=limit)

    async def get_odds(self, race_id, horse_id):
        if self.failing == "odds":
            raise self.error
        return await super().get_odds(race_id, horse_id)


async def test_a_generic_api_error_on_connect_is_reported_not_raised():
    from backend.services.racing_api.exceptions import RacingAPIServerError

    capabilities = await probe_capabilities(FakeClient(connect_error=RacingAPIServerError("502")))

    assert capabilities.reachable
    assert not capabilities.subscription_active
    assert "502" in capabilities.message


async def test_no_accessible_racecard_tier_is_reported():
    client = BrokenClient(failing="racecards", error=SubscriptionInactiveError("no racecards"))
    capabilities = await probe_capabilities(client)

    assert capabilities.best_tier is None
    assert not capabilities.probe("racecards").available


async def test_an_unexpected_racecard_error_does_not_crash_the_probe():
    client = BrokenClient(failing="racecards", error=ValueError("malformed payload"))
    capabilities = await probe_capabilities(client)

    assert not capabilities.probe("racecards").available
    assert "malformed payload" in capabilities.probe("racecards").detail


async def test_an_unexpected_results_error_does_not_crash_the_probe():
    client = BrokenClient(failing="results", error=ValueError("bad json"))
    capabilities = await probe_capabilities(client)

    assert not capabilities.probe("results").available
    assert "bad json" in capabilities.probe("results").detail


async def test_an_unexpected_history_error_does_not_crash_the_probe():
    client = BrokenClient(failing="historical", error=ValueError("timeout"))
    capabilities = await probe_capabilities(client)

    assert capabilities.probe("results").available
    assert not capabilities.probe("historical").available


async def test_history_refused_by_the_plan_is_reported_with_its_reason():
    client = BrokenClient(failing="historical", error=SubscriptionInactiveError("history not in this plan"))
    capabilities = await probe_capabilities(client)

    probe = capabilities.probe("historical")
    assert not probe.available
    assert "history not in this plan" in probe.detail
    assert probe.error_code == "racing_api_subscription_inactive"


async def test_the_odds_probe_copes_with_a_day_that_has_no_races():
    client = BrokenClient(failing="no_races", error=ValueError(), allowed_tiers=("free", "basic", "standard"))
    capabilities = await probe_capabilities(client)
    assert not capabilities.probe("odds").available


async def test_the_odds_probe_copes_with_a_race_that_has_no_runners():
    client = BrokenClient(
        failing="no_runners", error=ValueError(), allowed_tiers=("free", "basic", "standard")
    )
    capabilities = await probe_capabilities(client)

    probe = capabilities.probe("odds")
    assert not probe.available
    assert "no runners" in probe.detail


async def test_odds_refused_by_the_plan_is_reported():
    client = BrokenClient(
        failing="odds",
        error=SubscriptionInactiveError("odds not in this plan"),
        allowed_tiers=("free", "basic", "standard"),
    )
    capabilities = await probe_capabilities(client)

    assert not capabilities.probe("odds").available
    assert "odds not in this plan" in capabilities.probe("odds").detail


async def test_an_unexpected_odds_error_does_not_crash_the_probe():
    client = BrokenClient(
        failing="odds", error=ValueError("garbled"), allowed_tiers=("free", "basic", "standard")
    )
    capabilities = await probe_capabilities(client)
    assert not capabilities.probe("odds").available


async def test_a_blocked_plan_renders_its_reason():
    client = FakeClient(connect_error=SubscriptionInactiveError("Subscription inactive"))
    rendered = (await probe_capabilities(client)).render()

    assert "Ready for backfill NO" in rendered
    assert "Blocked by" in rendered


# ---------------------------------------------------------------------------
# The leakage gate must not cry wolf
# ---------------------------------------------------------------------------
def test_settlement_columns_are_not_mistaken_for_leaked_features():
    """A false positive here aborts every real run.

    The prediction frame carries outcomes so bets can be settled, plus the
    model's own probability. All of them separate winners perfectly by
    construction. Treating them as leaks would stop the pipeline on data that
    is completely fine — which is exactly what happened the first time this ran.
    """
    from backend.features.base import LABEL_COLUMNS
    from backend.research.validation_report import _feature_columns

    frame = pd.DataFrame(
        {
            **{column: [1, 0, 1, 0] for column in LABEL_COLUMNS},
            "model_probability": [0.9, 0.1, 0.8, 0.2],
            "market_probability": [0.8, 0.2, 0.7, 0.3],
            "race_id": ["a", "a", "b", "b"],
            "horse_id": ["h1", "h2", "h3", "h4"],
            "genuine_feature": [3.0, 1.0, 2.0, 4.0],
        }
    )
    kept = _feature_columns(frame)

    assert kept == ["genuine_feature"]
    assert gate_no_leakage(frame, kept).passed


def test_the_exclusion_list_is_the_authoritative_one():
    """Not a hand-written copy — that is how `is_winner` leaked in Phase 4."""
    from backend.features.base import LABEL_COLUMNS
    from backend.research.validation_report import _feature_columns

    frame = pd.DataFrame({column: [1.0, 0.0] for column in (*LABEL_COLUMNS, "keep_me")})
    assert _feature_columns(frame) == ["keep_me"]


# ---------------------------------------------------------------------------
# 6.3 — timestamp consistency
#
# Every leakage guarantee in this project rests on knowing when a price was
# taken relative to the off. These checks are what make that knowable, so a
# missing or contradictory off time is a hole in the guarantee rather than a
# cosmetic gap.
# ---------------------------------------------------------------------------
def test_a_few_races_without_an_off_time_is_a_warning(db_session):
    from backend.models import Race
    from backend.research.audit import audit_dataset
    from backend.research.synthetic import SyntheticConfig, generate_synthetic_data

    generate_synthetic_data(db_session, SyntheticConfig(n_days=100, races_per_day=2, seed=21))
    # One race in two hundred: below the tolerance, so it must not block.
    race = db_session.query(Race).first()
    race.off_time = None
    db_session.commit()

    audit = audit_dataset(db_session)

    assert audit.races_without_off_time == 1
    assert any("no off time" in warning for warning in audit.readiness.warnings)
    assert not any("no off time" in reason for reason in audit.readiness.blocking)


def test_many_races_without_an_off_time_blocks(db_session):
    from backend.models import Race
    from backend.research.audit import audit_dataset
    from backend.research.synthetic import SyntheticConfig, generate_synthetic_data

    generate_synthetic_data(db_session, SyntheticConfig(n_days=20, races_per_day=2, seed=22))
    for race in db_session.query(Race).all():
        race.off_time = None
    db_session.commit()

    audit = audit_dataset(db_session)

    assert audit.races_without_off_time == audit.races
    assert any("cannot be proven to be pre-race" in reason for reason in audit.readiness.blocking)


def test_an_off_time_on_the_wrong_day_is_detected(db_session):
    from datetime import timedelta

    from backend.models import Race
    from backend.research.audit import audit_dataset
    from backend.research.synthetic import SyntheticConfig, generate_synthetic_data

    generate_synthetic_data(db_session, SyntheticConfig(n_days=100, races_per_day=2, seed=23))
    race = db_session.query(Race).first()
    race.off_time = race.off_time + timedelta(days=3)
    db_session.commit()

    audit = audit_dataset(db_session)

    assert audit.off_time_date_mismatch == 1
    assert any("different day than race_date" in text for text in audit.readiness.warnings)


def test_systematic_off_time_disagreement_blocks(db_session):
    """One late meeting crossing midnight is noise. All of them is a bug."""
    from datetime import timedelta

    from backend.models import Race
    from backend.research.audit import audit_dataset
    from backend.research.synthetic import SyntheticConfig, generate_synthetic_data

    generate_synthetic_data(db_session, SyntheticConfig(n_days=20, races_per_day=2, seed=24))
    for race in db_session.query(Race).all():
        race.off_time = race.off_time + timedelta(days=1)
    db_session.commit()

    audit = audit_dataset(db_session)

    assert audit.off_time_date_mismatch == audit.races
    assert any("shifts the feature cutoff" in reason for reason in audit.readiness.blocking)


def test_clean_timestamps_raise_nothing(db_session):
    from backend.research.audit import audit_dataset
    from backend.research.synthetic import SyntheticConfig, generate_synthetic_data

    generate_synthetic_data(db_session, SyntheticConfig(n_days=40, races_per_day=2, seed=25))
    audit = audit_dataset(db_session)

    assert audit.races_without_off_time == 0
    assert audit.off_time_date_mismatch == 0
    assert not any("off time" in text for text in audit.readiness.warnings)


def test_the_audit_report_shows_the_timestamp_section(db_session):
    from backend.research.audit import audit_dataset, render_markdown
    from backend.research.synthetic import SyntheticConfig, generate_synthetic_data

    generate_synthetic_data(db_session, SyntheticConfig(n_days=30, races_per_day=2, seed=26))
    markdown = render_markdown(audit_dataset(db_session))

    for expected in (
        "# Real data audit",
        "## Coverage",
        "## Quality",
        "### Timestamp consistency",
        "Quotes recorded after the off",
        "Races with no off time",
        "Off time on a different day than race_date",
    ):
        assert expected in markdown, f"missing {expected}"


def test_the_audit_warns_about_integrity_problems_without_blocking(db_session):
    """Warnings are for things worth knowing that do not invalidate a study."""
    from backend.models import OddsHistory, RaceResult
    from backend.research.audit import audit_dataset, coverage_frame
    from backend.research.synthetic import SyntheticConfig, generate_synthetic_data

    generate_synthetic_data(db_session, SyntheticConfig(n_days=40, races_per_day=2, seed=27))

    # An impossible price, and a finished race whose winner was never recorded.
    db_session.query(OddsHistory).first().decimal_odds = 0.5
    winner = db_session.query(RaceResult).filter_by(is_winner=True).first()
    winner.is_winner = False
    db_session.commit()

    audit = audit_dataset(db_session)

    assert audit.invalid_prices >= 1
    assert audit.races_without_winner >= 1
    assert any("evens" in warning for warning in audit.readiness.warnings)
    assert any("no winner recorded" in warning for warning in audit.readiness.warnings)

    frame = coverage_frame(audit)
    assert not frame.empty
    assert "period" in frame.columns


# ---------------------------------------------------------------------------
# 6.7.1 — the data range the plan actually serves
#
# "Is there a lot of history" is the wrong question. The question is whether
# there is history *where the protocol needs it*: a plan serving 2021 onward has
# five years and still cannot run a study that trains from 2018.
# ---------------------------------------------------------------------------
async def test_the_served_history_window_is_discovered():
    client = FakeClient(history_from=date(2014, 3, 1))
    capabilities = await probe_capabilities(client)

    assert capabilities.earliest_result is not None
    assert capabilities.earliest_result.year == 2014
    assert capabilities.latest_result is not None
    assert capabilities.available_years > 10
    assert capabilities.enough_for_protocol
    assert capabilities.ready_for_backfill


async def test_the_range_search_is_a_bisection_not_a_scan():
    """Twenty-five years must not cost twenty-five calls."""
    client = FakeClient(history_from=date(2013, 1, 1))
    await probe_capabilities(client)

    range_calls = [call for call in client.calls if call.startswith("results:")]
    assert len(range_calls) < 15, f"probed {len(range_calls)} times: {range_calls}"


async def test_history_starting_after_the_protocol_is_not_enough():
    """The failure this probe exists to catch."""
    client = FakeClient(history_from=date(2021, 1, 1))
    capabilities = await probe_capabilities(client)

    assert capabilities.earliest_result is not None
    assert capabilities.earliest_result.year == 2021
    assert capabilities.available_years >= 5, "five years — and still not the right five"
    assert not capabilities.enough_for_protocol
    assert not capabilities.ready_for_backfill
    assert "trains from" in capabilities.blocking_reason


async def test_a_plan_with_barely_any_history_is_rejected():
    client = FakeClient(history_from=date.today() - timedelta(days=300))
    capabilities = await probe_capabilities(client)

    assert capabilities.available_years < 5
    assert not capabilities.enough_for_protocol


async def test_the_status_block_shows_the_range_and_the_coverage_verdict():
    capabilities = await probe_capabilities(FakeClient(history_from=date(2014, 1, 1)))
    rendered = capabilities.render()

    for expected in (
        "Authentication     PASS",
        "Subscription       ACTIVE",
        "Data range",
        "Earliest result",
        "Latest result",
        "Enough for protocol  YES",
        "Ready for backfill YES",
    ):
        assert expected in rendered, f"missing {expected}"


async def test_the_status_payload_carries_the_range():
    payload = (await probe_capabilities(FakeClient(history_from=date(2014, 1, 1)))).as_dict()

    assert payload["earliest_result"].startswith("2014")
    assert payload["latest_result"]
    assert payload["enough_for_protocol"] is True
    assert payload["available_years"] > 10


async def test_blocking_reasons_are_specific():
    inactive = await probe_capabilities(
        FakeClient(connect_error=SubscriptionInactiveError("Subscription inactive"))
    )
    assert "inactive" in inactive.blocking_reason

    no_results = await probe_capabilities(
        BrokenClient(failing="results", error=SubscriptionInactiveError("no results"))
    )
    assert "does not serve results" in no_results.blocking_reason


def test_a_plan_with_no_discovered_range_is_not_ready():
    capabilities = ApiCapabilities(subscription_active=True)
    assert not capabilities.enough_for_protocol
    assert capabilities.available_years == 0.0


# ---------------------------------------------------------------------------
# 6.7.2 — the API gates, and the full ordered check
# ---------------------------------------------------------------------------
async def _capabilities(**kwargs):
    return await probe_capabilities(FakeClient(**kwargs))


async def test_a_live_plan_passes_all_three_api_gates():
    from backend.research.preflight import (
        gate_api_endpoints,
        gate_api_subscription,
        gate_history_range,
    )

    capabilities = await _capabilities(history_from=date(2014, 1, 1))

    assert gate_api_subscription(capabilities).passed
    assert gate_api_endpoints(capabilities).passed
    assert gate_history_range(capabilities).passed


async def test_an_inactive_subscription_stops_at_the_first_api_gate():
    from backend.research.preflight import gate_api_subscription

    capabilities = await _capabilities(connect_error=SubscriptionInactiveError("Subscription inactive"))
    gate = gate_api_subscription(capabilities)

    assert not gate.passed
    assert "subscription inactive" in gate.detail


async def test_bad_credentials_fail_the_subscription_gate():
    from backend.research.preflight import gate_api_subscription

    capabilities = await _capabilities(connect_error=RacingAPIAuthenticationError("Incorrect password"))
    gate = gate_api_subscription(capabilities)

    assert not gate.passed
    assert "authentication failed" in gate.detail


async def test_a_plan_without_results_fails_the_endpoint_gate():
    from backend.research.preflight import gate_api_endpoints

    capabilities = await probe_capabilities(
        BrokenClient(failing="results", error=SubscriptionInactiveError("no results"))
    )
    gate = gate_api_endpoints(capabilities)

    assert not gate.passed
    assert "results" in gate.detail


async def test_short_history_fails_the_range_gate():
    from backend.research.preflight import gate_history_range

    gate = gate_history_range(await _capabilities(history_from=date(2021, 1, 1)))

    assert not gate.passed
    assert "trains from" in gate.detail


def test_unprobed_api_gates_say_so_rather_than_silently_passing():
    """An unchecked gate reports itself as unchecked, never as verified."""
    from backend.research.preflight import (
        gate_api_endpoints,
        gate_api_subscription,
        gate_history_range,
    )

    for gate in (gate_api_subscription(None), gate_api_endpoints(None), gate_history_range(None)):
        assert gate.passed
        assert "not probed" in gate.detail


async def test_full_preflight_checks_the_api_before_the_database():
    """Cheapest and most likely to fail goes first."""
    from backend.research.preflight import full_preflight

    capabilities = await _capabilities(connect_error=SubscriptionInactiveError("Subscription inactive"))
    result = full_preflight(healthy_audit(), PROTOCOL, capabilities=capabilities)

    assert len(result.gates) == 1, "must not audit a database it could never have filled"
    assert result.failure is not None
    assert result.failure.name == "api subscription"


async def test_full_preflight_passes_everything_when_all_is_well():
    from backend.research.preflight import full_preflight

    capabilities = await _capabilities(history_from=date(2014, 1, 1))
    result = full_preflight(
        healthy_audit(),
        PROTOCOL,
        capabilities=capabilities,
        expected_fingerprint=PROTOCOL.fingerprint,
    )

    assert result.passed
    assert len(result.gates) == 8
    assert "READY" not in result.render()  # the CLI adds that, not the result


def test_full_preflight_offline_still_checks_the_data():
    from backend.research.preflight import full_preflight

    result = full_preflight(healthy_audit(source="synthetic"), PROTOCOL, capabilities=None)

    assert not result.passed
    assert result.failure is not None
    assert result.failure.name == "data source"


def test_full_preflight_catches_an_edited_protocol_before_touching_the_data():
    from backend.research.preflight import full_preflight

    result = full_preflight(
        healthy_audit(), PROTOCOL, capabilities=None, expected_fingerprint="deadbeefdeadbeef"
    )

    assert result.failure is not None
    assert result.failure.name == "protocol integrity"
