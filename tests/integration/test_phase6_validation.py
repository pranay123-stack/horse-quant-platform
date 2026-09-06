"""The Phase 6 validation pipeline, end to end, against a real PostgreSQL.

This is a *harness* test, not a validation. It runs on synthetic data with the
pre-flight gates deliberately overridden, so nothing it produces says anything
about whether an edge exists. What it does prove is that the machinery which
will one day run on real data works: walk-forward prediction feeds the market
frame, the market frame feeds the baselines and the encompassing test, the
strategies settle into ledgers, the ledgers segment, and the frozen rule turns
all of that into exactly one verdict.

The protocol is compressed to the synthetic season's dates. That is the one
substitution allowed — the *windows* move, the *decision rule* does not.
"""

from __future__ import annotations

import os
from datetime import date

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.database.base import Base
from backend.research.protocol import ValidationProtocol, Verdict
from backend.research.synthetic import SyntheticConfig, generate_synthetic_data
from backend.research.validation_report import (
    BACKTEST_STAKING,
    BACKTEST_STRATEGIES,
    render_markdown,
    run_validation,
    segments_to_frame,
    write_report,
)
from backend.utils.config import Settings
from backend.utils.exceptions import PipelineError

pytestmark = pytest.mark.integration

SCHEMA = "phase6_test"

SEASON = SyntheticConfig(
    n_horses=350,
    n_jockeys=30,
    n_trainers=25,
    n_courses=6,
    start_date=date(2021, 1, 1),
    n_days=1300,
    races_per_day=2,
    runners_per_race=8,
    seed=606,
)

#: The real protocol with its windows moved onto the synthetic season. Every
#: threshold that decides the verdict is left exactly as pre-registered.
TEST_PROTOCOL = ValidationProtocol(
    train_start=date(2021, 1, 1),
    train_end=date(2022, 12, 31),
    valid_start=date(2023, 1, 1),
    valid_end=date(2023, 12, 31),
    test_start=date(2024, 1, 1),
    test_end=date(2024, 7, 31),
)


def _settings() -> Settings:
    return Settings(
        postgres_host=os.environ.get("TEST_POSTGRES_HOST", "localhost"),
        postgres_port=int(os.environ.get("TEST_POSTGRES_PORT", "5432")),
        postgres_user=os.environ.get("TEST_POSTGRES_USER", "horse_quant"),
        postgres_password=os.environ.get("TEST_POSTGRES_PASSWORD", "horse_quant"),
        postgres_db=os.environ.get("TEST_POSTGRES_DB", "horse_quant"),
        DATABASE_URL=None,
    )


@pytest.fixture(scope="module")
def pg_session():
    settings = _settings()
    admin = create_engine(settings.database_url, pool_pre_ping=True)
    try:
        with admin.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL not available: {exc}")

    with admin.begin() as connection:
        connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        connection.execute(text(f"CREATE SCHEMA {SCHEMA}"))

    engine = create_engine(settings.database_url, connect_args={"options": f"-csearch_path={SCHEMA}"})
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as session:
        generate_synthetic_data(session, SEASON)
        yield session

    engine.dispose()
    with admin.begin() as connection:
        connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    admin.dispose()


@pytest.fixture(scope="module")
def validation(pg_session):
    """One expensive run, shared. ``allow_unready`` because this is synthetic."""
    return run_validation(
        pg_session,
        TEST_PROTOCOL,
        allow_unready=True,
        retrain_months=12,
        model_params={"n_estimators": 120},
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def test_synthetic_data_is_refused_without_the_override(pg_session):
    """The single most important behaviour in Phase 6."""
    with pytest.raises(PipelineError, match="pre-flight gate"):
        run_validation(pg_session, TEST_PROTOCOL)


def test_an_overridden_run_is_stamped_as_worthless(validation):
    assert validation.notes, "an unready run must say so"
    assert any("not findings" in note for note in validation.notes)
    assert "synthetic" in " ".join(validation.notes)


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------
def test_predictions_land_inside_the_test_window(validation):
    assert validation.predictions_rows > 0


def test_every_baseline_is_scored(validation):
    forecasters = set(validation.baselines["forecaster"])
    assert {"random", "uniform", "favourite", "market", "model"} <= forecasters


def test_the_market_beats_the_uninformed_baselines(validation):
    """A control. If the market does not beat uniform, the data is broken."""
    losses = validation.baselines.set_index("forecaster")["race_log_loss"]
    assert losses["market"] < losses["uniform"] < losses["random"]


def test_the_encompassing_test_ran_on_the_real_predictions(validation):
    result = validation.encompassing
    assert result.rows > 0
    assert result.converged or result.collinear
    assert result.verdict


def test_every_strategy_and_staking_combination_is_backtested(validation):
    names = {report.metrics.strategy for report in validation.strategy_reports}
    expected = {f"{s}/{k}" for s in BACKTEST_STRATEGIES for k in BACKTEST_STAKING}
    assert names == expected


def test_the_primary_hypothesis_is_the_pre_registered_one(validation):
    assert validation.primary is not None
    assert validation.primary.metrics.strategy == (
        f"{TEST_PROTOCOL.primary_strategy}/{TEST_PROTOCOL.primary_staking}"
    )


def test_segments_carry_their_own_arithmetic(validation):
    if not validation.segments:
        pytest.skip("primary strategy placed too few bets to segment")

    frame = segments_to_frame(validation)
    assert {"dimension", "segment", "bets", "roi", "t_statistic"} <= set(frame.columns)
    for segment in validation.segments:
        assert segment.bets > 0
        assert -1.0 <= segment.roi <= 50.0


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------
def test_exactly_one_verdict_is_produced(validation):
    assert validation.verdict is not None
    assert validation.verdict.verdict in set(Verdict)
    assert validation.verdict.headline
    assert validation.verdict.reasons


def test_the_verdict_is_reproducible_from_the_frozen_rule(validation):
    """Re-applying the rule to the same numbers must give the same answer.

    This is what "the verdict is computed, not written" means in practice.
    """
    replayed = TEST_PROTOCOL.verdict(_inputs_from(validation, validation.primary.metrics))

    assert replayed.verdict is validation.verdict.verdict
    assert replayed.headline == validation.verdict.headline


def _inputs_from(validation, metrics):
    from backend.research.protocol import VerdictInputs

    return VerdictInputs(
        bets=metrics.bets,
        roi=metrics.roi,
        t_statistic=metrics.t_statistic,
        profitable_year_fraction=validation.consistency.profitable_year_fraction,
        years_covered=validation.consistency.years_covered,
        passing_segments=tuple(validation.consistency.passing_nominal),
        segment_t_statistics=tuple(
            segment.t_statistic
            for segment in validation.segments
            if segment.label in validation.consistency.passing_nominal
        ),
    )


def test_a_thin_run_cannot_confirm_an_edge(validation):
    """Guards the direction that matters: no edge claim without enough bets."""
    if validation.primary.metrics.bets < TEST_PROTOCOL.min_bets:
        assert validation.verdict.verdict is Verdict.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# The deliverable
# ---------------------------------------------------------------------------
def test_the_report_renders_every_section(validation):
    markdown = render_markdown(validation)
    for section in (
        "# Phase 6 — final validation report",
        "## 0. Pre-flight gates",
        "## 1. Dataset",
        "## 2. Model performance",
        "## 3. Probability calibration",
        "## 4. Betting strategy performance",
        "## 5. Year-by-year results",
        "## 6. Segment analysis",
        "## 7. Statistical significance",
        "## 8. Final verdict",
        "Does the model know anything the price does not?",
    ):
        assert section in markdown, f"missing {section}"

    assert TEST_PROTOCOL.fingerprint in markdown
    assert "⚠️" in markdown, "an unready run must carry its warning into the report"


def test_the_report_writes_to_disk(validation, tmp_path):
    path = write_report(validation, tmp_path / "REPORT.md")
    assert path.exists()
    assert "Phase 6 — final validation report" in path.read_text(encoding="utf-8")
