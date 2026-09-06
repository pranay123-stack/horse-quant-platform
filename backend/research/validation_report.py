"""The Phase 6 deliverable: ``Phase6_Final_Report.md``.

One command runs the frozen protocol end to end and writes a single markdown
document containing the dataset audit, the model comparison against baselines,
the strategy backtests, the robustness breakdown, and a mechanically-derived
verdict.

Two design choices are worth stating.

**The verdict is computed, not written.** :meth:`ValidationProtocol.verdict`
takes a fixed set of numbers and returns one of four conclusions. Nothing in
this module can nudge it. That is the whole point: by the time results exist,
the rule that interprets them is already settled.

**The report refuses to run on synthetic data.** The pre-flight gates in
:mod:`backend.research.preflight` stop it. A validation report generated from
generated data would look exactly like a real one, and would be worthless — the
most dangerous artefact this project could produce.

The verdict *labels* printed here (``EDGE_CONFIRMED`` and friends) are display
strings and nothing more. They map from the frozen :class:`Verdict` enum; they
do not participate in deciding anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from backend.features.base import LABEL_COLUMNS
from backend.ml.baselines import EncompassingResult, compare_baselines, market_encompassing_test
from backend.ml.metrics.classification import (
    ReliabilityCurve,
    expected_calibration_error,
    reliability_curve,
)
from backend.research.audit import DatasetAudit, audit_dataset
from backend.research.preflight import PreflightResult, gate_no_leakage, preflight
from backend.research.protocol import (
    PROTOCOL,
    ValidationProtocol,
    Verdict,
    VerdictInputs,
    VerdictResult,
)
from backend.strategy import (
    STRATEGY_LIBRARY,
    BacktestConfig,
    StakingConfig,
    StrategyBacktester,
    WalkForwardPredictor,
    build_report,
    comparison_table,
)
from backend.strategy.odds import build_market_frame
from backend.strategy.reports import StrategyReport
from backend.strategy.robustness import (
    ConsistencySummary,
    SegmentResult,
    consistency_summary,
    render_segments,
    segment_analysis,
    segment_frame,
)
from backend.utils.exceptions import PipelineError
from backend.utils.logging import get_logger, safe_extra
from backend.utils.timeutils import utcnow

logger = get_logger(__name__, channel="model")

REPORT_FILENAME = "Phase6_Final_Report.md"

#: Display names for the four conclusions, as specified for the Phase 6
#: report. These are labels only — the decision function in
#: :mod:`backend.research.protocol` is frozen and is not touched by them.
VERDICT_LABELS: dict[Verdict, str] = {
    Verdict.EDGE_CONFIRMED: "EDGE_CONFIRMED",
    Verdict.NO_EDGE: "NO_EDGE_FOUND",
    Verdict.SEGMENT_ONLY: "SEGMENT_EDGE",
    Verdict.INSUFFICIENT_DATA: "INCONCLUSIVE",
}

#: The strategies the protocol backtests. EV thresholds per Phase 6.6.
BACKTEST_STRATEGIES = ("A_ev5", "B_ev10", "G_ev15")
BACKTEST_STAKING = ("flat", "kelly")


@dataclass(slots=True)
class ValidationRun:
    """Everything the report needs, and nothing that could change the verdict."""

    protocol: ValidationProtocol
    audit: DatasetAudit
    generated_at: str = field(default_factory=lambda: utcnow().isoformat())

    preflight: PreflightResult = field(default_factory=PreflightResult)
    baselines: pd.DataFrame = field(default_factory=pd.DataFrame)
    encompassing: EncompassingResult = field(default_factory=EncompassingResult)
    calibration: ReliabilityCurve = field(default_factory=ReliabilityCurve)
    calibration_error: float = 0.0
    strategy_reports: list[StrategyReport] = field(default_factory=list)
    primary: StrategyReport | None = None
    segments: list[SegmentResult] = field(default_factory=list)
    consistency: ConsistencySummary = field(default_factory=ConsistencySummary)
    verdict: VerdictResult | None = None
    predictions_rows: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "protocol": self.protocol.as_dict(),
            "audit": self.audit.as_dict(),
            "preflight": self.preflight.as_dict(),
            "baselines": self.baselines.to_dict(orient="records"),
            "calibration_error": round(self.calibration_error, 5),
            "encompassing": self.encompassing.as_dict(),
            "strategies": [report.metrics.as_dict() for report in self.strategy_reports],
            "consistency": self.consistency.as_dict(),
            "segments": [segment.as_dict() for segment in self.segments],
            "verdict": self.verdict.as_dict() if self.verdict else None,
            "notes": self.notes,
        }


def generate_predictions(
    session: Session,
    protocol: ValidationProtocol = PROTOCOL,
    *,
    retrain_months: int = 12,
    model_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Steps 4 and 5: walk-forward retraining, then out-of-sample scoring.

    Split out from :func:`run_validation` so the execution workflow can cache
    the result. Producing these rows is by far the most expensive part of the
    phase — hours on eight years of racing — and a crash during the cheap
    analysis that follows should not cost all of it.
    """
    predictor = WalkForwardPredictor(
        session,
        model_type=protocol.primary_model,
        model_params=model_params,
        embargo_days=protocol.embargo_days,
    )
    return predictor.run(
        history_start=protocol.train_start,
        test_start=protocol.test_start,
        test_end=protocol.test_end,
        retrain_months=retrain_months,
    )


def evaluate_predictions(
    predictions: pd.DataFrame,
    audit: DatasetAudit,
    protocol: ValidationProtocol = PROTOCOL,
    *,
    preflight_result: PreflightResult | None = None,
    allow_unready: bool = False,
    notes: list[str] | None = None,
) -> ValidationRun:
    """Step 6 onward: baselines, calibration, backtests, segments, verdict.

    Pure with respect to the database — everything it needs is in the frame it
    is handed. That is what lets the workflow resume here from cached
    predictions without re-reading a row.
    """
    run = ValidationRun(protocol=protocol, audit=audit)
    run.preflight = preflight_result or PreflightResult()
    run.notes = list(notes or [])
    run.predictions_rows = len(predictions)

    # The leakage gate needs a built feature matrix, so it runs here rather
    # than in pre-flight. A model trained on a leaked target would sail through
    # every earlier check and then report a spectacular, worthless edge.
    leakage = gate_no_leakage(predictions, _feature_columns(predictions))
    run.preflight.gates.append(leakage)
    if not leakage.passed and not allow_unready:
        raise PipelineError(f"validation stopped at pre-flight gate '{leakage.detail}'")

    market = build_market_frame(predictions, method="power")

    # --- model validation ---------------------------------------------------
    run.baselines = compare_baselines(market)
    run.encompassing = market_encompassing_test(market)
    run.calibration, run.calibration_error = _assess_calibration(market)

    # --- strategy backtests -------------------------------------------------
    for staking_method in BACKTEST_STAKING:
        for strategy_name in BACKTEST_STRATEGIES:
            config = BacktestConfig(
                strategy=STRATEGY_LIBRARY[strategy_name],
                staking=StakingConfig(method=staking_method),  # type: ignore[arg-type]
                execution=protocol.execution_config(),
            )
            report = build_report(StrategyBacktester(config).run(predictions))
            report.metrics.strategy = f"{strategy_name}/{staking_method}"
            run.strategy_reports.append(report)

            if strategy_name == protocol.primary_strategy and staking_method == protocol.primary_staking:
                run.primary = report

    # --- robustness ---------------------------------------------------------
    if run.primary is not None and not run.primary.equity.empty:
        primary_config = BacktestConfig(
            strategy=STRATEGY_LIBRARY[protocol.primary_strategy],
            staking=StakingConfig(method=protocol.primary_staking),  # type: ignore[arg-type]
            execution=protocol.execution_config(),
        )
        ledger = StrategyBacktester(primary_config).run(predictions).ledger
        run.segments = segment_analysis(ledger, predictions)
        run.consistency = consistency_summary(run.segments, nominal_threshold=protocol.min_t_statistic)

    # --- verdict, computed by the frozen rule -------------------------------
    metrics = run.primary.metrics if run.primary else None
    delta = None
    if not run.baselines.empty and {"forecaster", "race_log_loss"} <= set(run.baselines.columns):
        lookup = run.baselines.set_index("forecaster")["race_log_loss"].to_dict()
        if "model" in lookup and "market" in lookup:
            delta = float(lookup["model"] - lookup["market"])

    run.verdict = protocol.verdict(
        VerdictInputs(
            bets=metrics.bets if metrics else 0,
            roi=metrics.roi if metrics else 0.0,
            t_statistic=metrics.t_statistic if metrics else 0.0,
            profitable_year_fraction=run.consistency.profitable_year_fraction,
            years_covered=run.consistency.years_covered,
            passing_segments=tuple(run.consistency.passing_nominal),
            segment_t_statistics=tuple(
                segment.t_statistic
                for segment in run.segments
                if segment.label in run.consistency.passing_nominal
            ),
            model_vs_market_log_loss_delta=delta,
        )
    )
    logger.info(
        "validation complete",
        extra=safe_extra({"verdict": str(run.verdict.verdict), "bets": metrics.bets if metrics else 0}),
    )
    return run


def run_validation(
    session: Session,
    protocol: ValidationProtocol = PROTOCOL,
    *,
    allow_unready: bool = False,
    retrain_months: int = 12,
    model_params: dict[str, Any] | None = None,
    expected_fingerprint: str | None = None,
) -> ValidationRun:
    """Execute the frozen protocol, end to end.

    The pipeline is: audit → pre-flight gates → walk-forward prediction →
    baselines and encompassing → calibration → strategy backtests → segments →
    verdict. Each stage depends on the one before it, and the gates come first
    because every later stage is meaningless if they fail.

    ``allow_unready`` exists solely so the machinery can be exercised on
    synthetic data in tests. It stamps the run with a loud note; it does not
    make the output meaningful.
    """
    audit = audit_dataset(session)
    gates = preflight(audit, protocol, expected_fingerprint=expected_fingerprint)
    notes: list[str] = []

    if not gates.passed:
        reason = gates.reason
        if not allow_unready:
            raise PipelineError(
                f"validation stopped at pre-flight gate '{reason}'. "
                "Fix the blocking issue, or pass allow_unready=True to exercise "
                "the machinery on data that cannot support a conclusion."
            )
        notes.append(
            f"RUN ON A DATASET THAT FAILED A PRE-FLIGHT GATE — ({reason}). "
            "Results are a harness check, not findings."
        )
        logger.warning("validation run past a failed gate", extra=safe_extra({"gate": reason}))

    predictions = generate_predictions(
        session, protocol, retrain_months=retrain_months, model_params=model_params
    )
    return evaluate_predictions(
        predictions,
        audit,
        protocol,
        preflight_result=gates,
        allow_unready=allow_unready,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_markdown(run: ValidationRun) -> str:
    """The Phase 6 final report, in the eight sections the phase specifies."""
    protocol, audit, verdict = run.protocol, run.audit, run.verdict
    label = VERDICT_LABELS.get(verdict.verdict, "INCONCLUSIVE") if verdict else "INCONCLUSIVE"

    lines = [
        "# Phase 6 — final validation report",
        "",
        f"*Generated {run.generated_at}*",
        f"*Protocol `{protocol.version}` · fingerprint `{protocol.fingerprint}` · "
        f"registered {protocol.registered_at}*",
        "",
    ]
    if run.notes:
        lines += ["> ⚠️ " + note for note in run.notes] + [""]

    lines += [
        f"## Verdict: `{label}`",
        "",
        f"**{verdict.headline if verdict else 'not computed'}**",
        "",
    ]
    if verdict:
        lines += [f"- {reason}" for reason in verdict.reasons]
        lines += [
            "",
            "*Produced by the pre-registered decision function, which was fixed "
            "before any data existed and is applied mechanically. Nothing in this "
            "report can move it.*",
            "",
        ]

    # --- pre-flight --------------------------------------------------------
    lines += ["## 0. Pre-flight gates", "", "```", run.preflight.render(), "```", ""]

    # --- 1. dataset --------------------------------------------------------
    lines += [
        "## 1. Dataset",
        "",
        f"- Source: **{audit.source}**",
        f"- Span: **{audit.date_from} → {audit.date_to}** "
        f"({audit.years_covered} years, {audit.racing_days:,} racing days)",
        f"- {audit.races:,} races · {audit.runners:,} runners · {audit.horses:,} horses",
        f"- Odds coverage {audit.odds_coverage:.1%} · result coverage {audit.result_coverage:.1%}",
        f"- Out-of-sample rows scored: {run.predictions_rows:,}",
        f"- Readiness: **{'READY' if audit.readiness.ready else 'NOT READY'}**",
        "",
        "### Splits (pre-registered)",
        "",
        "| Split | From | To |",
        "|---|---|---|",
        f"| Train | {protocol.train_start} | {protocol.train_end} |",
        f"| Validate | {protocol.valid_start} | {protocol.valid_end} |",
        f"| Test (out of sample) | {protocol.test_start} | {protocol.test_end} |",
        "",
        f"Embargo between splits: **{protocol.embargo_days} days**.",
        "",
    ]
    if audit.readiness.blocking:
        lines += [f"  - blocking: {reason}" for reason in audit.readiness.blocking] + [""]

    # --- 2. model performance ---------------------------------------------
    lines += [
        "## 2. Model performance",
        "",
        "Race-level log loss on the out-of-sample window, against every baseline. "
        "Lower is better. The `market` row is the only comparison that decides "
        "anything — beating `uniform` or `random` is table stakes.",
        "",
    ]
    lines += _markdown_table(run.baselines)
    lines.append("")

    # --- 3. calibration ----------------------------------------------------
    lines += [
        "## 3. Probability calibration",
        "",
        "Ranking well is not enough: stakes are computed from the probability, so "
        "a 20% shout that wins 12% of the time turns a real edge into a real loss. "
        "Measured on the test window, with the calibrator fitted on an earlier slice.",
        "",
        f"**Expected calibration error: {run.calibration_error:.4f}**",
        "",
    ]
    if run.calibration.predicted:
        lines += ["```", run.calibration.render(), "```", ""]
    else:
        lines += ["*(no calibration curve — no predictions to bin)*", ""]

    # --- 4. strategy performance ------------------------------------------
    lines += [
        "## 4. Betting strategy performance",
        "",
        f"Identical predictions replayed through every strategy, at "
        f"{protocol.slippage:.0%} slippage and {protocol.commission:.0%} commission.",
        "",
    ]
    if run.strategy_reports:
        lines += _markdown_table(comparison_table(run.strategy_reports))
    else:
        lines.append("*(no strategy placed a bet)*")
    lines.append("")

    if run.primary is not None:
        lines += [
            f"### Primary hypothesis — {protocol.primary_strategy} / {protocol.primary_staking}",
            "",
            "The one strategy named in advance. Every other row above is a "
            "secondary test and carries the corrected significance bar.",
            "",
            "```",
            run.primary.render(sparkline_width=50),
            "```",
            "",
        ]

    # --- 5. year by year ---------------------------------------------------
    lines += [
        "## 5. Year-by-year results",
        "",
        "An edge that lives in one year is an artefact. The protocol requires "
        f"profit in at least {protocol.min_profitable_year_fraction:.0%} of years.",
        "",
    ]
    year_segments = render_segments(run.segments, dimension="year")
    if "(no segments)" in year_segments:
        lines += ["*(too few bets to break down by year)*", ""]
    else:
        lines += ["```", year_segments, "```", ""]

    # --- 6. segments -------------------------------------------------------
    lines += ["## 6. Segment analysis", "", "```", run.consistency.render(), "```", ""]
    for dimension, title in (
        ("code", "By code"),
        ("handicap", "Handicap versus non-handicap"),
        ("odds_band", "By price band"),
        ("field_size_band", "By field size"),
    ):
        rendered = render_segments(run.segments, dimension=dimension)
        if "(no segments)" in rendered:
            continue
        lines += [f"### {title}", "", "```", rendered, "```", ""]

    # --- 7. significance ---------------------------------------------------
    primary_metrics = run.primary.metrics if run.primary else None
    lines += [
        "## 7. Statistical significance",
        "",
        "| Quantity | Value | Required |",
        "|---|---:|---:|",
        f"| Bets (primary) | {primary_metrics.bets if primary_metrics else 0:,} | {protocol.min_bets:,} |",
        f"| ROI (primary) | {primary_metrics.roi if primary_metrics else 0.0:+.2%} | > 0 |",
        f"| t-statistic (primary) | {primary_metrics.t_statistic if primary_metrics else 0.0:+.2f} | "
        f"{protocol.min_t_statistic:.2f} |",
        f"| Profitable years | {run.consistency.profitable_year_fraction:.0%} | "
        f"{protocol.min_profitable_year_fraction:.0%} |",
        f"| Secondary-test bar (Bonferroni, {protocol.secondary_tests} tests) | "
        f"{run.consistency.corrected_threshold:.2f} | — |",
        "",
        "### Does the model know anything the price does not?",
        "",
        f"**{run.encompassing.verdict}**",
        "",
        "Encompassing regression `logit(win) ~ logit(p_market) + logit(p_model)`. "
        "The market's forecast is already in the equation, so the model's "
        "coefficient measures what it adds *conditional on the price being known*.",
        "",
        "| Term | Coefficient | z |",
        "|---|---:|---:|",
        f"| market | {run.encompassing.market_coefficient:+.4f} | {run.encompassing.market_z:+.2f} |",
        f"| model | {run.encompassing.model_coefficient:+.4f} | {run.encompassing.model_z:+.2f} |",
        "",
        f"Log loss: market only {run.encompassing.log_loss_market_only:.5f} → "
        f"combined {run.encompassing.log_loss_combined:.5f} "
        f"(n = {run.encompassing.rows:,})",
        "",
    ]

    # --- 8. final verdict --------------------------------------------------
    lines += [
        "## 8. Final verdict",
        "",
        f"# `{label}`",
        "",
        f"{verdict.headline if verdict else 'not computed'}",
        "",
    ]
    if verdict:
        lines += ["Reasoning, in the order the frozen rule evaluates it:", ""]
        lines += [f"{index}. {reason}" for index, reason in enumerate(verdict.reasons, start=1)]
        lines.append("")

    lines += [
        "### Limitations",
        "",
        "- Backtests are an upper bound: no model of account restriction, and "
        "liquidity is a flat cap rather than a real book.",
        "- Bets settle at the last captured pre-race price less slippage; real "
        "execution is worse and prices move against a value bettor specifically.",
        "- Walk-forward retrains annually. More frequent retraining would change "
        "results in an unknown direction.",
        "- The out-of-sample window is scored once. It is one draw from a noisy "
        "distribution, which is what the t-statistic is there to express.",
        "",
    ]
    return "\n".join(lines)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric columns that a model could actually have been fed.

    The exclusion list is :data:`LABEL_COLUMNS`, which is authoritative and
    lives next to the query that produces those columns, rather than a set
    hand-written here — a hand-written list is exactly how ``is_winner`` leaked
    into the feature matrix in Phase 4.

    The prediction frame also carries outcomes for settlement and the model's
    own probability. Both separate winners perfectly by construction, so
    checking them would raise a false alarm on every single run.
    """
    excluded = {
        *LABEL_COLUMNS,
        "position",
        "model_probability",
        "market_probability",
        "race_id",
        "horse_id",
        "race_date",
    }
    return [
        str(column)
        for column in frame.columns
        if str(column) not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _assess_calibration(frame: pd.DataFrame) -> tuple[ReliabilityCurve, float]:
    """Do the out-of-sample probabilities mean what they say?

    A model can rank runners well and still be badly calibrated, and staking is
    computed from the probability, not the rank — so a 20% shout that wins 12%
    of the time turns a real edge into a real loss. This is measured on the test
    window, after calibration was fitted on an earlier slice.
    """
    if frame.empty or "model_probability" not in frame.columns:
        return ReliabilityCurve(), 0.0
    truth = frame["won"].to_numpy(dtype=float)
    probabilities = frame["model_probability"].to_numpy(dtype=float)
    return (
        reliability_curve(truth, probabilities),
        float(expected_calibration_error(truth, probabilities)),
    )


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["*(no data)*"]
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in record) + " |"
        for record in frame.itertuples(index=False)
    ]
    return [header, divider, *rows]


def write_report(run: ValidationRun, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(run), encoding="utf-8")
    logger.info("validation report written", extra={"path": str(target)})
    return target


def segments_to_frame(run: ValidationRun) -> pd.DataFrame:
    return segment_frame(run.segments)


__all__ = [
    "BACKTEST_STAKING",
    "BACKTEST_STRATEGIES",
    "REPORT_FILENAME",
    "VERDICT_LABELS",
    "ValidationRun",
    "evaluate_predictions",
    "generate_predictions",
    "render_markdown",
    "run_validation",
    "segments_to_frame",
    "write_report",
]
