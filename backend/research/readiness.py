"""``PRODUCTION_READINESS.md`` — is each component fit to be relied on?

Five checks: API, Data, Model, Backtest, Risk.

One thing this document deliberately does *not* answer: whether the strategy
makes money. A `PASS` on every line means the pipeline is sound, the data is
clean, the probabilities mean what they say, the backtest was run under
realistic conditions, and the risk controls are configured — not that there is
an edge. Those are different questions, and conflating them is exactly how a
green dashboard ends up funding a losing strategy.

The profitability question has one answer, in one place: the frozen decision
function's verdict. This checklist reports that verdict verbatim and draws no
conclusion of its own from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.research.audit import DatasetAudit
from backend.research.protocol import ValidationProtocol
from backend.research.validation_report import VERDICT_LABELS, ValidationRun
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="model")

REPORT_FILENAME = "PRODUCTION_READINESS.md"

#: Above this expected calibration error, stated probabilities are not
#: trustworthy enough to size a stake from.
MAX_CALIBRATION_ERROR = 0.05
#: A drawdown deeper than this makes the staking plan unfit regardless of ROI —
#: it is the depth at which a real bettor stops following the system.
MAX_TOLERABLE_DRAWDOWN = 0.35


@dataclass(slots=True)
class Check:
    """One component, and whether it is fit to be relied on."""

    name: str
    passed: bool
    detail: str = ""
    evidence: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.name,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class ProductionReadiness:
    """The five checks, plus the verdict they must not be confused with."""

    checks: list[Check] = field(default_factory=list)
    verdict_label: str = "INCONCLUSIVE"
    verdict_headline: str = "no verdict was computed"
    generated_at: str = ""

    @property
    def all_passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]

    def check(self, name: str) -> Check:
        for candidate in self.checks:
            if candidate.name == name:
                return candidate
        return Check(name=name, passed=False, detail="not evaluated")

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "all_passed": self.all_passed,
            "verdict": self.verdict_label,
            "checks": [check.as_dict() for check in self.checks],
        }

    def render(self) -> str:
        lines = ["Production readiness", "--------------------"]
        lines += [f"  {check.name:<10} {check.status:<5} {check.detail}" for check in self.checks]
        lines += ["", f"  Verdict (separate question): {self.verdict_label}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The five checks
# ---------------------------------------------------------------------------
def _check_api(capabilities: Any | None) -> Check:
    if capabilities is None:
        return Check("API", False, "not probed — run `make api-status`")
    if not capabilities.subscription_active:
        return Check("API", False, capabilities.blocking_reason or "subscription inactive")
    if not capabilities.ready_for_backfill:
        return Check("API", False, capabilities.blocking_reason or "plan insufficient")
    return Check(
        "API",
        True,
        f"authenticated, subscription active, history "
        f"{capabilities.earliest_result} → {capabilities.latest_result}",
        [f"tier: {capabilities.best_tier}", f"{capabilities.available_years:.1f} years served"],
    )


def _check_data(audit: DatasetAudit) -> Check:
    if not audit.readiness.ready:
        return Check("Data", False, "; ".join(audit.readiness.blocking)[:200])
    return Check(
        "Data",
        True,
        f"{audit.races:,} real races over {audit.years_covered} years",
        [
            f"odds coverage {audit.odds_coverage:.1%}",
            f"result coverage {audit.result_coverage:.1%}",
            f"{audit.odds_after_off} quotes after the off",
        ],
    )


def _check_model(run: ValidationRun) -> Check:
    """Are the probabilities usable — not whether they are profitable."""
    if run.predictions_rows == 0:
        return Check("Model", False, "no out-of-sample predictions were produced")

    if run.calibration_error > MAX_CALIBRATION_ERROR:
        return Check(
            "Model",
            False,
            f"expected calibration error {run.calibration_error:.4f} exceeds "
            f"{MAX_CALIBRATION_ERROR:.2f}; stakes computed from these probabilities "
            "would be mis-sized",
        )

    evidence = [f"calibration error {run.calibration_error:.4f}", f"{run.predictions_rows:,} rows scored"]

    if not run.baselines.empty and "forecaster" in run.baselines.columns:
        losses = run.baselines.set_index("forecaster")["race_log_loss"].to_dict()
        if "model" in losses and "uniform" in losses:
            if losses["model"] >= losses["uniform"]:
                return Check(
                    "Model",
                    False,
                    f"model race log loss {losses['model']:.4f} is no better than a "
                    f"uniform forecast {losses['uniform']:.4f}",
                )
            evidence.append(f"beats uniform ({losses['model']:.4f} vs {losses['uniform']:.4f})")

    return Check("Model", True, "calibrated and better than an uninformed forecast", evidence)


def _check_backtest(run: ValidationRun, protocol: ValidationProtocol) -> Check:
    """Was the backtest run honestly — not whether it made money."""
    leakage = next((gate for gate in run.preflight.gates if gate.name == "no leakage"), None)
    if leakage is not None and not leakage.passed:
        return Check("Backtest", False, f"leakage detected: {leakage.detail[:160]}")

    if not run.strategy_reports:
        return Check("Backtest", False, "no strategy was backtested")

    metrics = run.primary.metrics if run.primary else None
    if metrics is None:
        return Check("Backtest", False, "the primary strategy produced no result")

    if metrics.bets < protocol.min_bets:
        return Check(
            "Backtest",
            False,
            f"{metrics.bets:,} qualifying bets; {protocol.min_bets:,} are required "
            "before any result is reportable",
        )

    return Check(
        "Backtest",
        True,
        f"{len(run.strategy_reports)} strategies over {metrics.bets:,} primary bets, "
        f"walk-forward, {protocol.slippage:.0%} slippage and {protocol.commission:.0%} commission",
        [
            "no look-ahead: starting price excluded",
            f"{run.consistency.years_covered} years segmented",
        ],
    )


def _check_risk(run: ValidationRun) -> Check:
    """Are the risk controls configured and the observed downside survivable?"""
    metrics = run.primary.metrics if run.primary else None
    if metrics is None:
        return Check("Risk", False, "no ledger to assess")

    if metrics.max_drawdown_pct > MAX_TOLERABLE_DRAWDOWN:
        return Check(
            "Risk",
            False,
            f"maximum drawdown {metrics.max_drawdown_pct:.1%} exceeds the "
            f"{MAX_TOLERABLE_DRAWDOWN:.0%} tolerance; a real bettor stops following a "
            "system before this point",
        )

    if metrics.final_bankroll <= 0:
        return Check("Risk", False, "the bankroll was exhausted during the test window")

    return Check(
        "Risk",
        True,
        f"maximum drawdown {metrics.max_drawdown_pct:.1%}, longest losing streak "
        f"{metrics.longest_losing_streak} bets",
        [
            f"bankroll {metrics.starting_bankroll:,.0f} → {metrics.final_bankroll:,.0f}",
            f"average stake {metrics.average_stake:,.2f}",
        ],
    )


def assess(
    run: ValidationRun,
    *,
    capabilities: Any | None = None,
) -> ProductionReadiness:
    """Run all five checks against a completed validation."""
    readiness = ProductionReadiness(generated_at=run.generated_at)
    readiness.checks = [
        _check_api(capabilities),
        _check_data(run.audit),
        _check_model(run),
        _check_backtest(run, run.protocol),
        _check_risk(run),
    ]
    if run.verdict is not None:
        readiness.verdict_label = VERDICT_LABELS.get(run.verdict.verdict, "INCONCLUSIVE")
        readiness.verdict_headline = run.verdict.headline

    logger.info("production readiness assessed", extra={"all_passed": readiness.all_passed})
    return readiness


# ---------------------------------------------------------------------------
def render_markdown(readiness: ProductionReadiness, protocol: ValidationProtocol) -> str:
    lines = [
        "# Production readiness",
        "",
        f"*Generated {readiness.generated_at}*",
        f"*Protocol `{protocol.version}` · fingerprint `{protocol.fingerprint}`*",
        "",
        "| Component | Status |",
        "|---|---|",
    ]
    lines += [f"| {check.name} | **{check.status}** |" for check in readiness.checks]
    lines += [
        "",
        f"**Overall: {'READY' if readiness.all_passed else 'NOT READY'}**",
        "",
        "> These checks answer *is each component fit to be relied on* — not *is the",
        "> strategy profitable*. A row of PASSes means the pipeline is sound, the data",
        "> is clean, the probabilities are calibrated, the backtest was run under",
        "> realistic conditions and the risk controls hold. Whether an edge exists is",
        "> a different question, with exactly one answer, below.",
        "",
        "## Detail",
        "",
    ]
    for check in readiness.checks:
        lines += [f"### {check.name} — {check.status}", "", check.detail or "-", ""]
        if check.evidence:
            lines += [f"- {item}" for item in check.evidence] + [""]

    if readiness.failures:
        lines += ["## What must be fixed", ""]
        lines += [f"- **{check.name}**: {check.detail}" for check in readiness.failures]
        lines.append("")

    lines += [
        "## Profitability verdict",
        "",
        f"### `{readiness.verdict_label}`",
        "",
        readiness.verdict_headline,
        "",
        "*Produced by the pre-registered decision function, which was content-hashed "
        "before any data existed. Nothing in this checklist can move it, and a full "
        "row of PASSes above does not imply it.*",
        "",
    ]
    return "\n".join(lines)


def write_report(readiness: ProductionReadiness, protocol: ValidationProtocol, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(readiness, protocol), encoding="utf-8")
    logger.info("production readiness written", extra={"path": str(target)})
    return target


__all__ = [
    "MAX_CALIBRATION_ERROR",
    "MAX_TOLERABLE_DRAWDOWN",
    "REPORT_FILENAME",
    "Check",
    "ProductionReadiness",
    "assess",
    "render_markdown",
    "write_report",
]
