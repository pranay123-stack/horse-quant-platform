"""The gates that must pass before a validation is allowed to run.

Phase 6 names five conditions that stop the study outright: synthetic data, a
protocol fingerprint mismatch, insufficient data, missing odds coverage, and
detected leakage. They were already enforced in scattered places — the audit's
readiness check, ``assert_protocol_unchanged``, ``ModelTrainer``. This module
collects them into one named, ordered list so the runner can say *which* gate
stopped it rather than surfacing whichever exception happened to fire first.

That distinction matters operationally. "Validation failed" sends someone
reading tracebacks. "Gate 2 of 6 — data source — failed: the database contains
synthetic races" sends them to fix the right thing.

Gates are ordered cheapest-first, and evaluation stops at the first failure.
There is no point auditing eight years of odds coverage if the protocol has been
edited, because the run is void either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backend.ml.dataset import detect_leaky_features
from backend.research.audit import (
    MIN_ODDS_COVERAGE,
    MIN_RESULT_COVERAGE,
    MIN_YEARS,
    DatasetAudit,
)
from backend.research.protocol import PROTOCOL, ValidationProtocol

#: The minimum rows worth attempting a study on. Far below this the splits are
#: empty rather than merely noisy.
MIN_ROWS_FOR_STUDY = 5_000


@dataclass(slots=True)
class Gate:
    """One precondition, and whether the data met it."""

    name: str
    passed: bool
    detail: str = ""

    @property
    def symbol(self) -> str:
        return "PASS" if self.passed else "STOP"

    def as_dict(self) -> dict[str, Any]:
        return {"gate": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(slots=True)
class PreflightResult:
    """The outcome of every gate that was evaluated."""

    gates: list[Gate] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(gate.passed for gate in self.gates)

    @property
    def failure(self) -> Gate | None:
        for gate in self.gates:
            if not gate.passed:
                return gate
        return None

    @property
    def reason(self) -> str:
        failure = self.failure
        return f"{failure.name}: {failure.detail}" if failure else ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed_gate": self.failure.name if self.failure else None,
            "gates": [gate.as_dict() for gate in self.gates],
        }

    def render(self) -> str:
        lines = ["Pre-flight gates", "----------------"]
        lines += [f"  [{gate.symbol}] {gate.name:<22} {gate.detail}" for gate in self.gates]
        if not self.passed:
            lines += ["", f"  STOPPED at '{self.failure.name if self.failure else '?'}'"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------
def gate_protocol_integrity(protocol: ValidationProtocol, expected_fingerprint: str | None) -> Gate:
    """Has the pre-registered plan been edited since it was registered?

    A protocol that changed after results were seen is not a protocol. When no
    expected fingerprint is supplied there is nothing to compare against, which
    is reported honestly rather than treated as a pass in disguise.
    """
    if expected_fingerprint is None:
        return Gate(
            "protocol integrity",
            True,
            f"fingerprint {protocol.fingerprint} (not pinned by the caller)",
        )
    if protocol.fingerprint != expected_fingerprint:
        return Gate(
            "protocol integrity",
            False,
            f"protocol fingerprint is {protocol.fingerprint}, expected "
            f"{expected_fingerprint} — the frozen analysis plan has been edited, "
            "so any result from this run is uninterpretable",
        )
    return Gate("protocol integrity", True, f"fingerprint {protocol.fingerprint} unchanged")


def gate_data_source(audit: DatasetAudit) -> Gate:
    """Real data, or nothing."""
    if audit.source == "synthetic":
        return Gate(
            "data source",
            False,
            "the database contains only synthetic races — generated data cannot "
            "validate anything about real markets",
        )
    if audit.source == "mixed":
        return Gate(
            "data source",
            False,
            "the database mixes real and synthetic races; the synthetic rows must "
            "be purged before a study can run",
        )
    if audit.source == "empty":
        return Gate("data source", False, "the database contains no races")
    return Gate("data source", True, f"source is '{audit.source}'")


def gate_sufficient_data(audit: DatasetAudit) -> Gate:
    """Enough history, and enough of it, to support the pre-registered splits."""
    complete_years = [row for row in audit.coverage if row.racing_days >= 200]
    if len(complete_years) < MIN_YEARS:
        return Gate(
            "sufficient data",
            False,
            f"{len(complete_years)} year(s) with a full racing calendar; the protocol requires {MIN_YEARS}+",
        )
    if audit.runners < MIN_ROWS_FOR_STUDY:
        return Gate(
            "sufficient data",
            False,
            f"only {audit.runners:,} runner rows; {MIN_ROWS_FOR_STUDY:,} is the floor",
        )
    return Gate(
        "sufficient data",
        True,
        f"{len(complete_years)} full years, {audit.runners:,} runner rows",
    )


def gate_odds_coverage(audit: DatasetAudit) -> Gate:
    """A value study on a thin price subset measures the subset, not the market."""
    if audit.odds_coverage < MIN_ODDS_COVERAGE:
        return Gate(
            "odds coverage",
            False,
            f"only {audit.odds_coverage:.1%} of runners have a price "
            f"(need {MIN_ODDS_COVERAGE:.0%}); the priced subset is not a random sample",
        )
    if audit.result_coverage < MIN_RESULT_COVERAGE:
        return Gate(
            "odds coverage",
            False,
            f"only {audit.result_coverage:.1%} of races have results (need {MIN_RESULT_COVERAGE:.0%})",
        )
    return Gate(
        "odds coverage",
        True,
        f"{audit.odds_coverage:.1%} priced, {audit.result_coverage:.1%} settled",
    )


def gate_timestamp_integrity(audit: DatasetAudit) -> Gate:
    """No price may be timestamped after the race it applies to."""
    if audit.odds_after_off:
        return Gate(
            "timestamp integrity",
            False,
            f"{audit.odds_after_off:,} quotes are recorded after the off — a "
            "post-race price is not something anyone could have bet at",
        )
    return Gate("timestamp integrity", True, "no quotes recorded after the off")


def gate_no_leakage(frame: pd.DataFrame, feature_names: list[str], target: str = "won") -> Gate:
    """Does any single feature separate winners almost perfectly?

    Caught a real bug once already: ``is_winner`` reached the feature matrix as
    a boolean and every model scored a perfect AUC. A denylist of names would
    never have found it; this finds it by behaviour.
    """
    if frame.empty or not feature_names:
        return Gate("no leakage", True, "nothing to check")

    suspects = detect_leaky_features(frame, feature_names, target)
    if suspects:
        return Gate(
            "no leakage",
            False,
            "target leakage detected: "
            + ", ".join(str(suspect) for suspect in suspects[:5])
            + " — these separate winners almost perfectly and cannot be inputs",
        )
    return Gate("no leakage", True, f"{len(feature_names)} features, none suspicious")


# ---------------------------------------------------------------------------
def gate_api_subscription(capabilities: Any | None) -> Gate:
    """Is the plan live, and does it serve what the protocol needs?

    ``None`` means the caller did not probe — reported as such rather than
    treated as a pass, because an unchecked gate is not a passed gate.
    """
    if capabilities is None:
        return Gate("api subscription", True, "not probed (offline check)")
    if not capabilities.authenticated:
        return Gate("api subscription", False, f"authentication failed: {capabilities.message}")
    if not capabilities.subscription_active:
        return Gate("api subscription", False, f"subscription inactive: {capabilities.message}")
    return Gate("api subscription", True, "authenticated, subscription active")


def gate_api_endpoints(capabilities: Any | None) -> Gate:
    """Results and history are the two the study cannot proceed without."""
    if capabilities is None:
        return Gate("api endpoints", True, "not probed (offline check)")

    missing = [name for name in ("results", "historical") if not capabilities.probe(name).available]
    if missing:
        return Gate("api endpoints", False, f"plan does not serve: {', '.join(missing)}")
    served = [
        name for name in ("racecards", "results", "odds", "historical") if capabilities.probe(name).available
    ]
    return Gate("api endpoints", True, f"available: {', '.join(served)}")


def gate_history_range(capabilities: Any | None) -> Gate:
    """Does the served window actually cover the pre-registered study?"""
    if capabilities is None:
        return Gate("history range", True, "not probed (offline check)")
    if not capabilities.enough_for_protocol:
        return Gate("history range", False, capabilities.blocking_reason or "insufficient history")
    return Gate(
        "history range",
        True,
        f"{capabilities.earliest_result} → {capabilities.latest_result} "
        f"({capabilities.available_years:.1f} years)",
    )


def preflight(
    audit: DatasetAudit,
    protocol: ValidationProtocol = PROTOCOL,
    *,
    expected_fingerprint: str | None = None,
) -> PreflightResult:
    """Run the cheap gates in order, stopping at the first failure.

    The leakage gate is not here: it needs a built feature matrix, so it runs
    later via :func:`gate_no_leakage` once the dataset exists.
    """
    result = PreflightResult()
    checks = (
        lambda: gate_protocol_integrity(protocol, expected_fingerprint),
        lambda: gate_data_source(audit),
        lambda: gate_sufficient_data(audit),
        lambda: gate_odds_coverage(audit),
        lambda: gate_timestamp_integrity(audit),
    )
    for check in checks:
        gate = check()
        result.gates.append(gate)
        if not gate.passed:
            break
    return result


def full_preflight(
    audit: DatasetAudit,
    protocol: ValidationProtocol = PROTOCOL,
    *,
    capabilities: Any | None = None,
    expected_fingerprint: str | None = None,
) -> PreflightResult:
    """Every gate: the API first, then the data already downloaded.

    API gates come first because they are the cheapest and the most likely to
    fail — there is no point auditing a database that could not have been filled
    in the first place.
    """
    result = PreflightResult()
    checks = (
        lambda: gate_api_subscription(capabilities),
        lambda: gate_api_endpoints(capabilities),
        lambda: gate_history_range(capabilities),
        lambda: gate_protocol_integrity(protocol, expected_fingerprint),
        lambda: gate_data_source(audit),
        lambda: gate_sufficient_data(audit),
        lambda: gate_odds_coverage(audit),
        lambda: gate_timestamp_integrity(audit),
    )
    for check in checks:
        gate = check()
        result.gates.append(gate)
        if not gate.passed:
            break
    return result


__all__ = [
    "MIN_ROWS_FOR_STUDY",
    "Gate",
    "PreflightResult",
    "full_preflight",
    "gate_api_endpoints",
    "gate_api_subscription",
    "gate_data_source",
    "gate_history_range",
    "gate_no_leakage",
    "gate_odds_coverage",
    "gate_protocol_integrity",
    "gate_sufficient_data",
    "gate_timestamp_integrity",
    "preflight",
]
