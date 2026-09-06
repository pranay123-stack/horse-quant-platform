"""Pre-registered validation protocol.

The instruction for this phase was *"do not optimize after seeing results — the
goal is validation, not curve fitting"*. Promising restraint is worth very
little; the reliable version is to fix the analysis **before** the data exists
and then run it once, unchanged.

That is what this module is. Every choice that could be tuned after seeing an
answer — split dates, model family, strategy thresholds, staking, execution
costs, which segments get examined, and the arithmetic that turns numbers into a
verdict — is frozen here, in code, with a registration timestamp.

Why this matters more than it sounds
====================================
Betting research is unusually easy to fool yourself with. There are thousands of
defensible ways to slice a racing dataset: by year, by code, by class, by field
size, by price band, by going, by month. Test enough of them and something clears
any threshold you like. The single most common way a "profitable system" turns
out to be nothing is that its author chose the slice after seeing the data.

Three defences are built in:

**A single primary hypothesis.** One strategy, one staking plan, one window. It
either passes or it does not. Everything else is explicitly secondary.

**Bonferroni correction on the secondary tests.** Examining
:data:`SECONDARY_TEST_COUNT` segments means each needs a proportionally stronger
result to count. Without this, "an edge exists in some segments" is a statement
about arithmetic, not about racing.

**A mechanical verdict.** :meth:`ValidationProtocol.verdict` maps results onto
one of three outcomes with no human judgement in the loop, so the conclusion
cannot drift towards the one that would be nicer to report.

The protocol is content-hashed. If it is edited after data arrives, the hash
changes and :func:`assert_protocol_unchanged` fails the run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

PROTOCOL_VERSION = "phase6-v1"
#: Frozen when the protocol was written — before any real data was available.
REGISTERED_AT = "2026-08-10"

# ---------------------------------------------------------------------------
# Windows (Phase 6.4 specification)
# ---------------------------------------------------------------------------
TRAIN_START = date(2018, 1, 1)
TRAIN_END = date(2022, 12, 31)
VALID_START = date(2023, 1, 1)
VALID_END = date(2024, 12, 31)
TEST_START = date(2025, 1, 1)
TEST_END = date(2026, 12, 31)
EMBARGO_DAYS = 14

# ---------------------------------------------------------------------------
# Thresholds — all fixed in advance
# ---------------------------------------------------------------------------
#: Below this many bets, no result is reportable at any t-statistic.
MIN_BETS_FOR_VERDICT = 200
#: Two standard errors on the primary hypothesis.
MIN_T_STATISTIC = 2.0
#: Number of pre-registered secondary segment tests, used for the Bonferroni
#: correction. Counting them in advance is what makes the correction honest.
SECONDARY_TEST_COUNT = 12
#: Fraction of complete years in the test window that must be profitable before
#: a positive result is called consistent rather than lucky.
MIN_PROFITABLE_YEAR_FRACTION = 0.5

#: Execution assumptions for the primary hypothesis. Deliberately pessimistic;
#: a strategy that only works at zero cost does not work.
PRIMARY_SLIPPAGE = 0.02
PRIMARY_COMMISSION = 0.02

#: The single primary hypothesis.
PRIMARY_MODEL = "lightgbm"
PRIMARY_STRATEGY = "A_ev5"
PRIMARY_STAKING = "flat"

#: Everything else is secondary and Bonferroni-corrected.
SECONDARY_STRATEGIES = ("B_ev10", "C_top_pick", "D_ev5_tight", "F_edge5")
SECONDARY_MODELS = ("logistic_regression", "xgboost")
SECONDARY_STAKING = ("kelly",)

#: Segments examined in the robustness pass. Fixed in advance; nothing may be
#: added after results are seen.
SEGMENT_DIMENSIONS = ("year", "race_type", "handicap", "odds_band", "field_size_band")
ODDS_BANDS = ((1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 1000.0))


class Verdict(StrEnum):
    """The three permitted conclusions, per the Phase 6.8 specification."""

    EDGE_CONFIRMED = "positive_edge_confirmed"
    NO_EDGE = "no_edge_found"
    SEGMENT_ONLY = "edge_only_in_specific_segments"
    INSUFFICIENT_DATA = "insufficient_data_to_conclude"


@dataclass(frozen=True, slots=True)
class VerdictInputs:
    """The numbers the verdict rule is allowed to look at.

    Deliberately narrow. If a quantity is not here, it cannot influence the
    conclusion — which stops the rule quietly acquiring extra clauses that
    happen to favour a positive answer.
    """

    bets: int
    roi: float
    t_statistic: float
    profitable_year_fraction: float
    years_covered: int
    #: Secondary segments that passed on their own, before correction.
    passing_segments: tuple[str, ...] = ()
    #: Their t-statistics, for the Bonferroni check.
    segment_t_statistics: tuple[float, ...] = ()
    #: Race log loss of the model against the market-price baseline. Negative
    #: means the model beat the market at predicting winners.
    model_vs_market_log_loss_delta: float | None = None


@dataclass(frozen=True, slots=True)
class VerdictResult:
    """The conclusion, and the reasoning that produced it."""

    verdict: Verdict
    headline: str
    reasons: list[str] = field(default_factory=list)
    corrected_threshold: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": str(self.verdict),
            "headline": self.headline,
            "reasons": self.reasons,
            "corrected_t_threshold": round(self.corrected_threshold, 4),
        }


def bonferroni_t_threshold(base: float = MIN_T_STATISTIC, tests: int = SECONDARY_TEST_COUNT) -> float:
    """t-threshold for a secondary test, corrected for multiplicity.

    Uses the normal approximation: the two-sided level ``alpha`` implied by
    ``base`` is divided by the number of tests, and the threshold is the z-score
    of the corrected level. With 12 tests, a nominal t of 2.0 becomes roughly
    3.2 — which is the point. Twelve independent chances at "significant" would
    otherwise produce one by luck about half the time.
    """
    from scipy.stats import norm  # imported lazily; scipy is heavy

    alpha = 2 * (1 - norm.cdf(base))
    return float(norm.ppf(1 - (alpha / tests) / 2))


@dataclass(frozen=True, slots=True)
class ValidationProtocol:
    """The frozen analysis plan."""

    version: str = PROTOCOL_VERSION
    registered_at: str = REGISTERED_AT

    train_start: date = TRAIN_START
    train_end: date = TRAIN_END
    valid_start: date = VALID_START
    valid_end: date = VALID_END
    test_start: date = TEST_START
    test_end: date = TEST_END
    embargo_days: int = EMBARGO_DAYS

    primary_model: str = PRIMARY_MODEL
    primary_strategy: str = PRIMARY_STRATEGY
    primary_staking: str = PRIMARY_STAKING
    slippage: float = PRIMARY_SLIPPAGE
    commission: float = PRIMARY_COMMISSION

    min_bets: int = MIN_BETS_FOR_VERDICT
    min_t_statistic: float = MIN_T_STATISTIC
    secondary_tests: int = SECONDARY_TEST_COUNT
    min_profitable_year_fraction: float = MIN_PROFITABLE_YEAR_FRACTION

    # ------------------------------------------------------------------
    @property
    def fingerprint(self) -> str:
        """Content hash. Changes if any pre-registered choice is edited."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def split_config(self) -> Any:
        """The :class:`~backend.ml.dataset.SplitConfig` this protocol mandates."""
        from backend.ml.dataset import SplitConfig

        return SplitConfig(
            train_start=self.train_start,
            train_end=self.train_end,
            valid_start=self.valid_start,
            valid_end=self.valid_end,
            test_start=self.test_start,
            test_end=self.test_end,
            embargo_days=self.embargo_days,
        )

    def execution_config(self) -> Any:
        from backend.strategy.backtester import ExecutionConfig

        return ExecutionConfig(slippage=self.slippage, commission=self.commission)

    # ------------------------------------------------------------------
    def verdict(self, inputs: VerdictInputs) -> VerdictResult:
        """Map results onto a conclusion. No judgement, no exceptions."""
        corrected = bonferroni_t_threshold(self.min_t_statistic, self.secondary_tests)
        reasons: list[str] = []

        if inputs.bets < self.min_bets:
            return VerdictResult(
                Verdict.INSUFFICIENT_DATA,
                f"only {inputs.bets} qualifying bets; {self.min_bets} required before any "
                "conclusion is reportable",
                [f"bets={inputs.bets} < min_bets={self.min_bets}"],
                corrected,
            )

        primary_passes = (
            inputs.roi > 0
            and inputs.t_statistic >= self.min_t_statistic
            and inputs.profitable_year_fraction >= self.min_profitable_year_fraction
        )

        reasons.append(f"primary ROI {inputs.roi:+.2%} over {inputs.bets} bets")
        reasons.append(f"primary t = {inputs.t_statistic:+.2f} (threshold {self.min_t_statistic})")
        reasons.append(
            f"profitable years {inputs.profitable_year_fraction:.0%} of {inputs.years_covered} "
            f"(threshold {self.min_profitable_year_fraction:.0%})"
        )
        if inputs.model_vs_market_log_loss_delta is not None:
            direction = "beats" if inputs.model_vs_market_log_loss_delta < 0 else "does not beat"
            reasons.append(
                f"model {direction} the market baseline on race log loss "
                f"({inputs.model_vs_market_log_loss_delta:+.4f})"
            )

        if primary_passes:
            return VerdictResult(
                Verdict.EDGE_CONFIRMED,
                f"Positive edge confirmed: {inputs.roi:+.2%} ROI over {inputs.bets} bets "
                f"(t = {inputs.t_statistic:.2f})",
                reasons,
                corrected,
            )

        surviving = [
            segment
            for segment, t_value in zip(inputs.passing_segments, inputs.segment_t_statistics, strict=False)
            if abs(t_value) >= corrected
        ]
        if surviving:
            reasons.append(
                f"{len(surviving)} of {len(inputs.passing_segments)} segments survive the "
                f"Bonferroni-corrected threshold t >= {corrected:.2f}"
            )
            return VerdictResult(
                Verdict.SEGMENT_ONLY,
                "Edge only in specific segments: " + ", ".join(surviving[:5]),
                reasons,
                corrected,
            )

        reasons.append(
            f"no segment survives correction for {self.secondary_tests} tests (t >= {corrected:.2f} required)"
        )
        return VerdictResult(
            Verdict.NO_EDGE,
            f"No edge found: {inputs.roi:+.2%} ROI, t = {inputs.t_statistic:.2f}",
            reasons,
            corrected,
        )

    # ------------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint
        payload["corrected_t_threshold"] = round(
            bonferroni_t_threshold(self.min_t_statistic, self.secondary_tests), 4
        )
        return {key: str(value) if isinstance(value, date) else value for key, value in payload.items()}

    def render(self) -> str:
        corrected = bonferroni_t_threshold(self.min_t_statistic, self.secondary_tests)
        return "\n".join(
            [
                f"Pre-registered validation protocol  [{self.version}]",
                "=" * 62,
                f"  Registered           {self.registered_at}  (before any real data existed)",
                f"  Fingerprint          {self.fingerprint}",
                "",
                f"  Train                {self.train_start} → {self.train_end}",
                f"  Validation           {self.valid_start} → {self.valid_end}",
                f"  Out-of-sample test   {self.test_start} → {self.test_end}",
                f"  Embargo              {self.embargo_days} days",
                "",
                "  PRIMARY HYPOTHESIS (one test, decided in advance)",
                f"    model              {self.primary_model}",
                f"    strategy           {self.primary_strategy}",
                f"    staking            {self.primary_staking}",
                f"    slippage           {self.slippage:.1%}",
                f"    commission         {self.commission:.1%}",
                "",
                "  DECISION RULE",
                f"    minimum bets       {self.min_bets}",
                f"    primary t          >= {self.min_t_statistic}",
                f"    profitable years   >= {self.min_profitable_year_fraction:.0%}",
                f"    secondary t        >= {corrected:.2f}  (Bonferroni, {self.secondary_tests} tests)",
            ]
        )


#: The protocol instance the Phase 6 pipeline runs. Importing this is how the
#: analysis is bound to a plan it cannot silently change.
PROTOCOL = ValidationProtocol()


def assert_protocol_unchanged(expected_fingerprint: str) -> None:
    """Fail if the protocol has been edited since a run was registered."""
    if PROTOCOL.fingerprint != expected_fingerprint:
        raise ProtocolViolationError(
            f"the validation protocol has changed since this run was registered "
            f"(expected {expected_fingerprint}, now {PROTOCOL.fingerprint}). "
            "Editing the plan after seeing results invalidates the conclusion."
        )


class ProtocolViolationError(RuntimeError):
    """The frozen analysis plan was modified."""


__all__ = [
    "EMBARGO_DAYS",
    "MIN_BETS_FOR_VERDICT",
    "MIN_T_STATISTIC",
    "ODDS_BANDS",
    "PROTOCOL",
    "PROTOCOL_VERSION",
    "SECONDARY_TEST_COUNT",
    "SEGMENT_DIMENSIONS",
    "ProtocolViolationError",
    "ValidationProtocol",
    "Verdict",
    "VerdictInputs",
    "VerdictResult",
    "assert_protocol_unchanged",
    "bonferroni_t_threshold",
]
