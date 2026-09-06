"""Model evaluation and comparison.

Every model is judged on three axes, because each answers a different question:

**Probability quality** (log loss, Brier, ECE) — can the number be trusted as a
probability? This is the axis Phase 5 depends on.

**Discrimination** (ROC-AUC, precision, recall) — can the model tell winners from
losers at all? Useful, but insufficient: AUC is invariant to any monotone
rescaling of the output, so a hopelessly over-confident model scores identically
to a perfectly calibrated one.

**Betting relevance** (race log loss, top-1 hit rate) — does it pick winners in
the shape the sport actually has? A per-runner metric is dominated by the seven
easy negatives in an eight-runner field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backend.ml.calibration.calibrators import CalibrationComparison, compare_calibration
from backend.ml.metrics.classification import (
    ClassificationMetrics,
    ReliabilityCurve,
    compute_classification_metrics,
    reliability_curve,
)
from backend.ml.metrics.racing import RacingMetrics, compute_racing_metrics
from backend.ml.models.calibrated import CalibratedRacingModel
from backend.utils.logging import get_logger, safe_extra

logger = get_logger(__name__, channel="model")


@dataclass(slots=True)
class EvaluationReport:
    """Everything measured about one model on one split."""

    model: str
    split: str
    classification: ClassificationMetrics
    racing: RacingMetrics
    #: Race-normalised probabilities scored per runner. Normalisation usually
    #: improves these, and by how much is itself informative.
    normalised_classification: ClassificationMetrics | None = None
    curve: ReliabilityCurve = field(default_factory=ReliabilityCurve)
    calibration: CalibrationComparison | None = None
    importance: pd.DataFrame | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "split": self.split,
            "classification": self.classification.as_dict(),
            "racing": self.racing.as_dict(),
            "reliability_curve": self.curve.as_dict(),
        }
        if self.normalised_classification is not None:
            payload["classification_normalised"] = self.normalised_classification.as_dict()
        if self.calibration is not None:
            payload["calibration"] = self.calibration.as_dict()
        if self.importance is not None and not self.importance.empty:
            payload["top_features"] = self.importance.head(20).to_dict(orient="records")
        return payload

    def render(self, *, show_curve: bool = True, show_importance: int = 15) -> str:
        c, r = self.classification, self.racing
        lines = [
            f"Model: {self.model}   ({self.split} split)",
            "=" * 60,
            "Probability quality",
            f"  Log loss                 {c.log_loss:.5f}",
            f"  Brier score              {c.brier:.5f}",
            f"  Expected calib. error    {c.expected_calibration_error:.5f}",
            f"  Max calib. error         {c.maximum_calibration_error:.5f}",
            f"  Mean predicted / actual  {c.mean_predicted:.4f} / {c.base_rate:.4f}"
            f"  (ratio {c.calibration_ratio:.3f})",
            "",
            "Discrimination",
            f"  ROC-AUC                  {c.roc_auc:.5f}",
            f"  Average precision        {c.average_precision:.5f}",
            f"  Precision @ {c.threshold:.2f}         {c.precision:.5f}",
            f"  Recall    @ {c.threshold:.2f}         {c.recall:.5f}",
            "",
            "Betting relevance (race level)",
            f"  Races                    {r.races}",
            f"  Race log loss            {r.race_log_loss:.5f}   (uniform {r.uniform_log_loss:.5f})",
            f"  Skill vs uniform         {r.skill_score:+.2%}",
            f"  Top-1 hit rate           {r.top1_hit_rate:.2%}",
            f"  Top-3 hit rate           {r.top3_hit_rate:.2%}",
            f"  Mean winner rank         {r.mean_winner_rank:.2f} of {r.mean_field_size:.1f}",
        ]
        if self.normalised_classification is not None:
            n = self.normalised_classification
            lines += [
                "",
                "After within-race normalisation",
                f"  Log loss                 {n.log_loss:.5f}",
                f"  Brier score              {n.brier:.5f}",
                f"  Expected calib. error    {n.expected_calibration_error:.5f}",
            ]
        if show_curve and self.curve.counts:
            lines += ["", "Reliability", self.curve.render()]
        if self.importance is not None and not self.importance.empty and show_importance:
            lines += ["", f"Top {show_importance} features"]
            for row in self.importance.head(show_importance).itertuples(index=False):
                lines.append(f"  {row.importance_pct:>7.2%}  {row.feature}")
        return "\n".join(lines)


def evaluate_model(
    model: CalibratedRacingModel,
    frame: pd.DataFrame,
    *,
    split: str = "test",
    target: str = "won",
    bins: int = 10,
    include_importance: bool = True,
) -> EvaluationReport:
    """Score ``model`` on ``frame`` across all three axes."""
    if frame.empty:
        return EvaluationReport(
            model=model.name, split=split, classification=ClassificationMetrics(), racing=RacingMetrics()
        )

    truth = frame[target].to_numpy()
    raw = model.raw_proba(frame)
    calibrated = model.predict_proba(frame)
    normalised = model.predict_race_proba(frame)

    report = EvaluationReport(
        model=model.name,
        split=split,
        classification=compute_classification_metrics(truth, calibrated, bins=bins),
        racing=compute_racing_metrics(frame["race_id"], truth, normalised, already_normalised=True),
        normalised_classification=compute_classification_metrics(truth, normalised, bins=bins),
        curve=reliability_curve(truth, calibrated, bins=bins),
    )
    if model.is_calibrated:
        report.calibration = compare_calibration(
            truth, raw, calibrated, method=model.calibrator.method, bins=bins
        )
    if include_importance:
        try:
            report.importance = model.feature_importance()
        except Exception as exc:
            logger.warning("feature importance unavailable", extra={"error": str(exc)})

    logger.info(
        "model evaluated",
        extra=safe_extra(
            {
                "model": model.name,
                "split": split,
                "log_loss": round(report.classification.log_loss, 5),
                "brier": round(report.classification.brier, 5),
                "roc_auc": round(report.classification.roc_auc, 5),
                "top1": round(report.racing.top1_hit_rate, 5),
            }
        ),
    )
    return report


def comparison_table(reports: list[EvaluationReport]) -> pd.DataFrame:
    """The Phase 4.7 table: one row per model, best log loss first."""
    rows = [
        {
            "model": report.model,
            "log_loss": round(report.classification.log_loss, 5),
            "brier": round(report.classification.brier, 5),
            "roc_auc": round(report.classification.roc_auc, 5),
            "ece": round(report.classification.expected_calibration_error, 5),
            "race_log_loss": round(report.racing.race_log_loss, 5),
            "top1_hit_rate": round(report.racing.top1_hit_rate, 4),
            "top3_hit_rate": round(report.racing.top3_hit_rate, 4),
            "skill_vs_uniform": round(report.racing.skill_score, 4),
        }
        for report in reports
    ]
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def select_champion(
    reports: list[EvaluationReport], *, criterion: str = "race_log_loss"
) -> EvaluationReport | None:
    """Pick the best model.

    The default criterion is **race log loss**, not AUC and not per-runner log
    loss. It is the only one of the three that is both a proper scoring rule and
    measured on the unit the platform actually bets on — a race.
    """
    if not reports:
        return None

    def score(report: EvaluationReport) -> float:
        values = {
            "race_log_loss": report.racing.race_log_loss,
            "log_loss": report.classification.log_loss,
            "brier": report.classification.brier,
            "ece": report.classification.expected_calibration_error,
            # Negated so that "lower is better" holds for every criterion.
            "roc_auc": -report.classification.roc_auc,
            "top1_hit_rate": -report.racing.top1_hit_rate,
        }
        if criterion not in values:
            raise ValueError(f"unknown criterion {criterion!r}; expected one of {sorted(values)}")
        value = values[criterion]
        return float("inf") if value is None or np.isnan(value) else value

    return min(reports, key=score)


__all__ = ["EvaluationReport", "comparison_table", "evaluate_model", "select_champion"]
