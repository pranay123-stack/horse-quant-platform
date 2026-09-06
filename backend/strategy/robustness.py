"""Robustness: does the edge hold up when the data is cut differently?

A single headline ROI is the least informative number a backtest produces. An
edge that is real shows up broadly — across years, codes and price bands. An
edge that is an artefact concentrates: one profitable season carrying six flat
ones, or every pound of profit coming from 33/1 shots in small fields.

So this module slices the ledger along the dimensions **pre-registered** in
:mod:`backend.research.protocol` and reports each one with its own t-statistic.

The multiplicity trap
=====================
Slicing invites false positives. Twelve independent segments tested at the usual
two-standard-error bar will throw up roughly one "significant" result by chance
about half the time. Every segment here is therefore reported against the
Bonferroni-corrected threshold as well as the nominal one, and
:func:`consistency_summary` scores *breadth* — how much of the edge survives
across slices — rather than picking the best slice and calling it a finding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backend.research.protocol import ODDS_BANDS, bonferroni_t_threshold

#: A segment below this many bets is reported but never counted as evidence.
MIN_SEGMENT_BETS = 100


@dataclass(slots=True)
class SegmentResult:
    """One slice of the ledger."""

    dimension: str
    segment: str
    bets: int
    wins: int
    staked: float
    profit: float
    roi: float
    strike_rate: float
    average_odds: float
    t_statistic: float

    @property
    def is_reportable(self) -> bool:
        return self.bets >= MIN_SEGMENT_BETS

    def passes(self, threshold: float) -> bool:
        return self.is_reportable and self.roi > 0 and self.t_statistic >= threshold

    @property
    def label(self) -> str:
        return f"{self.dimension}={self.segment}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "segment": self.segment,
            "bets": self.bets,
            "wins": self.wins,
            "staked": round(self.staked, 2),
            "profit": round(self.profit, 2),
            "roi": round(self.roi, 6),
            "strike_rate": round(self.strike_rate, 4),
            "average_odds": round(self.average_odds, 3),
            "t_statistic": round(self.t_statistic, 3),
            "reportable": self.is_reportable,
        }


def _segment_metrics(dimension: str, segment: str, group: pd.DataFrame) -> SegmentResult:
    stakes = pd.to_numeric(group["stake"], errors="coerce").fillna(0.0)
    profits = pd.to_numeric(group["profit_loss"], errors="coerce").fillna(0.0)
    won = pd.to_numeric(group["won"], errors="coerce").fillna(0).astype(int)

    staked = float(stakes.sum())
    profit = float(profits.sum())
    returns = (profits / stakes.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()

    t_statistic = 0.0
    if len(returns) > 1:
        spread = float(returns.std(ddof=1))
        if spread > 0:
            t_statistic = float(returns.mean() / (spread / math.sqrt(len(returns))))

    return SegmentResult(
        dimension=dimension,
        segment=segment,
        bets=len(group),
        wins=int(won.sum()),
        staked=staked,
        profit=profit,
        roi=profit / staked if staked > 0 else 0.0,
        strike_rate=float(won.mean()) if len(group) else 0.0,
        average_odds=float(pd.to_numeric(group["odds"], errors="coerce").mean()),
        t_statistic=t_statistic,
    )


def _annotate(ledger: pd.DataFrame, context: pd.DataFrame | None) -> pd.DataFrame:
    """Attach the segmentation columns, joining race context where needed."""
    frame = ledger.copy()
    frame["year"] = pd.to_datetime(frame["race_date"], errors="coerce").dt.year.astype("Int64").astype(str)

    labels = [f"{low:g}-{high:g}" for low, high in ODDS_BANDS]
    frame["odds_band"] = pd.cut(
        pd.to_numeric(frame["odds"], errors="coerce"),
        bins=[low for low, _ in ODDS_BANDS] + [ODDS_BANDS[-1][1]],
        labels=labels,
        right=False,
    ).astype(str)

    if context is not None and not context.empty:
        wanted = [
            column
            for column in ("race_id", "race_type", "race_is_handicap", "race_field_size")
            if column in context.columns
        ]
        frame = frame.merge(context[wanted].drop_duplicates("race_id"), on="race_id", how="left")

    if "race_type" in frame.columns:
        # UK racing splits into Flat and National Hunt; the distinction changes
        # everything from form cycles to field composition.
        lowered = frame["race_type"].fillna("unknown").str.lower()
        frame["code"] = np.where(
            lowered.str.contains("chase|hurdle|nh|bumper", regex=True), "national_hunt", "flat"
        )
    else:
        frame["code"] = "unknown"

    if "race_is_handicap" in frame.columns:
        frame["handicap"] = np.where(
            pd.to_numeric(frame["race_is_handicap"], errors="coerce").fillna(0) > 0,
            "handicap",
            "non_handicap",
        )
    else:
        frame["handicap"] = "unknown"

    if "race_field_size" in frame.columns:
        sizes = pd.to_numeric(frame["race_field_size"], errors="coerce")
        frame["field_size_band"] = pd.cut(
            sizes, bins=[0, 8, 12, 16, 100], labels=["<=8", "9-12", "13-16", "17+"], right=True
        ).astype(str)
    else:
        frame["field_size_band"] = "unknown"

    return frame


DIMENSIONS = ("year", "code", "handicap", "odds_band", "field_size_band")


def segment_analysis(
    ledger: pd.DataFrame,
    context: pd.DataFrame | None = None,
    *,
    dimensions: tuple[str, ...] = DIMENSIONS,
) -> list[SegmentResult]:
    """Break the ledger down along each pre-registered dimension."""
    if ledger.empty:
        return []

    annotated = _annotate(ledger, context)
    results: list[SegmentResult] = []
    for dimension in dimensions:
        if dimension not in annotated.columns:
            continue
        for segment, group in annotated.groupby(dimension, sort=True, observed=True):
            if str(segment) in ("nan", "<NA>", "None"):
                continue
            results.append(_segment_metrics(dimension, str(segment), group))
    return results


def segment_frame(results: list[SegmentResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(
            columns=["dimension", "segment", "bets", "roi", "strike_rate", "average_odds", "t_statistic"]
        )
    return pd.DataFrame([result.as_dict() for result in results])


@dataclass(slots=True)
class ConsistencySummary:
    """How broadly the edge holds, rather than how strong its best slice is."""

    years_covered: int = 0
    profitable_years: int = 0
    worst_year_roi: float = 0.0
    best_year_roi: float = 0.0
    reportable_segments: int = 0
    positive_segments: int = 0
    passing_nominal: list[str] = field(default_factory=list)
    passing_corrected: list[str] = field(default_factory=list)
    corrected_threshold: float = 0.0

    @property
    def profitable_year_fraction(self) -> float:
        return self.profitable_years / self.years_covered if self.years_covered else 0.0

    @property
    def positive_segment_fraction(self) -> float:
        return self.positive_segments / self.reportable_segments if self.reportable_segments else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "years_covered": self.years_covered,
            "profitable_years": self.profitable_years,
            "profitable_year_fraction": round(self.profitable_year_fraction, 4),
            "worst_year_roi": round(self.worst_year_roi, 4),
            "best_year_roi": round(self.best_year_roi, 4),
            "reportable_segments": self.reportable_segments,
            "positive_segments": self.positive_segments,
            "positive_segment_fraction": round(self.positive_segment_fraction, 4),
            "passing_nominal": self.passing_nominal,
            "passing_corrected": self.passing_corrected,
            "corrected_threshold": round(self.corrected_threshold, 3),
        }

    def render(self) -> str:
        return "\n".join(
            [
                "Consistency",
                f"  Years covered        {self.years_covered}",
                f"  Profitable years     {self.profitable_years} ({self.profitable_year_fraction:.0%})",
                f"  Worst / best year    {self.worst_year_roi:+.2%} / {self.best_year_roi:+.2%}",
                f"  Reportable segments  {self.reportable_segments}",
                f"  Positive segments    {self.positive_segments} ({self.positive_segment_fraction:.0%})",
                f"  Pass at t>=2.0       {len(self.passing_nominal)}",
                f"  Pass at t>={self.corrected_threshold:.2f} (corrected)  {len(self.passing_corrected)}",
            ]
        )


def consistency_summary(
    results: list[SegmentResult], *, nominal_threshold: float = 2.0
) -> ConsistencySummary:
    """Aggregate the segment results into a breadth-of-edge verdict."""
    corrected = bonferroni_t_threshold(nominal_threshold)
    summary = ConsistencySummary(corrected_threshold=corrected)

    years = [result for result in results if result.dimension == "year" and result.is_reportable]
    summary.years_covered = len(years)
    summary.profitable_years = sum(1 for result in years if result.roi > 0)
    if years:
        summary.worst_year_roi = min(result.roi for result in years)
        summary.best_year_roi = max(result.roi for result in years)

    reportable = [result for result in results if result.is_reportable]
    summary.reportable_segments = len(reportable)
    summary.positive_segments = sum(1 for result in reportable if result.roi > 0)
    summary.passing_nominal = [result.label for result in reportable if result.passes(nominal_threshold)]
    summary.passing_corrected = [result.label for result in reportable if result.passes(corrected)]
    return summary


def render_segments(results: list[SegmentResult], *, dimension: str | None = None) -> str:
    """A readable table, optionally for one dimension."""
    selected = [r for r in results if dimension is None or r.dimension == dimension]
    if not selected:
        return "  (no segments)"

    lines = [
        f"  {'segment':<22}{'bets':>7}{'strike':>9}{'avg odds':>10}{'roi':>10}{'t':>8}",
        "  " + "-" * 66,
    ]
    for result in selected:
        marker = "" if result.is_reportable else "  (thin)"
        lines.append(
            f"  {result.label:<22}{result.bets:>7}{result.strike_rate:>8.1%}"
            f"{result.average_odds:>10.2f}{result.roi:>+9.1%}{result.t_statistic:>+8.2f}{marker}"
        )
    return "\n".join(lines)


__all__ = [
    "DIMENSIONS",
    "MIN_SEGMENT_BETS",
    "ConsistencySummary",
    "SegmentResult",
    "consistency_summary",
    "render_segments",
    "segment_analysis",
    "segment_frame",
]
