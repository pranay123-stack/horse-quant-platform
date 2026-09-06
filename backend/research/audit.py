"""Dataset audit and readiness gate.

Phase 3's :class:`~backend.data_quality.DataValidator` answers *"which individual
races are trustworthy?"*. This module answers a different and blunter question:
**is this dataset fit to draw a conclusion from at all?**

They are not the same. A dataset can be composed entirely of individually valid
races and still be useless — because it covers three years rather than eight,
because two-thirds of runners have no price, or because the odds only start
appearing in 2023. Modelling on it would produce numbers, and the numbers would
be meaningless.

The audit therefore reports coverage rather than correctness, and ends with a
hard :class:`Readiness` verdict against the pre-registered protocol's windows.
Running the validation on a dataset that fails the gate is how a research project
talks itself into a false positive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.models import Horse, Jockey, OddsHistory, Race, RaceResult, RaceRunner, Trainer
from backend.utils.logging import get_logger, safe_extra
from backend.utils.timeutils import utcnow

logger = get_logger(__name__, channel="pipeline")

#: Minimum share of runners that need a usable pre-race price before a
#: value-betting study is worth running. Below this the strategy is selecting
#: from a biased subset of the card rather than from the card.
MIN_ODDS_COVERAGE = 0.60
#: Minimum share of races with results — the labels.
MIN_RESULT_COVERAGE = 0.95
#: Minimum distinct racing days per year before a year counts as covered.
MIN_DAYS_PER_YEAR = 200
#: The specification asks for five or more years.
MIN_YEARS = 5

#: Above this share of races missing an off time, the pre-race guarantee is
#: unverifiable for too much of the data to proceed.
MAX_MISSING_OFF_TIME = 0.05
#: Above this share, off-time/date disagreement is systematic rather than the
#: occasional late meeting crossing midnight.
MAX_OFF_TIME_MISMATCH = 0.01


@dataclass(slots=True)
class CoverageRow:
    """One period's completeness."""

    period: str
    races: int
    runners: int
    results: int
    with_odds: int
    racing_days: int

    @property
    def odds_coverage(self) -> float:
        return self.with_odds / self.runners if self.runners else 0.0

    @property
    def result_coverage(self) -> float:
        return self.results / self.races if self.races else 0.0


@dataclass(slots=True)
class Readiness:
    """Whether the dataset can support the pre-registered study."""

    ready: bool = False
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def block(self, reason: str) -> None:
        self.blocking.append(reason)

    def warn(self, reason: str) -> None:
        self.warnings.append(reason)


@dataclass(slots=True)
class DatasetAudit:
    """Everything the Phase 6.2 report needs."""

    generated_at: str = field(default_factory=lambda: utcnow().isoformat())
    source: str = "unknown"

    races: int = 0
    runners: int = 0
    results: int = 0
    horses: int = 0
    jockeys: int = 0
    trainers: int = 0
    odds_quotes: int = 0

    date_from: date | None = None
    date_to: date | None = None
    racing_days: int = 0

    races_without_results: int = 0
    races_without_runners: int = 0
    runners_without_odds: int = 0
    duplicate_race_slots: int = 0
    invalid_prices: int = 0
    odds_after_off: int = 0
    #: Races with no off time at all. These are a blind spot rather than an
    #: error: without an off time we cannot prove their odds are pre-race.
    races_without_off_time: int = 0
    #: Off time falling on a different calendar day than race_date. Features
    #: are cut at race_date, so a mismatch silently shifts the cutoff.
    off_time_date_mismatch: int = 0
    races_without_winner: int = 0
    horses_with_one_run: int = 0

    coverage: list[CoverageRow] = field(default_factory=list)
    by_race_type: dict[str, int] = field(default_factory=dict)
    readiness: Readiness = field(default_factory=Readiness)

    # ------------------------------------------------------------------
    @property
    def years_covered(self) -> int:
        return len({row.period[:4] for row in self.coverage})

    @property
    def odds_coverage(self) -> float:
        return (self.runners - self.runners_without_odds) / self.runners if self.runners else 0.0

    @property
    def result_coverage(self) -> float:
        return (self.races - self.races_without_results) / self.races if self.races else 0.0

    @property
    def is_synthetic(self) -> bool:
        """Synthetic rows use a reserved id prefix and must never be validated on."""
        return self.source == "synthetic"

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "source": self.source,
            "totals": {
                "races": self.races,
                "runners": self.runners,
                "results": self.results,
                "horses": self.horses,
                "jockeys": self.jockeys,
                "trainers": self.trainers,
                "odds_quotes": self.odds_quotes,
            },
            "span": {
                "from": str(self.date_from),
                "to": str(self.date_to),
                "racing_days": self.racing_days,
                "years": self.years_covered,
            },
            "completeness": {
                "odds_coverage": round(self.odds_coverage, 4),
                "result_coverage": round(self.result_coverage, 4),
                "races_without_results": self.races_without_results,
                "races_without_runners": self.races_without_runners,
                "runners_without_odds": self.runners_without_odds,
            },
            "integrity": {
                "duplicate_race_slots": self.duplicate_race_slots,
                "invalid_prices": self.invalid_prices,
                "odds_after_off": self.odds_after_off,
                "races_without_winner": self.races_without_winner,
            },
            "readiness": {
                "ready": self.readiness.ready,
                "blocking": self.readiness.blocking,
                "warnings": self.readiness.warnings,
            },
        }


def audit_dataset(session: Session, *, source: str | None = None) -> DatasetAudit:
    """Audit whatever is currently in the warehouse."""
    audit = DatasetAudit()

    audit.races = session.execute(select(func.count()).select_from(Race)).scalar_one()
    if audit.races == 0:
        audit.source = source or "empty"
        audit.readiness.block("the database contains no races")
        return audit

    audit.runners = session.execute(select(func.count()).select_from(RaceRunner)).scalar_one()
    audit.results = session.execute(select(func.count()).select_from(RaceResult)).scalar_one()
    audit.horses = session.execute(select(func.count()).select_from(Horse)).scalar_one()
    audit.jockeys = session.execute(select(func.count()).select_from(Jockey)).scalar_one()
    audit.trainers = session.execute(select(func.count()).select_from(Trainer)).scalar_one()
    audit.odds_quotes = session.execute(select(func.count()).select_from(OddsHistory)).scalar_one()

    audit.source = source or _detect_source(session)

    audit.date_from, audit.date_to = session.execute(
        select(func.min(Race.race_date), func.max(Race.race_date))
    ).one()
    audit.racing_days = session.execute(select(func.count(func.distinct(Race.race_date)))).scalar_one()

    audit.races_without_results = session.execute(
        select(func.count()).select_from(Race).where(Race.has_result.is_(False))
    ).scalar_one()
    audit.races_without_runners = session.execute(
        select(func.count()).select_from(Race).where(~Race.race_id.in_(select(RaceRunner.race_id).distinct()))
    ).scalar_one()

    runners_with_odds = session.execute(
        select(func.count(func.distinct(func.concat(OddsHistory.race_id, OddsHistory.horse_id))))
    ).scalar_one()
    audit.runners_without_odds = max(0, audit.runners - runners_with_odds)

    audit.invalid_prices = session.execute(
        select(func.count())
        .select_from(OddsHistory)
        .where((OddsHistory.decimal_odds.is_(None)) | (OddsHistory.decimal_odds <= 1))
    ).scalar_one()

    audit.odds_after_off = session.execute(
        select(func.count())
        .select_from(OddsHistory)
        .join(Race, Race.race_id == OddsHistory.race_id)
        .where(Race.off_time.isnot(None), OddsHistory.recorded_at > Race.off_time)
    ).scalar_one()

    audit.races_without_off_time = session.execute(
        select(func.count()).select_from(Race).where(Race.off_time.is_(None))
    ).scalar_one()

    # An off time on a different calendar day than race_date moves the
    # point-in-time cutoff without anything noticing.
    audit.off_time_date_mismatch = session.execute(
        select(func.count())
        .select_from(Race)
        .where(Race.off_time.isnot(None), func.date(Race.off_time) != Race.race_date)
    ).scalar_one()

    # Two races at the same course, date and time are the same race twice.
    duplicate_slots = session.execute(
        select(func.count()).select_from(
            select(Race.course_id, Race.race_date, Race.off_time)
            .where(Race.course_id.isnot(None), Race.off_time.isnot(None))
            .group_by(Race.course_id, Race.race_date, Race.off_time)
            .having(func.count() > 1)
            .subquery()
        )
    ).scalar_one()
    audit.duplicate_race_slots = duplicate_slots

    with_result = select(Race.race_id).where(Race.has_result.is_(True)).subquery()
    winners = (
        select(RaceResult.race_id)
        .where(RaceResult.is_winner.is_(True))
        .group_by(RaceResult.race_id)
        .subquery()
    )
    audit.races_without_winner = session.execute(
        select(func.count())
        .select_from(with_result)
        .where(~with_result.c.race_id.in_(select(winners.c.race_id)))
    ).scalar_one()

    audit.horses_with_one_run = session.execute(
        select(func.count()).select_from(
            select(RaceResult.horse_id).group_by(RaceResult.horse_id).having(func.count() == 1).subquery()
        )
    ).scalar_one()

    audit.by_race_type = {
        str(race_type or "unknown"): count
        for race_type, count in session.execute(
            select(Race.race_type, func.count()).group_by(Race.race_type).order_by(func.count().desc())
        ).all()
    }

    audit.coverage = _coverage_by_year(session)
    _assess_readiness(audit)

    logger.info("dataset audit complete", extra=safe_extra(audit.as_dict()["completeness"]))
    return audit


def _detect_source(session: Session) -> str:
    """Synthetic rows carry a reserved id prefix; real ones never do."""
    synthetic = session.execute(
        select(func.count()).select_from(Race).where(Race.race_id.like("rac_s%"))
    ).scalar_one()
    total = session.execute(select(func.count()).select_from(Race)).scalar_one()
    if synthetic == 0:
        return "real"
    if synthetic == total:
        return "synthetic"
    return "mixed"


def _year_counts(rows: Sequence[Any]) -> dict[int, int]:
    """SQL ``extract(year ...)`` returns Decimal on Postgres, int on SQLite."""
    return {int(year): int(count) for year, count in rows}


def _coverage_by_year(session: Session) -> list[CoverageRow]:
    rows = session.execute(
        select(
            func.extract("year", Race.race_date).label("year"),
            func.count(func.distinct(Race.race_id)),
            func.count(func.distinct(Race.race_date)),
            func.sum(func.cast(Race.has_result, __import__("sqlalchemy").Integer)),
        )
        .where(Race.race_date.isnot(None))
        .group_by("year")
        .order_by("year")
    ).all()

    runner_counts: dict[int, int] = _year_counts(
        session.execute(
            select(func.extract("year", Race.race_date), func.count())
            .join(RaceRunner, RaceRunner.race_id == Race.race_id)
            .group_by(func.extract("year", Race.race_date))
        ).all()
    )
    odds_counts: dict[int, int] = _year_counts(
        session.execute(
            select(
                func.extract("year", Race.race_date),
                func.count(func.distinct(func.concat(OddsHistory.race_id, OddsHistory.horse_id))),
            )
            .join(OddsHistory, OddsHistory.race_id == Race.race_id)
            .group_by(func.extract("year", Race.race_date))
        ).all()
    )

    coverage = []
    for year, races, days, results in rows:
        key = int(year)
        coverage.append(
            CoverageRow(
                period=str(key),
                races=int(races),
                runners=int(runner_counts.get(year, 0) or 0),
                results=int(results or 0),
                with_odds=int(odds_counts.get(year, 0) or 0),
                racing_days=int(days),
            )
        )
    return coverage


def _assess_readiness(audit: DatasetAudit) -> None:
    """Apply the hard gate. Blocking reasons stop the study; warnings do not."""
    readiness = audit.readiness

    if audit.is_synthetic:
        readiness.block("this dataset is synthetic — it cannot validate anything about real markets")
    elif audit.source == "mixed":
        readiness.block("the database mixes real and synthetic races; synthetic rows must be purged first")

    complete_years = [row for row in audit.coverage if row.racing_days >= MIN_DAYS_PER_YEAR]
    if len(complete_years) < MIN_YEARS:
        readiness.block(
            f"only {len(complete_years)} year(s) with >= {MIN_DAYS_PER_YEAR} racing days; "
            f"the specification requires {MIN_YEARS}+"
        )

    if audit.odds_coverage < MIN_ODDS_COVERAGE:
        readiness.block(
            f"only {audit.odds_coverage:.1%} of runners have a price "
            f"(need {MIN_ODDS_COVERAGE:.0%}); a value study on this subset would be biased"
        )
    if audit.result_coverage < MIN_RESULT_COVERAGE:
        readiness.block(
            f"only {audit.result_coverage:.1%} of races have results (need {MIN_RESULT_COVERAGE:.0%})"
        )

    if audit.duplicate_race_slots:
        readiness.warn(f"{audit.duplicate_race_slots} duplicated course/date/time slots")
    if audit.invalid_prices:
        readiness.warn(f"{audit.invalid_prices} quotes at or below evens")
    if audit.odds_after_off:
        readiness.block(
            f"{audit.odds_after_off} quotes are timestamped after the off — post-race "
            "prices must be excluded before any backtest"
        )
    # A race with no off time cannot have its odds proven pre-race, so a large
    # share of them is a hole in the leakage guarantee, not a cosmetic gap.
    if audit.races_without_off_time:
        share = audit.races_without_off_time / max(audit.races, 1)
        message = (
            f"{audit.races_without_off_time:,} races ({share:.1%}) have no off time, so their "
            "quotes cannot be proven to be pre-race"
        )
        if share > MAX_MISSING_OFF_TIME:
            readiness.block(message)
        else:
            readiness.warn(message)

    if audit.off_time_date_mismatch:
        share = audit.off_time_date_mismatch / max(audit.races, 1)
        message = (
            f"{audit.off_time_date_mismatch:,} races ({share:.1%}) have an off time on a "
            "different day than race_date, which shifts the feature cutoff"
        )
        if share > MAX_OFF_TIME_MISMATCH:
            readiness.block(message)
        else:
            readiness.warn(message)

    if audit.races_without_winner:
        readiness.warn(f"{audit.races_without_winner} finished races have no winner recorded")
    if audit.races_without_runners:
        readiness.warn(f"{audit.races_without_runners} races have no declared runners")

    readiness.ready = not readiness.blocking


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_markdown(audit: DatasetAudit) -> str:
    """The Phase 6.3 deliverable: ``REAL_DATA_AUDIT.md``."""
    status = "READY" if audit.readiness.ready else "NOT READY"
    lines = [
        "# Real data audit",
        "",
        f"*Generated {audit.generated_at}*",
        "",
        f"**Dataset source: `{audit.source}`**",
        f"**Modelling readiness: {status}**",
        "",
    ]

    if audit.readiness.blocking:
        lines += ["## Blocking issues", ""]
        lines += [f"- {reason}" for reason in audit.readiness.blocking]
        lines += [
            "",
            "> A study run on this dataset cannot support a conclusion about real markets.",
            "",
        ]

    lines += [
        "## Totals",
        "",
        "| Entity | Count |",
        "|---|---:|",
        f"| Races | {audit.races:,} |",
        f"| Runners (declarations) | {audit.runners:,} |",
        f"| Results | {audit.results:,} |",
        f"| Horses | {audit.horses:,} |",
        f"| Jockeys | {audit.jockeys:,} |",
        f"| Trainers | {audit.trainers:,} |",
        f"| Odds quotes | {audit.odds_quotes:,} |",
        "",
        "## Span",
        "",
        f"- **{audit.date_from} → {audit.date_to}**",
        f"- {audit.racing_days:,} distinct racing days across {audit.years_covered} year(s)",
        "",
        "## Coverage",
        "",
        "| Check | Value | Threshold |",
        "|---|---:|---:|",
        f"| Runners with a price | {audit.odds_coverage:.1%} | {MIN_ODDS_COVERAGE:.0%} |",
        f"| Races with results | {audit.result_coverage:.1%} | {MIN_RESULT_COVERAGE:.0%} |",
        f"| Races missing results | {audit.races_without_results:,} | — |",
        f"| Races missing runners | {audit.races_without_runners:,} | — |",
        f"| Runners missing odds | {audit.runners_without_odds:,} | — |",
        "",
        "## Quality",
        "",
        "| Check | Count | Why it matters |",
        "|---|---:|---|",
        f"| Duplicate course/date/time slots | {audit.duplicate_race_slots:,} | "
        "the same race counted twice |",
        f"| Prices at or below evens | {audit.invalid_prices:,} | an impossible decimal price |",
        f"| Finished races with no winner | {audit.races_without_winner:,} | "
        "an unlabelled row in the training set |",
        f"| Horses with a single career run | {audit.horses_with_one_run:,} | "
        "no form history to build features from |",
        "",
        "### Timestamp consistency",
        "",
        "Every leakage guarantee in this project rests on knowing when a price was "
        "taken relative to the off. These three checks are what make that knowable.",
        "",
        "| Check | Count | Consequence |",
        "|---|---:|---|",
        f"| Quotes recorded after the off | {audit.odds_after_off:,} | "
        "a post-race price is not bettable — **blocking** |",
        f"| Races with no off time | {audit.races_without_off_time:,} | "
        "their quotes cannot be proven pre-race |",
        f"| Off time on a different day than race_date | {audit.off_time_date_mismatch:,} | "
        "shifts the point-in-time feature cutoff |",
        "",
    ]

    if audit.coverage:
        lines += [
            "## Coverage by year",
            "",
            "| Year | Races | Days | Runners | Result coverage | Odds coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in audit.coverage:
            lines.append(
                f"| {row.period} | {row.races:,} | {row.racing_days} | {row.runners:,} | "
                f"{row.result_coverage:.1%} | {row.odds_coverage:.1%} |"
            )
        lines.append("")

    if audit.by_race_type:
        lines += ["## Race types", "", "| Type | Races |", "|---|---:|"]
        for race_type, count in list(audit.by_race_type.items())[:12]:
            lines.append(f"| {race_type} | {count:,} |")
        lines.append("")

    if audit.readiness.warnings:
        lines += ["## Warnings", ""]
        lines += [f"- {warning}" for warning in audit.readiness.warnings]
        lines.append("")

    return "\n".join(lines)


def write_report(audit: DatasetAudit, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(audit), encoding="utf-8")
    logger.info("data quality report written", extra={"path": str(target)})
    return target


def coverage_frame(audit: DatasetAudit) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period": row.period,
                "races": row.races,
                "racing_days": row.racing_days,
                "runners": row.runners,
                "result_coverage": round(row.result_coverage, 4),
                "odds_coverage": round(row.odds_coverage, 4),
            }
            for row in audit.coverage
        ]
    )


__all__ = [
    "MIN_DAYS_PER_YEAR",
    "MIN_ODDS_COVERAGE",
    "MIN_RESULT_COVERAGE",
    "MIN_YEARS",
    "CoverageRow",
    "DatasetAudit",
    "Readiness",
    "audit_dataset",
    "coverage_frame",
    "render_markdown",
    "write_report",
]
