"""Database-wide data-quality auditing.

The validator answers one question the research layer cannot proceed without:
*which races are trustworthy enough to learn from?*

It is deliberately run **after** ingestion rather than inside it. Ingestion's job
is to capture what the upstream said, faithfully and idempotently; quality
judgement is a separate, re-runnable concern with thresholds that will change as
we learn more. Conflating the two would mean every threshold change required a
re-ingest.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.data_quality import rules
from backend.data_quality.report import DataQualityReport
from backend.data_quality.rules import ValidationIssue
from backend.models import OddsHistory, Race, RaceResult, RaceRunner
from backend.utils.logging import get_logger, safe_extra

logger = get_logger(__name__, channel="pipeline")


class DataValidator:
    """Audits stored racing data and reports which races are usable."""

    def __init__(self, session: Session, *, today: date | None = None) -> None:
        self.session = session
        self.today = today

    # ------------------------------------------------------------------
    # Record-level validation (also usable standalone, pre-ingest)
    # ------------------------------------------------------------------
    def validate_race(
        self,
        race: Race,
        *,
        runner_count: int = 0,
        seen_slots: dict[tuple, str] | None = None,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        issues += rules.check_race_date(race.race_id, race.race_date, today=self.today)
        issues += rules.check_race_distance(race.race_id, race.distance_yards)
        issues += rules.check_race_class(race.race_id, race.race_class)
        if runner_count:
            issues += rules.check_field_size(race.race_id, race.field_size, runner_count)
        if seen_slots is not None:
            issues += rules.check_duplicate_race(
                race.race_id, race.course_id, race.race_date, race.off_time, seen_slots
            )
        return issues

    def validate_runner(self, runner: RaceRunner) -> list[ValidationIssue]:
        runner_id = f"{runner.race_id}/{runner.horse_id}"
        issues = rules.check_horse_id(runner.race_id, runner.horse_id)
        issues += rules.check_horse_age(runner_id, runner.age)
        issues += rules.check_weight(runner_id, runner.weight_lbs)
        return issues

    def validate_result(self, result: RaceResult, *, runner_count: int) -> list[ValidationIssue]:
        result_id = f"{result.race_id}/{result.horse_id}"
        issues = rules.check_horse_id(result.race_id, result.horse_id)
        issues += rules.check_horse_age(result_id, result.age)
        issues += rules.check_weight(result_id, result.weight_lbs)
        issues += rules.check_finishing_position(
            result_id, result.finishing_position, result.finishing_status, runner_count
        )
        return issues

    def validate_odds(self, quote: OddsHistory, *, off_time: object = None) -> list[ValidationIssue]:
        quote_id = f"{quote.race_id}/{quote.horse_id}/{quote.bookmaker}"
        issues = rules.check_decimal_odds(quote_id, quote.decimal_odds)
        issues += rules.check_odds_timestamp(quote_id, quote.recorded_at)
        issues += rules.check_odds_before_off(quote_id, quote.recorded_at, off_time)
        return issues

    # ------------------------------------------------------------------
    # Whole-database audit
    # ------------------------------------------------------------------
    def validate_database(
        self,
        *,
        date_from: date | None = None,
        date_to: date | None = None,
        race_ids: Sequence[str] | None = None,
        check_odds: bool = True,
    ) -> DataQualityReport:
        """Audit every race in the window and return a full report."""
        report = DataQualityReport(
            date_from=date_from.isoformat() if date_from else None,
            date_to=date_to.isoformat() if date_to else None,
        )

        races = list(self._iter_races(date_from, date_to, race_ids))
        report.total_races = len(races)
        if not races:
            logger.info("data quality audit found no races in window")
            return report

        selected_ids = [race.race_id for race in races]
        runner_counts = self._runner_counts(selected_ids)
        result_counts = self._result_counts(selected_ids)
        winner_counts = self._winner_counts(selected_ids)

        seen_slots: dict[tuple, str] = {}

        for race in races:
            runner_count = runner_counts.get(race.race_id, 0)
            result_count = result_counts.get(race.race_id, 0)
            # Prefer the number of actual finishers when validating positions.
            field_count = result_count or runner_count

            report.add(
                self.validate_race(race, runner_count=field_count, seen_slots=seen_slots),
                race_id=race.race_id,
            )

            for runner in race.runners:
                report.total_runners += 1
                report.add(self.validate_runner(runner), race_id=race.race_id)

            for result in race.results:
                report.total_results += 1
                report.add(self.validate_result(result, runner_count=field_count), race_id=race.race_id)

            if race.has_result:
                report.add(
                    rules.check_single_winner(race.race_id, winner_counts.get(race.race_id, 0)),
                    race_id=race.race_id,
                )

            if check_odds:
                for quote in race.odds:
                    report.total_odds += 1
                    report.add(self.validate_odds(quote, off_time=race.off_time), race_id=race.race_id)

        logger.info(
            "data quality audit complete",
            extra=safe_extra(
                {
                    "races": report.total_races,
                    "valid": report.valid_races,
                    "rejected": report.rejected_races,
                    "errors": report.error_count,
                    "warnings": report.warning_count,
                }
            ),
        )
        return report

    # ------------------------------------------------------------------
    def _iter_races(
        self,
        date_from: date | None,
        date_to: date | None,
        race_ids: Sequence[str] | None,
    ) -> Iterable[Race]:
        statement = select(Race)
        if race_ids is not None:
            statement = statement.where(Race.race_id.in_(race_ids))
        if date_from is not None:
            statement = statement.where(Race.race_date >= date_from)
        if date_to is not None:
            statement = statement.where(Race.race_date <= date_to)
        return self.session.execute(statement.order_by(Race.race_date, Race.off_time)).scalars().all()

    def _runner_counts(self, race_ids: Sequence[str]) -> dict[str, int]:
        rows = self.session.execute(
            select(RaceRunner.race_id, func.count())
            .where(RaceRunner.race_id.in_(race_ids))
            .group_by(RaceRunner.race_id)
        ).all()
        counts: dict[str, int] = {}
        for race_id, count in rows:
            counts[race_id] = count
        return counts

    def _result_counts(self, race_ids: Sequence[str]) -> dict[str, int]:
        rows = self.session.execute(
            select(RaceResult.race_id, func.count())
            .where(RaceResult.race_id.in_(race_ids))
            .group_by(RaceResult.race_id)
        ).all()
        counts: dict[str, int] = {}
        for race_id, count in rows:
            counts[race_id] = count
        return counts

    def _winner_counts(self, race_ids: Sequence[str]) -> dict[str, int]:
        rows = self.session.execute(
            select(RaceResult.race_id, func.count())
            .where(RaceResult.race_id.in_(race_ids), RaceResult.is_winner.is_(True))
            .group_by(RaceResult.race_id)
        ).all()
        counts: dict[str, int] = {}
        for race_id, count in rows:
            counts[race_id] = count
        return counts


def usable_race_ids(session: Session, **kwargs: object) -> set[str]:
    """Convenience: the set of race ids that pass validation."""
    report = DataValidator(session).validate_database(**kwargs)  # type: ignore[arg-type]
    all_ids = set(session.execute(select(Race.race_id)).scalars().all())
    return all_ids - report.rejected_race_ids


__all__ = ["DataValidator", "usable_race_ids"]
