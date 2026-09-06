"""Race data ingestion: API -> validation -> parser -> database.

    API (validated payloads)
        |
    parsers -> RaceSchema (typed domain objects)
        |
    RaceDataImporter  -- upsert, per-race savepoint, failure capture
        |
    PostgreSQL

Two properties matter more than throughput here:

**Idempotence.** Re-running a day's import must not duplicate anything. Every
write is an upsert keyed on the Racing API's own identifiers, so a re-run is a
no-op plus whatever genuinely changed (a non-runner withdrawn, a price moved).

**Isolation of failure.** One malformed race must not lose the other thirty-nine
on the card. Each race is written inside its own ``SAVEPOINT``; a failure rolls
back that race alone, records it, and the run continues.

A deliberate subtlety: updates never overwrite a populated column with ``NULL``.
Racecards and results carry different subsets of a horse's attributes, so a
blind overwrite would erase a horse's colour every time a result was imported.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.data_pipeline.report import ImportSummary
from backend.models import Course, Horse, Jockey, OddsHistory, Race, RaceResult, RaceRunner, Trainer
from backend.services.racing_api.client import DEFAULT_REGIONS, RacingAPIClient
from backend.services.racing_api.schemas import (
    HorseSchema,
    JockeySchema,
    OddsSchema,
    RaceSchema,
    ResultRunnerSchema,
    RunnerSchema,
    TrainerSchema,
)
from backend.utils.exceptions import PipelineError
from backend.utils.logging import get_logger, safe_extra
from backend.utils.timeutils import date_range, format_date, today_uk

logger = get_logger(__name__, channel="pipeline")

TModel = TypeVar("TModel")


def _apply(instance: object, values: dict[str, Any]) -> bool:
    """Assign non-``None`` values, returning True if anything actually changed.

    Skipping ``None`` is what stops a sparse payload from erasing data a richer
    payload already supplied.
    """
    changed = False
    for field, value in values.items():
        if value is None:
            continue
        if getattr(instance, field, None) != value:
            setattr(instance, field, value)
            changed = True
    return changed


class RaceDataImporter:
    """Persists :class:`RaceSchema` objects into the warehouse."""

    def __init__(self, session: Session, *, summary: ImportSummary | None = None) -> None:
        self.session = session
        self.summary = summary or ImportSummary()
        # Entities already upserted in this run — avoids a SELECT per runner for
        # the trainer who saddles twelve horses on the same card.
        self._seen_horses: set[str] = set()
        self._seen_jockeys: set[str] = set()
        self._seen_trainers: set[str] = set()
        self._seen_courses: set[str] = set()

    # ------------------------------------------------------------------
    # Entity upserts
    # ------------------------------------------------------------------
    def _upsert_course(self, course_id: str | None, name: str | None, region: str | None) -> None:
        if not course_id or course_id in self._seen_courses:
            return
        course = self.session.get(Course, course_id)
        if course is None:
            course = Course(course_id=course_id, name=name, region=region)
            self.session.add(course)
            self.summary.courses_upserted += 1
        elif _apply(course, {"name": name, "region": region}):
            self.summary.courses_upserted += 1
        self._seen_courses.add(course_id)

    def _upsert_horse(self, horse: HorseSchema) -> None:
        if horse.horse_id in self._seen_horses:
            return
        existing = self.session.get(Horse, horse.horse_id)
        values = {
            "name": horse.name,
            "sex": horse.sex,
            "sex_code": horse.sex_code,
            "colour": horse.colour,
            "date_of_birth": horse.date_of_birth,
            "region": horse.region,
            "sire": horse.sire,
            "sire_id": horse.sire_id,
            "dam": horse.dam,
            "dam_id": horse.dam_id,
            "damsire": horse.damsire,
            "damsire_id": horse.damsire_id,
        }
        if existing is None:
            record = Horse(horse_id=horse.horse_id)
            _apply(record, values)
            self.session.add(record)
            self.summary.horses_upserted += 1
        elif _apply(existing, values):
            self.summary.horses_upserted += 1
        self._seen_horses.add(horse.horse_id)

    def _upsert_jockey(self, jockey: JockeySchema | None) -> str | None:
        if jockey is None:
            return None
        if jockey.jockey_id not in self._seen_jockeys:
            existing = self.session.get(Jockey, jockey.jockey_id)
            if existing is None:
                self.session.add(Jockey(jockey_id=jockey.jockey_id, name=jockey.name))
                self.summary.jockeys_upserted += 1
            elif _apply(existing, {"name": jockey.name}):
                self.summary.jockeys_upserted += 1
            self._seen_jockeys.add(jockey.jockey_id)
        return jockey.jockey_id

    def _upsert_trainer(self, trainer: TrainerSchema | None) -> str | None:
        if trainer is None:
            return None
        if trainer.trainer_id not in self._seen_trainers:
            existing = self.session.get(Trainer, trainer.trainer_id)
            values = {"name": trainer.name, "location": trainer.location}
            if existing is None:
                record = Trainer(trainer_id=trainer.trainer_id)
                _apply(record, values)
                self.session.add(record)
                self.summary.trainers_upserted += 1
            elif _apply(existing, values):
                self.summary.trainers_upserted += 1
            self._seen_trainers.add(trainer.trainer_id)
        return trainer.trainer_id

    # ------------------------------------------------------------------
    # Race, runners, results, odds
    # ------------------------------------------------------------------
    def _upsert_race(self, race: RaceSchema) -> tuple[Race, bool]:
        """Insert or update the race row. Returns ``(race, created)``."""
        values = {
            "race_date": race.race_date,
            "off_time": race.off_time,
            "course_id": race.course_id,
            "course_name": race.course,
            "race_name": race.race_name,
            "race_class": race.race_class,
            "race_type": race.race_type,
            "pattern": race.pattern,
            "age_band": race.age_band,
            "rating_band": race.rating_band,
            "sex_restriction": race.sex_restriction,
            "distance_yards": race.distance_yards,
            "distance_furlongs": race.distance_furlongs,
            "going": race.going,
            "going_detailed": race.going_detailed,
            "surface": race.surface,
            "jumps": race.jumps,
            "weather": race.weather,
            "region": race.region,
            "prize_money": race.prize_money,
            "field_size": race.field_size,
            "winning_time_detail": race.winning_time_detail,
        }
        existing = self.session.get(Race, race.race_id)
        if existing is None:
            record = Race(race_id=race.race_id)
            _apply(record, values)
            # Booleans are set explicitly: _apply skips falsey-but-meaningful None
            # semantics, and False here is a real value, not "unknown".
            record.is_abandoned = race.is_abandoned
            record.has_result = race.is_result
            self.session.add(record)
            self.session.flush()
            return record, True

        _apply(existing, values)
        if race.is_abandoned:
            existing.is_abandoned = True
        if race.is_result:
            existing.has_result = True
        return existing, False

    def _upsert_runner(self, race_id: str, runner: RunnerSchema) -> None:
        self._upsert_horse(runner.horse)
        jockey_id = self._upsert_jockey(runner.jockey)
        trainer_id = self._upsert_trainer(runner.trainer)

        values = {
            "jockey_id": jockey_id,
            "trainer_id": trainer_id,
            "owner": runner.owner,
            "owner_id": runner.owner_id,
            "saddle_cloth": runner.saddle_cloth,
            "draw": runner.draw,
            "age": runner.age,
            "weight_lbs": runner.weight_lbs,
            "headgear": runner.headgear,
            "official_rating": runner.official_rating,
            "rpr": runner.rpr,
            "topspeed": runner.topspeed,
            "days_since_last_run": runner.days_since_last_run,
            "form": runner.form,
            "comment": runner.comment,
        }
        existing = self.session.execute(
            select(RaceRunner).where(
                RaceRunner.race_id == race_id, RaceRunner.horse_id == runner.horse.horse_id
            )
        ).scalar_one_or_none()

        if existing is None:
            record = RaceRunner(race_id=race_id, horse_id=runner.horse.horse_id)
            _apply(record, values)
            self.session.add(record)
            self.summary.runners_created += 1
        elif _apply(existing, values):
            self.summary.runners_updated += 1

        for quote in runner.odds:
            self._upsert_odds(quote)

    def _upsert_result(self, race_id: str, result: ResultRunnerSchema) -> None:
        self._upsert_horse(result.horse)
        jockey_id = self._upsert_jockey(result.jockey)
        trainer_id = self._upsert_trainer(result.trainer)

        values = {
            "jockey_id": jockey_id,
            "trainer_id": trainer_id,
            "owner": result.owner,
            "owner_id": result.owner_id,
            "finishing_position": result.finishing_position,
            "finishing_status": result.finishing_status,
            "saddle_cloth": result.saddle_cloth,
            "draw": result.draw,
            "age": result.age,
            "weight_lbs": result.weight_lbs,
            "headgear": result.headgear,
            "starting_price": result.starting_price,
            "beaten_lengths": result.beaten_lengths,
            "overall_beaten_lengths": result.overall_beaten_lengths,
            "official_rating": result.official_rating,
            "rpr": result.rpr,
            "topspeed": result.topspeed,
            "finishing_time": result.finishing_time,
            "prize_won": result.prize_won,
            "comment": result.comment,
        }
        existing = self.session.execute(
            select(RaceResult).where(
                RaceResult.race_id == race_id, RaceResult.horse_id == result.horse.horse_id
            )
        ).scalar_one_or_none()

        is_winner = result.finishing_position == 1

        if existing is None:
            record = RaceResult(race_id=race_id, horse_id=result.horse.horse_id)
            _apply(record, values)
            record.is_winner = is_winner
            self.session.add(record)
            self.summary.results_created += 1
        else:
            changed = _apply(existing, values)
            if existing.is_winner != is_winner:
                existing.is_winner = is_winner
                changed = True
            if changed:
                self.summary.results_updated += 1

    def _upsert_odds(self, quote: OddsSchema) -> None:
        """Insert a price only if that exact quote is not already stored."""
        existing = self.session.execute(
            select(OddsHistory.id).where(
                OddsHistory.race_id == quote.race_id,
                OddsHistory.horse_id == quote.horse_id,
                OddsHistory.bookmaker == quote.bookmaker,
                OddsHistory.recorded_at == quote.recorded_at,
            )
        ).first()
        if existing is not None:
            self.summary.odds_skipped += 1
            return

        self.session.add(
            OddsHistory(
                race_id=quote.race_id,
                horse_id=quote.horse_id,
                bookmaker=quote.bookmaker,
                decimal_odds=quote.decimal_odds,
                fractional_odds=quote.fractional_odds,
                implied_probability=quote.implied_probability,
                each_way_places=quote.each_way_places,
                each_way_denominator=quote.each_way_denominator,
                recorded_at=quote.recorded_at,
            )
        )
        self.summary.odds_created += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def import_race(self, race: RaceSchema) -> bool:
        """Import one race inside its own savepoint.

        Returns ``True`` when the race was written, ``False`` when it failed and
        was rolled back. Never raises for a single bad record.
        """
        self.summary.races_seen += 1
        try:
            with self.session.begin_nested():
                self._upsert_course(race.course_id, race.course, race.region)
                record, created = self._upsert_race(race)

                for runner in race.runners:
                    self._upsert_runner(record.race_id, runner)
                for result in race.results:
                    self._upsert_result(record.race_id, result)

                if created:
                    self.summary.races_created += 1
                else:
                    self.summary.races_updated += 1
            return True
        except SQLAlchemyError as exc:
            logger.warning(
                "race import failed; rolled back",
                extra={"race_id": race.race_id, "error": str(exc)[:200]},
            )
            self.summary.record_failure(race.race_id, "persist", exc)
            return False
        except Exception as exc:
            logger.warning(
                "race import failed (unexpected)",
                extra={"race_id": race.race_id, "error": str(exc)[:200]},
            )
            self.summary.record_failure(race.race_id, "persist", exc)
            return False

    def import_races(self, races: Iterable[RaceSchema]) -> ImportSummary:
        """Import many races, committing once at the end."""
        for race in races:
            self.import_race(race)
        self.session.commit()
        self.summary.finish()
        logger.info("import complete", extra=safe_extra(self.summary.as_dict()["races"]))
        return self.summary

    def import_odds(self, quotes: Sequence[OddsSchema]) -> ImportSummary:
        """Import bookmaker quotes for races that already exist."""
        known: set[str] = set()
        for quote in quotes:
            if quote.race_id not in known:
                if self.session.get(Race, quote.race_id) is None:
                    self.summary.record_failure(
                        quote.race_id,
                        "odds",
                        PipelineError("race not in database; import the racecard first"),
                    )
                    continue
                known.add(quote.race_id)
            try:
                with self.session.begin_nested():
                    self._upsert_odds(quote)
            except SQLAlchemyError as exc:
                self.summary.record_failure(f"{quote.race_id}/{quote.horse_id}", "odds", exc)
        self.session.commit()
        self.summary.finish()
        return self.summary


# ---------------------------------------------------------------------------
# Orchestrators -- fetch from the API, then import
# ---------------------------------------------------------------------------
async def import_racecards_for_date(
    session: Session,
    client: RacingAPIClient,
    *,
    race_date: str | None = None,
    regions: Sequence[str] = DEFAULT_REGIONS,
    tier: str = "standard",
) -> ImportSummary:
    """Fetch and store every racecard for one day."""
    race_date = race_date or format_date(today_uk())
    logger.info("importing racecards", extra={"race_date": race_date, "tier": tier})

    races = [race async for race in client.iter_races(date=race_date, region_codes=regions, tier=tier)]
    importer = RaceDataImporter(session)
    importer.summary.api_requests = client.request_count
    importer.summary.api_retries = client.retry_count
    return importer.import_races(races)


async def import_results_for_range(
    session: Session,
    client: RacingAPIClient,
    *,
    start_date: str,
    end_date: str,
    regions: Sequence[str] = DEFAULT_REGIONS,
) -> ImportSummary:
    """Backfill finished races across a date range."""
    logger.info("importing results", extra={"start_date": start_date, "end_date": end_date})

    importer = RaceDataImporter(session)
    async for race in client.iter_results(start_date=start_date, end_date=end_date, region=regions):
        importer.import_race(race)
    session.commit()

    importer.summary.api_requests = client.request_count
    importer.summary.api_retries = client.retry_count
    importer.summary.finish()
    logger.info("results import complete", extra=safe_extra(importer.summary.as_dict()["races"]))
    return importer.summary


async def backfill_day_by_day(
    session: Session,
    client: RacingAPIClient,
    *,
    start_date: str,
    end_date: str,
    regions: Sequence[str] = DEFAULT_REGIONS,
) -> ImportSummary:
    """Backfill one day at a time.

    Slower than a single wide range, but a failure loses one day rather than the
    whole request, and progress is visible in the log as it goes.
    """
    total = ImportSummary(source="racing_api:backfill")
    for day in date_range(start_date, end_date):
        day_string = format_date(day)
        try:
            summary = await import_results_for_range(
                session, client, start_date=day_string, end_date=day_string, regions=regions
            )
            total.merge(summary)
            logger.info(
                "backfill day complete",
                extra={"date": day_string, "races": summary.races_seen, "failed": summary.failed},
            )
        except Exception as exc:
            logger.error("backfill day failed", extra={"date": day_string, "error": str(exc)[:200]})
            total.record_failure(day_string, "backfill", exc)
    return total.finish()


__all__ = [
    "RaceDataImporter",
    "backfill_day_by_day",
    "import_racecards_for_date",
    "import_results_for_range",
]
