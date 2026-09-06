"""Individual data-quality rules.

Each rule is a small, pure function over one record. Keeping them separate from
the orchestration in :mod:`backend.data_quality.validator` means every threshold
is independently testable and independently arguable — which matters, because
the thresholds below are *domain judgements*, not universal truths.

Severity has a precise operational meaning:

``ERROR``    the record is unusable; it is excluded from the research dataset.
``WARNING``  the record is suspicious but usable; it is kept and flagged.

Getting that split wrong is expensive in both directions. Rejecting too much
silently shrinks the training set and biases it (bad data is rarely random).
Rejecting too little poisons the model with impossible rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Domain bounds. Every number here is a claim about UK racing, so each is cited.
# ---------------------------------------------------------------------------

#: Thoroughbreds race from 2yo (Flat) to roughly 14yo (jumps). Beyond that the
#: value is far more likely to be a data error than a genuine veteran.
MIN_HORSE_AGE = 2
MAX_HORSE_AGE = 15

#: Carried weight in pounds. Apprentice claims bottom out near 7st 7lb (105lb);
#: top weights in staying chases reach about 12st 7lb (175lb).
MIN_WEIGHT_LBS = 95
MAX_WEIGHT_LBS = 190

#: UK race distances: 5 furlongs (1100y) to the Grand National (4m2f, ~7480y).
MIN_DISTANCE_YARDS = 1000
MAX_DISTANCE_YARDS = 8000

#: UK Flat and Jumps classes are 1-7 (Class 1 is the highest).
MIN_RACE_CLASS = 1
MAX_RACE_CLASS = 7

#: Decimal odds. 1.0 is a free bet; anything at or below it is corrupt.
#: The upper bound is generous — 1000/1 shows up in big-field handicaps.
MIN_DECIMAL_ODDS = Decimal("1.01")
MAX_DECIMAL_ODDS = Decimal("1001")

#: A field of one is a walkover (real, but useless for modelling); 40 is the
#: Grand National maximum.
MIN_FIELD_SIZE = 2
MAX_FIELD_SIZE = 40

#: Racing data older than this is likely a typo; the Racing API's history does
#: not reach back to the 19th century.
EARLIEST_PLAUSIBLE_RACE_DATE = date(1990, 1, 1)

#: A race date this far in the future is a parsing error, not a declaration.
MAX_FUTURE_DAYS = 400


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One rule violation on one record."""

    entity: str
    entity_id: str
    rule: str
    severity: Severity
    message: str
    value: Any = None

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "entity_id": self.entity_id,
            "rule": self.rule,
            "severity": str(self.severity),
            "message": self.message,
            "value": str(self.value) if self.value is not None else None,
        }


def _issue(
    entity: str,
    entity_id: str,
    rule: str,
    severity: Severity,
    message: str,
    value: Any = None,
) -> ValidationIssue:
    return ValidationIssue(entity, entity_id or "<unknown>", rule, severity, message, value)


# ---------------------------------------------------------------------------
# Race rules
# ---------------------------------------------------------------------------
def check_race_date(
    race_id: str, race_date: date | None, *, today: date | None = None
) -> list[ValidationIssue]:
    """A race without a usable date cannot be time-ordered, so it cannot be used.

    Everything in this platform is point-in-time: features read "all races before
    this one". A missing or absurd date makes that ordering meaningless, which is
    why this is an ERROR rather than a warning.
    """
    if race_date is None:
        return [_issue("race", race_id, "race_date_missing", Severity.ERROR, "race has no date")]

    issues: list[ValidationIssue] = []
    if race_date < EARLIEST_PLAUSIBLE_RACE_DATE:
        issues.append(
            _issue(
                "race",
                race_id,
                "race_date_implausible",
                Severity.ERROR,
                f"race date {race_date} predates {EARLIEST_PLAUSIBLE_RACE_DATE}",
                race_date,
            )
        )
    horizon = (today or date.today()) + timedelta(days=MAX_FUTURE_DAYS)
    if race_date > horizon:
        issues.append(
            _issue(
                "race",
                race_id,
                "race_date_too_far_ahead",
                Severity.ERROR,
                f"race date {race_date} is more than {MAX_FUTURE_DAYS} days ahead",
                race_date,
            )
        )
    return issues


def check_race_distance(race_id: str, distance_yards: int | None) -> list[ValidationIssue]:
    if distance_yards is None:
        return [
            _issue(
                "race",
                race_id,
                "distance_missing",
                Severity.ERROR,
                "race has no distance; distance features cannot be computed",
            )
        ]
    if not MIN_DISTANCE_YARDS <= distance_yards <= MAX_DISTANCE_YARDS:
        return [
            _issue(
                "race",
                race_id,
                "distance_out_of_range",
                Severity.ERROR,
                f"distance {distance_yards}y outside {MIN_DISTANCE_YARDS}-{MAX_DISTANCE_YARDS}y",
                distance_yards,
            )
        ]
    return []


def check_race_class(race_id: str, race_class: int | None) -> list[ValidationIssue]:
    """Missing class is a WARNING, not an ERROR.

    A large share of genuine UK jumps races carry no class at all. Rejecting them
    would discard real data and skew the sample towards Flat racing.
    """
    if race_class is None:
        return [
            _issue(
                "race", race_id, "race_class_missing", Severity.WARNING, "race has no class (common in jumps)"
            )
        ]
    if not MIN_RACE_CLASS <= race_class <= MAX_RACE_CLASS:
        return [
            _issue(
                "race",
                race_id,
                "race_class_out_of_range",
                Severity.ERROR,
                f"class {race_class} outside {MIN_RACE_CLASS}-{MAX_RACE_CLASS}",
                race_class,
            )
        ]
    return []


def check_field_size(race_id: str, field_size: int | None, runner_count: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if runner_count < MIN_FIELD_SIZE:
        issues.append(
            _issue(
                "race",
                race_id,
                "field_too_small",
                Severity.ERROR,
                f"only {runner_count} runner(s); a race needs at least {MIN_FIELD_SIZE}",
                runner_count,
            )
        )
    elif runner_count > MAX_FIELD_SIZE:
        issues.append(
            _issue(
                "race",
                race_id,
                "field_too_large",
                Severity.WARNING,
                f"{runner_count} runners exceeds the usual maximum of {MAX_FIELD_SIZE}",
                runner_count,
            )
        )
    if field_size is not None and runner_count and abs(field_size - runner_count) > 2:
        issues.append(
            _issue(
                "race",
                race_id,
                "field_size_mismatch",
                Severity.WARNING,
                f"declared field size {field_size} disagrees with {runner_count} stored runners",
                field_size,
            )
        )
    return issues


def check_duplicate_race(
    race_id: str, course_id: str | None, race_date: date | None, off_time: Any, seen: dict[tuple, str]
) -> list[ValidationIssue]:
    """Two race ids at the same course, date and time are the same race.

    ``seen`` is mutated: the first occurrence claims the slot, later ones are
    reported as duplicates.
    """
    if course_id is None or race_date is None or off_time is None:
        return []
    key = (course_id, race_date, off_time)
    first = seen.get(key)
    if first is None:
        seen[key] = race_id
        return []
    if first == race_id:
        return []
    return [
        _issue(
            "race",
            race_id,
            "duplicate_race",
            Severity.ERROR,
            f"same course/date/off-time as race {first}",
            first,
        )
    ]


# ---------------------------------------------------------------------------
# Horse / runner rules
# ---------------------------------------------------------------------------
def check_horse_id(race_id: str, horse_id: str | None) -> list[ValidationIssue]:
    if not horse_id:
        return [_issue("runner", race_id, "horse_id_missing", Severity.ERROR, "runner has no horse id")]
    return []


def check_horse_age(runner_id: str, age: int | None) -> list[ValidationIssue]:
    if age is None:
        return [_issue("runner", runner_id, "age_missing", Severity.WARNING, "runner has no age")]
    if not MIN_HORSE_AGE <= age <= MAX_HORSE_AGE:
        return [
            _issue(
                "runner",
                runner_id,
                "age_impossible",
                Severity.ERROR,
                f"age {age} outside {MIN_HORSE_AGE}-{MAX_HORSE_AGE}",
                age,
            )
        ]
    return []


def check_weight(runner_id: str, weight_lbs: int | None) -> list[ValidationIssue]:
    if weight_lbs is None:
        return [_issue("runner", runner_id, "weight_missing", Severity.WARNING, "runner has no weight")]
    if not MIN_WEIGHT_LBS <= weight_lbs <= MAX_WEIGHT_LBS:
        return [
            _issue(
                "runner",
                runner_id,
                "weight_out_of_range",
                Severity.ERROR,
                f"weight {weight_lbs}lb outside {MIN_WEIGHT_LBS}-{MAX_WEIGHT_LBS}lb",
                weight_lbs,
            )
        ]
    return []


def check_finishing_position(
    runner_id: str, position: int | None, status: str, runner_count: int
) -> list[ValidationIssue]:
    """A non-finisher legitimately has no position; a finisher must have one."""
    if position is None:
        if status in {"unknown", "finished"}:
            return [
                _issue(
                    "result",
                    runner_id,
                    "position_missing",
                    Severity.WARNING,
                    f"no finishing position but status is {status!r}",
                    status,
                )
            ]
        return []
    if position < 1:
        return [
            _issue(
                "result", runner_id, "position_invalid", Severity.ERROR, f"position {position} < 1", position
            )
        ]
    if runner_count and position > runner_count:
        return [
            _issue(
                "result",
                runner_id,
                "position_exceeds_field",
                Severity.ERROR,
                f"position {position} in a field of {runner_count}",
                position,
            )
        ]
    return []


def check_single_winner(race_id: str, winner_count: int) -> list[ValidationIssue]:
    """Exactly one winner, except for genuine dead heats."""
    if winner_count == 0:
        return [_issue("race", race_id, "no_winner", Severity.ERROR, "result has no winner", winner_count)]
    if winner_count > 1:
        return [
            _issue(
                "race",
                race_id,
                "multiple_winners",
                Severity.WARNING,
                f"{winner_count} winners — dead heat, or duplicated results",
                winner_count,
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Odds rules
# ---------------------------------------------------------------------------
def check_decimal_odds(quote_id: str, odds: Decimal | float | None) -> list[ValidationIssue]:
    if odds is None:
        return [_issue("odds", quote_id, "odds_missing", Severity.ERROR, "quote has no decimal odds")]
    value = Decimal(str(odds))
    if value <= 1:
        return [
            _issue(
                "odds",
                quote_id,
                "odds_not_above_one",
                Severity.ERROR,
                f"decimal odds {value} imply a risk-free return",
                value,
            )
        ]
    if value < MIN_DECIMAL_ODDS or value > MAX_DECIMAL_ODDS:
        return [
            _issue(
                "odds",
                quote_id,
                "odds_out_of_range",
                Severity.ERROR,
                f"decimal odds {value} outside {MIN_DECIMAL_ODDS}-{MAX_DECIMAL_ODDS}",
                value,
            )
        ]
    return []


def check_odds_timestamp(quote_id: str, recorded_at: Any) -> list[ValidationIssue]:
    """An undated quote cannot be placed before or after the off.

    That makes it unusable for market-drift features and, worse, a look-ahead
    risk: we cannot prove it was available pre-race. Excluded.
    """
    if recorded_at is None:
        return [
            _issue(
                "odds",
                quote_id,
                "odds_timestamp_missing",
                Severity.ERROR,
                "quote has no timestamp; cannot prove it predates the off",
            )
        ]
    return []


def check_odds_before_off(quote_id: str, recorded_at: Any, off_time: Any) -> list[ValidationIssue]:
    """A quote recorded after the off is post-race information."""
    if recorded_at is None or off_time is None:
        return []
    if recorded_at > off_time:
        return [
            _issue(
                "odds",
                quote_id,
                "odds_after_off",
                Severity.ERROR,
                f"quote recorded at {recorded_at} is after the off at {off_time}",
                recorded_at,
            )
        ]
    return []


__all__ = [
    "EARLIEST_PLAUSIBLE_RACE_DATE",
    "MAX_DECIMAL_ODDS",
    "MAX_DISTANCE_YARDS",
    "MAX_FIELD_SIZE",
    "MAX_FUTURE_DAYS",
    "MAX_HORSE_AGE",
    "MAX_RACE_CLASS",
    "MAX_WEIGHT_LBS",
    "MIN_DECIMAL_ODDS",
    "MIN_DISTANCE_YARDS",
    "MIN_FIELD_SIZE",
    "MIN_HORSE_AGE",
    "MIN_RACE_CLASS",
    "MIN_WEIGHT_LBS",
    "Severity",
    "ValidationIssue",
    "check_decimal_odds",
    "check_duplicate_race",
    "check_field_size",
    "check_finishing_position",
    "check_horse_age",
    "check_horse_id",
    "check_odds_before_off",
    "check_odds_timestamp",
    "check_race_class",
    "check_race_date",
    "check_race_distance",
    "check_single_winner",
    "check_weight",
]
