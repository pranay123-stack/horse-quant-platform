"""Coercion helpers for The Racing API's string-typed payloads.

Every field the API returns is a JSON **string**, including quantities we need
as numbers: ``age``, ``lbs``, ``ofr``, ``rpr``, ``ts``, ``draw``, and decimal
odds. Missing values are not ``null`` -- they arrive as ``""``, ``"-"``, ``"–"``
or ``"N/A"`` depending on the field and the endpoint.

Naively calling ``int(runner["ofr"])`` therefore raises on perfectly normal rows.
Everything here is total: it returns ``None`` rather than raising, because a
single odd runner must never abort the ingestion of an entire race day.

Domain-specific notes encoded below:

* **Weights** are ``"9-7"`` (stone-pounds) -> 133 lbs.
* **Fractional odds** are ``"5/2"`` -> 3.5 decimal; ``"evens"`` -> 2.0.
* **Distances** are ``"1m2f110y"`` -> 2310 yards.
* **Finishing positions** carry non-finisher codes: ``PU``, ``F``, ``UR``, ``BD``…
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from backend.services.racing_api.schemas import (
    CoursePayload,
    CourseSchema,
    HorsePayload,
    HorseSchema,
    JockeySchema,
    OddsEntryPayload,
    OddsSchema,
    PersonPayload,
    RacecardPayload,
    RacecardRunnerPayload,
    RaceSchema,
    ResultRacePayload,
    ResultRunnerPayload,
    ResultRunnerSchema,
    RunnerOddsPayload,
    RunnerPayloadBase,
    RunnerSchema,
    TrainerSchema,
)
from backend.utils.timeutils import UK_TZ, parse_date

#: Strings the API uses to mean "no value".
NULL_SENTINELS: Final[frozenset[str]] = frozenset(
    {"", "-", "--", "–", "—", "n/a", "na", "none", "null", "nan", "?", "unknown", "tbc"}
)

#: Non-finisher codes seen in ``position`` on results runners.
NON_FINISHER_CODES: Final[dict[str, str]] = {
    "PU": "pulled_up",
    "P": "pulled_up",
    "F": "fell",
    "FL": "fell",
    "UR": "unseated_rider",
    "U": "unseated_rider",
    "BD": "brought_down",
    "B": "brought_down",
    "RR": "refused_to_race",
    "REF": "refused",
    "R": "refused",
    "SU": "slipped_up",
    "S": "slipped_up",
    "RO": "ran_out",
    "CO": "carried_out",
    "LFT": "left_at_start",
    "DSQ": "disqualified",
    "DISQ": "disqualified",
    "VOI": "void",
    "NR": "non_runner",
    "WD": "withdrawn",
}

_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")
_DISTANCE_RE = re.compile(r"(?:(\d+)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*f)?\s*(?:(\d+)\s*y)?", re.IGNORECASE)
_WEIGHT_RE = re.compile(r"^\s*(\d+)\s*[-–]\s*(\d+)\s*$")

YARDS_PER_MILE: Final[int] = 1760
YARDS_PER_FURLONG: Final[int] = 220


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def clean_str(value: Any) -> str | None:
    """Trim a value, mapping the API's null sentinels onto ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in NULL_SENTINELS:
            return None
        return stripped or None
    return str(value)


def to_int(value: Any) -> int | None:
    """Best-effort integer. ``"7"`` -> 7, ``"7th"`` -> 7, ``"-"`` -> None."""
    cleaned = clean_str(value)
    if cleaned is None:
        return None
    if isinstance(value, bool):
        return int(value)
    match = _NUMERIC_RE.search(cleaned.replace(",", ""))
    if not match:
        return None
    try:
        return int(float(match.group()))
    except (ValueError, OverflowError):
        return None


def to_float(value: Any) -> float | None:
    cleaned = clean_str(value)
    if cleaned is None:
        return None
    match = _NUMERIC_RE.search(cleaned.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group())
    except (ValueError, OverflowError):
        return None


def to_decimal(value: Any) -> Decimal | None:
    """Exact decimal -- used for money and odds, where float drift is unacceptable."""
    cleaned = clean_str(value)
    if cleaned is None:
        return None
    match = _NUMERIC_RE.search(cleaned.replace(",", ""))
    if not match:
        return None
    try:
        return Decimal(match.group())
    except (InvalidOperation, ValueError):
        return None


def to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    cleaned = clean_str(value)
    if cleaned is None:
        return None
    lowered = cleaned.lower()
    if lowered in {"true", "yes", "y", "1"}:
        return True
    if lowered in {"false", "no", "n", "0"}:
        return False
    return None


# ---------------------------------------------------------------------------
# Dates and times
# ---------------------------------------------------------------------------
def to_date(value: Any) -> date | None:
    """Parse ``YYYY-MM-DD`` (the format used by ``date`` on every endpoint)."""
    cleaned = clean_str(value)
    if cleaned is None:
        return None
    try:
        return parse_date(cleaned[:10])
    except ValueError:
        return None


def to_datetime(value: Any) -> datetime | None:
    """Parse ``off_dt`` (ISO 8601). Naive values are assumed UK local."""
    cleaned = clean_str(value)
    if cleaned is None:
        return None
    text = cleaned.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UK_TZ) if parsed.tzinfo is None else parsed


def combine_race_datetime(race_date: date | None, off_time: str | None) -> datetime | None:
    """Build an off-time from ``date`` + ``off_time`` when ``off_dt`` is absent.

    UK racecards print 24-hour times without a date (``"14:35"``). Afternoon
    cards occasionally appear as ``"2:35"``; races before 10:00 do not exist in
    UK racing, so a sub-10 hour is unambiguously an afternoon time.
    """
    if race_date is None:
        return None
    cleaned = clean_str(off_time)
    if cleaned is None:
        return None
    match = re.match(r"^(\d{1,2})[:.](\d{2})", cleaned)
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour < 10:
        hour += 12
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return datetime(race_date.year, race_date.month, race_date.day, hour, minute, tzinfo=UK_TZ)


# ---------------------------------------------------------------------------
# Racing-specific quantities
# ---------------------------------------------------------------------------
def parse_prize(value: Any) -> Decimal | None:
    """``"£12,345"`` -> ``Decimal("12345")``. Currency symbols are discarded."""
    cleaned = clean_str(value)
    if cleaned is None:
        return None
    stripped = re.sub(r"[^\d.]", "", cleaned)
    if not stripped or stripped == ".":
        return None
    try:
        return Decimal(stripped)
    except InvalidOperation:
        return None


def parse_weight_lbs(value: Any) -> int | None:
    """``"9-7"`` (9 stone 7 lb) -> 133. A plain number is taken as pounds."""
    cleaned = clean_str(value)
    if cleaned is None:
        return None
    match = _WEIGHT_RE.match(cleaned)
    if match:
        return int(match.group(1)) * 14 + int(match.group(2))
    return to_int(cleaned)


def parse_fractional_odds(value: Any) -> Decimal | None:
    """``"5/2"`` -> ``Decimal("3.5")``. ``"evens"`` -> 2. Returns *decimal* odds."""
    cleaned = clean_str(value)
    if cleaned is None:
        return None
    lowered = cleaned.lower()
    if lowered in {"evens", "evs", "even", "1/1"}:
        return Decimal("2")
    if lowered in {"sp", "n/r", "nr"}:
        return None
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[/-]\s*(\d+(?:\.\d+)?)\s*$", cleaned)
    if not match:
        return None
    numerator, denominator = Decimal(match.group(1)), Decimal(match.group(2))
    if denominator == 0:
        return None
    return (numerator / denominator) + 1


def parse_decimal_odds(value: Any) -> Decimal | None:
    """Decimal odds, rejecting anything below 1.0 (an impossible price)."""
    odds = to_decimal(value)
    if odds is None or odds < 1:
        return None
    return odds


def resolve_odds(decimal_value: Any, fractional_value: Any) -> Decimal | None:
    """Prefer the decimal price; fall back to converting the fraction."""
    return parse_decimal_odds(decimal_value) or parse_fractional_odds(fractional_value)


def implied_probability(decimal_odds: Decimal | float | None) -> float | None:
    """``1 / odds`` -- the bookmaker's price expressed as a probability.

    Includes the overround; normalising across a race happens in Phase 5.
    """
    if decimal_odds is None:
        return None
    odds = float(decimal_odds)
    if odds < 1:
        return None
    return 1.0 / odds


def parse_distance_yards(*values: Any) -> int | None:
    """Yards from any of the distance representations the API returns.

    Accepts an explicit yardage (``dist_y``), a compound string
    (``"1m2f110y"``), or furlongs (``"10.5"``). The first value that yields a
    plausible distance wins, so callers can pass their preference order.
    """
    for value in values:
        cleaned = clean_str(value)
        if cleaned is None:
            continue

        # Pure number: yards if large, furlongs if small.
        if re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
            number = float(cleaned)
            if number > 100:
                return int(number)
            if number > 0:
                return round(number * YARDS_PER_FURLONG)
            continue

        match = _DISTANCE_RE.fullmatch(cleaned.replace(" ", ""))
        if match and any(match.groups()):
            miles = float(match.group(1) or 0)
            furlongs = float(match.group(2) or 0)
            yards = float(match.group(3) or 0)
            total = miles * YARDS_PER_MILE + furlongs * YARDS_PER_FURLONG + yards
            if total > 0:
                return round(total)
    return None


def yards_to_furlongs(yards: int | None) -> float | None:
    if yards is None:
        return None
    return round(yards / YARDS_PER_FURLONG, 2)


def parse_position(value: Any) -> tuple[int | None, str]:
    """Split a finishing position into ``(position, status)``.

    ``"1"`` -> ``(1, "finished")``; ``"PU"`` -> ``(None, "pulled_up")``.
    Dead heats arrive as ``"1="`` and keep their numeric position.
    """
    cleaned = clean_str(value)
    if cleaned is None:
        return None, "unknown"

    normalised = cleaned.upper().replace("=", "").strip()
    if normalised in NON_FINISHER_CODES:
        return None, NON_FINISHER_CODES[normalised]

    position = to_int(normalised)
    if position is not None and position > 0:
        return position, "finished"
    return None, "unknown"


def parse_going(value: Any) -> str | None:
    """Normalise going descriptions to lower-case, whitespace-collapsed text."""
    cleaned = clean_str(value)
    if cleaned is None:
        return None
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def parse_race_class(value: Any) -> int | None:
    """``"Class 4"`` -> 4. Non-class races (many jumps races) return ``None``."""
    return to_int(value)


def parse_form_string(value: Any) -> list[str]:
    """Split a form string like ``"1-P23"`` into per-run tokens, newest last.

    Separators (``-`` season break, ``/`` year break) are dropped; the caller
    gets only the run outcomes, which is what the Phase 5 form features need.
    """
    cleaned = clean_str(value)
    if cleaned is None:
        return []
    return [char for char in cleaned.upper() if char.isalnum()]


# ---------------------------------------------------------------------------
# Field-name normalisation
# ---------------------------------------------------------------------------
#: The same concept is named differently on racecards and results. Left-hand
#: side is our canonical name; the tuple lists the wire names in priority order.
RACE_FIELD_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "race_id": ("race_id",),
    "course": ("course",),
    "course_id": ("course_id",),
    "race_date": ("date",),
    "off_time": ("off_time", "off"),
    "off_dt": ("off_dt",),
    "race_name": ("race_name",),
    "race_class": ("race_class", "class"),
    "race_type": ("type",),
    "pattern": ("pattern",),
    "age_band": ("age_band",),
    "rating_band": ("rating_band",),
    "sex_restriction": ("sex_restriction", "sex_rest"),
    "going": ("going",),
    "going_detailed": ("going_detailed",),
    "surface": ("surface",),
    "jumps": ("jumps",),
    "region": ("region",),
    "prize": ("prize",),
    "field_size": ("field_size",),
    "weather": ("weather",),
    "stalls": ("stalls",),
    "rail_movements": ("rail_movements",),
    "is_abandoned": ("is_abandoned",),
    "big_race": ("big_race",),
    "race_status": ("race_status",),
    "winning_time_detail": ("winning_time_detail",),
    "comments": ("comments",),
    "non_runners": ("non_runners",),
    "tote_win": ("tote_win",),
}

#: Distance fields, in the order we prefer to read them.
DISTANCE_FIELDS: Final[tuple[str, ...]] = (
    "dist_y",
    "distance_round",
    "dist",
    "distance",
    "dist_f",
    "distance_f",
)


def pick(payload: dict[str, Any], *names: str) -> Any:
    """First present, non-sentinel value among ``names``."""
    for name in names:
        if name in payload:
            value = payload[name]
            if clean_str(value) is not None or isinstance(value, bool):
                return value
    return None


def normalise_race_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Map a racecard *or* a result race onto one canonical field set."""
    return {canonical: pick(payload, *aliases) for canonical, aliases in RACE_FIELD_ALIASES.items()}


# ---------------------------------------------------------------------------
# Wire payload -> domain schema
# ---------------------------------------------------------------------------
def build_horse(runner: RunnerPayloadBase) -> HorseSchema:
    return HorseSchema(
        horse_id=runner.horse_id,
        name=clean_str(runner.horse),
        sex=clean_str(runner.sex),
        sex_code=clean_str(runner.sex_code),
        colour=clean_str(runner.colour),
        date_of_birth=to_date(runner.dob),
        region=clean_str(runner.region),
        sire=clean_str(runner.sire),
        sire_id=clean_str(runner.sire_id),
        dam=clean_str(runner.dam),
        dam_id=clean_str(runner.dam_id),
        damsire=clean_str(runner.damsire),
        damsire_id=clean_str(runner.damsire_id),
    )


def build_jockey(runner: RunnerPayloadBase) -> JockeySchema | None:
    jockey_id = clean_str(runner.jockey_id)
    if jockey_id is None:
        return None
    return JockeySchema(jockey_id=jockey_id, name=clean_str(runner.jockey))


def build_trainer(runner: RunnerPayloadBase) -> TrainerSchema | None:
    trainer_id = clean_str(runner.trainer_id)
    if trainer_id is None:
        return None
    return TrainerSchema(
        trainer_id=trainer_id,
        name=clean_str(runner.trainer),
        location=clean_str(getattr(runner, "trainer_location", None)),
    )


def build_odds(
    race_id: str,
    horse_id: str,
    entry: OddsEntryPayload,
    *,
    fallback_time: datetime | None = None,
) -> OddsSchema | None:
    """Convert one bookmaker quote. Returns ``None`` when there is no usable price."""
    bookmaker = clean_str(entry.bookmaker)
    if bookmaker is None:
        return None
    decimal_odds = resolve_odds(entry.decimal, entry.fractional)
    if decimal_odds is None:
        return None
    return OddsSchema(
        race_id=race_id,
        horse_id=horse_id,
        bookmaker=bookmaker,
        decimal_odds=decimal_odds,
        fractional_odds=clean_str(entry.fractional),
        implied_probability=implied_probability(decimal_odds),
        each_way_places=to_int(entry.ew_places),
        each_way_denominator=to_int(entry.ew_denom),
        recorded_at=to_datetime(entry.updated) or fallback_time,
    )


def build_runner(race_id: str, runner: RacecardRunnerPayload) -> RunnerSchema:
    return RunnerSchema(
        race_id=race_id,
        horse=build_horse(runner),
        jockey=build_jockey(runner),
        trainer=build_trainer(runner),
        owner=clean_str(runner.owner),
        owner_id=clean_str(runner.owner_id),
        saddle_cloth=to_int(runner.number),
        draw=to_int(runner.draw),
        age=to_int(runner.age),
        weight_lbs=parse_weight_lbs(runner.lbs),
        headgear=clean_str(runner.headgear),
        official_rating=to_int(runner.official_rating),
        rpr=to_int(runner.rpr),
        topspeed=to_int(runner.ts),
        days_since_last_run=to_int(runner.last_run),
        form=clean_str(runner.form),
        form_tokens=parse_form_string(runner.form),
        comment=clean_str(runner.comment),
        odds=[
            odds
            for odds in (build_odds(race_id, runner.horse_id, entry) for entry in (runner.odds or []))
            if odds is not None
        ],
    )


def build_result_runner(race_id: str, runner: ResultRunnerPayload) -> ResultRunnerSchema:
    position, status = parse_position(runner.position)
    return ResultRunnerSchema(
        race_id=race_id,
        horse=build_horse(runner),
        jockey=build_jockey(runner),
        trainer=build_trainer(runner),
        owner=clean_str(runner.owner),
        owner_id=clean_str(runner.owner_id),
        finishing_position=position,
        finishing_status=status,
        saddle_cloth=to_int(runner.number),
        draw=to_int(runner.draw),
        age=to_int(runner.age),
        weight_lbs=parse_weight_lbs(runner.weight_lbs) or parse_weight_lbs(runner.weight),
        headgear=clean_str(runner.headgear),
        starting_price=resolve_odds(runner.sp_dec, runner.sp),
        beaten_lengths=to_float(runner.btn),
        overall_beaten_lengths=to_float(runner.ovr_btn),
        official_rating=to_int(runner.official_rating),
        rpr=to_int(runner.rpr),
        topspeed=to_int(runner.tsr),
        finishing_time=clean_str(runner.time),
        prize_won=parse_prize(runner.prize),
        comment=clean_str(runner.comment),
    )


def parse_racecard(payload: RacecardPayload) -> RaceSchema:
    """Racecard payload -> canonical :class:`RaceSchema` (pre-race)."""
    race_date = to_date(payload.date)
    distance_yards = parse_distance_yards(payload.distance_round, payload.distance, payload.distance_f)
    return RaceSchema(
        race_id=payload.race_id,
        race_date=race_date,
        off_time=to_datetime(payload.off_dt) or combine_race_datetime(race_date, payload.off_time),
        course=clean_str(payload.course),
        course_id=clean_str(payload.course_id),
        race_name=clean_str(payload.race_name),
        race_class=parse_race_class(payload.race_class),
        race_type=clean_str(payload.type),
        pattern=clean_str(payload.pattern),
        age_band=clean_str(payload.age_band),
        rating_band=clean_str(payload.rating_band),
        sex_restriction=clean_str(payload.sex_restriction),
        distance_yards=distance_yards,
        distance_furlongs=yards_to_furlongs(distance_yards),
        going=parse_going(payload.going),
        going_detailed=clean_str(payload.going_detailed),
        surface=clean_str(payload.surface),
        jumps=clean_str(payload.jumps),
        region=clean_str(payload.region),
        prize_money=parse_prize(payload.prize),
        field_size=to_int(payload.field_size),
        weather=clean_str(payload.weather),
        is_abandoned=bool(payload.is_abandoned),
        is_result=False,
        runners=[build_runner(payload.race_id, runner) for runner in payload.runners],
    )


def parse_result(payload: ResultRacePayload) -> RaceSchema:
    """Result payload -> canonical :class:`RaceSchema` (post-race)."""
    race_date = to_date(payload.date)
    distance_yards = parse_distance_yards(payload.dist_y, payload.dist, payload.dist_f)
    return RaceSchema(
        race_id=payload.race_id,
        race_date=race_date,
        off_time=to_datetime(payload.off_dt) or combine_race_datetime(race_date, payload.off_time),
        course=clean_str(payload.course),
        course_id=clean_str(payload.course_id),
        race_name=clean_str(payload.race_name),
        race_class=parse_race_class(payload.race_class),
        race_type=clean_str(payload.type),
        pattern=clean_str(payload.pattern),
        age_band=clean_str(payload.age_band),
        rating_band=clean_str(payload.rating_band),
        sex_restriction=clean_str(payload.sex_restriction),
        distance_yards=distance_yards,
        distance_furlongs=yards_to_furlongs(distance_yards),
        going=parse_going(payload.going),
        surface=clean_str(payload.surface),
        jumps=clean_str(payload.jumps),
        region=clean_str(payload.region),
        winning_time_detail=clean_str(payload.winning_time_detail),
        field_size=len(payload.runners) or None,
        is_result=True,
        results=[build_result_runner(payload.race_id, runner) for runner in payload.runners],
    )


def parse_course(payload: CoursePayload) -> CourseSchema:
    return CourseSchema(
        course_id=payload.id,
        name=clean_str(payload.course),
        region=clean_str(payload.region),
        region_code=clean_str(payload.region_code),
    )


def parse_horse(payload: HorsePayload) -> HorseSchema:
    return HorseSchema(
        horse_id=payload.id,
        name=clean_str(payload.name),
        sex=clean_str(payload.sex),
        sex_code=clean_str(payload.sex_code),
        colour=clean_str(payload.colour),
        date_of_birth=to_date(payload.dob),
        sire=clean_str(payload.sire),
        sire_id=clean_str(payload.sire_id),
        dam=clean_str(payload.dam),
        dam_id=clean_str(payload.dam_id),
        damsire=clean_str(payload.damsire),
        damsire_id=clean_str(payload.damsire_id),
    )


def parse_jockey(payload: PersonPayload) -> JockeySchema:
    return JockeySchema(jockey_id=payload.id, name=clean_str(payload.name))


def parse_trainer(payload: PersonPayload) -> TrainerSchema:
    return TrainerSchema(trainer_id=payload.id, name=clean_str(payload.name))


def parse_runner_odds(payload: RunnerOddsPayload) -> list[OddsSchema]:
    race_id = clean_str(payload.race_id)
    horse_id = clean_str(payload.horse_id)
    if race_id is None or horse_id is None:
        return []
    return [
        odds for odds in (build_odds(race_id, horse_id, entry) for entry in payload.odds) if odds is not None
    ]


__all__ = [
    "DISTANCE_FIELDS",
    "NON_FINISHER_CODES",
    "NULL_SENTINELS",
    "RACE_FIELD_ALIASES",
    "build_horse",
    "build_jockey",
    "build_odds",
    "build_result_runner",
    "build_runner",
    "build_trainer",
    "clean_str",
    "combine_race_datetime",
    "implied_probability",
    "normalise_race_fields",
    "parse_course",
    "parse_decimal_odds",
    "parse_distance_yards",
    "parse_form_string",
    "parse_fractional_odds",
    "parse_going",
    "parse_horse",
    "parse_jockey",
    "parse_position",
    "parse_prize",
    "parse_race_class",
    "parse_racecard",
    "parse_result",
    "parse_runner_odds",
    "parse_trainer",
    "parse_weight_lbs",
    "pick",
    "resolve_odds",
    "to_bool",
    "to_date",
    "to_datetime",
    "to_decimal",
    "to_float",
    "to_int",
    "yards_to_furlongs",
]
