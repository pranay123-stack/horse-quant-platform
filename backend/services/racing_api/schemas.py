"""Pydantic contracts for The Racing API.

Two layers, deliberately separated:

**Wire payloads** (``*Payload``) mirror exactly what the API sends -- every field
optional, every value a string, ``extra="ignore"`` so that a new upstream field
never breaks ingestion. Their only job is to prove the response is *shaped*
right before we touch it.

**Domain schemas** (``RaceSchema``, ``HorseSchema``, ``OddsSchema``…) are what
the rest of the platform consumes: real ``int``/``Decimal``/``date`` types, one
canonical field name per concept. :mod:`backend.services.racing_api.parsers`
converts between them.

Schema field names verified against The Racing API OpenAPI spec v1.4.3.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Wire payloads -- lenient by design
# ---------------------------------------------------------------------------
WIRE_CONFIG = ConfigDict(extra="ignore", populate_by_name=True, str_strip_whitespace=True)


class OddsEntryPayload(BaseModel):
    """One bookmaker's price for one runner."""

    model_config = WIRE_CONFIG

    bookmaker: str | None = None
    fractional: str | None = None
    decimal: str | None = None
    ew_places: str | None = None
    ew_denom: str | None = None
    updated: str | None = None
    history: list[dict[str, Any]] | None = None


class RunnerPayloadBase(BaseModel):
    """Fields common to racecard runners and result runners."""

    model_config = WIRE_CONFIG

    horse_id: str
    horse: str | None = None
    age: str | None = None
    sex: str | None = None
    sex_code: str | None = None
    colour: str | None = None
    region: str | None = None
    dob: str | None = None
    number: str | None = None
    draw: str | None = None
    headgear: str | None = None

    jockey: str | None = None
    jockey_id: str | None = None
    trainer: str | None = None
    trainer_id: str | None = None
    owner: str | None = None
    owner_id: str | None = None

    sire: str | None = None
    sire_id: str | None = None
    dam: str | None = None
    dam_id: str | None = None
    damsire: str | None = None
    damsire_id: str | None = None

    silk_url: str | None = None
    comment: str | None = None


class RacecardRunnerPayload(RunnerPayloadBase):
    """A declared runner on a racecard (pre-race)."""

    # Racecards call the official rating "ofr"; results call it "or".
    official_rating: str | None = Field(default=None, validation_alias=AliasChoices("ofr", "or"))
    rpr: str | None = None
    ts: str | None = None
    lbs: str | None = None
    last_run: str | None = None
    form: str | None = None
    trainer_rtf: str | None = None
    trainer_location: str | None = None
    spotlight: str | None = None
    wind_surgery: str | None = None
    past_results_flags: list[str] | None = None
    odds: list[OddsEntryPayload] | None = None


class ResultRunnerPayload(RunnerPayloadBase):
    """A runner as it finished (post-race)."""

    official_rating: str | None = Field(default=None, validation_alias=AliasChoices("or", "ofr"))
    position: str | None = None
    sp: str | None = None
    sp_dec: str | None = None
    btn: str | None = None
    ovr_btn: str | None = None
    weight: str | None = None
    weight_lbs: str | None = None
    time: str | None = None
    prize: str | None = None
    rpr: str | None = None
    tsr: str | None = None
    jockey_claim_lbs: str | None = None


class RacecardPayload(BaseModel):
    """A racecard (pre-race declaration) as returned by ``/v1/racecards/*``."""

    model_config = WIRE_CONFIG

    race_id: str
    course: str | None = None
    course_id: str | None = None
    date: str | None = None
    off_time: str | None = None
    off_dt: str | None = None
    race_name: str | None = None
    race_class: str | None = None
    type: str | None = None
    pattern: str | None = None
    age_band: str | None = None
    rating_band: str | None = None
    sex_restriction: str | None = None
    distance: str | None = None
    distance_round: str | None = None
    distance_f: str | None = None
    region: str | None = None
    prize: str | None = None
    field_size: str | None = None
    going: str | None = None
    going_detailed: str | None = None
    surface: str | None = None
    jumps: str | None = None
    weather: str | None = None
    stalls: str | None = None
    rail_movements: str | None = None
    big_race: bool | None = None
    is_abandoned: bool | None = None
    race_status: str | None = None
    runners: list[RacecardRunnerPayload] = Field(default_factory=list)


class ResultRacePayload(BaseModel):
    """A finished race as returned by ``/v1/results*``.

    Note the field names differ from :class:`RacecardPayload` for identical
    concepts -- ``class``/``dist``/``off``/``sex_rest``. The aliases below make
    both endpoints land on the same attribute names.
    """

    model_config = WIRE_CONFIG

    race_id: str
    course: str | None = None
    course_id: str | None = None
    date: str | None = None
    off_time: str | None = Field(default=None, validation_alias=AliasChoices("off", "off_time"))
    off_dt: str | None = None
    race_name: str | None = None
    race_class: str | None = Field(default=None, validation_alias=AliasChoices("class", "race_class"))
    type: str | None = None
    pattern: str | None = None
    age_band: str | None = None
    rating_band: str | None = None
    sex_restriction: str | None = Field(
        default=None, validation_alias=AliasChoices("sex_rest", "sex_restriction")
    )
    dist: str | None = None
    dist_y: str | None = None
    dist_m: str | None = None
    dist_f: str | None = None
    region: str | None = None
    going: str | None = None
    surface: str | None = None
    jumps: str | None = None
    winning_time_detail: str | None = None
    comments: str | None = None
    non_runners: str | None = None
    tote_win: str | None = None
    runners: list[ResultRunnerPayload] = Field(default_factory=list)


class RacecardsPagePayload(BaseModel):
    model_config = WIRE_CONFIG

    racecards: list[RacecardPayload] = Field(default_factory=list)
    total: int | None = None
    limit: int | None = None
    skip: int | None = None


class ResultsPagePayload(BaseModel):
    model_config = WIRE_CONFIG

    results: list[ResultRacePayload] = Field(default_factory=list)
    total: int | None = None
    limit: int | None = None
    skip: int | None = None


class RunnerOddsPayload(BaseModel):
    """Response of ``/v1/odds/{race_id}/{horse_id}``."""

    model_config = WIRE_CONFIG

    race_id: str | None = None
    horse_id: str | None = None
    horse: str | None = None
    odds: list[OddsEntryPayload] = Field(default_factory=list)


class CoursePayload(BaseModel):
    model_config = WIRE_CONFIG

    id: str
    course: str | None = None
    region_code: str | None = None
    region: str | None = None


class CoursesPagePayload(BaseModel):
    model_config = WIRE_CONFIG

    courses: list[CoursePayload] = Field(default_factory=list)


class HorsePayload(BaseModel):
    model_config = WIRE_CONFIG

    id: str
    name: str | None = None
    sex: str | None = None
    sex_code: str | None = None
    dob: str | None = None
    colour: str | None = None
    breeder: str | None = None
    sire: str | None = None
    sire_id: str | None = None
    dam: str | None = None
    dam_id: str | None = None
    damsire: str | None = None
    damsire_id: str | None = None


class HorseSearchPagePayload(BaseModel):
    model_config = WIRE_CONFIG

    search_results: list[HorsePayload] = Field(
        default_factory=list, validation_alias=AliasChoices("search_results", "horses")
    )
    total: int | None = None
    limit: int | None = None
    skip: int | None = None


class PersonPayload(BaseModel):
    """A jockey or trainer -- the API returns the same ``{id, name}`` shape for both."""

    model_config = WIRE_CONFIG

    id: str
    name: str | None = None


class PersonSearchPagePayload(BaseModel):
    model_config = WIRE_CONFIG

    search_results: list[PersonPayload] = Field(
        default_factory=list,
        validation_alias=AliasChoices("search_results", "jockeys", "trainers"),
    )
    total: int | None = None
    limit: int | None = None
    skip: int | None = None


# ---------------------------------------------------------------------------
# Domain schemas -- typed, canonical, what the platform actually uses
# ---------------------------------------------------------------------------
DOMAIN_CONFIG = ConfigDict(extra="forbid", from_attributes=True)


class OddsSchema(BaseModel):
    """One bookmaker price for one runner at one instant."""

    model_config = DOMAIN_CONFIG

    race_id: str
    horse_id: str
    bookmaker: str
    decimal_odds: Decimal | None = None
    fractional_odds: str | None = None
    implied_probability: float | None = None
    each_way_places: int | None = None
    each_way_denominator: int | None = None
    recorded_at: datetime | None = None


class JockeySchema(BaseModel):
    model_config = DOMAIN_CONFIG

    jockey_id: str
    name: str | None = None


class TrainerSchema(BaseModel):
    model_config = DOMAIN_CONFIG

    trainer_id: str
    name: str | None = None
    location: str | None = None


class HorseSchema(BaseModel):
    model_config = DOMAIN_CONFIG

    horse_id: str
    name: str | None = None
    sex: str | None = None
    sex_code: str | None = None
    colour: str | None = None
    date_of_birth: date | None = None
    region: str | None = None
    sire: str | None = None
    sire_id: str | None = None
    dam: str | None = None
    dam_id: str | None = None
    damsire: str | None = None
    damsire_id: str | None = None


class CourseSchema(BaseModel):
    model_config = DOMAIN_CONFIG

    course_id: str
    name: str | None = None
    region: str | None = None
    region_code: str | None = None


class RunnerSchema(BaseModel):
    """A declared runner, pre-race."""

    model_config = DOMAIN_CONFIG

    race_id: str
    horse: HorseSchema
    jockey: JockeySchema | None = None
    trainer: TrainerSchema | None = None
    owner: str | None = None
    owner_id: str | None = None

    saddle_cloth: int | None = None
    draw: int | None = None
    age: int | None = None
    weight_lbs: int | None = None
    headgear: str | None = None
    official_rating: int | None = None
    rpr: int | None = None
    topspeed: int | None = None
    days_since_last_run: int | None = None
    form: str | None = None
    form_tokens: list[str] = Field(default_factory=list)
    comment: str | None = None
    odds: list[OddsSchema] = Field(default_factory=list)


class ResultRunnerSchema(BaseModel):
    """A runner as it finished."""

    model_config = DOMAIN_CONFIG

    race_id: str
    horse: HorseSchema
    jockey: JockeySchema | None = None
    trainer: TrainerSchema | None = None
    owner: str | None = None
    owner_id: str | None = None

    finishing_position: int | None = None
    finishing_status: str = "unknown"
    saddle_cloth: int | None = None
    draw: int | None = None
    age: int | None = None
    weight_lbs: int | None = None
    headgear: str | None = None
    starting_price: Decimal | None = None
    beaten_lengths: float | None = None
    overall_beaten_lengths: float | None = None
    official_rating: int | None = None
    rpr: int | None = None
    topspeed: int | None = None
    finishing_time: str | None = None
    prize_won: Decimal | None = None
    comment: str | None = None


class RaceSchema(BaseModel):
    """A race, whether declared or finished."""

    model_config = DOMAIN_CONFIG

    race_id: str
    race_date: date | None = None
    off_time: datetime | None = None
    course: str | None = None
    course_id: str | None = None
    race_name: str | None = None
    race_class: int | None = None
    race_type: str | None = None
    pattern: str | None = None
    age_band: str | None = None
    rating_band: str | None = None
    sex_restriction: str | None = None
    distance_yards: int | None = None
    distance_furlongs: float | None = None
    going: str | None = None
    going_detailed: str | None = None
    surface: str | None = None
    jumps: str | None = None
    region: str | None = None
    prize_money: Decimal | None = None
    field_size: int | None = None
    weather: str | None = None
    is_abandoned: bool = False
    is_result: bool = False
    winning_time_detail: str | None = None

    runners: list[RunnerSchema] = Field(default_factory=list)
    results: list[ResultRunnerSchema] = Field(default_factory=list)

    @property
    def runner_count(self) -> int:
        return len(self.results) if self.is_result else len(self.runners)


__all__ = [
    "CoursePayload",
    "CourseSchema",
    "CoursesPagePayload",
    "HorsePayload",
    "HorseSchema",
    "HorseSearchPagePayload",
    "JockeySchema",
    "OddsEntryPayload",
    "OddsSchema",
    "PersonPayload",
    "PersonSearchPagePayload",
    "RaceSchema",
    "RacecardPayload",
    "RacecardRunnerPayload",
    "RacecardsPagePayload",
    "ResultRacePayload",
    "ResultRunnerPayload",
    "ResultRunnerSchema",
    "ResultsPagePayload",
    "RunnerOddsPayload",
    "RunnerSchema",
    "TrainerSchema",
]
