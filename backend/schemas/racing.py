"""Read models for the platform's own HTTP API.

Separate from :mod:`backend.services.racing_api.schemas`: those describe what
the *upstream* sends, these describe what *we* return. Keeping them apart means
an upstream field rename never silently changes our public contract.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

READ_CONFIG = ConfigDict(from_attributes=True)


class Page[T](BaseModel):
    """A slice of a result set, with enough metadata to page through it."""

    items: list[T]
    total: int = Field(..., description="Total rows matching the filter, ignoring pagination")
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total


class HorseRead(BaseModel):
    model_config = READ_CONFIG

    horse_id: str
    name: str | None = None
    sex: str | None = None
    colour: str | None = None
    date_of_birth: date | None = None
    region: str | None = None
    sire: str | None = None
    dam: str | None = None
    damsire: str | None = None


class JockeyRead(BaseModel):
    model_config = READ_CONFIG

    jockey_id: str
    name: str | None = None
    win_rate: float | None = None


class TrainerRead(BaseModel):
    model_config = READ_CONFIG

    trainer_id: str
    name: str | None = None
    strike_rate: float | None = None


class RunnerRead(BaseModel):
    """A declared runner — safe to expose before the off."""

    model_config = READ_CONFIG

    horse_id: str
    horse_name: str | None = None
    jockey_id: str | None = None
    trainer_id: str | None = None
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


class ResultRead(BaseModel):
    """A finished runner."""

    model_config = READ_CONFIG

    horse_id: str
    horse_name: str | None = None
    jockey_id: str | None = None
    trainer_id: str | None = None
    finishing_position: int | None = None
    finishing_status: str = "unknown"
    is_winner: bool = False
    starting_price: Decimal | None = None
    beaten_lengths: float | None = None
    official_rating: int | None = None
    rpr: int | None = None
    topspeed: int | None = None
    prize_won: Decimal | None = None


class RaceSummaryRead(BaseModel):
    """One line of a race listing."""

    model_config = READ_CONFIG

    race_id: str
    race_date: date | None = None
    off_time: datetime | None = None
    course_id: str | None = None
    course_name: str | None = None
    race_name: str | None = None
    race_class: int | None = None
    race_type: str | None = None
    distance_furlongs: float | None = None
    going: str | None = None
    region: str | None = None
    prize_money: Decimal | None = None
    field_size: int | None = None
    is_abandoned: bool = False
    has_result: bool = False


class RaceDetailRead(RaceSummaryRead):
    """A race with its full card and, once run, its result."""

    distance_yards: int | None = None
    going_detailed: str | None = None
    surface: str | None = None
    jumps: str | None = None
    weather: str | None = None
    pattern: str | None = None
    age_band: str | None = None
    rating_band: str | None = None
    sex_restriction: str | None = None
    winning_time_detail: str | None = None

    runners: list[RunnerRead] = Field(default_factory=list)
    results: list[ResultRead] = Field(default_factory=list)


class OddsRead(BaseModel):
    model_config = READ_CONFIG

    horse_id: str
    bookmaker: str
    decimal_odds: Decimal | None = None
    fractional_odds: str | None = None
    implied_probability: float | None = None
    recorded_at: datetime | None = None


class RunnerOddsRead(BaseModel):
    """Every bookmaker's latest price for one runner, plus the best available."""

    horse_id: str
    horse_name: str | None = None
    best_decimal_odds: Decimal | None = None
    best_bookmaker: str | None = None
    #: Market-implied win probability at the best price, *including* overround.
    implied_probability: float | None = None
    quotes: list[OddsRead] = Field(default_factory=list)


class RaceOddsRead(BaseModel):
    """The market for a whole race."""

    race_id: str
    race_name: str | None = None
    off_time: datetime | None = None
    runners: list[RunnerOddsRead] = Field(default_factory=list)
    #: Sum of implied probabilities at best prices. >1 is the bookmaker margin;
    #: <1 across different books would be an arbitrage.
    overround: float | None = None
    quote_count: int = 0


class HorseFormRead(BaseModel):
    """A horse with its recent runs — the human-readable form line."""

    horse: HorseRead
    runs: list[ResultRead] = Field(default_factory=list)
    total_runs: int = 0
    wins: int = 0
    places: int = 0

    @property
    def strike_rate(self) -> float | None:
        return (self.wins / self.total_runs) if self.total_runs else None


class ImportSummaryRead(BaseModel):
    """The outcome of an ingestion run, as returned by the ingest endpoints."""

    source: str
    started_at: str
    finished_at: str | None = None
    races: dict[str, int]
    runners: dict[str, int]
    results: dict[str, int]
    odds: dict[str, int]
    entities: dict[str, int]
    api: dict[str, int]
    failed: int
    failures: list[dict[str, str]] = Field(default_factory=list)


__all__ = [
    "HorseFormRead",
    "HorseRead",
    "ImportSummaryRead",
    "JockeyRead",
    "OddsRead",
    "Page",
    "RaceDetailRead",
    "RaceOddsRead",
    "RaceSummaryRead",
    "ResultRead",
    "RunnerOddsRead",
    "RunnerRead",
    "TrainerRead",
]
