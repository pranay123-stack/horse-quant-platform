"""Point-in-time foundations for every feature in the platform.

The single most important property of this layer
================================================

A feature for a race at time ``T`` may use **only** information that existed
before ``T``. Violate that and a backtest becomes a fiction: the model appears to
predict results it was quietly shown.

Most implementations try to achieve this by being careful. This one achieves it
*structurally*, with one pattern applied everywhere:

1. Build a **history** frame — one row per past event (a horse's run, a jockey's
   ride), ordered in time.
2. Compute each entity's cumulative state **inclusive of that row**: after run
   *k*, what do we know about this horse?
3. Attach state to a target race with :func:`as_of_join`, a backward
   ``merge_asof`` with ``allow_exact_matches=False``.

Step 3 is the guarantee. For a target at ``T`` it selects the last history row
strictly before ``T``, whose state summarises every event before ``T`` and no
event after it. There is no code path by which a later row can be selected, so
leakage cannot be introduced by forgetting a filter — only by deliberately
bypassing this function.

``allow_exact_matches=False`` is not a detail: it is what excludes the race's own
result from its own features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Final

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import OddsHistory, Race, RaceResult, RaceRunner
from backend.utils.logging import get_logger
from backend.utils.timeutils import UK_TZ

logger = get_logger(__name__, channel="model")

#: The canonical time axis. Every frame carries it; every as-of join uses it.
EVENT_TIME: Final[str] = "event_time"

#: When a race has no off-time, assume a mid-afternoon UK off. This only ever
#: orders races *within* a day; cross-day ordering is unaffected.
ASSUMED_OFF_TIME: Final[time] = time(14, 0)

YARDS_PER_FURLONG: Final[int] = 220

#: Going mapped onto a firmness scale: 0.0 = bottomless, 1.0 = rock hard.
#: Numeric encoding lets the model learn a monotonic ground preference instead of
#: treating "soft" and "good to soft" as unrelated categories.
GOING_SCALE: Final[dict[str, float]] = {
    "heavy": 0.0,
    "very soft": 0.1,
    "soft": 0.2,
    "yielding to soft": 0.25,
    "good to soft": 0.35,
    "yielding": 0.35,
    "soft to good": 0.35,
    "good": 0.5,
    "standard": 0.5,
    "standard to slow": 0.4,
    "standard to fast": 0.6,
    "slow": 0.35,
    "good to firm": 0.65,
    "firm": 0.8,
    "hard": 1.0,
    "fast": 0.85,
}

#: Coarse ground buckets used for conditional form ("record on soft").
GOING_BANDS: Final[tuple[str, ...]] = ("soft", "good", "firm")

#: Distance buckets. Sprinters and stayers are close to different sports; a
#: horse's record over 5f says little about its chance over 3 miles.
DISTANCE_BANDS: Final[tuple[str, ...]] = ("sprint", "mile", "middle", "staying")


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """Weights for the composite ``horse_form_score``.

    Defaults follow the Phase 3 specification exactly. They are a starting
    hypothesis, not a fitted result — Phase 6 replaces this hand-weighted score
    with a learned model, and the score remains useful as an interpretable
    baseline to beat.
    """

    recent_form: float = 0.30
    speed_rating: float = 0.25
    distance_fit: float = 0.20
    going_fit: float = 0.15
    class_rating: float = 0.10

    def total(self) -> float:
        return self.recent_form + self.speed_rating + self.distance_fit + self.going_fit + self.class_rating

    def __post_init__(self) -> None:
        if abs(self.total() - 1.0) > 1e-9:
            raise ValueError(f"score weights must sum to 1.0, got {self.total()}")


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Window sizes and thresholds for the feature pipeline."""

    form_window_short: int = 3
    form_window_long: int = 5
    speed_window: int = 5
    jockey_rides_window: int = 30
    trainer_runner_window: int = 100
    trainer_days_window: int = 30
    place_threshold: int = 3
    #: Below this many prior runs, form features are unreliable; the pipeline
    #: still emits them but flags the row via ``horse_runs_before``.
    min_history_runs: int = 3
    weights: ScoreWeights = field(default_factory=ScoreWeights)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a column to numeric, returning an all-NaN column when it is absent.

    ``frame.get(column)`` returns ``Series | None``, so every caller had to guard
    the ``None`` case -- or, more often, quietly not guard it. This always returns
    a Series aligned to ``frame``, which is what the callers actually want.
    """
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def going_value(going: str | None) -> float | None:
    """Map a going description onto the 0-1 firmness scale."""
    if not going or not isinstance(going, str):
        return None
    text = going.strip().lower()
    if text in GOING_SCALE:
        return GOING_SCALE[text]
    # "good to soft (soft in places)" and similar: match the longest known prefix.
    matches = [key for key in GOING_SCALE if text.startswith(key)]
    if matches:
        return GOING_SCALE[max(matches, key=len)]
    contained = [key for key in GOING_SCALE if key in text]
    if contained:
        return GOING_SCALE[max(contained, key=len)]
    return None


def going_band(going: str | None) -> str:
    """Bucket going into ``soft`` / ``good`` / ``firm`` (or ``unknown``)."""
    value = going_value(going)
    if value is None:
        return "unknown"
    if value < 0.45:
        return "soft"
    if value < 0.62:
        return "good"
    return "firm"


def distance_band(distance_yards: float | None) -> str:
    """Bucket a distance into sprint / mile / middle / staying."""
    if distance_yards is None or pd.isna(distance_yards):
        return "unknown"
    furlongs = float(distance_yards) / YARDS_PER_FURLONG
    if furlongs < 7:
        return "sprint"
    if furlongs < 10:
        return "mile"
    if furlongs < 14:
        return "middle"
    return "staying"


def _event_time(race_date: date | None, off_time: datetime | None) -> datetime | None:
    """One reliable timestamp per race, whatever the source supplied."""
    if off_time is not None:
        return off_time
    if race_date is None:
        return None
    return datetime.combine(race_date, ASSUMED_OFF_TIME, tzinfo=UK_TZ)


def _event_time_column(frame: pd.DataFrame) -> list[datetime | None]:
    """One timestamp per row: ``off_time`` where present, else the race date."""
    if frame.empty:
        return []
    dates = frame["race_date"].tolist() if "race_date" in frame.columns else [None] * len(frame)
    offs = frame["off_time"].tolist() if "off_time" in frame.columns else [None] * len(frame)
    return [
        _event_time(
            race_date if isinstance(race_date, date) else None,
            off if isinstance(off, datetime) else None,
        )
        for race_date, off in zip(dates, offs, strict=True)
    ]


def _finalise_time_axis(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise ``event_time`` to sorted, tz-aware UTC and drop undated rows.

    A row with no usable timestamp cannot be placed on the time axis, so it can
    neither receive nor contribute point-in-time state. Dropping it is the only
    safe option — carrying it would mean guessing its position in history.
    """
    if frame.empty:
        frame[EVENT_TIME] = pd.Series(dtype="datetime64[ns, UTC]")
        return frame

    frame[EVENT_TIME] = pd.to_datetime(frame[EVENT_TIME], utc=True, errors="coerce")
    undated = int(frame[EVENT_TIME].isna().sum())
    if undated:
        logger.warning("dropping rows with no usable timestamp", extra={"row_count": undated})
        frame = frame[frame[EVENT_TIME].notna()]
    return frame.sort_values(EVENT_TIME, kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_history(
    session: Session,
    *,
    date_from: date | None = None,
    date_to: date | None = None,
    race_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Every finished run, with the race context needed to condition on it.

    This is the raw material for all form features: one row per horse per past
    race, in time order.
    """
    statement = (
        select(
            RaceResult.race_id,
            RaceResult.horse_id,
            RaceResult.jockey_id,
            RaceResult.trainer_id,
            RaceResult.finishing_position,
            RaceResult.finishing_status,
            RaceResult.is_winner,
            RaceResult.starting_price,
            RaceResult.rpr,
            RaceResult.topspeed,
            RaceResult.official_rating,
            RaceResult.weight_lbs,
            RaceResult.beaten_lengths,
            RaceResult.prize_won,
            Race.race_date,
            Race.off_time,
            Race.course_id,
            Race.race_class,
            Race.race_type,
            Race.distance_yards,
            Race.going,
            Race.region,
        )
        .join(Race, Race.race_id == RaceResult.race_id)
        .where(Race.has_result.is_(True))
    )
    if race_ids is not None:
        statement = statement.where(Race.race_id.in_(race_ids))
    if date_from is not None:
        statement = statement.where(Race.race_date >= date_from)
    if date_to is not None:
        statement = statement.where(Race.race_date <= date_to)

    rows = session.execute(statement).mappings().all()
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        frame = pd.DataFrame(columns=_HISTORY_COLUMNS)

    frame[EVENT_TIME] = _event_time_column(frame)
    frame = _finalise_time_axis(frame)
    return _coerce_history_types(frame)


_HISTORY_COLUMNS = [
    "race_id",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "finishing_position",
    "finishing_status",
    "is_winner",
    "starting_price",
    "rpr",
    "topspeed",
    "official_rating",
    "weight_lbs",
    "beaten_lengths",
    "prize_won",
    "race_date",
    "off_time",
    "course_id",
    "race_class",
    "race_type",
    "distance_yards",
    "going",
    "region",
]

_TARGET_COLUMNS = [
    "race_id",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "saddle_cloth",
    "draw",
    "age",
    "weight_lbs",
    "official_rating",
    "rpr",
    "topspeed",
    "days_since_last_run",
    "form",
    "race_date",
    "off_time",
    "course_id",
    "race_class",
    "race_type",
    "distance_yards",
    "going",
    "region",
    "field_size",
    "prize_money",
    "has_result",
]


def _coerce_history_types(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "finishing_position",
        "rpr",
        "topspeed",
        "official_rating",
        "weight_lbs",
        "beaten_lengths",
        "race_class",
        "distance_yards",
    ]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("starting_price", "prize_won"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "is_winner" in frame:
        frame["is_winner"] = frame["is_winner"].fillna(False).astype(bool)
    if "finishing_position" in frame:
        frame["is_placed"] = frame["finishing_position"].le(3).fillna(False)
    if "going" in frame:
        frame["going_band"] = frame["going"].map(going_band)
        frame["going_value"] = frame["going"].map(going_value)
    if "distance_yards" in frame:
        frame["distance_band"] = frame["distance_yards"].map(distance_band)
    return frame


def load_targets(
    session: Session,
    *,
    date_from: date | None = None,
    date_to: date | None = None,
    race_ids: list[str] | None = None,
    require_result: bool | None = None,
) -> pd.DataFrame:
    """The runners we want features for — declarations, never outcomes.

    Deliberately sourced from ``race_runners`` (the declaration) rather than
    ``race_results``: every column here was published before the off.
    """
    statement = (
        select(
            RaceRunner.race_id,
            RaceRunner.horse_id,
            RaceRunner.jockey_id,
            RaceRunner.trainer_id,
            RaceRunner.saddle_cloth,
            RaceRunner.draw,
            RaceRunner.age,
            RaceRunner.weight_lbs,
            RaceRunner.official_rating,
            RaceRunner.rpr,
            RaceRunner.topspeed,
            RaceRunner.days_since_last_run,
            RaceRunner.form,
            Race.race_date,
            Race.off_time,
            Race.course_id,
            Race.race_class,
            Race.race_type,
            Race.distance_yards,
            Race.going,
            Race.region,
            Race.field_size,
            Race.prize_money,
            Race.has_result,
        )
        .join(Race, Race.race_id == RaceRunner.race_id)
        .where(Race.is_abandoned.is_(False))
    )
    if race_ids is not None:
        statement = statement.where(Race.race_id.in_(race_ids))
    if date_from is not None:
        statement = statement.where(Race.race_date >= date_from)
    if date_to is not None:
        statement = statement.where(Race.race_date <= date_to)
    if require_result is not None:
        statement = statement.where(Race.has_result.is_(require_result))

    rows = session.execute(statement).mappings().all()
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        frame = pd.DataFrame(columns=_TARGET_COLUMNS)

    frame[EVENT_TIME] = _event_time_column(frame)
    frame = _finalise_time_axis(frame)

    for column in (
        "draw",
        "age",
        "weight_lbs",
        "official_rating",
        "rpr",
        "topspeed",
        "days_since_last_run",
        "race_class",
        "distance_yards",
        "field_size",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "prize_money" in frame:
        frame["prize_money"] = pd.to_numeric(frame["prize_money"], errors="coerce")
    if "going" in frame:
        frame["going_band"] = frame["going"].map(going_band)
        frame["going_value"] = frame["going"].map(going_value)
    if "distance_yards" in frame:
        frame["distance_band"] = frame["distance_yards"].map(distance_band)
        frame["distance_furlongs"] = frame["distance_yards"] / YARDS_PER_FURLONG
    return frame


#: Every column :func:`load_labels` produces. Anything here is an OUTCOME and
#: must never reach a model. Defined next to the query that creates them so the
#: two cannot drift apart -- an earlier version listed them by hand in another
#: module, missed ``is_winner``, and the target leaked in as a feature.
LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "finishing_position",
    "finishing_status",
    "is_winner",
    "starting_price",
    "won",
    "placed",
)


def load_labels(session: Session, race_ids: list[str] | None = None) -> pd.DataFrame:
    """Outcomes, kept strictly separate from features.

    Labels are loaded by a different function, from a different table, and are
    joined only at the very end of dataset construction. Keeping them physically
    apart from the feature path is a cheap structural safeguard.
    """
    statement = select(
        RaceResult.race_id,
        RaceResult.horse_id,
        RaceResult.finishing_position,
        RaceResult.finishing_status,
        RaceResult.is_winner,
        RaceResult.starting_price,
    )
    if race_ids is not None:
        statement = statement.where(RaceResult.race_id.in_(race_ids))

    rows = session.execute(statement).mappings().all()
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "race_id",
                "horse_id",
                "finishing_position",
                "finishing_status",
                "is_winner",
                "starting_price",
                "won",
                "placed",
            ]
        )
    frame["is_winner"] = frame["is_winner"].fillna(False).astype(bool)
    frame["won"] = frame["is_winner"].astype(int)
    frame["placed"] = (
        pd.to_numeric(frame["finishing_position"], errors="coerce").le(3).fillna(False).astype(int)
    )
    frame["starting_price"] = pd.to_numeric(frame["starting_price"], errors="coerce")
    return frame


def load_odds(
    session: Session,
    *,
    race_ids: list[str] | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> pd.DataFrame:
    """Bookmaker quotes joined to their race's off-time.

    ``off_time`` travels with each quote so the market-feature layer can discard
    anything recorded after the off — post-race prices are not market signal,
    they are the answer.
    """
    statement = select(
        OddsHistory.race_id,
        OddsHistory.horse_id,
        OddsHistory.bookmaker,
        OddsHistory.decimal_odds,
        OddsHistory.recorded_at,
        Race.off_time,
        Race.race_date,
    ).join(Race, Race.race_id == OddsHistory.race_id)
    if race_ids is not None:
        statement = statement.where(OddsHistory.race_id.in_(race_ids))
    if date_from is not None:
        statement = statement.where(Race.race_date >= date_from)
    if date_to is not None:
        statement = statement.where(Race.race_date <= date_to)

    rows = session.execute(statement).mappings().all()
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "race_id",
                "horse_id",
                "bookmaker",
                "decimal_odds",
                "recorded_at",
                "off_time",
                "race_date",
            ]
        )
    frame["decimal_odds"] = pd.to_numeric(frame["decimal_odds"], errors="coerce")
    frame["recorded_at"] = pd.to_datetime(frame["recorded_at"], utc=True, errors="coerce")
    frame["off_time"] = pd.to_datetime(frame["off_time"], utc=True, errors="coerce")
    return frame


# ---------------------------------------------------------------------------
# The point-in-time join
# ---------------------------------------------------------------------------
def as_of_join(
    targets: pd.DataFrame,
    state: pd.DataFrame,
    *,
    by: str | list[str],
    columns: list[str],
    suffix: str = "",
) -> pd.DataFrame:
    """Attach each target the entity state as of the last event strictly before it.

    This is the only sanctioned way to move information from history into a
    feature. ``allow_exact_matches=False`` excludes an event that happens at the
    target's own timestamp — which is precisely the race's own result.

    Parameters
    ----------
    targets:
        Rows needing features. Must contain ``by`` and :data:`EVENT_TIME`.
    state:
        History rows carrying cumulative state, inclusive of each row.
    by:
        Entity key(s) to match on — ``horse_id``, ``jockey_id``, or a composite.
    columns:
        State columns to bring across.
    """
    keys = [by] if isinstance(by, str) else list(by)
    wanted = [column for column in columns if column in state.columns]

    if targets.empty:
        # Iterate ``columns``, not ``wanted``: with an empty state frame nothing
        # is "wanted", and the caller would get back a frame missing the very
        # columns it asked for -- a KeyError several stages downstream.
        for column in columns:
            targets[f"{column}{suffix}"] = pd.Series(dtype="float64")
        return targets

    if state.empty or not wanted:
        for column in columns:
            targets[f"{column}{suffix}"] = pd.NA
        return targets

    left = targets.sort_values(EVENT_TIME, kind="stable")
    right = state[[*keys, EVENT_TIME, *wanted]].sort_values(EVENT_TIME, kind="stable")

    # merge_asof requires matching key dtypes; ids may be object vs category.
    for key in keys:
        left[key] = left[key].astype("object")
        right[key] = right[key].astype("object")

    merged = pd.merge_asof(
        left,
        right,
        on=EVENT_TIME,
        by=keys,
        direction="backward",
        allow_exact_matches=False,  # <-- excludes the race's own outcome
        suffixes=("", "_state"),
    )
    if suffix:
        merged = merged.rename(columns={column: f"{column}{suffix}" for column in wanted})
    return merged


def cumulative_state(
    history: pd.DataFrame,
    *,
    by: str | list[str],
    builders: dict[str, pd.Series],
) -> pd.DataFrame:
    """Assemble a state frame from pre-computed per-row series."""
    keys = [by] if isinstance(by, str) else list(by)
    state = history[[*keys, EVENT_TIME]].copy()
    for name, series in builders.items():
        state[name] = series.to_numpy()
    return state


__all__ = [
    "ASSUMED_OFF_TIME",
    "DISTANCE_BANDS",
    "EVENT_TIME",
    "GOING_BANDS",
    "GOING_SCALE",
    "LABEL_COLUMNS",
    "YARDS_PER_FURLONG",
    "FeatureConfig",
    "ScoreWeights",
    "as_of_join",
    "cumulative_state",
    "distance_band",
    "going_band",
    "going_value",
    "load_history",
    "load_labels",
    "load_odds",
    "load_targets",
    "numeric_column",
]
