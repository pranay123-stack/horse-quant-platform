"""Synthetic race generator — **development and testing only**.

.. warning::
   Nothing produced here is real. It exists so the research pipeline can be
   exercised, demonstrated and tested end to end without a live Racing API
   subscription. **No conclusion drawn from synthetic data says anything about
   real betting markets.** A model that scores well here has learned the
   generator, not racing.

What it is useful for:

* proving the feature pipeline, dataset builder and backtester run end to end
* giving leakage tests a world where the answer is known by construction
* letting the backtest arithmetic be checked against a market whose margin we set

The generative model is deliberately simple and explicit:

    performance = horse_ability + jockey_skill + trainer_skill
                  + going_fit + distance_fit + noise

The winner is the highest performance. Bookmaker prices are the true win
probabilities, blurred by ``market_noise`` and inflated by ``overround`` — so a
market-following strategy loses exactly the margin, which is the correct null
result for validating the backtester.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import numpy as np
from sqlalchemy import insert
from sqlalchemy.orm import Session

from backend.models import Course, Horse, Jockey, OddsHistory, Race, RaceResult, RaceRunner, Trainer
from backend.utils.logging import get_logger
from backend.utils.timeutils import UK_TZ

logger = get_logger(__name__, channel="pipeline")

GOINGS = ("Good", "Good To Soft", "Soft", "Heavy", "Good To Firm", "Firm")
DISTANCES = (1100, 1320, 1760, 2200, 2860, 3520, 4400)
RACE_TYPES = ("Flat", "Chase", "Hurdle")
BOOKMAKERS = ("Bet365", "Sky Bet", "William Hill", "Paddy Power")


@dataclass(slots=True)
class SyntheticConfig:
    """Knobs for the generator. Defaults produce ~1 season of a small circuit."""

    n_horses: int = 300
    n_jockeys: int = 40
    n_trainers: int = 30
    n_courses: int = 6

    start_date: date = date(2025, 1, 1)
    n_days: int = 180
    races_per_day: int = 4
    runners_per_race: int = 8

    #: How closely bookmaker prices track the true probability. 0 = pure noise,
    #: 1 = perfectly informed.
    market_efficiency: float = 0.85
    #: Bookmaker margin. 1.15 means implied probabilities sum to 115%.
    overround: float = 1.15
    #: Fraction of races that get a market at all.
    market_coverage: float = 0.9
    #: Relative price dispersion between bookmakers. Kept well below the
    #: overround so that best-price overround stays above 1.0 -- i.e. no
    #: risk-free arbitrage exists in the generated market.
    book_dispersion: float = 0.02

    seed: int = 42


@dataclass(slots=True)
class SyntheticSummary:
    horses: int = 0
    jockeys: int = 0
    trainers: int = 0
    courses: int = 0
    races: int = 0
    runners: int = 0
    results: int = 0
    odds: int = 0

    def render(self) -> str:
        return (
            "Synthetic dataset generated (NOT REAL DATA)\n"
            "-------------------------------------------\n"
            f"  Courses : {self.courses}\n"
            f"  Horses  : {self.horses}\n"
            f"  Jockeys : {self.jockeys}\n"
            f"  Trainers: {self.trainers}\n"
            f"  Races   : {self.races}\n"
            f"  Runners : {self.runners}\n"
            f"  Results : {self.results}\n"
            f"  Odds    : {self.odds}"
        )


def generate_synthetic_data(session: Session, config: SyntheticConfig | None = None) -> SyntheticSummary:
    """Populate the database with a deterministic synthetic season."""
    config = config or SyntheticConfig()
    rng = np.random.default_rng(config.seed)
    summary = SyntheticSummary()

    logger.warning(
        "generating SYNTHETIC racing data -- not real, never use for research conclusions",
        extra={"days": config.n_days, "races_per_day": config.races_per_day},
    )

    # --- latent skill, fixed for the whole season -------------------------
    horse_ability = rng.normal(0, 1, config.n_horses)
    horse_going_pref = rng.uniform(0, 1, config.n_horses)
    horse_distance_pref = rng.integers(0, len(DISTANCES), config.n_horses)
    jockey_skill = rng.normal(0, 0.35, config.n_jockeys)
    trainer_skill = rng.normal(0, 0.30, config.n_trainers)

    courses = [
        Course(
            course_id=f"crs_s{i:02d}", name=f"Synthetic Park {i}", region="Great Britain", region_code="gb"
        )
        for i in range(config.n_courses)
    ]
    horses = [
        Horse(
            horse_id=f"hrs_s{i:04d}",
            name=f"Synthetic Horse {i}",
            sex="gelding" if i % 2 else "mare",
            region="GB",
            date_of_birth=date(2018 + (i % 5), 1 + (i % 12), 1 + (i % 28)),
        )
        for i in range(config.n_horses)
    ]
    jockeys = [
        Jockey(jockey_id=f"jky_s{i:03d}", name=f"Synthetic Jockey {i}") for i in range(config.n_jockeys)
    ]
    trainers = [
        Trainer(trainer_id=f"trn_s{i:03d}", name=f"Synthetic Trainer {i}") for i in range(config.n_trainers)
    ]

    session.add_all([*courses, *horses, *jockeys, *trainers])
    session.flush()
    summary.courses, summary.horses = len(courses), len(horses)
    summary.jockeys, summary.trainers = len(jockeys), len(trainers)

    # Rows accumulate as plain dicts and go in via Core bulk inserts. Building
    # 100k+ mapped objects would spend most of the runtime in the identity map.
    race_rows: list[dict] = []
    runner_rows: list[dict] = []
    result_rows: list[dict] = []
    odds_rows: list[dict] = []

    def flush() -> None:
        for model, rows in (
            (Race, race_rows),
            (RaceRunner, runner_rows),
            (RaceResult, result_rows),
            (OddsHistory, odds_rows),
        ):
            if rows:
                session.execute(insert(model), rows)
                rows.clear()
        session.flush()

    race_counter = 0
    for day_offset in range(config.n_days):
        race_date = config.start_date + timedelta(days=day_offset)

        for race_index in range(config.races_per_day):
            race_counter += 1
            race_id = f"rac_s{race_counter:06d}"
            course = courses[rng.integers(0, len(courses))]
            off = datetime.combine(race_date, time(13 + race_index, 30), tzinfo=UK_TZ)

            distance_index = int(rng.integers(0, len(DISTANCES)))
            distance = DISTANCES[distance_index]
            going = GOINGS[int(rng.integers(0, len(GOINGS)))]
            going_firmness = _going_firmness(going)
            race_class = int(rng.integers(1, 8))

            field = rng.choice(config.n_horses, size=config.runners_per_race, replace=False)
            jockey_pick = rng.integers(0, config.n_jockeys, size=config.runners_per_race)
            trainer_pick = rng.integers(0, config.n_trainers, size=config.runners_per_race)

            # --- latent performance ---------------------------------------
            going_fit = -np.abs(horse_going_pref[field] - going_firmness) * 0.8
            distance_fit = -np.abs(horse_distance_pref[field] - distance_index) * 0.18
            performance = (
                horse_ability[field]
                + jockey_skill[jockey_pick]
                + trainer_skill[trainer_pick]
                + going_fit
                + distance_fit
                + rng.normal(0, 1.0, config.runners_per_race)
            )
            order = np.argsort(-performance)
            positions = np.empty(config.runners_per_race, dtype=int)
            positions[order] = np.arange(1, config.runners_per_race + 1)

            # --- the market: true probabilities, blurred and marked up -----
            strength = np.exp(
                horse_ability[field] + jockey_skill[jockey_pick] + trainer_skill[trainer_pick] + going_fit
            )
            true_prob = strength / strength.sum()
            blurred = config.market_efficiency * true_prob + (1 - config.market_efficiency) * rng.dirichlet(
                np.ones(config.runners_per_race)
            )
            blurred = blurred / blurred.sum()
            quoted_prob = blurred * config.overround
            decimal_odds = np.clip(1.0 / quoted_prob, 1.05, 500.0)

            race_rows.append(
                dict(
                    race_id=race_id,
                    race_date=race_date,
                    off_time=off,
                    course_id=course.course_id,
                    course_name=course.name,
                    race_name=f"Synthetic {RACE_TYPES[race_index % len(RACE_TYPES)]} {race_counter}",
                    race_class=race_class,
                    race_type=RACE_TYPES[race_index % len(RACE_TYPES)],
                    distance_yards=distance,
                    distance_furlongs=round(distance / 220, 2),
                    going=going,
                    region="GB",
                    prize_money=Decimal(str(2000 + race_class * 1500)),
                    field_size=config.runners_per_race,
                    has_result=True,
                    is_abandoned=False,
                )
            )
            summary.races += 1

            has_market = rng.random() < config.market_coverage
            for slot, horse_index in enumerate(field):
                horse_id = horses[int(horse_index)].horse_id
                jockey_id = jockeys[int(jockey_pick[slot])].jockey_id
                trainer_id = trainers[int(trainer_pick[slot])].trainer_id
                position = int(positions[slot])
                rating = int(np.clip(70 + horse_ability[horse_index] * 12 + rng.normal(0, 3), 40, 130))

                runner_rows.append(
                    dict(
                        race_id=race_id,
                        horse_id=horse_id,
                        jockey_id=jockey_id,
                        trainer_id=trainer_id,
                        saddle_cloth=slot + 1,
                        draw=slot + 1,
                        age=int(3 + (horse_index % 8)),
                        weight_lbs=int(rng.integers(120, 160)),
                        official_rating=rating,
                        rpr=int(np.clip(rating + rng.normal(0, 5), 40, 140)),
                        topspeed=int(np.clip(rating + rng.normal(0, 8), 30, 140)),
                    )
                )
                result_rows.append(
                    dict(
                        race_id=race_id,
                        horse_id=horse_id,
                        jockey_id=jockey_id,
                        trainer_id=trainer_id,
                        finishing_position=position,
                        finishing_status="finished",
                        is_winner=position == 1,
                        saddle_cloth=slot + 1,
                        draw=slot + 1,
                        age=int(3 + (horse_index % 8)),
                        weight_lbs=int(rng.integers(120, 160)),
                        starting_price=Decimal(str(round(float(decimal_odds[slot]), 2))),
                        beaten_lengths=float(round(max(0.0, (position - 1) * rng.uniform(0.5, 3.0)), 2)),
                        official_rating=rating,
                        rpr=int(np.clip(rating + (config.runners_per_race - position) * 2, 40, 150)),
                        topspeed=int(np.clip(rating + rng.normal(0, 8), 30, 140)),
                    )
                )
                summary.runners += 1
                summary.results += 1

                if has_market:
                    # Two snapshots -- an opening show and a final pre-off price --
                    # quoted by EVERY bookmaker, with small dispersion between them.
                    #
                    # Quoting only one randomly chosen book per snapshot would be
                    # cheaper, but it manufactures phantom arbitrage: taking the
                    # best price across two *different* books is biased long, so
                    # the best-price overround drifts below 1.0 and a
                    # market-following strategy appears to find value. The
                    # negative control has to be clean, so every book quotes.
                    drift = float(rng.normal(1.0, 0.12))
                    for offset_minutes, base_price in (
                        (120, float(decimal_odds[slot]) * drift),
                        (5, float(decimal_odds[slot])),
                    ):
                        for book in BOOKMAKERS:
                            price = base_price * float(rng.normal(1.0, config.book_dispersion))
                            odds_rows.append(
                                dict(
                                    race_id=race_id,
                                    horse_id=horse_id,
                                    bookmaker=book,
                                    decimal_odds=Decimal(str(round(max(1.05, price), 2))),
                                    implied_probability=1.0 / max(1.05, price),
                                    recorded_at=off - timedelta(minutes=offset_minutes),
                                )
                            )
                            summary.odds += 1

        if day_offset % 30 == 0:
            flush()

    flush()
    session.commit()
    logger.info("synthetic generation complete", extra={"races": summary.races, "runners": summary.runners})
    return summary


def _going_firmness(going: str) -> float:
    from backend.features.base import going_value

    return going_value(going) or 0.5


__all__ = ["SyntheticConfig", "SyntheticSummary", "generate_synthetic_data"]
