"""Small, fast fixtures for the ML tests.

Training a model in a unit test has to be cheap, so these build a deliberately
tiny synthetic season — enough rows for the machinery to run and for the
assertions to mean something, few enough to finish in seconds.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.ml.dataset import SplitConfig, TrainingData, TrainingDatasetBuilder
from backend.models import Base
from backend.research.synthetic import SyntheticConfig, generate_synthetic_data

#: Small enough to build in a couple of seconds, large enough that the
#: validation and calibration halves each contain winners.
SMALL_SEASON = SyntheticConfig(
    n_horses=120,
    n_jockeys=15,
    n_trainers=12,
    n_courses=4,
    start_date=date(2024, 1, 1),
    n_days=240,
    races_per_day=2,
    runners_per_race=6,
    seed=99,
)

SMALL_SPLIT = SplitConfig(
    train_start=date(2024, 1, 1),
    train_end=date(2024, 5, 31),
    valid_start=date(2024, 6, 1),
    valid_end=date(2024, 7, 15),
    test_start=date(2024, 7, 16),
    test_end=date(2024, 8, 31),
    embargo_days=0,
)


def make_ml_session(config: SyntheticConfig | None = None) -> Session:
    """A private in-memory database populated with a small synthetic season."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    generate_synthetic_data(session, config or SMALL_SEASON)
    return session


def make_training_data(session: Session, **kwargs: object) -> TrainingData:
    return TrainingDatasetBuilder(session).build(SMALL_SPLIT, **kwargs)  # type: ignore[arg-type]


def synthetic_frame(
    *,
    races: int = 40,
    runners: int = 6,
    seed: int = 0,
    signal: float = 1.5,
) -> pd.DataFrame:
    """A tabular frame with one honest predictive feature and pure noise.

    Used where a test needs known ground truth without touching the database:
    ``signal_feature`` genuinely predicts the winner, ``noise_feature`` does not.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for race_index in range(races):
        strengths = rng.normal(0, 1, runners)
        exponentials = np.exp(strengths * signal)
        winner = rng.choice(runners, p=exponentials / exponentials.sum())
        for runner_index in range(runners):
            rows.append(
                {
                    "race_id": f"rac_{race_index:04d}",
                    "horse_id": f"hrs_{race_index:04d}_{runner_index}",
                    "race_date": date(2025, 1, 1) + pd.Timedelta(days=race_index).to_pytimedelta(),
                    "signal_feature": float(strengths[runner_index]),
                    "noise_feature": float(rng.normal()),
                    "won": int(runner_index == winner),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "SMALL_SEASON",
    "SMALL_SPLIT",
    "make_ml_session",
    "make_training_data",
    "synthetic_frame",
]
