"""Dataset construction, time splits and the feature store."""

from __future__ import annotations

import itertools
from datetime import date, timedelta

import pandas as pd
import pytest

from backend.features import FeaturePipeline, load_features, save_features
from backend.features.feature_pipeline import feature_columns, feature_manifest
from backend.research import (
    ResearchDatasetBuilder,
    TemporalLeakageError,
    iter_race_groups,
    time_split,
    walk_forward_splits,
)
from backend.research.splits import DatasetSplit, assert_no_leakage_between
from backend.research.synthetic import SyntheticConfig, generate_synthetic_data

pytestmark = pytest.mark.unit


@pytest.fixture
def seeded_session(db_session):
    generate_synthetic_data(
        db_session, SyntheticConfig(n_days=60, races_per_day=3, runners_per_race=6, seed=5)
    )
    return db_session


def _dated_frame(days: int = 100, rows_per_day: int = 2) -> pd.DataFrame:
    records = []
    for offset in range(days):
        day = date(2025, 1, 1) + timedelta(days=offset)
        for index in range(rows_per_day):
            records.append(
                {
                    "race_id": f"rac_{offset:03d}_{index}",
                    "horse_id": f"hrs_{index}",
                    "race_date": day,
                    "feature": float(offset),
                    "won": index % 2,
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Time splits
# ---------------------------------------------------------------------------
def test_time_split_divides_on_the_date():
    frame = _dated_frame()
    split = time_split(frame, train_end=date(2025, 2, 1))

    assert split.train["race_date"].max() == date(2025, 2, 1)
    assert split.test["race_date"].min() == date(2025, 2, 2)
    assert split.train_rows + split.test_rows == len(frame)


def test_embargo_removes_the_races_straddling_the_boundary():
    frame = _dated_frame()
    split = time_split(frame, train_end=date(2025, 2, 1), embargo_days=7)

    gap = (split.test["race_date"].min() - split.train["race_date"].max()).days
    assert gap == 8, "the embargo window was not applied"


def test_split_rejects_a_test_start_before_training_ends():
    with pytest.raises(TemporalLeakageError):
        time_split(_dated_frame(), train_end=date(2025, 3, 1), test_start=date(2025, 2, 1))


def test_split_requires_a_date_column():
    with pytest.raises(KeyError, match="race_date"):
        time_split(pd.DataFrame({"a": [1]}), train_end=date(2025, 1, 1))


def test_leakage_assertion_catches_a_shared_race():
    shared = pd.DataFrame({"race_id": ["r1"], "race_date": [date(2025, 1, 1)], "won": [1]})
    later = pd.DataFrame({"race_id": ["r1"], "race_date": [date(2025, 6, 1)], "won": [0]})
    split = DatasetSplit(
        train=shared,
        test=later,
        train_end=date(2025, 1, 1),
        test_start=date(2025, 6, 1),
        test_end=date(2025, 6, 1),
    )
    with pytest.raises(TemporalLeakageError, match="both train and test"):
        assert_no_leakage_between(split)


def test_walk_forward_folds_move_forward_in_time():
    folds = walk_forward_splits(_dated_frame(days=300), n_splits=4, test_days=30, min_train_days=60)

    assert len(folds) == 4
    for earlier, later in itertools.pairwise(folds):
        assert earlier.test_start < later.test_start
        assert earlier.train_end < later.train_end


def test_every_walk_forward_fold_is_leak_free():
    for fold in walk_forward_splits(_dated_frame(days=300), n_splits=4, test_days=30, min_train_days=60):
        assert fold.train["race_date"].max() < fold.test["race_date"].min()
        assert not set(fold.train["race_id"]) & set(fold.test["race_id"])


def test_walk_forward_returns_nothing_when_history_is_too_short():
    assert walk_forward_splits(_dated_frame(days=20), n_splits=5, test_days=30, min_train_days=90) == []


def test_iter_race_groups_is_chronological():
    frame = _dated_frame(days=5).sample(frac=1, random_state=0)
    dates = [group["race_date"].iloc[0] for _, group in iter_race_groups(frame)]
    assert dates == sorted(dates)


def test_iter_race_groups_handles_an_empty_frame():
    assert list(iter_race_groups(pd.DataFrame())) == []


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
def test_dataset_has_features_and_a_target(seeded_session):
    dataset = ResearchDatasetBuilder(seeded_session).build()

    assert dataset.rows > 0
    assert dataset.races > 0
    assert len(dataset.feature_names) > 40
    assert dataset.target == "won"
    assert set(dataset.frame["won"].unique()) <= {0, 1}


def test_target_is_never_a_feature(seeded_session):
    """The most basic leak of all: predicting the label from the label."""
    dataset = ResearchDatasetBuilder(seeded_session).build()
    for forbidden in ("won", "placed", "finishing_position", "starting_price"):
        assert forbidden not in dataset.feature_names


def test_base_rate_matches_one_winner_per_race(seeded_session):
    dataset = ResearchDatasetBuilder(seeded_session).build()
    # Six runners per race in this fixture, exactly one winner.
    assert dataset.base_rate == pytest.approx(1 / 6, abs=0.02)


def test_every_race_has_exactly_one_winner(seeded_session):
    dataset = ResearchDatasetBuilder(seeded_session).build()
    winners = dataset.frame.groupby("race_id")["won"].sum()
    assert (winners == 1).all()


def test_dataset_is_sorted_chronologically(seeded_session):
    dataset = ResearchDatasetBuilder(seeded_session).build()
    dates = pd.to_datetime(dataset.frame["race_date"])
    assert dates.is_monotonic_increasing


def test_min_runs_filter_drops_debutants(seeded_session):
    everything = ResearchDatasetBuilder(seeded_session).build()
    experienced = ResearchDatasetBuilder(seeded_session).build(min_runs=3)

    assert experienced.rows < everything.rows
    assert experienced.frame["horse_runs_before"].min() >= 3


def test_require_market_keeps_only_priced_runners(seeded_session):
    dataset = ResearchDatasetBuilder(seeded_session).build(require_market=True)
    assert (dataset.frame["mkt_has_market"] == 1).all()
    assert dataset.frame["mkt_latest_odds"].notna().all()


def test_date_window_is_respected(seeded_session):
    dataset = ResearchDatasetBuilder(seeded_session).build(
        date_from=date(2025, 1, 15), date_to=date(2025, 1, 31)
    )
    start, end = dataset.date_range
    assert start >= date(2025, 1, 15)
    assert end <= date(2025, 1, 31)


def test_dataset_excludes_races_failing_data_quality(db_session):
    generate_synthetic_data(db_session, SyntheticConfig(n_days=10, races_per_day=2, seed=3))
    # Corrupt one race so validation rejects it.
    from backend.models import Race

    race = db_session.query(Race).first()
    race.distance_yards = 5
    db_session.commit()

    dataset = ResearchDatasetBuilder(db_session).build()
    assert dataset.excluded_races >= 1
    assert race.race_id not in set(dataset.frame["race_id"])


def test_data_quality_filtering_can_be_disabled(db_session):
    generate_synthetic_data(db_session, SyntheticConfig(n_days=10, races_per_day=2, seed=3))
    dataset = ResearchDatasetBuilder(db_session, apply_data_quality=False).build()
    assert dataset.excluded_races == 0


def test_dataset_round_trips_through_parquet_and_csv(seeded_session, tmp_path):
    dataset = ResearchDatasetBuilder(seeded_session).build()

    parquet = dataset.to_parquet(tmp_path / "d.parquet")
    csv = dataset.to_csv(tmp_path / "d.csv")

    assert pd.read_parquet(parquet).shape == dataset.frame.shape
    assert len(pd.read_csv(csv)) == dataset.rows


def test_dataset_split_is_leak_free(seeded_session):
    dataset = ResearchDatasetBuilder(seeded_session).build()
    split = dataset.split(train_end=date(2025, 2, 1), embargo_days=3)

    assert split.train_rows > 0
    assert split.test_rows > 0
    assert not set(split.train["race_id"]) & set(split.test["race_id"])


def test_dataset_summary_and_render(seeded_session):
    dataset = ResearchDatasetBuilder(seeded_session).build()
    summary = dataset.summary()

    assert summary["rows"] == dataset.rows
    assert summary["features"] == len(dataset.feature_names)
    assert "Research dataset" in dataset.render()


def test_x_and_y_line_up(seeded_session):
    dataset = ResearchDatasetBuilder(seeded_session).build()
    assert list(dataset.X().columns) == dataset.feature_names
    assert len(dataset.y()) == dataset.rows


def test_empty_database_yields_an_empty_dataset(db_session):
    dataset = ResearchDatasetBuilder(db_session).build()
    assert dataset.rows == 0
    assert dataset.feature_names == []


# ---------------------------------------------------------------------------
# Feature manifest and store
# ---------------------------------------------------------------------------
def test_feature_manifest_reports_coverage(seeded_session):
    frame = FeaturePipeline(seeded_session).build()
    manifest = feature_manifest(frame)

    assert not manifest.empty
    assert set(manifest["group"]) <= {"horse", "jockey", "trainer", "market", "race", "score", "other"}
    assert manifest["coverage"].between(0, 1).all()


def test_feature_columns_are_stable_and_numeric(seeded_session):
    frame = FeaturePipeline(seeded_session).build()
    columns = feature_columns(frame)

    assert columns == sorted(columns), "column order must be deterministic"
    assert "race_id" not in columns
    assert "off_time" not in columns


def test_features_save_and_reload(seeded_session):
    frame = FeaturePipeline(seeded_session).build()
    written = save_features(seeded_session, frame.head(50))
    assert written == 50

    reloaded = load_features(seeded_session)
    assert len(reloaded) == 50
    assert "horse_form_score" in reloaded.columns
    assert "horse_win_rate" in reloaded.columns  # from the JSON vector


def test_saving_features_twice_updates_in_place(seeded_session):
    frame = FeaturePipeline(seeded_session).build().head(20)
    save_features(seeded_session, frame)
    save_features(seeded_session, frame)

    assert len(load_features(seeded_session)) == 20


def test_stored_vector_contains_no_nan_tokens(seeded_session):
    """``json.dumps`` emits a bare ``NaN``, which is invalid JSON on read-back."""
    import json

    from backend.models import RaceFeatures

    frame = FeaturePipeline(seeded_session).build()
    save_features(seeded_session, frame.head(30))

    for row in seeded_session.query(RaceFeatures).limit(30):
        serialised = json.dumps(row.features)
        assert "NaN" not in serialised
        assert "Infinity" not in serialised


def test_saving_an_empty_frame_is_a_no_op(db_session):
    assert save_features(db_session, pd.DataFrame()) == 0


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def test_synthetic_generation_is_deterministic(db_session, db_engine):
    from sqlalchemy.orm import Session

    from backend.models import RaceResult

    first = generate_synthetic_data(db_session, SyntheticConfig(n_days=5, seed=99))
    positions_first = [
        (r.race_id, r.horse_id, r.finishing_position)
        for r in db_session.query(RaceResult).order_by(RaceResult.race_id, RaceResult.horse_id)
    ]

    from backend.models import Base

    Base.metadata.drop_all(db_engine)
    Base.metadata.create_all(db_engine)
    with Session(db_engine, expire_on_commit=False) as fresh:
        second = generate_synthetic_data(fresh, SyntheticConfig(n_days=5, seed=99))
        positions_second = [
            (r.race_id, r.horse_id, r.finishing_position)
            for r in fresh.query(RaceResult).order_by(RaceResult.race_id, RaceResult.horse_id)
        ]

    assert first.races == second.races
    assert positions_first == positions_second


def test_synthetic_market_has_no_risk_free_arbitrage(seeded_session):
    """Best-price overround must stay above 1.0, or the negative control breaks."""
    dataset = ResearchDatasetBuilder(seeded_session).build(require_market=True)
    overround = dataset.frame.groupby("race_id")["mkt_overround"].first().dropna()

    assert (overround > 1.0).all(), "generated market contains free money"


def test_synthetic_races_have_exactly_one_winner(db_session):
    from backend.models import RaceResult

    generate_synthetic_data(db_session, SyntheticConfig(n_days=5, seed=1))
    winners = (
        pd.DataFrame(
            [(r.race_id, int(r.is_winner)) for r in db_session.query(RaceResult)],
            columns=["race_id", "won"],
        )
        .groupby("race_id")["won"]
        .sum()
    )
    assert (winners == 1).all()


def test_synthetic_data_passes_its_own_quality_checks(db_session):
    from backend.data_quality import DataValidator

    generate_synthetic_data(db_session, SyntheticConfig(n_days=10, seed=2))
    report = DataValidator(db_session, today=date(2025, 12, 31)).validate_database()
    assert report.is_clean, report.render()
