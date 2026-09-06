"""Edge cases across the research layer.

Empty inputs, absent columns and unserialisable values. These paths are the ones
that only fire on the first day of a new deployment, in the middle of a backfill,
or on the one race that has no market — exactly when nobody is watching.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from backend.backtesting import BacktestConfig, BacktestEngine, UniformPredictor
from backend.features import FeaturePipeline, load_features
from backend.features.base import (
    EVENT_TIME,
    as_of_join,
    cumulative_state,
    load_history,
    load_labels,
    load_odds,
    load_targets,
    numeric_column,
)
from backend.features.horse_features import build_horse_conditional_state, build_horse_state
from backend.features.jockey_features import build_jockey_course_state, build_jockey_state
from backend.features.market_features import build_market_state
from backend.features.race_features import build_race_features
from backend.features.store import _clean, save_features
from backend.features.trainer_features import build_trainer_jockey_state, build_trainer_state
from backend.research import ResearchDataset, iter_race_groups, walk_forward_splits
from backend.research.synthetic import SyntheticConfig, generate_synthetic_data
from backend.utils.timeutils import UK_TZ
from tests.fixtures.racing_builders import build_race_with_results

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Loaders on an empty database
# ---------------------------------------------------------------------------
def test_loaders_return_empty_frames_not_errors(db_session):
    for loader in (load_history, load_targets, load_odds):
        frame = loader(db_session)
        assert frame.empty
        assert isinstance(frame, pd.DataFrame)

    labels = load_labels(db_session)
    assert labels.empty
    assert "won" in labels.columns


def test_loaders_respect_a_race_id_filter(db_session):
    build_race_with_results(
        db_session,
        race_id="rac_keep",
        off_time=datetime(2025, 3, 1, 14, 0, tzinfo=UK_TZ),
        runners=[("hrs_a", 1), ("hrs_b", 2)],
    )
    build_race_with_results(
        db_session,
        race_id="rac_drop",
        off_time=datetime(2025, 3, 2, 14, 0, tzinfo=UK_TZ),
        runners=[("hrs_a", 2), ("hrs_c", 1)],
    )
    db_session.commit()

    assert set(load_history(db_session, race_ids=["rac_keep"])["race_id"]) == {"rac_keep"}
    assert set(load_targets(db_session, race_ids=["rac_keep"])["race_id"]) == {"rac_keep"}
    assert set(load_labels(db_session, race_ids=["rac_keep"])["race_id"]) == {"rac_keep"}


def test_races_without_a_date_are_dropped_from_the_time_axis(db_session):
    """A row with no timestamp cannot be placed in history, so it must not be."""
    from backend.models import Race

    build_race_with_results(
        db_session,
        race_id="rac_undated",
        off_time=datetime(2025, 3, 1, 14, 0, tzinfo=UK_TZ),
        runners=[("hrs_a", 1), ("hrs_b", 2)],
    )
    db_session.commit()
    race = db_session.get(Race, "rac_undated")
    race.off_time = None
    race.race_date = None
    db_session.commit()

    assert load_history(db_session).empty
    assert load_targets(db_session).empty


# ---------------------------------------------------------------------------
# Feature builders with nothing to build from
# ---------------------------------------------------------------------------
def test_state_builders_handle_empty_history():
    empty = pd.DataFrame()
    assert build_horse_state(empty).empty
    assert build_jockey_state(empty).empty
    assert build_trainer_state(empty).empty
    assert build_jockey_course_state(empty).empty
    assert build_trainer_jockey_state(empty).empty
    assert build_horse_conditional_state(empty, "going_band", "horse_going").empty
    assert build_market_state(empty).empty


def test_state_builders_handle_history_without_the_entity():
    frame = pd.DataFrame(
        {
            "horse_id": ["h1"],
            "jockey_id": [None],
            "trainer_id": [None],
            "course_id": [None],
            "is_winner": [True],
            "finishing_position": [1],
            "starting_price": [3.0],
            EVENT_TIME: pd.to_datetime(["2025-01-01"], utc=True),
        }
    )
    assert build_jockey_state(frame).empty
    assert build_trainer_state(frame).empty
    assert build_jockey_course_state(frame).empty
    assert build_trainer_jockey_state(frame).empty


def test_race_features_on_an_empty_frame():
    frame = build_race_features(pd.DataFrame(columns=["race_id", "horse_id"]))
    assert "race_field_size" in frame
    assert frame.empty


def test_race_features_tolerate_absent_optional_columns():
    minimal = pd.DataFrame({"race_id": ["r1", "r1"], "horse_id": ["h1", "h2"]})
    frame = build_race_features(minimal)

    assert frame["race_field_size"].tolist() == [2, 2]
    assert frame["runner_or_rank"].isna().all()


def test_numeric_column_returns_nan_for_missing_columns():
    frame = pd.DataFrame({"a": [1, 2]})
    assert numeric_column(frame, "a").tolist() == [1, 2]

    missing = numeric_column(frame, "nope")
    assert len(missing) == 2
    assert missing.isna().all()


def test_cumulative_state_assembles_from_series():
    history = pd.DataFrame(
        {"horse_id": ["h1", "h1"], EVENT_TIME: pd.to_datetime(["2025-01-01", "2025-02-01"], utc=True)}
    )
    state = cumulative_state(history, by="horse_id", builders={"stat": pd.Series([1.0, 2.0])})
    assert state["stat"].tolist() == [1.0, 2.0]


def test_as_of_join_on_empty_targets():
    empty = pd.DataFrame(columns=["horse_id", EVENT_TIME])
    merged = as_of_join(empty, pd.DataFrame(), by="horse_id", columns=["stat"])
    assert "stat" in merged.columns


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def test_pipeline_on_an_empty_database(db_session):
    pipeline = FeaturePipeline(db_session)
    frame = pipeline.build()

    assert frame.empty
    assert "no target runners in window" in pipeline.stats.warnings


def test_pipeline_warns_when_there_is_no_market_or_history(db_session):
    build_race_with_results(
        db_session,
        race_id="rac_solo",
        off_time=datetime(2025, 3, 1, 14, 0, tzinfo=UK_TZ),
        runners=[("hrs_a", 1), ("hrs_b", 2)],
    )
    db_session.commit()

    pipeline = FeaturePipeline(db_session)
    pipeline.build()
    assert any("no market data" in warning for warning in pipeline.stats.warnings)


def test_build_for_race_targets_one_race(db_session):
    for index, day in enumerate([1, 2]):
        build_race_with_results(
            db_session,
            race_id=f"rac_{index}",
            off_time=datetime(2025, 3, day, 14, 0, tzinfo=UK_TZ),
            runners=[("hrs_a", 1), (f"hrs_b{index}", 2)],
        )
    db_session.commit()

    frame = FeaturePipeline(db_session).build_for_race("rac_1")
    assert set(frame["race_id"]) == {"rac_1"}


# ---------------------------------------------------------------------------
# Feature store
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        (np.float64(1.5), 1.5),
        (np.int64(3), 3.0),
        (True, True),
        ("text", "text"),
        (date(2025, 1, 1), "2025-01-01"),
    ],
)
def test_clean_produces_json_safe_scalars(value, expected):
    assert _clean(value) == expected


def test_clean_handles_pandas_timestamps():
    assert _clean(pd.Timestamp("2025-01-01")).startswith("2025-01-01")
    assert _clean(pd.NaT) is None


def test_load_features_returns_an_empty_frame_when_nothing_is_stored(db_session):
    frame = load_features(db_session)
    assert frame.empty
    assert "horse_form_score" in frame.columns


def test_save_features_batches(db_session):
    generate_synthetic_data(db_session, SyntheticConfig(n_days=6, races_per_day=2, seed=8))
    frame = FeaturePipeline(db_session).build()

    written = save_features(db_session, frame, batch_size=7)
    assert written == len(frame)
    assert len(load_features(db_session)) == len(frame)


# ---------------------------------------------------------------------------
# Dataset and splits
# ---------------------------------------------------------------------------
def test_empty_dataset_properties():
    dataset = ResearchDataset(frame=pd.DataFrame(), feature_names=[])
    assert dataset.rows == 0
    assert dataset.races == 0
    assert dataset.date_range == (None, None)
    assert dataset.base_rate == 0.0
    assert "Research dataset" in dataset.render()


def test_dataset_date_range_ignores_unparseable_dates():
    frame = pd.DataFrame({"race_id": ["r1"], "race_date": ["not-a-date"], "won": [1]})
    dataset = ResearchDataset(frame=frame, feature_names=[])
    assert dataset.date_range == (None, None)


def test_walk_forward_on_an_empty_frame():
    assert walk_forward_splits(pd.DataFrame({"race_date": []})) == []


def test_iter_race_groups_without_a_date_column():
    """With no date to order by it falls back to race_id -- still deterministic."""
    frame = pd.DataFrame({"race_id": ["r2", "r1"], "value": [1, 2]})
    assert [race_id for race_id, _ in iter_race_groups(frame)] == ["r1", "r2"]


# ---------------------------------------------------------------------------
# Backtester edge cases
# ---------------------------------------------------------------------------
def test_predictor_returning_the_wrong_length_is_skipped():
    frame = pd.DataFrame(
        {
            "race_id": ["r1", "r1"],
            "horse_id": ["h1", "h2"],
            "race_date": [date(2025, 1, 1)] * 2,
            "mkt_latest_odds": [3.0, 4.0],
            "won": [1, 0],
        }
    )

    class Broken:
        name = "broken"

        def __call__(self, race: pd.DataFrame) -> pd.Series:
            return pd.Series([1.0])  # wrong length

    result = BacktestEngine(BacktestConfig(min_expected_value=0.0)).run(frame, Broken())
    assert result.skipped_races == 1
    assert result.metrics.total_bets == 0


def test_zero_stake_places_no_bet():
    frame = pd.DataFrame(
        {
            "race_id": ["r1", "r1"],
            "horse_id": ["h1", "h2"],
            "race_date": [date(2025, 1, 1)] * 2,
            "mkt_latest_odds": [3.0, 4.0],
            "won": [1, 0],
        }
    )
    config = BacktestConfig(staking="flat", flat_stake=0.0, min_expected_value=0.0)
    assert BacktestEngine(config).run(frame, UniformPredictor()).metrics.total_bets == 0


def test_metrics_render_and_dict_on_an_empty_run():
    result = BacktestEngine().run(pd.DataFrame(), UniformPredictor())
    assert "Backtest results" in result.render()
    assert result.as_dict()["metrics"]["bets"]["total"] == 0
