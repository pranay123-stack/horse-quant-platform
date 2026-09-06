"""Training dataset construction, split integrity and leak detection."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.ml.dataset import (
    LEAKAGE_AUC_THRESHOLD,
    SplitConfig,
    TrainingData,
    TrainingDatasetBuilder,
    detect_leaky_features,
)
from backend.research.splits import TemporalLeakageError
from tests.fixtures.ml_builders import SMALL_SPLIT, make_ml_session, make_training_data

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def ml_session():
    session = make_ml_session()
    yield session
    session.close()


@pytest.fixture(scope="module")
def training_data(ml_session):
    return make_training_data(ml_session)


# ---------------------------------------------------------------------------
# SplitConfig
# ---------------------------------------------------------------------------
def test_default_windows_match_the_specification():
    config = SplitConfig()
    assert config.train_start == date(2018, 1, 1)
    assert config.train_end == date(2023, 12, 31)
    assert config.valid_start == date(2024, 1, 1)
    assert config.valid_end == date(2024, 12, 31)
    assert config.test_start == date(2025, 1, 1)


def test_out_of_order_windows_are_rejected():
    with pytest.raises(TemporalLeakageError, match="strictly ordered"):
        SplitConfig(train_end=date(2025, 1, 1), valid_start=date(2020, 1, 1))


def test_test_window_must_follow_validation():
    with pytest.raises(TemporalLeakageError):
        SplitConfig(
            train_start=date(2020, 1, 1),
            train_end=date(2021, 1, 1),
            valid_start=date(2021, 1, 2),
            valid_end=date(2022, 1, 1),
            test_start=date(2021, 6, 1),
        )


@pytest.mark.parametrize("fraction", [0, 1, 1.5, -0.2])
def test_calibration_fraction_bounds(fraction):
    with pytest.raises(ValueError, match="calibration_fraction"):
        SplitConfig(calibration_fraction=fraction)


def test_adaptive_split_derives_ordered_windows():
    frame = pd.DataFrame({"race_date": pd.date_range("2025-01-01", periods=100, freq="D").date})
    config = SplitConfig.adaptive(frame)

    assert config.train_start < config.train_end < config.valid_start
    assert config.valid_end < config.test_start <= config.test_end


def test_adaptive_split_needs_dates():
    with pytest.raises(ValueError, match="no usable dates"):
        SplitConfig.adaptive(pd.DataFrame({"race_date": [None, None]}))


# ---------------------------------------------------------------------------
# Split correctness
# ---------------------------------------------------------------------------
def test_all_four_windows_are_populated(training_data):
    for name in ("train", "valid_stop", "valid_calib", "test"):
        assert not training_data.frame(name).empty, f"{name} split is empty"


def test_splits_are_strictly_ordered_in_time(training_data):
    boundaries = [
        (training_data.train, training_data.valid_stop),
        (training_data.valid_stop, training_data.valid_calib),
        (training_data.valid_calib, training_data.test),
    ]
    for earlier, later in boundaries:
        assert pd.to_datetime(earlier["race_date"]).max() < pd.to_datetime(later["race_date"]).min()


def test_no_race_appears_in_two_splits(training_data):
    seen: set[str] = set()
    for name in ("train", "valid_stop", "valid_calib", "test"):
        races = set(training_data.frame(name)["race_id"])
        assert not races & seen, f"{name} shares races with an earlier split"
        seen |= races


def test_assert_no_leakage_catches_an_overlap(training_data):
    broken = TrainingData(
        train=training_data.test,  # deliberately inverted
        valid_stop=training_data.valid_stop,
        valid_calib=training_data.valid_calib,
        test=training_data.train,
        numeric_features=training_data.numeric_features,
        categorical_features=training_data.categorical_features,
    )
    with pytest.raises(TemporalLeakageError):
        broken.assert_no_leakage()


def test_validation_halves_never_split_a_race(training_data):
    """A race cut in half would break every race-level metric."""
    stop = set(training_data.valid_stop["race_id"])
    calib = set(training_data.valid_calib["race_id"])
    assert not stop & calib


def test_embargo_creates_a_gap_after_training(ml_session):
    embargoed = TrainingDatasetBuilder(ml_session).build(
        SplitConfig(
            train_start=date(2024, 1, 1),
            train_end=date(2024, 5, 31),
            valid_start=date(2024, 6, 1),
            valid_end=date(2024, 7, 15),
            test_start=date(2024, 7, 16),
            test_end=date(2024, 8, 31),
            embargo_days=10,
        )
    )
    gap = (
        pd.to_datetime(embargoed.valid_stop["race_date"]).min()
        - pd.to_datetime(embargoed.train["race_date"]).max()
    ).days
    assert gap > 10


def test_target_is_not_in_the_feature_list(training_data):
    for forbidden in ("won", "placed", "is_winner", "finishing_position", "starting_price"):
        assert forbidden not in training_data.feature_names


def test_categorical_features_are_carried_through(training_data):
    assert "course_id" in training_data.categorical_features
    assert "going_band" in training_data.categorical_features


def test_feature_families_can_be_excluded(ml_session):
    """The market-free control: can the other features predict anything alone?"""
    full = make_training_data(ml_session)
    market_free = TrainingDatasetBuilder(ml_session).build(
        SMALL_SPLIT, drop_feature_prefixes=("mkt_", "market_")
    )

    assert len(market_free.numeric_features) < len(full.numeric_features)
    assert not any(name.startswith(("mkt_", "market_")) for name in market_free.numeric_features)


def test_summary_and_render(training_data):
    summary = training_data.summary()
    assert summary["splits"]["train"]["rows"] == len(training_data.train)
    assert "Training data" in training_data.render()


def test_frame_rejects_an_unknown_split(training_data):
    with pytest.raises(KeyError, match="unknown split"):
        training_data.frame("nonsense")


def test_valid_property_concatenates_both_halves(training_data):
    assert len(training_data.valid) == len(training_data.valid_stop) + len(training_data.valid_calib)


def test_empty_database_raises(db_session):
    with pytest.raises(ValueError, match="no rows"):
        TrainingDatasetBuilder(db_session).build()


# ---------------------------------------------------------------------------
# Leak detection
# ---------------------------------------------------------------------------
def _leak_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    won = rng.integers(0, 2, 1000)
    return pd.DataFrame(
        {
            "won": won,
            "copy_of_target": won.astype(float),
            "inverted_target": 1 - won,
            "weak_signal": won * 0.4 + rng.normal(size=1000),
            "pure_noise": rng.normal(size=1000),
            "constant": np.ones(1000),
        }
    )


def test_detector_finds_a_renamed_target():
    suspects = detect_leaky_features(_leak_frame(), ["copy_of_target", "weak_signal", "pure_noise"])
    assert [suspect.feature for suspect in suspects] == ["copy_of_target"]


def test_detector_finds_a_perfectly_inverted_target():
    """A perfectly *negative* separation is just as much a leak."""
    suspects = detect_leaky_features(_leak_frame(), ["inverted_target"])
    assert suspects and suspects[0].auc >= LEAKAGE_AUC_THRESHOLD


def test_detector_ignores_genuine_signal():
    assert detect_leaky_features(_leak_frame(), ["weak_signal", "pure_noise"]) == []


def test_detector_ignores_constant_and_missing_columns():
    assert detect_leaky_features(_leak_frame(), ["constant", "does_not_exist"]) == []


def test_detector_handles_degenerate_inputs():
    assert detect_leaky_features(pd.DataFrame(), ["a"]) == []
    single_class = pd.DataFrame({"won": [0, 0, 0], "feature": [1.0, 2.0, 3.0]})
    assert detect_leaky_features(single_class, ["feature"]) == []


def test_real_dataset_has_no_leaky_features(training_data):
    """The regression test for the ``is_winner`` leak this detector was born from."""
    suspects = detect_leaky_features(training_data.train, training_data.feature_names)
    assert suspects == [], f"leaky features present: {[str(s) for s in suspects]}"


def test_leak_suspect_renders():
    suspects = detect_leaky_features(_leak_frame(), ["copy_of_target"])
    assert "copy_of_target" in str(suspects[0])
    assert "AUC" in str(suspects[0])
