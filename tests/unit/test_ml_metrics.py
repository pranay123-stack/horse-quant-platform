"""Metric correctness.

Values are computed by hand where possible. A metric that is merely "plausible"
will happily report a broken model as a good one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.ml.metrics.classification import (
    compute_classification_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
)
from backend.ml.metrics.racing import (
    compute_racing_metrics,
    normalise_within_race,
    race_probability_frame,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Within-race normalisation
# ---------------------------------------------------------------------------
def test_normalisation_makes_each_race_sum_to_one():
    races = ["r1", "r1", "r1", "r2", "r2"]
    normalised = normalise_within_race([0.2, 0.3, 0.5, 0.1, 0.1], races)

    frame = pd.DataFrame({"race_id": races, "p": normalised})
    totals = frame.groupby("race_id")["p"].sum()
    assert totals.round(9).tolist() == [1.0, 1.0]


def test_normalisation_preserves_relative_order():
    normalised = normalise_within_race([0.1, 0.4, 0.2], ["r1"] * 3)
    assert normalised.tolist() == pytest.approx([0.1 / 0.7, 0.4 / 0.7, 0.2 / 0.7])


def test_normalisation_falls_back_to_uniform_when_the_race_sums_to_zero():
    normalised = normalise_within_race([0.0, 0.0, 0.0, 0.5, 0.5], ["r1"] * 3 + ["r2"] * 2)
    assert normalised.tolist()[:3] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_normalisation_clips_negative_inputs():
    normalised = normalise_within_race([-0.5, 1.0], ["r1", "r1"])
    assert (normalised >= 0).all()
    assert normalised.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Race metrics
# ---------------------------------------------------------------------------
def _perfect_race(n_races: int = 10, runners: int = 8) -> pd.DataFrame:
    """Runner 0 always wins, and always has the highest probability."""
    rows = []
    for race in range(n_races):
        for runner in range(runners):
            rows.append(
                {
                    "race_id": f"r{race}",
                    "won": int(runner == 0),
                    "probability": 0.9 if runner == 0 else 0.1 / (runners - 1),
                }
            )
    return pd.DataFrame(rows)


def test_a_perfect_model_scores_top1_of_one():
    frame = _perfect_race()
    metrics = compute_racing_metrics(frame["race_id"], frame["won"], frame["probability"])

    assert metrics.top1_hit_rate == 1.0
    assert metrics.top3_hit_rate == 1.0
    assert metrics.mean_winner_rank == 1.0
    assert metrics.races == 10


def test_uniform_predictions_score_exactly_the_uniform_baseline():
    """The reference point: log(field size) for an eight-runner race."""
    rows = []
    for race in range(20):
        for runner in range(8):
            rows.append({"race_id": f"r{race}", "won": int(runner == 0), "probability": 0.125})
    frame = pd.DataFrame(rows)

    metrics = compute_racing_metrics(frame["race_id"], frame["won"], frame["probability"])
    assert metrics.race_log_loss == pytest.approx(np.log(8), abs=1e-9)
    assert metrics.uniform_log_loss == pytest.approx(np.log(8), abs=1e-9)
    assert metrics.skill_score == pytest.approx(0.0, abs=1e-9)


def test_a_better_than_uniform_model_has_positive_skill():
    frame = _perfect_race()
    metrics = compute_racing_metrics(frame["race_id"], frame["won"], frame["probability"])
    assert metrics.skill_score > 0.5


def test_a_worse_than_uniform_model_has_negative_skill():
    """Backing the wrong horse every time must score worse than knowing nothing."""
    rows = []
    for race in range(10):
        for runner in range(8):
            rows.append(
                {
                    "race_id": f"r{race}",
                    "won": int(runner == 0),
                    "probability": 0.01 if runner == 0 else 0.99 / 7,
                }
            )
    frame = pd.DataFrame(rows)

    metrics = compute_racing_metrics(frame["race_id"], frame["won"], frame["probability"])
    assert metrics.skill_score < 0
    assert metrics.top1_hit_rate == 0.0


def test_metrics_on_an_empty_frame():
    metrics = compute_racing_metrics(pd.Series(dtype=str), pd.Series(dtype=int), pd.Series(dtype=float))
    assert metrics.races == 0
    assert metrics.as_dict()["top1_hit_rate"] == 0.0


def test_race_without_a_winner_is_ignored_not_crashed():
    frame = pd.DataFrame({"race_id": ["r1", "r1"], "won": [0, 0], "probability": [0.5, 0.5]})
    metrics = compute_racing_metrics(frame["race_id"], frame["won"], frame["probability"])
    assert metrics.races == 1
    assert metrics.top1_hit_rate == 0.0


def test_race_probability_frame_adds_ranks():
    frame = pd.DataFrame({"race_id": ["r1"] * 3, "horse_id": ["a", "b", "c"]})
    scored = race_probability_frame(frame, [0.1, 0.6, 0.3])

    assert scored["probability_normalised"].sum() == pytest.approx(1.0)
    assert scored.set_index("horse_id")["model_rank"].to_dict() == {"b": 1, "c": 2, "a": 3}


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def test_perfect_predictions_score_perfectly():
    truth = np.array([0, 0, 1, 1])
    metrics = compute_classification_metrics(truth, np.array([0.0, 0.0, 1.0, 1.0]))

    assert metrics.roc_auc == 1.0
    assert metrics.brier == pytest.approx(0.0, abs=1e-6)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_base_rate_and_calibration_ratio():
    truth = np.array([0] * 9 + [1])
    metrics = compute_classification_metrics(truth, np.full(10, 0.2))

    assert metrics.base_rate == pytest.approx(0.1)
    assert metrics.mean_predicted == pytest.approx(0.2)
    assert metrics.calibration_ratio == pytest.approx(2.0)  # twice as confident as reality


def test_auc_is_undefined_with_one_class():
    metrics = compute_classification_metrics(np.zeros(10), np.full(10, 0.3))
    assert metrics.roc_auc == 0.0  # not computed rather than fabricated
    assert metrics.brier == pytest.approx(0.09)


def test_metrics_on_empty_input():
    metrics = compute_classification_metrics(np.array([]), np.array([]))
    assert metrics.rows == 0
    assert metrics.as_dict()["log_loss"] == 0.0


def test_log_loss_survives_a_confident_miss():
    """A prediction of exactly 0 on a winner must not produce infinity."""
    metrics = compute_classification_metrics(np.array([1, 0]), np.array([0.0, 1.0]))
    assert np.isfinite(metrics.log_loss)


# ---------------------------------------------------------------------------
# Calibration measurement
# ---------------------------------------------------------------------------
def test_perfectly_calibrated_predictions_have_near_zero_ece():
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0.05, 0.95, 20_000)
    truth = rng.binomial(1, probabilities)

    assert expected_calibration_error(truth, probabilities, bins=10) < 0.02


def test_systematically_overconfident_predictions_have_large_ece():
    rng = np.random.default_rng(1)
    true_probability = rng.uniform(0.05, 0.5, 20_000)
    truth = rng.binomial(1, true_probability)
    inflated = np.clip(true_probability * 2, 0, 1)

    assert expected_calibration_error(truth, inflated, bins=10) > 0.1


def test_maximum_calibration_error_is_at_least_the_mean():
    rng = np.random.default_rng(2)
    probabilities = rng.uniform(0, 1, 5000)
    truth = rng.binomial(1, probabilities * 0.5)

    assert maximum_calibration_error(truth, probabilities) >= expected_calibration_error(truth, probabilities)


def test_reliability_curve_bins_and_renders():
    rng = np.random.default_rng(3)
    probabilities = rng.uniform(0, 1, 1000)
    truth = rng.binomial(1, probabilities)

    curve = reliability_curve(truth, probabilities, bins=5)
    assert len(curve.predicted) <= 5
    assert sum(curve.counts) == 1000
    assert "predicted" in curve.render()


def test_reliability_curve_handles_a_constant_prediction():
    curve = reliability_curve(np.array([0, 1, 0, 1]), np.full(4, 0.5))
    assert curve.counts == [4]
    assert curve.predicted == [0.5]


def test_reliability_curve_on_empty_input():
    curve = reliability_curve(np.array([]), np.array([]))
    assert curve.counts == []
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0
