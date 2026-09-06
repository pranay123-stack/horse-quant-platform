"""Inference: the Phase 4 deliverable — a race in, probabilities out."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.ml.models.calibrated import CalibratedRacingModel
from backend.ml.models.registry import ModelRegistry
from backend.ml.predict import RacePredictor, load_champion
from backend.ml.train import ModelTrainer, TrainingConfig
from backend.utils.exceptions import ModelError, NotFoundError
from tests.fixtures.ml_builders import make_ml_session, make_training_data

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def ml_session():
    session = make_ml_session()
    yield session
    session.close()


@pytest.fixture(scope="module")
def trained(ml_session, tmp_path_factory):
    """One trained, registered champion shared by every test here."""
    registry = ModelRegistry(tmp_path_factory.mktemp("registry"))
    data = make_training_data(ml_session)
    config = TrainingConfig(
        models=("lightgbm",), calibrations=("none",), params={"lightgbm": {"n_estimators": 40}}
    )
    ModelTrainer(config, registry).train(data)
    return registry, data


@pytest.fixture
def predictor(ml_session, trained):
    registry, _ = trained
    return RacePredictor(ml_session, registry=registry)


# ---------------------------------------------------------------------------
def test_prediction_probabilities_sum_to_one(predictor, trained):
    _, data = trained
    race_id = data.test["race_id"].iloc[0]

    prediction = predictor.predict_race(race_id)
    assert prediction.probability_sum == pytest.approx(1.0, abs=1e-6)


def test_every_probability_is_in_range(predictor, trained):
    _, data = trained
    prediction = predictor.predict_race(data.test["race_id"].iloc[0])

    for runner in prediction.runners:
        assert 0.0 <= runner.probability <= 1.0


def test_runners_are_ranked_most_likely_first(predictor, trained):
    _, data = trained
    prediction = predictor.predict_race(data.test["race_id"].iloc[0])

    probabilities = [runner.probability for runner in prediction.runners]
    assert probabilities == sorted(probabilities, reverse=True)
    assert [runner.rank for runner in prediction.runners] == list(range(1, len(probabilities) + 1))
    assert prediction.favourite is prediction.runners[0]


def test_prediction_output_matches_the_specified_shape(predictor, trained):
    """``{"horse_id": ..., "probability": 0.285}``"""
    _, data = trained
    prediction = predictor.predict_race(data.test["race_id"].iloc[0])

    payload = prediction.runners[0].as_dict()
    assert set(payload) >= {"horse_id", "probability"}
    assert isinstance(payload["horse_id"], str)
    assert isinstance(payload["probability"], float)


def test_prediction_serialises_and_renders(predictor, trained):
    _, data = trained
    prediction = predictor.predict_race(data.test["race_id"].iloc[0])

    payload = prediction.as_dict()
    assert payload["race_id"] == prediction.race_id
    assert len(payload["runners"]) == len(prediction.runners)
    assert "probability" in prediction.render()


def test_predictions_are_repeatable(predictor, trained):
    _, data = trained
    race_id = data.test["race_id"].iloc[0]

    first = predictor.predict_race(race_id)
    second = predictor.predict_race(race_id)
    assert [r.probability for r in first.runners] == [r.probability for r in second.runners]


def test_unknown_race_raises_not_found(predictor):
    with pytest.raises(NotFoundError, match="no runners"):
        predictor.predict_race("rac_does_not_exist")


def test_predict_date_scores_every_race_that_day(predictor, trained):
    _, data = trained
    race_date = str(pd.to_datetime(data.test["race_date"]).dt.date.iloc[0])

    predictions = predictor.predict_date(race_date)
    assert predictions
    for prediction in predictions:
        assert prediction.probability_sum == pytest.approx(1.0, abs=1e-6)


def test_predict_date_with_no_racing_returns_nothing(predictor):
    assert predictor.predict_date("1999-01-01") == []


def test_predict_frame_on_empty_input(predictor):
    result = predictor.predict_frame(pd.DataFrame())
    assert result.empty


def test_predictor_loads_the_champion_lazily(ml_session, trained):
    registry, _ = trained
    predictor = RacePredictor(ml_session, registry=registry)

    assert predictor._model is None
    assert predictor.model.model.name == "lightgbm"
    assert predictor._model is not None


def test_an_explicit_model_overrides_the_registry(ml_session, trained):
    registry, _ = trained
    explicit = registry.load_champion()
    predictor = RacePredictor(ml_session, explicit, registry=registry)
    assert predictor.model is explicit


def test_load_champion_without_one_raises(tmp_path):
    with pytest.raises(ModelError, match="no champion"):
        load_champion(ModelRegistry(tmp_path))


def test_loaded_model_reproduces_training_time_predictions(ml_session, trained):
    """Model loading must be lossless: same inputs, same numbers."""
    registry, data = trained
    from_registry = registry.load_champion()
    from_disk = CalibratedRacingModel.load(registry.root / registry.champion().path)

    sample = data.test.head(30)
    assert from_registry.predict_proba(sample).tolist() == from_disk.predict_proba(sample).tolist()
