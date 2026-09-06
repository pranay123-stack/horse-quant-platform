"""The product: betting rules, storage, settlement, performance, replay.

The rules tests carry the most weight. They are the last thing between a model
output and a recommendation someone acts on with money, so each guard is tested
for the case it exists to catch — and, just as importantly, for *which* reason
it reports. "No price available" and "expected value too low" send a person to
completely different places.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from backend.models.predictions import Prediction, Recommendation
from backend.prediction_service.daily import (
    DailyJobResult,
    run_daily_job,
    settle_predictions,
)
from backend.prediction_service.performance import (
    MIN_BETS_FOR_INFERENCE,
    summarise_performance,
)
from backend.prediction_service.replay import ReplayResult, compare_frames
from backend.prediction_service.rules import (
    MAX_ODDS,
    MIN_EXPECTED_VALUE,
    MIN_FIELD_SIZE,
    MIN_PROBABILITY,
    RULES,
    BettingRules,
    apply_rules,
)
from backend.prediction_service.service import (
    DayPredictions,
    RaceSignals,
    RunnerSignal,
    store_predictions,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def runner_row(**overrides) -> dict:
    row = {
        "race_id": "r1",
        "horse_id": "h1",
        "model_probability": 0.30,
        "odds": 5.0,
        "expected_value": 0.50,
        "field_size": 8,
    }
    row.update(overrides)
    return row


def frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows) or [runner_row()])


def decide(**overrides) -> tuple[str, str | None]:
    """Apply the rules to one runner and return (recommendation, reason)."""
    result = apply_rules(frame(runner_row(**overrides)))
    return result["recommendation"].iloc[0], result["rejection_reason"].iloc[0]


# ---------------------------------------------------------------------------
# The production rules
# ---------------------------------------------------------------------------
def test_a_qualifying_runner_is_backed():
    recommendation, reason = decide()
    assert recommendation == Recommendation.BET
    assert reason is None


def test_expected_value_below_the_threshold_is_rejected():
    recommendation, reason = decide(expected_value=MIN_EXPECTED_VALUE - 0.001)
    assert recommendation == Recommendation.NO_BET
    assert reason == "expected value too low"


def test_expected_value_exactly_at_the_threshold_is_rejected():
    """`> 5%`, not `>= 5%` — the boundary should not be a coin flip."""
    recommendation, _ = decide(expected_value=MIN_EXPECTED_VALUE)
    assert recommendation == Recommendation.NO_BET


def test_a_probability_in_the_tail_is_rejected_however_good_the_ev():
    """The guard that matters most.

    A 3% shout at 40/1 shows a huge EV, but that number is built almost
    entirely from extrapolation in the region where the model is least
    reliable.
    """
    recommendation, reason = decide(model_probability=0.03, odds=40.0, expected_value=0.20)
    assert recommendation == Recommendation.NO_BET
    assert reason == "probability below the minimum"


def test_a_probability_just_below_the_floor_is_rejected():
    recommendation, _ = decide(model_probability=MIN_PROBABILITY - 0.001)
    assert recommendation == Recommendation.NO_BET


def test_a_missing_price_is_rejected_as_a_data_problem():
    """Not as 'EV too low' — a missing price sends someone somewhere else."""
    recommendation, reason = decide(odds=None, expected_value=None)
    assert recommendation == Recommendation.NO_BET
    assert reason == "no price available"


def test_an_extreme_price_is_rejected():
    recommendation, reason = decide(odds=MAX_ODDS + 1, expected_value=2.0)
    assert recommendation == Recommendation.NO_BET
    assert reason == "price above the maximum"


def test_a_price_below_evens_is_rejected():
    recommendation, reason = decide(odds=1.0, expected_value=0.5)
    assert recommendation == Recommendation.NO_BET
    assert reason == "price below the minimum"


def test_a_small_field_is_rejected():
    recommendation, reason = decide(field_size=MIN_FIELD_SIZE - 1)
    assert recommendation == Recommendation.NO_BET
    assert reason == "field too small"


def test_data_problems_are_reported_before_judgements():
    """A runner with no price AND a poor EV must report the price."""
    _, reason = decide(odds=None, expected_value=-0.9, model_probability=0.02)
    assert reason == "no price available"


def test_a_missing_probability_is_rejected():
    recommendation, reason = decide(model_probability=None)
    assert recommendation == Recommendation.NO_BET
    assert reason == "no model probability"


def test_field_size_falls_back_to_counting_the_runners():
    """A frame without a field_size column must still be judged, not crash."""
    rows = [runner_row(horse_id=f"h{index}") for index in range(3)]
    for row in rows:
        row.pop("field_size")
    result = apply_rules(pd.DataFrame(rows))

    assert (result["recommendation"] == Recommendation.NO_BET).all()
    assert (result["rejection_reason"] == "field too small").all()


def test_the_rules_are_applied_per_runner_not_per_race():
    result = apply_rules(
        frame(
            runner_row(horse_id="good"),
            runner_row(horse_id="thin", model_probability=0.02, expected_value=0.9),
        )
    )
    assert list(result["recommendation"]) == [Recommendation.BET, Recommendation.NO_BET]


def test_an_empty_frame_is_handled():
    result = apply_rules(pd.DataFrame())
    assert result.empty
    assert "recommendation" in result.columns


def test_the_rules_describe_themselves():
    assert RULES.describe().startswith("BET when EV > 5%")
    assert set(RULES.as_dict()) == {
        "min_expected_value",
        "min_probability",
        "min_field_size",
        "max_odds",
        "min_odds",
    }


def test_stricter_rules_can_be_supplied():
    """The thresholds are configuration, not hard-coded in the logic."""
    strict = BettingRules(min_expected_value=0.40)
    result = apply_rules(frame(runner_row(expected_value=0.20)), strict)
    assert result["recommendation"].iloc[0] == Recommendation.NO_BET


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def make_day(*, race_date: date = date(2026, 6, 15), bets: int = 2) -> DayPredictions:
    runners = [
        RunnerSignal(
            horse_id=f"h{index}",
            horse=f"Horse {index}",
            model_probability=0.3 - index * 0.05,
            market_probability=0.25,
            odds=5.0,
            expected_value=0.5,
            edge=0.05,
            recommendation=Recommendation.BET if index < bets else Recommendation.NO_BET,
            rejection_reason=None if index < bets else "expected value too low",
        )
        for index in range(4)
    ]
    race = RaceSignals(
        race_id="rac_1",
        race="Ascot 15:30",
        race_date=race_date,
        off_time="15:30",
        course="Ascot",
        runners=runners,
    )
    return DayPredictions(
        race_date=race_date,
        races=[race],
        model_name="lightgbm",
        model_version="v3",
        generated_at="2026-06-15T07:00:00",
    )


def test_a_day_reports_its_bets():
    day = make_day(bets=2)
    assert day.total_runners == 4
    assert len(day.bets) == 2
    assert day.as_dict()["bets"] == 2


def test_storing_a_day_writes_every_runner(db_session, seeded_race):
    day = make_day()
    day.races[0].race_id = seeded_race["race_id"]
    for index, runner in enumerate(day.races[0].runners):
        runner.horse_id = seeded_race["horse_ids"][index]

    written = store_predictions(db_session, day)

    assert written == 4
    rows = db_session.query(Prediction).all()
    assert len(rows) == 4
    assert sum(1 for row in rows if row.is_bet) == 2
    assert rows[0].model_name == "lightgbm"
    assert rows[0].feature_version


def test_re_running_the_job_replaces_rather_than_duplicates(db_session, seeded_race):
    """Prices move through the morning, so a later run is a better answer."""
    day = make_day()
    day.races[0].race_id = seeded_race["race_id"]
    for index, runner in enumerate(day.races[0].runners):
        runner.horse_id = seeded_race["horse_ids"][index]

    store_predictions(db_session, day)
    for runner in day.races[0].runners:
        runner.odds = 6.5
    store_predictions(db_session, day)

    rows = db_session.query(Prediction).all()
    assert len(rows) == 4, "the second run duplicated instead of replacing"
    assert all(float(row.odds) == 6.5 for row in rows)


def test_storing_an_empty_day_writes_nothing(db_session):
    empty = DayPredictions(race_date=date(2026, 6, 15))
    assert store_predictions(db_session, empty) == 0


def test_a_prediction_serialises_for_the_api(db_session, seeded_race):
    day = make_day()
    day.races[0].race_id = seeded_race["race_id"]
    for index, runner in enumerate(day.races[0].runners):
        runner.horse_id = seeded_race["horse_ids"][index]
    store_predictions(db_session, day)

    payload = db_session.query(Prediction).first().as_dict()
    assert {"horse", "probability", "odds", "expected_value", "recommendation"} <= set(payload)
    assert payload["model"] == "lightgbm:v3"


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------
def test_settlement_marks_winners_and_losers(db_session, seeded_race):
    day = make_day(race_date=seeded_race["race_date"])
    day.races[0].race_id = seeded_race["race_id"]
    for index, runner in enumerate(day.races[0].runners):
        runner.horse_id = seeded_race["horse_ids"][index]
    store_predictions(db_session, day)

    count = settle_predictions(db_session, seeded_race["race_date"])

    assert count == 4
    rows = db_session.query(Prediction).all()
    assert all(row.settled for row in rows)
    assert sum(1 for row in rows if row.won) == 1, "exactly one winner per race"


def test_settlement_skips_races_with_no_result_yet(db_session, seeded_race_unsettled):
    day = make_day(race_date=seeded_race_unsettled["race_date"])
    day.races[0].race_id = seeded_race_unsettled["race_id"]
    for index, runner in enumerate(day.races[0].runners):
        runner.horse_id = seeded_race_unsettled["horse_ids"][index]
    store_predictions(db_session, day)

    assert settle_predictions(db_session, seeded_race_unsettled["race_date"]) == 0
    assert not any(row.settled for row in db_session.query(Prediction).all())


def test_settling_a_day_with_nothing_stored_is_a_no_op(db_session):
    assert settle_predictions(db_session, date(2020, 1, 1)) == 0


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def add_settled_bet(session, *, index: int, won: bool, odds: float = 5.0) -> None:
    """A settled BET, with the race and horse rows its foreign keys require."""
    from backend.models import Horse, Race

    race_id, horse_id = f"rac_{index}", f"hrs_{index}"
    race_date = date(2026, 1, 1) + timedelta(days=index)
    if session.get(Race, race_id) is None:
        session.add(Race(race_id=race_id, race_date=race_date, has_result=True))
    if session.get(Horse, horse_id) is None:
        session.add(Horse(horse_id=horse_id, name=f"Horse {index}"))
    session.flush()

    session.add(
        Prediction(
            race_id=race_id,
            horse_id=horse_id,
            race_date=race_date,
            model_probability=0.25,
            odds=odds,
            expected_value=0.25,
            recommendation=Recommendation.BET,
            model_name="lightgbm",
            model_version="v1",
            settled=True,
            won=won,
        )
    )


def test_performance_with_nothing_settled():
    from backend.prediction_service.performance import PerformanceSummary

    summary = PerformanceSummary()
    assert summary.settled == 0
    assert summary.headline == "No settled bets yet."
    assert not summary.is_significant


def test_performance_scores_settled_bets_only(db_session):
    for index in range(10):
        add_settled_bet(db_session, index=index, won=index < 3)
    from backend.models import Horse, Race

    db_session.add(Race(race_id="rac_pending", race_date=date(2026, 2, 1)))
    db_session.add(Horse(horse_id="hrs_pending", name="Pending"))
    db_session.flush()
    db_session.add(
        Prediction(
            race_id="rac_pending",
            horse_id="hrs_pending",
            race_date=date(2026, 2, 1),
            model_probability=0.3,
            odds=4.0,
            recommendation=Recommendation.BET,
            model_name="lightgbm",
            model_version="v1",
            settled=False,
        )
    )
    db_session.commit()

    summary = summarise_performance(db_session)

    assert summary.bets == 11
    assert summary.settled == 10
    assert summary.pending == 1, "an unsettled bet is not a pending win"
    assert summary.wins == 3
    # 3 wins at 5.0 on a 10 stake = 150 returned, 100 staked.
    assert summary.staked == pytest.approx(100.0)
    assert summary.returned == pytest.approx(150.0)
    assert summary.roi == pytest.approx(0.5)
    assert summary.strike_rate == pytest.approx(0.3)


def test_a_losing_record_reports_a_loss(db_session):
    for index in range(10):
        add_settled_bet(db_session, index=index, won=False)
    db_session.commit()

    summary = summarise_performance(db_session)
    assert summary.roi == pytest.approx(-1.0)
    assert summary.profit == pytest.approx(-100.0)
    assert summary.max_drawdown == pytest.approx(100.0)


def test_a_thin_record_says_it_cannot_be_trusted(db_session):
    """The number is shown; the claim is not made."""
    for index in range(20):
        add_settled_bet(db_session, index=index, won=index < 8)
    db_session.commit()

    summary = summarise_performance(db_session)
    assert summary.settled < MIN_BETS_FOR_INFERENCE
    assert not summary.is_significant
    assert "too few to distinguish from chance" in summary.headline


def test_no_bet_rows_are_excluded_from_the_record(db_session):
    from backend.models import Horse, Race

    db_session.add(Race(race_id="rac_x", race_date=date(2026, 1, 1)))
    db_session.add(Horse(horse_id="hrs_x", name="Unbacked"))
    db_session.flush()
    db_session.add(
        Prediction(
            race_id="rac_x",
            horse_id="hrs_x",
            race_date=date(2026, 1, 1),
            model_probability=0.05,
            odds=20.0,
            recommendation=Recommendation.NO_BET,
            model_name="lightgbm",
            model_version="v1",
            settled=True,
            won=True,
        )
    )
    db_session.commit()

    summary = summarise_performance(db_session)
    assert summary.bets == 0, "a runner we did not back must not count as a win"


def test_performance_can_be_filtered_by_date(db_session):
    for index in range(10):
        add_settled_bet(db_session, index=index, won=True)
    db_session.commit()

    windowed = summarise_performance(db_session, date_from=date(2026, 1, 6))
    assert windowed.settled == 5


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def test_an_empty_replay_is_not_consistent():
    """Nothing compared is not the same as nothing wrong."""
    assert not ReplayResult().consistent


def test_a_clean_replay_is_consistent():
    result = ReplayResult(races_checked=2, runners_checked=16, max_absolute_difference=0.0)
    assert result.consistent
    assert "CONSISTENT" in result.render()


def test_a_mismatch_breaks_consistency():
    result = ReplayResult(
        races_checked=1,
        runners_checked=8,
        max_absolute_difference=0.01,
        mismatches=[{"race_id": "r1", "horse_id": "h1", "live": 0.31, "expected": 0.30}],
    )
    assert not result.consistent
    assert "DIVERGED" in result.render()


def test_a_runner_missing_from_the_live_path_breaks_consistency():
    result = ReplayResult(races_checked=1, runners_checked=7, missing_from_live=["r1/h8 was scored offline"])
    assert not result.consistent


def test_identical_frames_compare_to_zero():
    frame_a = pd.DataFrame(
        {"race_id": ["r1", "r1"], "horse_id": ["h1", "h2"], "model_probability": [0.3, 0.7]}
    )
    assert compare_frames(frame_a, frame_a.copy()) == 0.0


def test_differing_frames_report_the_gap():
    frame_a = pd.DataFrame({"race_id": ["r1"], "horse_id": ["h1"], "model_probability": [0.30]})
    frame_b = pd.DataFrame({"race_id": ["r1"], "horse_id": ["h1"], "model_probability": [0.35]})
    assert compare_frames(frame_a, frame_b) == pytest.approx(0.05)


def test_frames_that_share_no_rows_are_infinitely_apart():
    frame_a = pd.DataFrame({"race_id": ["r1"], "horse_id": ["h1"], "model_probability": [0.3]})
    frame_b = pd.DataFrame({"race_id": ["r2"], "horse_id": ["h9"], "model_probability": [0.3]})
    assert compare_frames(frame_a, frame_b) == float("inf")
    assert compare_frames(pd.DataFrame(), frame_b) == float("inf")


# ---------------------------------------------------------------------------
# The daily job
# ---------------------------------------------------------------------------
class StubService:
    def __init__(self, day: DayPredictions):
        self.day = day

    def signals_for_date(self, race_date=None) -> DayPredictions:
        return self.day


def test_the_daily_job_scores_and_reports(db_session, seeded_race):
    day = make_day(race_date=seeded_race["race_date"])
    day.races[0].race_id = seeded_race["race_id"]
    for index, runner in enumerate(day.races[0].runners):
        runner.horse_id = seeded_race["horse_ids"][index]

    result, returned = run_daily_job(
        db_session,
        race_date=seeded_race["race_date"],
        service=StubService(day),
        settle_previous=False,
    )

    assert result.succeeded
    assert result.races_scored == 1
    assert result.runners_scored == 4
    assert result.bets_recommended == 2
    assert returned is day
    assert db_session.query(Prediction).count() == 4


def test_a_failed_racecard_import_does_not_lose_the_rest(db_session, seeded_race):
    """A dead API should not stop us scoring data already in the database."""

    def boom() -> int:
        raise RuntimeError("subscription inactive")

    day = make_day(race_date=seeded_race["race_date"])
    day.races[0].race_id = seeded_race["race_id"]
    for index, runner in enumerate(day.races[0].runners):
        runner.horse_id = seeded_race["horse_ids"][index]

    result, _ = run_daily_job(
        db_session,
        race_date=seeded_race["race_date"],
        service=StubService(day),
        import_racecards=boom,
        settle_previous=False,
    )

    assert not result.succeeded
    assert any("subscription inactive" in error for error in result.errors)
    assert result.races_scored == 1, "scoring must still have happened"
    assert db_session.query(Prediction).count() == 4


def test_the_job_imports_when_a_fetcher_is_supplied(db_session):
    result, _ = run_daily_job(
        db_session,
        race_date=date(2026, 6, 15),
        service=StubService(DayPredictions(race_date=date(2026, 6, 15), note="none")),
        import_racecards=lambda: 42,
        settle_previous=False,
    )
    assert result.races_imported == 42
    assert result.note == "none"


def test_the_job_renders_a_summary():
    rendered = DailyJobResult(race_date=date(2026, 6, 15), races_scored=3).render()
    assert "Daily predictions — 2026-06-15" in rendered
    assert "Races scored" in rendered


# ---------------------------------------------------------------------------
# The API — what a dashboard actually receives
# ---------------------------------------------------------------------------
def store_a_day(session, seeded) -> DayPredictions:
    day = make_day(race_date=seeded["race_date"])
    day.races[0].race_id = seeded["race_id"]
    day.races[0].course = "Testfield Park"
    for index, runner in enumerate(day.races[0].runners):
        runner.horse_id = seeded["horse_ids"][index]
    store_predictions(session, day)
    return day


def test_todays_predictions_are_served(api_client, db_session, seeded_race):
    store_a_day(db_session, seeded_race)

    response = api_client.get(
        "/api/v1/predictions/today", params={"race_date": str(seeded_race["race_date"])}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["races"] == 1
    assert payload["runners"] == 4
    assert payload["bets"] == 2
    race = payload["predictions"][0]
    assert race["race"].startswith("Testfield Park")
    assert {"horse", "probability", "odds", "expected_value", "recommendation"} <= set(race["horses"][0])


def test_bets_only_filters_the_card(api_client, db_session, seeded_race):
    store_a_day(db_session, seeded_race)

    payload = api_client.get(
        "/api/v1/predictions/today",
        params={"race_date": str(seeded_race["race_date"]), "bets_only": True},
    ).json()

    assert payload["runners"] == 2
    assert all(
        horse["recommendation"] == "BET" for race in payload["predictions"] for horse in race["horses"]
    )


def test_a_day_with_nothing_scored_is_not_an_error(api_client):
    """'No races today' is a normal answer, not a 404."""
    response = api_client.get("/api/v1/predictions/today", params={"race_date": "2019-01-01"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["runners"] == 0
    assert "no stored predictions" in payload["note"]


def test_one_race_can_be_fetched(api_client, db_session, seeded_race):
    store_a_day(db_session, seeded_race)

    payload = api_client.get(f"/api/v1/predictions/{seeded_race['race_id']}").json()

    assert payload["race_id"] == seeded_race["race_id"]
    assert len(payload["horses"]) == 4
    assert payload["model"] == "lightgbm:v3"


def test_an_unknown_race_is_a_404(api_client):
    assert api_client.get("/api/v1/predictions/rac_nope").status_code == 404


def test_model_status_reports_when_nothing_is_promoted(api_client):
    payload = api_client.get("/api/v1/model/status").json()

    assert "ready" in payload
    assert "rules" in payload
    assert payload["rules"]["min_expected_value"] == MIN_EXPECTED_VALUE


def test_performance_is_served(api_client, db_session):
    for index in range(6):
        add_settled_bet(db_session, index=index, won=index < 2)
    db_session.commit()

    payload = api_client.get("/api/v1/performance").json()

    assert payload["settled"] == 6
    assert payload["wins"] == 2
    assert "headline" in payload
    assert isinstance(payload["equity"], list)


def test_the_dashboard_is_served(api_client):
    response = api_client.get("/dashboard/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    for expected in ("Today's Predictions", "Performance", "Model Information"):
        assert expected in body
    for column in ("Race", "Horse", "Probability", "Odds", "Expected Value", "Recommendation"):
        assert column in body


# ---------------------------------------------------------------------------
# Split construction — the guard that caught a real bug in this module
# ---------------------------------------------------------------------------
def test_a_split_carves_validation_from_the_end_of_training():
    from backend.prediction_service.training import build_split

    split = build_split(
        train_start=date(2018, 1, 1),
        train_end=date(2023, 12, 31),
        test_start=date(2024, 1, 1),
        test_end=date(2025, 6, 30),
        valid_months=12,
    )

    assert split.train_start == date(2018, 1, 1)
    assert split.train_end < split.valid_start < split.valid_end < split.test_start
    assert split.valid_end == date(2023, 12, 31)


def test_a_training_window_too_short_for_validation_is_refused():
    from backend.prediction_service.training import build_split
    from backend.utils.exceptions import ModelError

    with pytest.raises(ModelError, match="too short"):
        build_split(
            train_start=date(2023, 1, 1),
            train_end=date(2023, 6, 30),
            test_start=date(2024, 1, 1),
            test_end=date(2024, 6, 30),
            valid_months=12,
        )


def test_validation_may_not_overlap_the_test_window():
    """Calibrating on races the model is then scored against is leakage."""
    from backend.prediction_service.training import build_split
    from backend.utils.exceptions import ModelError

    with pytest.raises(ModelError, match="must not overlap"):
        build_split(
            train_start=date(2018, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 1, 1),
            test_end=date(2025, 1, 1),
            valid_months=12,
        )


def test_a_shorter_validation_window_can_be_requested():
    from backend.prediction_service.training import build_split

    split = build_split(
        train_start=date(2022, 1, 1),
        train_end=date(2023, 12, 31),
        test_start=date(2024, 1, 1),
        test_end=date(2024, 12, 31),
        valid_months=6,
    )
    assert (split.valid_end - split.valid_start).days < 200


# ---------------------------------------------------------------------------
# The production manifest
# ---------------------------------------------------------------------------
def make_production_model(**overrides):
    from backend.prediction_service.training import ProductionModel

    defaults = {
        "name": "lightgbm",
        "version": "v2",
        "trained_at": "2026-08-10T09:00:00+00:00",
        "feature_version": "v1",
        "dataset_hash": "abc123",
        "train_start": "2018-01-01",
        "train_end": "2022-12-31",
        "test_start": "2024-01-01",
        "test_end": "2025-06-30",
        "rows": 29_200,
        "metrics": {"log_loss": 0.31},
        "comparison": [
            {"model": "lightgbm", "race_log_loss": 1.371},
            {"model": "xgboost", "race_log_loss": 1.374},
        ],
    }
    defaults.update(overrides)
    return ProductionModel(**defaults)


def test_the_manifest_round_trips(tmp_path):
    from backend.prediction_service.training import _write_manifest, load_manifest

    model = make_production_model()
    path = _write_manifest(model, tmp_path)

    assert path.exists()
    payload = load_manifest(tmp_path)
    assert payload is not None
    assert payload["model"] == "lightgbm"
    assert payload["dataset_hash"] == "abc123"
    assert payload["rows"] == 29_200


def test_a_missing_manifest_reads_as_none(tmp_path):
    from backend.prediction_service.training import load_manifest

    assert load_manifest(tmp_path) is None


def test_a_corrupt_manifest_reads_as_none(tmp_path):
    from backend.prediction_service.training import (
        MANIFEST_FILENAME,
        PRODUCTION_DIRNAME,
        load_manifest,
    )

    target = tmp_path / PRODUCTION_DIRNAME
    target.mkdir()
    (target / MANIFEST_FILENAME).write_text("{ not json", encoding="utf-8")

    assert load_manifest(tmp_path) is None


def test_the_model_summary_marks_the_promoted_candidate():
    rendered = make_production_model().render()

    assert "lightgbm:v2" in rendered
    assert "* lightgbm" in rendered
    assert "29,200" in rendered
    assert "* promoted" in rendered


def test_the_model_serialises_for_the_api():
    payload = make_production_model().as_dict()

    assert payload["training_window"] == "2018-01-01 → 2022-12-31"
    assert payload["feature_version"] == "v1"
    assert len(payload["comparison"]) == 2


# ---------------------------------------------------------------------------
# Scoring — the path from a feature frame to a betting signal
# ---------------------------------------------------------------------------
class StubModel:
    """A model that ranks by a column already in the frame."""

    name = "stub"

    def predict_proba(self, frame):
        import numpy as np

        return np.asarray(frame["raw_score"], dtype=float)


def scoring_frame() -> pd.DataFrame:
    """Two races, priced, with a column the stub model reads."""
    rows = []
    for race in ("rac_t01", "rac_t02"):
        for index in range(6):
            rows.append(
                {
                    "race_id": race,
                    "horse_id": f"hrs_t{index:02d}",
                    "race_date": date(2026, 6, 15),
                    "off_time": pd.Timestamp("2026-06-15 15:30", tz="UTC"),
                    "course_id": "crs_t01",
                    "field_size": 6,
                    "raw_score": 0.4 if index == 0 else 0.12,
                    "mkt_mean_odds": 3.0 if index == 0 else 9.0,
                    "mkt_latest_odds": 3.2 if index == 0 else 9.5,
                }
            )
    return pd.DataFrame(rows)


def test_scoring_produces_races_runners_and_recommendations(db_session, seeded_race):
    from backend.prediction_service.service import PredictionService

    service = PredictionService(db_session, model=StubModel())
    races = service._score(scoring_frame())

    assert len(races) == 2
    for race in races:
        assert len(race.runners) == 6
        assert race.off_time == "15:30"
        assert race.course == "Testfield Park", "the course name must be looked up"
        assert race.race.startswith("Testfield Park")
        # Probabilities are normalised within the race.
        assert sum(runner.model_probability for runner in race.runners) == pytest.approx(1.0)
        # Sorted strongest first.
        probabilities = [runner.model_probability for runner in race.runners]
        assert probabilities == sorted(probabilities, reverse=True)


def test_scoring_attaches_prices_and_expected_value(db_session, seeded_race):
    from backend.prediction_service.service import PredictionService

    service = PredictionService(db_session, model=StubModel())
    race = service._score(scoring_frame())[0]
    favourite = race.runners[0]

    assert favourite.odds == pytest.approx(3.2), "the BEST price is what a stake is struck at"
    assert favourite.market_probability is not None
    assert favourite.expected_value == pytest.approx(favourite.model_probability * 3.2 - 1.0, abs=1e-9)
    assert favourite.edge == pytest.approx(
        favourite.model_probability - favourite.market_probability, abs=1e-9
    )


def test_scoring_names_the_horses(db_session, seeded_race):
    from backend.prediction_service.service import PredictionService

    service = PredictionService(db_session, model=StubModel())
    race = service._score(scoring_frame())[0]

    named = [runner for runner in race.runners if runner.horse]
    assert named, "horse names must be resolved, not left as ids"
    assert all(runner.horse.startswith("Test Horse") for runner in named)


def test_a_day_with_no_races_reports_why(db_session):
    from backend.prediction_service.service import PredictionService

    service = PredictionService(db_session, model=StubModel())
    day = service.signals_for_date(date(2019, 1, 1))

    assert day.races == []
    assert "no races found" in day.note
    assert day.total_runners == 0
    assert day.bets == []


def test_scoring_an_unknown_race_is_a_404(db_session):
    from backend.prediction_service.service import PredictionService
    from backend.utils.exceptions import NotFoundError

    service = PredictionService(db_session, model=StubModel())
    with pytest.raises(NotFoundError):
        service.signals_for_race("rac_does_not_exist")


def test_the_service_reports_its_model(db_session):
    from backend.prediction_service.service import PredictionService

    service = PredictionService(db_session, model=StubModel())
    assert service.model_name == "stub"
    assert service.model_version  # 'unregistered' when nothing is promoted


def test_a_runner_signal_serialises():
    runner = RunnerSignal(
        horse_id="h1",
        horse="Thunder King",
        model_probability=0.31,
        odds=6.0,
        expected_value=0.86,
        recommendation=Recommendation.BET,
    )
    payload = runner.as_dict()

    assert payload["horse"] == "Thunder King"
    assert payload["probability"] == 0.31
    assert payload["expected_value"] == 0.86
    assert payload["recommendation"] == "BET"


def test_a_runner_without_a_name_falls_back_to_its_id():
    assert RunnerSignal(horse_id="h9", horse=None, model_probability=0.1).as_dict()["horse"] == "h9"


# ---------------------------------------------------------------------------
# Replay against a real service
# ---------------------------------------------------------------------------
class StubPipeline:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def build(self, *, date_from=None, date_to=None) -> pd.DataFrame:
        return self.frame

    def build_for_race(self, race_id: str) -> pd.DataFrame:
        return self.frame[self.frame["race_id"] == race_id]


def stub_service(db_session, frame: pd.DataFrame):
    from backend.prediction_service.service import PredictionService

    service = PredictionService(db_session, model=StubModel())
    service.pipeline = StubPipeline(frame)
    service.predictor.pipeline = service.pipeline
    return service


def test_replay_confirms_the_live_path_matches_the_model(db_session, seeded_race):
    """The check that decides whether the backtest is evidence about the product."""
    from backend.prediction_service.replay import replay_predictions

    service = stub_service(db_session, scoring_frame())
    result = replay_predictions(db_session, date(2026, 6, 15), service=service)

    assert result.consistent
    assert result.races_checked == 2
    assert result.runners_checked == 12
    assert result.max_absolute_difference == 0.0


def test_replay_on_a_day_with_no_races_checks_nothing(db_session):
    from backend.prediction_service.replay import replay_predictions

    service = stub_service(db_session, pd.DataFrame())
    result = replay_predictions(db_session, date(2019, 1, 1), service=service)

    assert result.runners_checked == 0
    assert not result.consistent


def test_scoring_one_race_returns_just_that_race(db_session, seeded_race):
    service = stub_service(db_session, scoring_frame())
    race = service.signals_for_race("rac_t02")

    assert race.race_id == "rac_t02"
    assert len(race.runners) == 6


def test_a_full_day_is_scored_and_stored(db_session, seeded_race):
    """The whole product path, minus the model: score, decide, persist."""
    from backend.prediction_service.service import store_predictions

    service = stub_service(db_session, scoring_frame())
    day = service.signals_for_date(date(2026, 6, 15))

    assert len(day.races) == 2
    assert day.total_runners == 12
    assert day.model_name == "stub"
    assert day.as_dict()["races"] == 2

    # rac_t02 has no seeded Race row, so only rac_t01 can be persisted.
    day.races = [race for race in day.races if race.race_id == seeded_race["race_id"]]
    assert store_predictions(db_session, day) == 6


def test_the_daily_job_settles_yesterday(db_session, seeded_race):
    """Settlement is what turns stored intentions into a scoreable record."""
    day = make_day(race_date=seeded_race["race_date"])
    day.races[0].race_id = seeded_race["race_id"]
    for index, runner in enumerate(day.races[0].runners):
        runner.horse_id = seeded_race["horse_ids"][index]
    store_predictions(db_session, day)

    # Running the job for the following day settles the previous one.
    result, _ = run_daily_job(
        db_session,
        race_date=seeded_race["race_date"] + timedelta(days=1),
        service=StubService(DayPredictions(race_date=seeded_race["race_date"] + timedelta(days=1))),
        settle_previous=True,
    )

    assert result.settled == 4
    assert all(row.settled for row in db_session.query(Prediction).all())
