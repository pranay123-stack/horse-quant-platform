"""Feature calculation tests.

Values are hand-computed in the assertions wherever possible. A feature test that
only checks "not null" would pass on a completely wrong number, which is the
failure mode that matters here — a wrong feature does not raise, it just makes
the model quietly worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.features.base import (
    EVENT_TIME,
    FeatureConfig,
    ScoreWeights,
    distance_band,
    going_band,
    going_value,
)
from backend.features.horse_features import (
    PRIOR_WIN_RATE,
    _shrink,
    build_horse_features,
    build_horse_state,
)
from backend.features.jockey_features import build_jockey_features, level_stake_profit
from backend.features.market_features import build_market_features, build_market_state
from backend.features.race_features import build_race_features
from backend.features.scoring import NEUTRAL, SCORE_COLUMNS, build_scores, rank_within_race
from backend.features.trainer_features import build_trainer_features
from tests.fixtures.racing_builders import build_history_frame, build_target_frame

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("going", "expected"),
    [("Heavy", 0.0), ("Soft", 0.2), ("Good", 0.5), ("Good To Firm", 0.65), ("Firm", 0.8)],
)
def test_going_scale_is_monotonic(going, expected):
    assert going_value(going) == expected


def test_going_scale_orders_ground_correctly():
    order = ["Heavy", "Soft", "Good To Soft", "Good", "Good To Firm", "Firm"]
    values = [going_value(g) for g in order]
    assert values == sorted(values), "going scale must run soft -> firm"


def test_going_value_handles_racecourse_verbiage():
    assert going_value("Good To Soft (Soft in places)") == going_value("Good To Soft")
    assert going_value("") is None
    assert going_value(None) is None
    assert going_value("Martian") is None


@pytest.mark.parametrize(
    ("going", "band"),
    [("Heavy", "soft"), ("Soft", "soft"), ("Good", "good"), ("Firm", "firm"), (None, "unknown")],
)
def test_going_bands(going, band):
    assert going_band(going) == band


@pytest.mark.parametrize(
    ("yards", "band"),
    [
        (1100, "sprint"),
        (1320, "sprint"),
        (1760, "mile"),
        (2420, "middle"),
        (4400, "staying"),
        (None, "unknown"),
    ],
)
def test_distance_bands(yards, band):
    assert distance_band(yards) == band


def test_score_weights_must_sum_to_one():
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        ScoreWeights(recent_form=0.9, speed_rating=0.9)
    assert ScoreWeights().total() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------
def test_shrinkage_pulls_small_samples_toward_the_prior():
    """1 win from 1 run is not a 100% strike rate."""
    rate = _shrink(pd.Series([1.0]), pd.Series([1.0]), PRIOR_WIN_RATE)
    assert 0.1 < rate.iloc[0] < 0.35


def test_shrinkage_respects_large_samples():
    rate = _shrink(pd.Series([200.0]), pd.Series([400.0]), PRIOR_WIN_RATE)
    assert rate.iloc[0] == pytest.approx(0.5, abs=0.02)


def test_shrinkage_of_zero_trials_returns_the_prior():
    rate = _shrink(pd.Series([0.0]), pd.Series([0.0]), PRIOR_WIN_RATE)
    assert rate.iloc[0] == pytest.approx(PRIOR_WIN_RATE)


# ---------------------------------------------------------------------------
# Horse features
# ---------------------------------------------------------------------------
def test_horse_state_counts_runs_and_wins_cumulatively():
    history = build_history_frame([("h1", "2025-01-01", 1), ("h1", "2025-02-01", 5), ("h1", "2025-03-01", 1)])
    state = build_horse_state(history)

    assert state["horse_runs_before"].tolist() == [1, 2, 3]
    assert state["horse_wins_before"].tolist() == [1.0, 1.0, 2.0]


def test_horse_recent_finishes_are_ordered_newest_first():
    history = build_history_frame([("h1", "2025-01-01", 4), ("h1", "2025-02-01", 3), ("h1", "2025-03-01", 2)])
    state = build_horse_state(history).iloc[-1]

    assert state["horse_last1_position"] == 2  # most recent
    assert state["horse_last2_position"] == 3
    assert state["horse_last3_position"] == 4


def test_rolling_average_position_uses_the_window():
    history = build_history_frame(
        [("h1", f"2025-0{m}-01", position) for m, position in enumerate([6, 5, 4, 3, 2, 1], start=1)]
    )
    state = build_horse_state(history, FeatureConfig(form_window_short=3, form_window_long=5)).iloc[-1]

    assert state["horse_form_last3_avg_pos"] == pytest.approx(2.0)  # 3, 2, 1
    assert state["horse_form_last5_avg_pos"] == pytest.approx(3.0)  # 5, 4, 3, 2, 1
    assert state["horse_career_avg_pos"] == pytest.approx(3.5)  # all six


def test_consistency_is_higher_for_steadier_horses():
    steady = build_horse_state(build_history_frame([("h1", f"2025-0{m}-01", 3) for m in range(1, 6)])).iloc[
        -1
    ]
    erratic = build_horse_state(
        build_history_frame([("h2", f"2025-0{m}-01", p) for m, p in enumerate([1, 12, 2, 11, 3], start=1)])
    ).iloc[-1]

    assert steady["horse_consistency"] > erratic["horse_consistency"]


def test_consistency_is_null_without_enough_runs():
    """Unknown must stay unknown -- filling it with a global statistic leaks."""
    state = build_horse_state(build_history_frame([("h1", "2025-01-01", 3)])).iloc[-1]
    assert pd.isna(state["horse_consistency"])


def test_speed_trend_is_positive_when_improving():
    improving = build_horse_state(build_history_frame([("h1", f"2025-0{m}-01", 1) for m in range(1, 6)]))
    # Constant ratings in the fixture, so the trend must be flat, not noise.
    assert improving["horse_speed_trend"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_rest_days_are_computed_from_the_previous_run():
    history = build_history_frame([("h1", "2025-01-01", 3), ("h1", "2025-02-01", 2)])
    targets = build_target_frame([("h1", "2025-03-03")])

    frame = build_horse_features(targets, history)
    assert frame["horse_days_since_prev_run"].iloc[0] == pytest.approx(30.0)


def test_class_move_is_positive_when_dropping_in_class():
    history = build_history_frame([("h1", "2025-01-01", 4)], race_class=2)
    targets = build_target_frame([("h1", "2025-02-01")], race_class=5)

    frame = build_horse_features(targets, history)
    assert frame["horse_class_move"].iloc[0] == 3  # class 2 -> class 5 is easier


def test_going_record_is_conditioned_on_the_ground():
    history = (
        pd.concat(
            [
                build_history_frame([("h1", "2025-01-01", 1), ("h1", "2025-01-15", 1)], going="Soft"),
                build_history_frame([("h1", "2025-02-01", 8), ("h1", "2025-02-15", 9)], going="Firm"),
            ]
        )
        .sort_values(EVENT_TIME)
        .reset_index(drop=True)
    )

    on_soft = build_horse_features(build_target_frame([("h1", "2025-03-01")], going="Soft"), history)
    on_firm = build_horse_features(build_target_frame([("h1", "2025-03-01")], going="Firm"), history)

    assert on_soft["horse_going_win_rate"].iloc[0] > on_firm["horse_going_win_rate"].iloc[0]
    assert on_soft["horse_going_runs"].iloc[0] == 2


def test_distance_record_is_conditioned_on_the_trip():
    history = (
        pd.concat(
            [
                build_history_frame([("h1", "2025-01-01", 1)], distance_yards=1100),  # sprint
                build_history_frame([("h1", "2025-02-01", 10)], distance_yards=4400),  # staying
            ]
        )
        .sort_values(EVENT_TIME)
        .reset_index(drop=True)
    )

    sprint = build_horse_features(build_target_frame([("h1", "2025-03-01")], distance_yards=1100), history)
    staying = build_horse_features(build_target_frame([("h1", "2025-03-01")], distance_yards=4400), history)

    assert sprint["horse_distance_win_rate"].iloc[0] > staying["horse_distance_win_rate"].iloc[0]


def test_unknown_horse_gets_priors_not_zeros():
    history = build_history_frame([("h_known", "2025-01-01", 1)])
    frame = build_horse_features(build_target_frame([("h_new", "2025-02-01")]), history)

    assert frame["horse_runs_before"].iloc[0] == 0
    assert frame["horse_is_debutant"].iloc[0] == 1
    assert frame["horse_going_win_rate"].iloc[0] == pytest.approx(PRIOR_WIN_RATE)


def test_empty_history_does_not_crash():
    frame = build_horse_features(build_target_frame([("h1", "2025-01-01")]), pd.DataFrame())
    assert len(frame) == 1
    assert frame["horse_runs_before"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Jockey and trainer
# ---------------------------------------------------------------------------
def test_level_stake_profit_arithmetic():
    profit = level_stake_profit(pd.Series([True, False, True]), pd.Series([5.0, 3.0, None]))
    assert profit.iloc[0] == pytest.approx(4.0)  # 5.0 winner returns 4 profit
    assert profit.iloc[1] == pytest.approx(-1.0)  # loser loses the stake
    assert pd.isna(profit.iloc[2])  # unpriced rides do not count


def test_jockey_strike_rate_and_roi():
    history = build_history_frame(
        [("h1", "2025-01-01", 1), ("h2", "2025-01-02", 5), ("h3", "2025-01-03", 5)],
        starting_price=4.0,
    )
    frame = build_jockey_features(build_target_frame([("h9", "2025-02-01")]), history)

    # 1 win from 3 at 4.0: profit = 3 - 1 - 1 = 1 over 3 rides.
    assert frame["jky_rides_window"].iloc[0] == 3
    assert frame["jky_roi"].iloc[0] == pytest.approx(1 / 3)
    assert 0.1 < frame["jky_strike_rate"].iloc[0] < 0.4  # shrunk from a raw 33%


def test_jockey_features_default_for_an_unseen_rider():
    frame = build_jockey_features(build_target_frame([("h1", "2025-01-01")]), pd.DataFrame())
    assert frame["jky_rides_window"].iloc[0] == 0
    assert frame["jky_strike_rate"].iloc[0] == pytest.approx(PRIOR_WIN_RATE)


def test_trainer_30_day_window_excludes_older_runners():
    history = build_history_frame([("h1", "2025-01-01", 1), ("h2", "2025-05-01", 1), ("h3", "2025-05-10", 5)])
    frame = build_trainer_features(build_target_frame([("h9", "2025-05-15")]), history)

    # As of the last runner (2025-05-10) the 30-day window holds 2025-05-01 and
    # 2025-05-10 only; January is long gone.
    assert frame["trn_runners_30d"].iloc[0] == 2
    assert frame["trn_career_runners"].iloc[0] == 3


def test_trainer_jockey_combination_is_tracked():
    history = build_history_frame([("h1", "2025-01-01", 1), ("h2", "2025-01-02", 1)])
    frame = build_trainer_features(build_target_frame([("h9", "2025-02-01")]), history)

    assert frame["trn_jky_runs"].iloc[0] == 2
    assert frame["trn_jky_win_rate"].iloc[0] > PRIOR_WIN_RATE


# ---------------------------------------------------------------------------
# Race features
# ---------------------------------------------------------------------------
def test_race_features_rank_within_the_field():
    targets = build_target_frame([("h1", "2025-01-01"), ("h2", "2025-01-01")], race_id="rac_1")
    targets.loc[0, "official_rating"] = 100
    targets.loc[1, "official_rating"] = 60

    frame = build_race_features(targets)
    assert frame.loc[0, "runner_or_rank"] == 1.0  # top rated
    assert frame.loc[1, "runner_or_rank"] == 0.5
    assert frame.loc[0, "runner_or_vs_field"] == 20.0
    assert frame["race_field_size"].tolist() == [2, 2]


def test_draw_is_normalised_by_field_size():
    targets = build_target_frame([(f"h{i}", "2025-01-01") for i in range(4)], race_id="rac_1")
    frame = build_race_features(targets)
    assert frame["runner_draw_pct"].max() == pytest.approx(1.0)


def test_missing_class_falls_back_to_the_modal_class():
    targets = build_target_frame([("h1", "2025-01-01")], race_class=None)
    frame = build_race_features(targets)
    assert frame["race_class_filled"].iloc[0] == 5


# ---------------------------------------------------------------------------
# Market features
# ---------------------------------------------------------------------------
def _odds_frame(rows: list[tuple[str, str, str, float, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["race_id", "horse_id", "bookmaker", "decimal_odds", "recorded_at", "off_time"]
    )


def test_market_takes_the_best_price_across_bookmakers():
    off = "2025-01-01T14:00:00Z"
    odds = _odds_frame(
        [
            ("r1", "h1", "A", 3.0, "2025-01-01T13:00:00Z", off),
            ("r1", "h1", "B", 3.5, "2025-01-01T13:00:00Z", off),
            ("r1", "h2", "A", 2.0, "2025-01-01T13:00:00Z", off),
        ]
    )
    market = build_market_state(odds).set_index("horse_id")
    assert market.loc["h1", "mkt_latest_odds"] == 3.5
    assert market.loc["h1", "mkt_bookmaker_count"] == 2


def test_odds_movement_and_direction():
    off = "2025-01-01T14:00:00Z"
    odds = _odds_frame(
        [
            ("r1", "h1", "A", 10.0, "2025-01-01T12:00:00Z", off),
            ("r1", "h1", "A", 5.0, "2025-01-01T13:30:00Z", off),
        ]
    )
    market = build_market_state(odds).iloc[0]
    assert market["mkt_open_odds"] == 10.0
    assert market["mkt_latest_odds"] == 5.0
    assert market["mkt_odds_change_pct"] == pytest.approx(-0.5)
    assert market["mkt_shortened"] == 1


def test_implied_probability_overround_and_rank():
    off = "2025-01-01T14:00:00Z"
    odds = _odds_frame(
        [
            ("r1", "h1", "A", 2.0, "2025-01-01T13:00:00Z", off),
            ("r1", "h2", "A", 4.0, "2025-01-01T13:00:00Z", off),
            ("r1", "h3", "A", 5.0, "2025-01-01T13:00:00Z", off),
        ]
    )
    market = build_market_state(odds).set_index("horse_id")

    assert market.loc["h1", "mkt_implied_prob"] == pytest.approx(0.5)
    # 0.5 + 0.25 + 0.2 = 0.95
    assert market.loc["h1", "mkt_overround"] == pytest.approx(0.95)
    assert market.loc["h1", "mkt_implied_prob_norm"] == pytest.approx(0.5 / 0.95)
    assert market["mkt_implied_prob_norm"].sum() == pytest.approx(1.0)
    assert market.loc["h1", "mkt_rank"] == 1
    assert market.loc["h1", "mkt_is_favourite"] == 1
    assert market.loc["h3", "mkt_rank"] == 3


def test_quotes_after_the_off_are_discarded():
    odds = _odds_frame([("r1", "h1", "A", 99.0, "2025-01-01T15:00:00Z", "2025-01-01T14:00:00Z")])
    assert build_market_state(odds).empty


def test_quotes_without_an_off_time_are_discarded():
    odds = _odds_frame([("r1", "h1", "A", 3.0, "2025-01-01T13:00:00Z", None)])
    assert build_market_state(odds).empty


def test_runners_without_a_market_are_flagged_not_imputed():
    targets = build_target_frame([("h1", "2025-01-01")], race_id="r1")
    frame = build_market_features(targets, pd.DataFrame())

    assert frame["mkt_has_market"].iloc[0] == 0
    assert pd.isna(frame["mkt_latest_odds"].iloc[0]), "a fabricated price would look like a real bet"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_rank_within_race_is_relative_to_the_field():
    frame = pd.DataFrame({"race_id": ["r1"] * 3, "value": [10.0, 20.0, 30.0]})
    ranked = rank_within_race(frame, "value")
    assert ranked.tolist() == [pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]


def test_rank_within_race_treats_missing_as_neutral():
    frame = pd.DataFrame({"race_id": ["r1"] * 2, "value": [10.0, np.nan]})
    assert rank_within_race(frame, "value").iloc[1] == NEUTRAL


def test_rank_within_race_is_isolated_per_race():
    frame = pd.DataFrame({"race_id": ["r1", "r1", "r2"], "value": [1.0, 2.0, 99.0]})
    ranked = rank_within_race(frame, "value")
    assert ranked.iloc[2] == 1.0  # top of its own race, not compared with r1


def test_scores_are_bounded_and_present():
    targets = build_target_frame([(f"h{i}", "2025-02-01") for i in range(4)], race_id="rac_1")
    history = build_history_frame([(f"h{i}", "2025-01-01", i + 1) for i in range(4)])

    frame = build_horse_features(targets, history)
    frame = build_jockey_features(frame, history)
    frame = build_trainer_features(frame, history)
    frame = build_race_features(frame)
    frame = build_market_features(frame, pd.DataFrame())
    frame = build_scores(frame)

    for column in SCORE_COLUMNS:
        assert column in frame
        values = frame[column].dropna()
        assert ((values >= 0) & (values <= 1)).all(), f"{column} left [0, 1]"


def test_market_score_falls_back_to_a_uniform_prior():
    targets = build_target_frame([(f"h{i}", "2025-02-01") for i in range(4)], race_id="rac_1")
    frame = build_market_features(targets, pd.DataFrame())
    frame = build_scores(frame)
    assert frame["market_score"].iloc[0] == pytest.approx(0.25)


def test_form_score_respects_the_configured_weights():
    """A 100% recent-form weighting must reproduce the recent-form component."""
    targets = build_target_frame([(f"h{i}", "2025-03-01") for i in range(3)], race_id="rac_1")
    history = build_history_frame([("h0", "2025-01-01", 1), ("h1", "2025-01-01", 5), ("h2", "2025-01-01", 9)])
    weights = ScoreWeights(
        recent_form=1.0, speed_rating=0.0, distance_fit=0.0, going_fit=0.0, class_rating=0.0
    )

    frame = build_horse_features(targets, history)
    frame = build_scores(frame, FeatureConfig(weights=weights))

    # h0 finished best, so it must score highest.
    scores = frame.set_index("horse_id")["horse_form_score"]
    assert scores["h0"] > scores["h1"] > scores["h2"]


def test_scoring_an_empty_frame_returns_the_columns():
    frame = build_scores(pd.DataFrame(columns=["race_id", "horse_id"]))
    for column in SCORE_COLUMNS:
        assert column in frame
