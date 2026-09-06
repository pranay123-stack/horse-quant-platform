"""Backtesting engine, staking and metrics.

The engine is tested against situations whose answer is known in advance:
a strategy that bets the market must lose the overround; a strategy given a
guaranteed winner must win exactly the price. If those two do not hold, no
result the framework produces can be trusted.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from backend.backtesting import (
    BacktestConfig,
    BacktestEngine,
    FormScorePredictor,
    MarketPredictor,
    UniformPredictor,
    compare_predictors,
    compute_metrics,
    expected_value,
    kelly_fraction,
)
from backend.backtesting.engine import PRE_RACE_ODDS_COLUMNS
from backend.backtesting.predictors import BlendPredictor
from backend.utils.exceptions import StrategyError

pytestmark = pytest.mark.unit


def make_frame(races: list[dict]) -> pd.DataFrame:
    """Build a backtest frame from ``[{odds: [...], winner: idx}, ...]``."""
    rows = []
    for race_index, spec in enumerate(races):
        odds = spec["odds"]
        implied = [1 / o for o in odds]
        overround = sum(implied)
        for runner_index, price in enumerate(odds):
            rows.append(
                {
                    "race_id": f"rac_{race_index:04d}",
                    "horse_id": f"hrs_{race_index:04d}_{runner_index}",
                    "race_date": spec.get("date", date(2025, 1, 1) + timedelta(days=race_index)),
                    "mkt_latest_odds": price,
                    "mkt_open_odds": price,
                    "mkt_implied_prob": implied[runner_index],
                    "mkt_implied_prob_norm": implied[runner_index] / overround,
                    "mkt_overround": overround,
                    "horse_form_score": spec.get("scores", [0.5] * len(odds))[runner_index],
                    "won": int(runner_index == spec["winner"]),
                    "starting_price": price,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pure arithmetic
# ---------------------------------------------------------------------------
def test_expected_value_formula():
    assert expected_value(0.30, 5.0) == pytest.approx(0.5)  # the spec's example
    assert expected_value(0.20, 5.0) == pytest.approx(0.0)  # fair price
    assert expected_value(0.10, 5.0) == pytest.approx(-0.5)


def test_kelly_fraction_formula():
    # p=0.3, odds=5 -> b=4, edge = 0.3*4 - 0.7 = 0.5, f* = 0.125
    assert kelly_fraction(0.30, 5.0) == pytest.approx(0.125)


def test_kelly_never_backs_a_negative_edge():
    assert kelly_fraction(0.10, 5.0) == 0.0
    assert kelly_fraction(0.5, 1.0) == 0.0
    assert kelly_fraction(0.0, 5.0) == 0.0


def test_metrics_on_an_empty_ledger():
    metrics = compute_metrics(pd.DataFrame(), starting_bankroll=1000)
    assert metrics.total_bets == 0
    assert metrics.roi == 0.0
    assert metrics.final_bankroll == 1000


def test_metrics_arithmetic():
    bets = pd.DataFrame(
        {
            "race_id": ["r1", "r2", "r3", "r4"],
            "stake": [10.0, 10.0, 10.0, 10.0],
            "odds": [3.0, 4.0, 2.0, 5.0],
            "won": [1, 0, 0, 1],
            "profit": [20.0, -10.0, -10.0, 40.0],
            "bankroll_after": [1020.0, 1010.0, 1000.0, 1040.0],
        }
    )
    metrics = compute_metrics(bets, starting_bankroll=1000, races_considered=10)

    assert metrics.total_bets == 4
    assert metrics.wins == 2
    assert metrics.strike_rate == 0.5
    assert metrics.total_staked == 40.0
    assert metrics.profit == 40.0
    assert metrics.roi == pytest.approx(1.0)
    assert metrics.final_bankroll == 1040.0
    assert metrics.bet_rate == 0.4


def test_max_drawdown_is_measured_from_the_peak():
    bets = pd.DataFrame(
        {
            "race_id": list("abcd"),
            "stake": [100.0] * 4,
            "odds": [2.0] * 4,
            "won": [1, 0, 0, 0],
            "profit": [100.0, -100.0, -100.0, -100.0],
            "bankroll_after": [1100.0, 1000.0, 900.0, 800.0],
        }
    )
    metrics = compute_metrics(bets, starting_bankroll=1000)

    assert metrics.peak_bankroll == 1100.0
    assert metrics.max_drawdown == pytest.approx(300.0)
    assert metrics.max_drawdown_pct == pytest.approx(300 / 1100)


def test_longest_losing_streak():
    bets = pd.DataFrame(
        {
            "race_id": list("abcdefg"),
            "stake": [10.0] * 7,
            "odds": [2.0] * 7,
            "won": [1, 0, 0, 0, 1, 0, 0],
            "profit": [10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0],
            "bankroll_after": [10.0] * 7,
        }
    )
    assert compute_metrics(bets, starting_bankroll=1000).longest_losing_streak == 3


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def test_config_rejects_nonsense():
    with pytest.raises(StrategyError, match="staking"):
        BacktestConfig(staking="martingale")
    with pytest.raises(StrategyError, match="kelly"):
        BacktestConfig(kelly_fraction_multiplier=0)
    with pytest.raises(StrategyError, match="max_stake_fraction"):
        BacktestConfig(max_stake_fraction=2)
    with pytest.raises(StrategyError, match="bankroll"):
        BacktestConfig(starting_bankroll=0)


# ---------------------------------------------------------------------------
# Known-answer engine tests
# ---------------------------------------------------------------------------
def test_market_predictor_never_finds_value_in_the_market():
    """The negative control. If this ever bets, the framework is broken."""
    frame = make_frame([{"odds": [2.0, 4.0, 6.0, 8.0], "winner": 0} for _ in range(50)])
    result = BacktestEngine(BacktestConfig(min_expected_value=0.05)).run(frame, MarketPredictor())

    assert result.metrics.total_bets == 0
    assert result.metrics.profit == 0.0


def test_a_guaranteed_winner_returns_exactly_the_price():
    frame = make_frame([{"odds": [3.0, 10.0], "winner": 0}])

    class AlwaysFirst:
        name = "always_first"

        def __call__(self, race: pd.DataFrame) -> pd.Series:
            return pd.Series([0.99, 0.01], index=race.index)

    config = BacktestConfig(staking="flat", flat_stake=100, min_expected_value=0.05, min_odds=1.0)
    result = BacktestEngine(config).run(frame, AlwaysFirst())

    assert result.metrics.total_bets == 1
    assert result.metrics.profit == pytest.approx(200.0)  # 100 at 3.0 returns 200 profit
    assert result.metrics.final_bankroll == pytest.approx(config.starting_bankroll + 200)


def test_a_guaranteed_loser_loses_exactly_the_stake():
    frame = make_frame([{"odds": [3.0, 10.0], "winner": 1}])

    class AlwaysFirst:
        name = "always_first"

        def __call__(self, race: pd.DataFrame) -> pd.Series:
            return pd.Series([0.99, 0.01], index=race.index)

    result = BacktestEngine(BacktestConfig(flat_stake=100, min_odds=1.0)).run(frame, AlwaysFirst())
    assert result.metrics.profit == pytest.approx(-100.0)


def test_starting_price_is_not_used_as_a_bettable_price():
    """SP is only known at the off; betting at it is look-ahead."""
    assert "starting_price" not in PRE_RACE_ODDS_COLUMNS

    frame = make_frame([{"odds": [10.0, 10.0], "winner": 0} for _ in range(20)])
    frame["mkt_latest_odds"] = None  # no pre-race market captured
    frame["mkt_open_odds"] = None

    result = BacktestEngine(BacktestConfig(min_expected_value=0.0)).run(frame, UniformPredictor())
    assert result.metrics.total_bets == 0
    assert any("no pre-race price" in note for note in result.notes)


def test_starting_price_can_be_opted_into_with_a_warning():
    frame = make_frame([{"odds": [10.0, 10.0], "winner": 0} for _ in range(20)])
    frame["mkt_latest_odds"] = None
    frame["mkt_open_odds"] = None

    config = BacktestConfig(min_expected_value=0.0, allow_starting_price=True)
    result = BacktestEngine(config).run(frame, UniformPredictor())

    assert result.metrics.total_bets > 0
    assert any("unrealisable" in note for note in result.notes)


def test_ev_threshold_is_enforced():
    # Uniform p=0.5 on a two-runner race; a price of 2.2 gives EV = +10%.
    frame = make_frame([{"odds": [2.2, 2.2], "winner": 0} for _ in range(10)])

    strict = BacktestEngine(BacktestConfig(min_expected_value=0.20)).run(frame, UniformPredictor())
    loose = BacktestEngine(BacktestConfig(min_expected_value=0.05)).run(frame, UniformPredictor())

    assert strict.metrics.total_bets == 0
    assert loose.metrics.total_bets == 10


def test_odds_bounds_are_enforced():
    frame = make_frame([{"odds": [1.2, 100.0], "winner": 0} for _ in range(10)])
    config = BacktestConfig(min_expected_value=-1.0, min_odds=1.5, max_odds=51.0, min_probability=0.0)
    result = BacktestEngine(config).run(frame, UniformPredictor())

    assert result.metrics.total_bets == 0, "both runners are outside the price bounds"


def test_max_bets_per_race_is_respected():
    frame = make_frame([{"odds": [5.0, 5.0, 5.0, 5.0], "winner": 0}])
    config = BacktestConfig(min_expected_value=0.0, max_bets_per_race=2)
    result = BacktestEngine(config).run(frame, UniformPredictor())

    assert result.metrics.total_bets == 2


def test_kelly_stakes_scale_with_the_bankroll():
    frame = make_frame([{"odds": [4.0, 4.0], "winner": 0} for _ in range(5)])
    config = BacktestConfig(
        staking="kelly",
        kelly_fraction_multiplier=1.0,
        max_stake_fraction=1.0,
        min_expected_value=0.0,
        starting_bankroll=1000,
    )
    result = BacktestEngine(config).run(frame, UniformPredictor())

    # p=0.5, odds=4 -> b=3, edge = 0.5*3 - 0.5 = 1.0, f* = 1/3
    assert result.bets["stake"].iloc[0] == pytest.approx(1000 / 3, abs=0.5)
    # Bankroll grew after the first winner, so the next stake is larger.
    assert result.bets["stake"].iloc[1] > result.bets["stake"].iloc[0]


def test_max_stake_fraction_caps_kelly():
    frame = make_frame([{"odds": [4.0, 4.0], "winner": 0}])
    config = BacktestConfig(
        staking="kelly",
        kelly_fraction_multiplier=1.0,
        max_stake_fraction=0.02,
        min_expected_value=0.0,
        starting_bankroll=1000,
    )
    result = BacktestEngine(config).run(frame, UniformPredictor())
    assert result.bets["stake"].iloc[0] == pytest.approx(20.0)


def test_commission_reduces_winnings_only():
    frame = make_frame([{"odds": [3.0, 3.0], "winner": 0}])
    config = BacktestConfig(min_expected_value=0.0, flat_stake=100, commission=0.05)
    result = BacktestEngine(config).run(frame, UniformPredictor())

    assert result.metrics.profit == pytest.approx(200 * 0.95)


def test_engine_stops_at_the_ruin_threshold():
    frame = make_frame([{"odds": [3.0, 3.0], "winner": 1} for _ in range(200)])
    config = BacktestConfig(
        min_expected_value=0.0, flat_stake=500, starting_bankroll=1000, ruin_threshold=0.5
    )
    result = BacktestEngine(config).run(frame, UniformPredictor())

    assert result.stopped_early
    assert result.metrics.final_bankroll <= 1000 * 0.5


def test_races_are_processed_in_chronological_order():
    frame = make_frame([{"odds": [2.5, 2.5], "winner": 0} for _ in range(10)])
    shuffled = frame.sample(frac=1, random_state=1).reset_index(drop=True)

    result = BacktestEngine(BacktestConfig(min_expected_value=0.0)).run(shuffled, UniformPredictor())
    dates = pd.to_datetime(result.bets["race_date"])
    assert dates.is_monotonic_increasing


def test_frame_without_a_target_is_rejected():
    frame = make_frame([{"odds": [2.0, 2.0], "winner": 0}]).drop(columns=["won"])
    with pytest.raises(StrategyError, match="'won' column"):
        BacktestEngine().run(frame, UniformPredictor())


def test_empty_frame_produces_an_empty_result():
    result = BacktestEngine().run(pd.DataFrame(), UniformPredictor())
    assert result.metrics.total_bets == 0
    assert "no races to simulate" in result.notes


def test_small_sample_is_flagged():
    frame = make_frame([{"odds": [2.5, 2.5], "winner": 0} for _ in range(5)])
    result = BacktestEngine(BacktestConfig(min_expected_value=0.0)).run(frame, UniformPredictor())
    assert any("too few to distinguish edge from luck" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------
def test_predictors_return_probabilities_summing_to_one():
    frame = make_frame([{"odds": [2.0, 4.0, 8.0], "winner": 0, "scores": [0.9, 0.5, 0.1]}])
    race = frame[frame["race_id"] == "rac_0000"]

    for predictor in (UniformPredictor(), MarketPredictor(), FormScorePredictor()):
        probabilities = predictor(race)
        assert probabilities.sum() == pytest.approx(1.0), predictor.name
        assert (probabilities >= 0).all()


def test_form_score_predictor_prefers_higher_scores():
    frame = make_frame([{"odds": [5.0, 5.0, 5.0], "winner": 0, "scores": [0.9, 0.5, 0.1]}])
    probabilities = FormScorePredictor()(frame)
    assert probabilities.iloc[0] > probabilities.iloc[1] > probabilities.iloc[2]


def test_predictors_fall_back_to_uniform_without_their_input():
    frame = make_frame([{"odds": [2.0, 4.0], "winner": 0}]).drop(
        columns=["mkt_implied_prob_norm", "mkt_implied_prob", "horse_form_score"]
    )
    for predictor in (MarketPredictor(), FormScorePredictor()):
        assert predictor(frame).tolist() == [0.5, 0.5]


def test_blend_predictor_interpolates():
    frame = make_frame([{"odds": [2.0, 4.0, 8.0], "winner": 0}])
    market = MarketPredictor()(frame)
    uniform = UniformPredictor()(frame)
    blended = BlendPredictor(MarketPredictor(), UniformPredictor(), weight=0.5)(frame)

    assert blended.sum() == pytest.approx(1.0)
    assert blended.iloc[0] == pytest.approx((market.iloc[0] + uniform.iloc[0]) / 2)


def test_compare_predictors_tabulates():
    frame = make_frame([{"odds": [2.0, 4.0, 8.0], "winner": 0} for _ in range(30)])
    table = compare_predictors(
        frame, [MarketPredictor(), UniformPredictor()], BacktestConfig(min_expected_value=0.05)
    )
    assert list(table["predictor"]) == ["market", "uniform"]
    assert set(table.columns) >= {"bets", "roi", "strike_rate", "max_drawdown_pct"}


def test_backtest_result_renders():
    frame = make_frame([{"odds": [2.5, 2.5], "winner": 0} for _ in range(5)])
    result = BacktestEngine(BacktestConfig(min_expected_value=0.0)).run(frame, UniformPredictor())

    rendered = result.render()
    assert "Backtest results" in rendered
    assert "ROI" in rendered
    assert result.as_dict()["metrics"]["bets"]["total"] == result.metrics.total_bets
