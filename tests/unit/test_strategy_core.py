"""Odds, expected value, filters and staking.

Arithmetic first. Every number a strategy acts on — the fair probability, the
edge, the EV, the stake, the settlement — is checked by hand here, because a
quiet error in any of them produces a backtest that is confidently wrong rather
than obviously broken.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.strategy.expected_value import (
    ValueBetSignal,
    compute_signal_frame,
    edge,
    expected_value,
    fair_odds,
)
from backend.strategy.filters import (
    STRATEGY_A,
    STRATEGY_LIBRARY,
    STRATEGY_UNFILTERED,
    StrategyConfig,
    select_bets,
)
from backend.strategy.odds import (
    build_market_frame,
    implied_probability,
    overround,
    remove_margin,
)
from backend.strategy.staking import (
    BankrollState,
    StakingConfig,
    StakingPlan,
    kelly_fraction,
    settle,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Odds and margin removal
# ---------------------------------------------------------------------------
def test_implied_probability():
    assert implied_probability(4.0) == pytest.approx(0.25)
    assert implied_probability(2.0) == pytest.approx(0.5)
    assert np.isnan(implied_probability(1.0))
    assert np.isnan(implied_probability(0.0))


def test_overround_of_a_fair_book_is_one():
    assert overround(np.array([2.0, 2.0])) == pytest.approx(1.0)
    assert overround(np.array([1.9, 1.9])) == pytest.approx(2 / 1.9)


@pytest.mark.parametrize("method", ["proportional", "additive", "power"])
def test_every_margin_method_returns_a_distribution(method):
    fair = remove_margin(np.array([2.0, 4.0, 6.0, 10.0]), method)
    assert fair.sum() == pytest.approx(1.0)
    assert (fair > 0).all()


def test_margin_removal_preserves_the_favourite():
    """Whatever the method, the shortest price stays the most likely winner."""
    odds = np.array([2.0, 4.0, 6.0, 10.0])
    for method in ("proportional", "additive", "power"):
        fair = remove_margin(odds, method)
        assert np.argmax(fair) == 0
        assert list(fair) == sorted(fair, reverse=True)


def test_proportional_is_exact_division():
    odds = np.array([2.0, 4.0, 4.0])
    quoted = 1 / odds
    np.testing.assert_allclose(remove_margin(odds, "proportional"), quoted / quoted.sum())


def test_power_method_shrinks_longshots_more_than_proportional():
    """The favourite-longshot bias runs this way; proportional ignores it.

    Note the book must genuinely over-round. Given an *under*-round set of
    prices the power exponent drops below 1 and the effect reverses — which is
    correct, but is a book offering arbitrage, not a market.
    """
    odds = np.array([1.3, 4.0, 12.0])
    assert overround(odds) > 1.0
    proportional = remove_margin(odds, "proportional")
    power = remove_margin(odds, "power")

    assert power[-1] < proportional[-1], "power should discount the outsider further"
    assert power[0] > proportional[0]


def test_margin_removal_ignores_unpriced_runners():
    fair = remove_margin(np.array([2.0, np.nan, 4.0]), "power")
    assert np.isnan(fair[1])
    assert np.nansum(fair) == pytest.approx(1.0)


def test_margin_removal_with_no_prices():
    assert np.isnan(remove_margin(np.array([np.nan, np.nan]))).all()


def test_unknown_margin_method_rejected():
    with pytest.raises(ValueError, match="unknown margin method"):
        remove_margin(np.array([2.0, 2.0]), "magic")  # type: ignore[arg-type]


def test_market_frame_separates_consensus_from_best_price():
    """Betting the best price against a consensus probability is the point."""
    frame = pd.DataFrame(
        {
            "race_id": ["r1", "r1"],
            "horse_id": ["a", "b"],
            "mkt_mean_odds": [2.0, 4.0],
            "mkt_latest_odds": [2.2, 4.4],  # best prices, longer than consensus
        }
    )
    result = build_market_frame(frame, method="proportional")

    assert result["bet_odds"].tolist() == [2.2, 4.4]
    # Probabilities come from the consensus, not the best price.
    assert result["market_probability"].sum() == pytest.approx(1.0)
    assert result["market_probability"].iloc[0] == pytest.approx((1 / 2) / (1 / 2 + 1 / 4))


def test_market_frame_falls_back_when_consensus_is_missing():
    frame = pd.DataFrame(
        {"race_id": ["r1"], "horse_id": ["a"], "mkt_mean_odds": [np.nan], "mkt_latest_odds": [3.0]}
    )
    result = build_market_frame(frame)
    assert result["bet_odds"].iloc[0] == 3.0


def test_market_frame_on_empty_input():
    result = build_market_frame(pd.DataFrame())
    assert "market_probability" in result.columns


# ---------------------------------------------------------------------------
# Expected value
# ---------------------------------------------------------------------------
def test_expected_value_matches_the_specification_example():
    """p=0.30, odds=6.0 → EV = +0.8."""
    assert expected_value(0.30, 6.0) == pytest.approx(0.8)


def test_expected_value_of_a_fair_price_is_zero():
    assert expected_value(0.25, 4.0) == pytest.approx(0.0)


def test_expected_value_of_an_unusable_price_loses_the_stake():
    assert expected_value(0.5, 1.0) == -1.0
    assert expected_value(float("nan"), 4.0) == -1.0


def test_edge_matches_the_specification_example():
    """Model 30%, market 16.7% → edge 13.3 points."""
    assert edge(0.30, 1 / 6.0) == pytest.approx(0.1333, abs=1e-4)


def test_fair_odds_is_the_reciprocal():
    assert fair_odds(0.25) == pytest.approx(4.0)
    assert fair_odds(0.0) == float("inf")


def test_signal_frame_computes_edge_and_ev():
    frame = pd.DataFrame(
        {
            "race_id": ["r1", "r1"],
            "horse_id": ["a", "b"],
            "model_probability": [0.30, 0.10],
            "market_probability": [0.1667, 0.20],
            "bet_odds": [6.0, 4.0],
        }
    )
    result = compute_signal_frame(frame)

    assert result["expected_value"].iloc[0] == pytest.approx(0.8)
    assert result["edge"].iloc[0] == pytest.approx(0.1333, abs=1e-4)
    assert result["expected_value"].iloc[1] == pytest.approx(-0.6)


def test_signal_frame_on_empty_input():
    result = compute_signal_frame(pd.DataFrame(columns=["model_probability", "bet_odds"]))
    assert "expected_value" in result.columns


def test_signal_serialises_and_renders():
    signal = ValueBetSignal(
        race_id="r1",
        horse_id="h1",
        race_date=date(2025, 1, 1),
        model_probability=0.30,
        market_probability=0.167,
        odds=6.0,
        edge=0.133,
        expected_value=0.8,
        decision="bet",
        horse_name="Thunder King",
    )
    assert signal.is_bet
    assert signal.fair_odds == pytest.approx(3.333, abs=1e-3)
    assert signal.odds_ratio == pytest.approx(1.8, abs=1e-3)
    assert "Thunder King" in signal.render()
    assert signal.as_dict()["expected_value"] == 0.8


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def _race(**overrides) -> pd.DataFrame:
    base = {
        "race_id": ["r1"] * 4,
        "horse_id": ["a", "b", "c", "d"],
        "race_date": [date(2025, 1, 1)] * 4,
        "model_probability": [0.40, 0.25, 0.20, 0.15],
        "market_probability": [0.30, 0.25, 0.25, 0.20],
        "bet_odds": [3.0, 4.0, 4.0, 5.0],
        "edge": [0.10, 0.0, -0.05, -0.05],
        "expected_value": [0.20, 0.0, -0.20, -0.25],
        "book_overround": [1.15] * 4,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_only_positive_ev_runners_are_backed():
    result = select_bets(_race(), STRATEGY_A)
    assert [signal.horse_id for signal in result.bets] == ["a"]


def test_rejection_reasons_are_recorded():
    result = select_bets(_race(), STRATEGY_A)
    assert result.rejections.get("ev_below_threshold") == 3


def test_max_bets_per_race_caps_and_ranks_by_ev():
    race = _race(expected_value=[0.20, 0.30, 0.10, -0.5], model_probability=[0.4, 0.35, 0.3, 0.15])
    result = select_bets(race, StrategyConfig(max_bets_per_race=2))

    assert [signal.horse_id for signal in result.bets] == ["b", "a"]
    assert result.rejections.get("max_bets_per_race") == 1


def test_probability_floor_blocks_longshots():
    race = _race(model_probability=[0.05, 0.05, 0.05, 0.05], expected_value=[1.0, 1.0, 1.0, 1.0])
    result = select_bets(race, StrategyConfig(min_probability=0.10))
    assert result.bets == []
    assert result.rejections["probability_too_low"] == 4


def test_odds_ceiling_blocks_the_tail():
    race = _race(bet_odds=[50.0, 60.0, 70.0, 80.0], expected_value=[1.0] * 4)
    result = select_bets(race, StrategyConfig(max_odds=20.0))
    assert result.rejections["odds_too_high"] == 4


def test_missing_odds_are_never_backed():
    race = _race(bet_odds=[np.nan] * 4)
    result = select_bets(race, STRATEGY_A)
    assert result.bets == []
    assert result.rejections["no_odds"] == 4


def test_small_fields_are_skipped():
    race = _race().head(2)
    result = select_bets(race, StrategyConfig(min_field_size=4))
    assert result.rejections["field_too_small"] == 2


def test_implausible_overround_is_skipped():
    """A book summing to 3.0 is broken data, not an opportunity."""
    race = _race(book_overround=[3.0] * 4)
    result = select_bets(race, STRATEGY_A)
    assert result.rejections["overround_implausible"] == 4


def test_edge_filter_requires_beating_the_market_twice():
    race = _race(edge=[0.01, 0.0, 0.0, 0.0], expected_value=[0.20, 0, 0, 0])
    assert select_bets(race, StrategyConfig(min_edge=0.05)).bets == []
    assert len(select_bets(race, StrategyConfig(min_edge=0.0)).bets) == 1


def test_unfiltered_strategy_backs_everything_positive():
    race = _race(expected_value=[0.20, 0.10, 0.05, 0.01])
    assert len(select_bets(race, STRATEGY_UNFILTERED).bets) == 4


def test_strategy_library_is_complete():
    assert {"A_ev5", "B_ev10", "C_top_pick", "D_ev5_tight", "F_edge5", "E_unfiltered"} <= set(
        STRATEGY_LIBRARY
    )
    for config in STRATEGY_LIBRARY.values():
        assert config.as_dict()["name"] == config.name


def test_selecting_from_an_empty_race():
    assert select_bets(pd.DataFrame(), STRATEGY_A).bets == []


# ---------------------------------------------------------------------------
# Staking
# ---------------------------------------------------------------------------
def test_kelly_formula():
    # p=0.30, odds=6.0 -> b=5, f* = (0.3*5 - 0.7)/5 = 0.16
    assert kelly_fraction(0.30, 6.0) == pytest.approx(0.16)


def test_kelly_is_zero_without_an_edge():
    assert kelly_fraction(0.10, 5.0) == 0.0
    assert kelly_fraction(0.5, 1.0) == 0.0
    assert kelly_fraction(float("nan"), 5.0) == 0.0


def test_flat_staking_is_constant():
    plan = StakingPlan(StakingConfig(method="flat", flat_stake=10.0, max_stake_fraction=1.0))
    state = BankrollState.start(1000)

    for _ in range(3):
        stake, _ = plan.stake_for(state, 0.3, 6.0)
        assert stake == 10.0
        state.apply(stake, -stake)


def test_kelly_staking_scales_with_the_bankroll():
    plan = StakingPlan(
        StakingConfig(method="kelly", kelly_multiplier=1.0, max_stake_fraction=1.0, max_daily_exposure=1.0)
    )
    state = BankrollState.start(1000)
    stake, _ = plan.stake_for(state, 0.30, 6.0)
    assert stake == pytest.approx(160.0)  # 1000 * 0.16


def test_fractional_kelly_halves_the_stake():
    config = StakingConfig(
        method="kelly", kelly_multiplier=0.5, max_stake_fraction=1.0, max_daily_exposure=1.0
    )
    stake, _ = StakingPlan(config).stake_for(BankrollState.start(1000), 0.30, 6.0)
    assert stake == pytest.approx(80.0)


def test_max_stake_fraction_binds_and_is_reported():
    config = StakingConfig(method="kelly", kelly_multiplier=1.0, max_stake_fraction=0.02)
    stake, constraint = StakingPlan(config).stake_for(BankrollState.start(1000), 0.30, 6.0)

    assert stake == pytest.approx(20.0)
    assert constraint == "max_stake_fraction"


def test_daily_exposure_limit_stops_further_betting():
    config = StakingConfig(
        method="flat", flat_stake=100.0, max_stake_fraction=1.0, max_daily_exposure=0.15, min_stake=1.0
    )
    plan = StakingPlan(config)
    state = BankrollState.start(1000)
    day = date(2025, 1, 1)

    placed = []
    for _ in range(4):
        stake, constraint = plan.stake_for(state, 0.3, 6.0, day=day)
        if stake > 0:
            placed.append(stake)
            state.apply(stake, -stake)
        else:
            assert constraint == "daily_exposure"
            break
    assert sum(placed) <= 150.0


def test_new_day_resets_the_exposure_budget():
    config = StakingConfig(method="flat", flat_stake=100.0, max_stake_fraction=1.0, max_daily_exposure=0.15)
    plan = StakingPlan(config)
    state = BankrollState.start(1000)

    plan.stake_for(state, 0.3, 6.0, day=date(2025, 1, 1))
    state.apply(100, -100)
    stake, _ = plan.stake_for(state, 0.3, 6.0, day=date(2025, 1, 2))
    assert stake > 0


def test_drawdown_brake_reduces_stakes():
    config = StakingConfig(
        method="flat",
        flat_stake=100.0,
        max_stake_fraction=1.0,
        max_daily_exposure=1.0,
        drawdown_brake_at=0.9,
        drawdown_brake_factor=0.5,
    )
    plan = StakingPlan(config)
    state = BankrollState.start(1000)

    full, _ = plan.stake_for(state, 0.3, 6.0)
    assert full == 100.0

    state.apply(0, -200)  # 20% drawdown
    braked, constraint = plan.stake_for(state, 0.3, 6.0)
    assert braked == 50.0
    assert constraint == "drawdown_brake"


def test_ruin_threshold_stops_betting_entirely():
    plan = StakingPlan(StakingConfig(method="flat", flat_stake=10.0, ruin_threshold=0.5))
    state = BankrollState.start(1000)
    state.apply(0, -600)

    stake, constraint = plan.stake_for(state, 0.3, 6.0)
    assert stake == 0.0
    assert constraint == "ruined"
    assert state.is_ruined


def test_stakes_below_the_minimum_are_not_placed():
    config = StakingConfig(method="kelly", kelly_multiplier=0.01, min_stake=5.0, max_stake_fraction=1.0)
    stake, constraint = StakingPlan(config).stake_for(BankrollState.start(100), 0.30, 6.0)
    assert stake == 0.0
    assert constraint == "below_min_stake"


def test_no_edge_means_no_stake_under_kelly():
    config = StakingConfig(method="kelly")
    stake, constraint = StakingPlan(config).stake_for(BankrollState.start(1000), 0.10, 5.0)
    assert stake == 0.0
    assert constraint == "no_edge"


def test_invalid_staking_config_rejected():
    with pytest.raises(ValueError, match="unknown staking method"):
        StakingConfig(method="martingale")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kelly_multiplier"):
        StakingConfig(kelly_multiplier=0)
    with pytest.raises(ValueError, match="max_stake_fraction"):
        StakingConfig(max_stake_fraction=2)


def test_bankroll_tracks_peak_and_drawdown():
    state = BankrollState.start(1000)
    state.apply(100, 200)
    assert state.peak == 1200
    assert state.drawdown == 0.0

    state.apply(100, -300)
    assert state.current == 900
    assert state.drawdown == pytest.approx(300 / 1200)


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------
def test_winner_returns_stake_times_odds_minus_one():
    assert settle(10.0, 6.0, won=True) == pytest.approx(50.0)


def test_loser_loses_the_stake():
    assert settle(10.0, 6.0, won=False) == pytest.approx(-10.0)


def test_commission_is_charged_on_winnings_only():
    assert settle(10.0, 6.0, won=True, commission=0.05) == pytest.approx(47.5)
    assert settle(10.0, 6.0, won=False, commission=0.05) == pytest.approx(-10.0)


def test_full_commission_leaves_nothing():
    assert settle(10.0, 6.0, won=True, commission=1.0) == pytest.approx(0.0)
