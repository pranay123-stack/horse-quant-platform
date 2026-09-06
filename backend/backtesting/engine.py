"""Historical simulation.

For each race, in the order it was actually run:

1. Ask the predictor for win probabilities, using **only** that race's
   point-in-time features.
2. Compare each probability with the price that was available before the off.
3. Bet where ``EV = p * odds - 1`` clears the threshold.
4. Settle against the recorded result and update the bankroll.

Assumptions, stated because every one of them flatters or penalises the result
=============================================================================

* **We get the price we saw.** Bets settle at the last pre-race price captured,
  not at SP. Real execution is worse: prices move, and a bet placed on a horse
  because it looked overpriced is often exactly the bet a bookmaker cuts.
* **Unlimited liquidity.** No stake is ever refused or partially matched. On an
  exchange this is false at the extremes; with bookmakers, a consistently
  winning account gets restricted, which this cannot model.
* **No market impact.** Our bets do not move the price.
* **Bankroll updates race by race**, so Kelly stakes compound realistically.
* **Non-runners are already excluded** — they never reach the dataset.

Consequences: results here are an **upper bound** on what the same strategy
would have achieved live. Treat a marginal backtest edge as no edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backend.backtesting.metrics import BacktestMetrics, compute_metrics, expected_value, kelly_fraction
from backend.backtesting.predictors import Predictor
from backend.research.splits import iter_race_groups
from backend.utils.exceptions import StrategyError
from backend.utils.logging import get_logger, safe_extra

logger = get_logger(__name__, channel="model")

#: Prices we could actually have taken before the off, tried in order.
#:
#: ``starting_price`` is deliberately ABSENT. SP is not determined until the race
#: starts and reaches us through the *results* feed, so betting at SP is
#: look-ahead: it takes a price that did not exist when the bet was struck.
#: Silently falling back to it also converts "no market captured" into "bet at
#: SP", which quietly manufactures wagers the strategy never had the chance to
#: place. Enable it only via ``BacktestConfig.allow_starting_price``, knowing the
#: result is unrealisable.
PRE_RACE_ODDS_COLUMNS: tuple[str, ...] = ("mkt_latest_odds", "mkt_open_odds")
ODDS_COLUMNS = PRE_RACE_ODDS_COLUMNS  # backwards-compatible alias


@dataclass(slots=True)
class BacktestConfig:
    """Strategy and risk parameters."""

    starting_bankroll: float = 10_000.0
    #: Only bet when expected value clears this. 5% is the Phase 7 default and a
    #: deliberately high bar: at a lower threshold most "value" is estimation error.
    min_expected_value: float = 0.05
    staking: str = "flat"  # "flat" | "kelly"
    flat_stake: float = 10.0
    kelly_fraction_multiplier: float = 0.25
    max_stake_fraction: float = 0.05
    min_odds: float = 1.5
    max_odds: float = 51.0
    min_probability: float = 0.02
    #: Cap bets per race. One race, several "value" runners, is usually a sign
    #: the probabilities are miscalibrated rather than a basket of edges.
    max_bets_per_race: int = 1
    #: Exchange commission on winnings, if modelling an exchange.
    commission: float = 0.0
    #: Stop betting once the bankroll falls below this fraction of its start.
    ruin_threshold: float = 0.1
    #: Settle at starting price when no pre-race market was captured. OFF by
    #: default: SP is only known once the race is off, so a backtest that uses it
    #: is reporting a price the strategy could not have taken.
    allow_starting_price: bool = False

    def __post_init__(self) -> None:
        if self.staking not in {"flat", "kelly"}:
            raise StrategyError(f"unknown staking mode {self.staking!r}; expected 'flat' or 'kelly'")
        if not 0 < self.kelly_fraction_multiplier <= 1:
            raise StrategyError("kelly_fraction_multiplier must be in (0, 1]")
        if not 0 < self.max_stake_fraction <= 1:
            raise StrategyError("max_stake_fraction must be in (0, 1]")
        if self.starting_bankroll <= 0:
            raise StrategyError("starting_bankroll must be positive")


@dataclass(slots=True)
class BacktestResult:
    """Ledger, metrics and the parameters that produced them."""

    bets: pd.DataFrame
    metrics: BacktestMetrics
    config: BacktestConfig
    predictor_name: str = ""
    skipped_races: int = 0
    stopped_early: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "predictor": self.predictor_name,
            "metrics": self.metrics.as_dict(),
            "skipped_races": self.skipped_races,
            "stopped_early": self.stopped_early,
            "notes": self.notes,
        }

    def render(self) -> str:
        header = f"Strategy: {self.predictor_name}  |  staking: {self.config.staking}  |  min EV: {self.config.min_expected_value:.0%}"
        parts = [header, "", self.metrics.render()]
        if self.stopped_early:
            parts.append("\n  ** stopped early: bankroll fell below the ruin threshold **")
        for note in self.notes:
            parts.append(f"  note: {note}")
        return "\n".join(parts)


class BacktestEngine:
    """Runs a predictor over historical races and settles the bets."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    # ------------------------------------------------------------------
    def _price(self, runner: pd.Series) -> float | None:
        columns = (
            (*PRE_RACE_ODDS_COLUMNS, "starting_price")
            if self.config.allow_starting_price
            else PRE_RACE_ODDS_COLUMNS
        )
        for column in columns:
            value = runner.get(column)
            if value is None or pd.isna(value):
                continue
            price = float(value)
            if price > 1:
                return price
        return None

    def _stake(self, probability: float, odds: float, bankroll: float) -> float:
        if self.config.staking == "flat":
            stake = self.config.flat_stake
        else:
            fraction = kelly_fraction(probability, odds) * self.config.kelly_fraction_multiplier
            stake = bankroll * fraction

        stake = min(stake, bankroll * self.config.max_stake_fraction, bankroll)
        return max(0.0, round(stake, 2))

    # ------------------------------------------------------------------
    def run(self, frame: pd.DataFrame, predictor: Predictor) -> BacktestResult:
        """Simulate ``predictor`` over ``frame``, oldest race first."""
        config = self.config
        bankroll = config.starting_bankroll
        ruin_floor = config.starting_bankroll * config.ruin_threshold

        ledger: list[dict[str, Any]] = []
        races_considered = 0
        skipped = 0
        runners_without_price = 0
        stopped_early = False
        notes: list[str] = []

        if frame.empty:
            return BacktestResult(
                bets=_empty_ledger(),
                metrics=compute_metrics(_empty_ledger(), starting_bankroll=bankroll),
                config=config,
                predictor_name=getattr(predictor, "name", type(predictor).__name__),
                notes=["no races to simulate"],
            )

        if "won" not in frame.columns:
            raise StrategyError("backtest frame needs a 'won' column to settle bets")

        for race_id, race in iter_race_groups(frame):
            races_considered += 1

            if bankroll <= ruin_floor:
                stopped_early = True
                notes.append(f"bankroll hit the ruin threshold at race {race_id}")
                break

            probabilities = predictor(race)
            if probabilities is None or len(probabilities) != len(race):
                skipped += 1
                continue

            candidates: list[dict[str, Any]] = []
            for position, (_, runner) in enumerate(race.iterrows()):
                probability = float(probabilities.to_numpy()[position])
                odds = self._price(runner)
                if odds is None:
                    runners_without_price += 1
                    continue
                if not config.min_odds <= odds <= config.max_odds:
                    continue
                if probability < config.min_probability:
                    continue

                edge = expected_value(probability, odds)
                if edge < config.min_expected_value:
                    continue
                candidates.append(
                    {
                        "runner": runner,
                        "probability": probability,
                        "odds": odds,
                        "expected_value": edge,
                    }
                )

            if not candidates:
                continue

            # Strongest edge first, then take at most ``max_bets_per_race``.
            candidates.sort(key=lambda item: item["expected_value"], reverse=True)
            for candidate in candidates[: config.max_bets_per_race]:
                stake = self._stake(candidate["probability"], candidate["odds"], bankroll)
                if stake <= 0:
                    continue

                runner = candidate["runner"]
                won = bool(runner["won"])
                if won:
                    gross = stake * (candidate["odds"] - 1.0)
                    profit = gross * (1.0 - config.commission)
                else:
                    profit = -stake
                bankroll += profit

                ledger.append(
                    {
                        "race_id": race_id,
                        "horse_id": runner.get("horse_id"),
                        "race_date": runner.get("race_date"),
                        "probability": candidate["probability"],
                        "odds": candidate["odds"],
                        "expected_value": candidate["expected_value"],
                        "stake": stake,
                        "won": int(won),
                        "profit": round(profit, 4),
                        "bankroll_after": round(bankroll, 4),
                    }
                )

        bets = pd.DataFrame(ledger) if ledger else _empty_ledger()
        metrics = compute_metrics(
            bets, starting_bankroll=config.starting_bankroll, races_considered=races_considered
        )

        if runners_without_price:
            notes.append(f"{runners_without_price} runner(s) had no pre-race price and were skipped")
        if config.allow_starting_price:
            notes.append(
                "allow_starting_price is ON: some bets settle at SP, which was not "
                "knowable pre-race — results are unrealisable"
            )
        if metrics.total_bets == 0:
            notes.append(
                f"no bet cleared the {config.min_expected_value:.0%} EV threshold — "
                "expected when the predictor is no better than the market"
            )
        elif metrics.total_bets < 100:
            notes.append(f"only {metrics.total_bets} bets: too few to distinguish edge from luck")

        result = BacktestResult(
            bets=bets,
            metrics=metrics,
            config=config,
            predictor_name=getattr(predictor, "name", type(predictor).__name__),
            skipped_races=skipped,
            stopped_early=stopped_early,
            notes=notes,
        )
        logger.info(
            "backtest complete",
            extra=safe_extra(
                {
                    "predictor": result.predictor_name,
                    "bets": metrics.total_bets,
                    "roi": round(metrics.roi, 4),
                    "profit": round(metrics.profit, 2),
                }
            ),
        )
        return result


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "race_id",
            "horse_id",
            "race_date",
            "probability",
            "odds",
            "expected_value",
            "stake",
            "won",
            "profit",
            "bankroll_after",
        ]
    )


def compare_predictors(
    frame: pd.DataFrame, predictors: list[Predictor], config: BacktestConfig | None = None
) -> pd.DataFrame:
    """Run several predictors over the same races and tabulate the outcomes."""
    engine = BacktestEngine(config)
    rows = []
    for predictor in predictors:
        result = engine.run(frame, predictor)
        rows.append(
            {
                "predictor": result.predictor_name,
                "bets": result.metrics.total_bets,
                "strike_rate": round(result.metrics.strike_rate, 4),
                "roi": round(result.metrics.roi, 4),
                "profit": round(result.metrics.profit, 2),
                "max_drawdown_pct": round(result.metrics.max_drawdown_pct, 4),
                "sharpe_per_bet": round(result.metrics.sharpe_per_bet, 3),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ODDS_COLUMNS",
    "PRE_RACE_ODDS_COLUMNS",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "compare_predictors",
]
