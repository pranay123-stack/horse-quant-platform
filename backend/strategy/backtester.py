"""Walk-forward strategy simulation.

The question this answers is narrow and specific: *if we had followed these
signals, race by race, in the order they happened, what would have occurred?*

Two stages, deliberately separated
==================================

**1. Predictions** — :class:`WalkForwardPredictor` retrains the model
periodically and predicts only forward. Expensive, run once.

**2. Simulation** — :class:`StrategyBacktester` consumes those predictions and
applies a strategy. Cheap, run many times.

The split matters because Phase 5.6 compares several strategies. Re-running
walk-forward training for each would be slow *and* misleading: differences
between strategies would be confounded with differences between independently
retrained models. Sharing one prediction set isolates the strategy.

Why walk-forward at all
=======================
Phase 4 trained once on 2018-2023 and scored 2025-2026. That is the right way to
measure a *model*. It is the wrong way to measure a *strategy*, which would in
reality have been retrained as data arrived, and whose early years would have
been traded by a model fitted on much less history. Walk-forward reproduces that:
each period is predicted by a model that saw only what preceded it.

Execution realism
=================
Every assumption in :class:`ExecutionConfig` makes the simulation *worse* than
the naive version, because every one of them makes live betting worse than the
simulation:

* **Slippage** — the price on the screen is not the price you get.
* **Commission** — charged on winnings, as exchanges do.
* **Liquidity cap** — you cannot get £5,000 on a Class 6 seller at Sedgefield.
* **No SP** — starting price is set at the off and reaches us through the results
  feed; betting at it is look-ahead. Refused unless explicitly enabled.

Even with all of them, the result is an upper bound. Nothing here models a
bookmaker restricting a winning account, which in practice is what ends most
profitable betting operations.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.ml.dataset import SplitConfig, TrainingDatasetBuilder
from backend.ml.metrics.racing import normalise_within_race
from backend.ml.models.calibrated import CalibratedRacingModel
from backend.ml.preprocessing.schema import FeatureSchema
from backend.ml.train import MODEL_TYPES
from backend.strategy.expected_value import compute_signal_frame
from backend.strategy.filters import StrategyConfig, select_bets
from backend.strategy.odds import MarginMethod, build_market_frame
from backend.strategy.staking import BankrollState, StakingConfig, StakingPlan, settle
from backend.utils.exceptions import StrategyError
from backend.utils.logging import get_logger, safe_extra

logger = get_logger(__name__, channel="model")

MARKET_PREFIXES = ("mkt_", "market_")


# ---------------------------------------------------------------------------
# Execution assumptions
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """How much worse reality is than the screen."""

    #: Fraction by which the struck price is worse than the observed one. 2% on
    #: a 6.0 quote means the bet goes on at 5.88. Prices move between a signal
    #: firing and a stake landing, and they move against you more often than not
    #: — a horse looks like value partly because its price is about to be cut.
    slippage: float = 0.02
    #: Charged on winnings only.
    commission: float = 0.0
    #: Hard ceiling on any single stake, standing in for market depth.
    max_stake_liquidity: float | None = 500.0
    #: Refuse to bet at SP; it is not knowable before the off.
    allow_starting_price: bool = False
    #: Require at least this many bookmakers before treating quotes as a market.
    min_bookmakers: int = 1

    def struck_odds(self, observed: float) -> float:
        """Apply slippage. Never below 1.01, which would be a free bet."""
        if not np.isfinite(observed):
            return float("nan")
        return max(1.01, observed * (1.0 - self.slippage))

    def as_dict(self) -> dict[str, Any]:
        return {
            "slippage": self.slippage,
            "commission": self.commission,
            "max_stake_liquidity": self.max_stake_liquidity,
            "allow_starting_price": self.allow_starting_price,
            "min_bookmakers": self.min_bookmakers,
        }


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """A complete, reproducible strategy specification."""

    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    staking: StakingConfig = field(default_factory=StakingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    starting_bankroll: float = 10_000.0
    margin_method: MarginMethod = "power"

    @property
    def name(self) -> str:
        return f"{self.strategy.name}/{self.staking.method}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.as_dict(),
            "staking": self.staking.as_dict(),
            "execution": self.execution.as_dict(),
            "starting_bankroll": self.starting_bankroll,
            "margin_method": self.margin_method,
        }


# ---------------------------------------------------------------------------
# Stage 1: walk-forward predictions
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class WalkForwardWindow:
    """One retrain-and-predict period."""

    train_start: date
    train_end: date
    predict_start: date
    predict_end: date
    rows: int = 0
    races: int = 0

    def describe(self) -> str:
        return (
            f"train {self.train_start}→{self.train_end}  "
            f"predict {self.predict_start}→{self.predict_end}  ({self.races} races)"
        )


class WalkForwardPredictor:
    """Retrains periodically and predicts strictly forward."""

    def __init__(
        self,
        session: Session,
        *,
        model_type: str = "lightgbm",
        model_params: dict[str, Any] | None = None,
        drop_feature_prefixes: tuple[str, ...] = (),
        embargo_days: int = 14,
        min_train_days: int = 365,
        seed: int = 42,
        apply_data_quality: bool = False,
    ) -> None:
        if model_type not in MODEL_TYPES:
            raise StrategyError(f"unknown model type {model_type!r}")
        self.session = session
        self.model_type = model_type
        self.model_params = model_params
        self.drop_feature_prefixes = drop_feature_prefixes
        self.embargo_days = embargo_days
        self.min_train_days = min_train_days
        self.seed = seed
        # Off by default inside walk-forward: the quality audit is a
        # whole-database scan and re-running it for every retrain window is
        # pure waste. Audit once with `make quality`, then backtest.
        self.apply_data_quality = apply_data_quality
        self.windows: list[WalkForwardWindow] = []

    # ------------------------------------------------------------------
    def _periods(self, start: date, end: date, months: int) -> Iterator[tuple[date, date]]:
        cursor = start
        while cursor <= end:
            stop = min(end, _add_months(cursor, months) - timedelta(days=1))
            yield cursor, stop
            cursor = stop + timedelta(days=1)

    def run(
        self,
        *,
        history_start: date,
        test_start: date,
        test_end: date,
        retrain_months: int = 12,
    ) -> pd.DataFrame:
        """Predict every race in ``[test_start, test_end]``, retraining as we go."""
        if test_start <= history_start:
            raise StrategyError("test period must start after the history period")

        builder = TrainingDatasetBuilder(self.session, apply_data_quality=self.apply_data_quality)
        frames: list[pd.DataFrame] = []
        self.windows = []

        for period_start, period_end in self._periods(test_start, test_end, retrain_months):
            train_end = period_start - timedelta(days=self.embargo_days + 1)
            if (train_end - history_start).days < self.min_train_days:
                logger.warning(
                    "skipping period: not enough history to train",
                    extra={"period_start": str(period_start), "train_end": str(train_end)},
                )
                continue

            # Validation sits between training and prediction: later than the
            # training data, earlier than anything we bet on.
            valid_start = train_end - timedelta(days=90)
            split = SplitConfig(
                train_start=history_start,
                train_end=valid_start - timedelta(days=1),
                valid_start=valid_start,
                valid_end=train_end,
                test_start=period_start,
                test_end=period_end,
                embargo_days=0,
            )
            data = builder.build(split, drop_feature_prefixes=self.drop_feature_prefixes)
            if data.train.empty or data.test.empty:
                continue

            schema = FeatureSchema.from_frame(data.train, data.numeric_features, data.categorical_features)
            model = MODEL_TYPES[self.model_type](schema, self.model_params, seed=self.seed)
            model.fit(
                data.train,
                data.train[data.target],
                data.valid_stop if not data.valid_stop.empty else None,
                data.valid_stop[data.target] if not data.valid_stop.empty else None,
            )
            calibrated = CalibratedRacingModel(model)

            predictions = data.test.copy()
            predictions["model_probability"] = normalise_within_race(
                calibrated.predict_proba(data.test), data.test["race_id"]
            ).to_numpy()
            predictions["model_type"] = self.model_type
            frames.append(predictions)

            window = WalkForwardWindow(
                train_start=history_start,
                train_end=train_end,
                predict_start=period_start,
                predict_end=period_end,
                rows=len(predictions),
                races=int(predictions["race_id"].nunique()),
            )
            self.windows.append(window)
            logger.info("walk-forward window complete", extra=safe_extra({"window": window.describe()}))

        if not frames:
            raise StrategyError("walk-forward produced no predictions; check the date windows")

        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values(["race_date", "race_id", "horse_id"]).reset_index(drop=True)


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return date(year, month, day)


# ---------------------------------------------------------------------------
# Stage 2: strategy simulation
# ---------------------------------------------------------------------------
LEDGER_COLUMNS = [
    "run_id",
    "strategy",
    "race_id",
    "horse_id",
    "race_date",
    "model_probability",
    "market_probability",
    "edge",
    "expected_value",
    "observed_odds",
    "odds",
    "stake",
    "stake_constraint",
    "won",
    "profit_loss",
    "bankroll_after",
]


@dataclass(slots=True)
class StrategyRun:
    """A settled ledger plus the context needed to interpret it."""

    run_id: str
    config: BacktestConfig
    ledger: pd.DataFrame
    races_considered: int = 0
    runners_considered: int = 0
    rejections: dict[str, int] = field(default_factory=dict)
    stopped_early: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def bets(self) -> int:
        return len(self.ledger)


class StrategyBacktester:
    """Applies a strategy to walk-forward predictions and settles the bets."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    # ------------------------------------------------------------------
    def prepare(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Attach market probabilities, edge and EV to raw predictions."""
        if predictions.empty:
            return predictions

        frame = build_market_frame(predictions, method=self.config.margin_method)
        if not self.config.execution.allow_starting_price:
            # SP is post-race information; without a pre-race quote there is no bet.
            frame.loc[frame["bet_odds"].isna(), "bet_odds"] = np.nan
        elif "starting_price" in frame.columns:
            frame["bet_odds"] = frame["bet_odds"].fillna(
                pd.to_numeric(frame["starting_price"], errors="coerce")
            )

        if self.config.execution.min_bookmakers > 1 and "mkt_bookmaker_count" in frame.columns:
            thin = pd.to_numeric(frame["mkt_bookmaker_count"], errors="coerce").fillna(0)
            frame.loc[thin < self.config.execution.min_bookmakers, "bet_odds"] = np.nan

        return compute_signal_frame(frame)

    def run(self, predictions: pd.DataFrame, *, run_id: str | None = None) -> StrategyRun:
        """Simulate the strategy over ``predictions``, oldest race first."""
        config = self.config
        run_id = run_id or uuid.uuid4().hex[:12]

        if predictions.empty:
            return StrategyRun(run_id, config, _empty_ledger(), notes=["no predictions to simulate"])
        if "won" not in predictions.columns:
            raise StrategyError("backtest input needs a 'won' column to settle bets")

        frame = self.prepare(predictions)
        plan = StakingPlan(config.staking)
        state = BankrollState.start(config.starting_bankroll)

        rows: list[dict[str, Any]] = []
        rejections: dict[str, int] = {}
        races_considered = 0
        stopped_early = False

        order = [column for column in ("race_date", "off_time", "race_id") if column in frame.columns]
        for _, race in frame.sort_values(order, kind="stable").groupby("race_id", sort=False):
            races_considered += 1
            if state.is_ruined:
                stopped_early = True
                break

            selection = select_bets(race, config.strategy)
            for reason, count in selection.rejections.items():
                rejections[reason] = rejections.get(reason, 0) + count

            race_day = _race_day(race)
            for signal in selection.bets:
                observed = signal.odds
                struck = config.execution.struck_odds(observed)
                stake, constraint = plan.stake_for(state, signal.model_probability, struck, day=race_day)
                liquidity = config.execution.max_stake_liquidity
                if liquidity is not None and stake > liquidity:
                    stake, constraint = liquidity, "liquidity"
                if stake <= 0:
                    rejections[constraint or "zero_stake"] = rejections.get(constraint or "zero_stake", 0) + 1
                    continue

                won = bool(race.loc[race["horse_id"] == signal.horse_id, "won"].iloc[0])
                profit = settle(stake, struck, won, commission=config.execution.commission)
                state.apply(stake, profit)

                rows.append(
                    {
                        "run_id": run_id,
                        "strategy": config.strategy.name,
                        "race_id": signal.race_id,
                        "horse_id": signal.horse_id,
                        "race_date": race_day,
                        "model_probability": signal.model_probability,
                        "market_probability": signal.market_probability,
                        "edge": signal.edge,
                        "expected_value": signal.expected_value,
                        "observed_odds": observed,
                        "odds": struck,
                        "stake": stake,
                        "stake_constraint": constraint,
                        "won": int(won),
                        "profit_loss": round(profit, 4),
                        "bankroll_after": round(state.current, 4),
                    }
                )

        ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS) if rows else _empty_ledger()
        notes: list[str] = []
        if stopped_early:
            notes.append("stopped early: bankroll fell below the ruin threshold")
        if len(ledger) < 200:
            notes.append(
                f"only {len(ledger)} bets — far too few to separate edge from luck; "
                "treat any ROI here as noise"
            )

        run = StrategyRun(
            run_id=run_id,
            config=config,
            ledger=ledger,
            races_considered=races_considered,
            runners_considered=len(frame),
            rejections=dict(sorted(rejections.items(), key=lambda item: -item[1])),
            stopped_early=stopped_early,
            notes=notes,
        )
        logger.info(
            "strategy backtest complete",
            extra=safe_extra(
                {
                    "strategy": config.strategy.name,
                    "bets": run.bets,
                    "races": races_considered,
                    "final_bankroll": round(state.current, 2),
                }
            ),
        )
        return run


def _race_day(race: pd.DataFrame) -> date | None:
    if "race_date" not in race.columns or race.empty:
        return None
    value = race["race_date"].iloc[0]
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


__all__ = [
    "LEDGER_COLUMNS",
    "MARKET_PREFIXES",
    "BacktestConfig",
    "ExecutionConfig",
    "StrategyBacktester",
    "StrategyRun",
    "WalkForwardPredictor",
    "WalkForwardWindow",
]
