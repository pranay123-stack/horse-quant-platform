"""Performance reporting.

Metrics chosen for what they reveal, not for what flatters:

**ROI on turnover**, not on bankroll. Staking £10 a thousand times is a different
exposure from staking £10,000 once, and only turnover-based ROI treats them
differently.

**Profit factor** — gross winnings divided by gross losses. Immune to bet sizing,
so it separates "the picks were good" from "the staking was lucky".

**Maximum drawdown**, in cash and as a fraction of peak. Drawdown is what ends
strategies: a profitable plan with a 60% drawdown is one an operator abandons at
the worst possible moment.

**Sharpe per bet**, not annualised. Annualising needs an assumed bet frequency,
and a figure quoted from a few hundred bets implies precision that is not there.

**A t-statistic on mean profit per unit staked.** With a few hundred bets, an ROI
of +8% is entirely consistent with having no edge at all. Reporting ROI without
its uncertainty is the single most common way betting backtests mislead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

#: Below this, results are noise. Stated in the report rather than assumed known.
MIN_BETS_FOR_INFERENCE = 200


@dataclass(slots=True)
class StrategyMetrics:
    """Everything needed to judge a strategy, including whether to believe it."""

    strategy: str = ""
    bets: int = 0
    wins: int = 0
    losses: int = 0

    staked: float = 0.0
    returned: float = 0.0
    profit: float = 0.0
    roi: float = 0.0
    profit_factor: float = 0.0

    strike_rate: float = 0.0
    average_odds: float = 0.0
    average_stake: float = 0.0
    average_edge: float = 0.0
    average_ev: float = 0.0

    starting_bankroll: float = 0.0
    final_bankroll: float = 0.0
    peak_bankroll: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    longest_losing_streak: int = 0

    sharpe_per_bet: float = 0.0
    #: t = mean(return per unit staked) / standard error. |t| < 2 means the
    #: result is indistinguishable from no edge.
    t_statistic: float = 0.0
    roi_standard_error: float = 0.0

    races_considered: int = 0
    bet_rate: float = 0.0
    #: Realised strike rate divided by mean model probability. 1.0 means the
    #: model's confidence on the bets it placed was borne out.
    calibration_on_bets: float = 0.0

    @property
    def is_significant(self) -> bool:
        return self.bets >= MIN_BETS_FOR_INFERENCE and abs(self.t_statistic) >= 2.0

    @property
    def verdict(self) -> str:
        if self.bets == 0:
            return "no bets placed"
        if self.bets < MIN_BETS_FOR_INFERENCE:
            return f"inconclusive — only {self.bets} bets"
        if self.profit <= 0:
            return "negative expectancy"
        if not self.is_significant:
            return f"profitable but not significant (t={self.t_statistic:.2f})"
        return f"positive expectancy (t={self.t_statistic:.2f})"

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "bets": self.bets,
            "wins": self.wins,
            "strike_rate": round(self.strike_rate, 6),
            "average_odds": round(self.average_odds, 4),
            "staked": round(self.staked, 2),
            "profit": round(self.profit, 2),
            "roi": round(self.roi, 6),
            "profit_factor": round(self.profit_factor, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 6),
            "sharpe_per_bet": round(self.sharpe_per_bet, 4),
            "t_statistic": round(self.t_statistic, 4),
            "final_bankroll": round(self.final_bankroll, 2),
            "calibration_on_bets": round(self.calibration_on_bets, 4),
            "verdict": self.verdict,
        }


def compute_metrics(
    ledger: pd.DataFrame, *, starting_bankroll: float, races_considered: int = 0, strategy: str = ""
) -> StrategyMetrics:
    """Metrics from a settled ledger."""
    metrics = StrategyMetrics(
        strategy=strategy,
        starting_bankroll=starting_bankroll,
        final_bankroll=starting_bankroll,
        peak_bankroll=starting_bankroll,
        races_considered=races_considered,
    )
    if ledger.empty:
        return metrics

    stakes = pd.to_numeric(ledger["stake"], errors="coerce").fillna(0.0)
    profits = pd.to_numeric(ledger["profit_loss"], errors="coerce").fillna(0.0)
    won = pd.to_numeric(ledger["won"], errors="coerce").fillna(0).astype(int)

    metrics.bets = len(ledger)
    metrics.wins = int(won.sum())
    metrics.losses = metrics.bets - metrics.wins
    metrics.strike_rate = metrics.wins / metrics.bets

    metrics.staked = float(stakes.sum())
    metrics.profit = float(profits.sum())
    metrics.returned = metrics.staked + metrics.profit
    metrics.roi = metrics.profit / metrics.staked if metrics.staked > 0 else 0.0

    gross_win = float(profits[profits > 0].sum())
    gross_loss = float(-profits[profits < 0].sum())
    metrics.profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    metrics.average_odds = float(pd.to_numeric(ledger["odds"], errors="coerce").mean())
    metrics.average_stake = float(stakes.mean())
    if "edge" in ledger:
        metrics.average_edge = float(pd.to_numeric(ledger["edge"], errors="coerce").mean())
    if "expected_value" in ledger:
        metrics.average_ev = float(pd.to_numeric(ledger["expected_value"], errors="coerce").mean())
    if "model_probability" in ledger:
        mean_probability = float(pd.to_numeric(ledger["model_probability"], errors="coerce").mean())
        metrics.calibration_on_bets = metrics.strike_rate / mean_probability if mean_probability > 0 else 0.0

    equity = [starting_bankroll, *pd.to_numeric(ledger["bankroll_after"], errors="coerce").tolist()]
    metrics.final_bankroll = float(equity[-1])

    peak = starting_bankroll
    for value in equity:
        peak = max(peak, value)
        drawdown = peak - value
        if drawdown > metrics.max_drawdown:
            metrics.max_drawdown = drawdown
            metrics.max_drawdown_pct = drawdown / peak if peak > 0 else 0.0
    metrics.peak_bankroll = peak

    # Return on stake, so bets of different sizes are comparable.
    returns = (profits / stakes.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) > 1:
        spread = float(returns.std(ddof=1))
        mean_return = float(returns.mean())
        if spread > 0:
            metrics.sharpe_per_bet = mean_return / spread
            metrics.roi_standard_error = spread / math.sqrt(len(returns))
            metrics.t_statistic = mean_return / metrics.roi_standard_error

    metrics.longest_losing_streak = _longest_losing_streak(won.tolist())
    metrics.bet_rate = metrics.bets / races_considered if races_considered else 0.0
    return metrics


def _longest_losing_streak(outcomes: list[int]) -> int:
    longest = current = 0
    for outcome in outcomes:
        current = 0 if outcome else current + 1
        longest = max(longest, current)
    return longest


# ---------------------------------------------------------------------------
# Curves and distributions
# ---------------------------------------------------------------------------
def equity_curve(ledger: pd.DataFrame, starting_bankroll: float) -> pd.DataFrame:
    """Bankroll after every settled bet, with running peak and drawdown."""
    if ledger.empty:
        return pd.DataFrame(columns=["bet_number", "race_date", "bankroll", "peak", "drawdown_pct"])

    bankroll = pd.to_numeric(ledger["bankroll_after"], errors="coerce").ffill()
    peak = bankroll.cummax().clip(lower=starting_bankroll)
    return pd.DataFrame(
        {
            "bet_number": range(1, len(ledger) + 1),
            "race_date": ledger["race_date"].to_numpy(),
            "bankroll": bankroll.to_numpy(),
            "peak": peak.to_numpy(),
            "drawdown_pct": ((peak - bankroll) / peak).to_numpy(),
        }
    )


def monthly_returns(ledger: pd.DataFrame) -> pd.DataFrame:
    """Profit, turnover and ROI by calendar month."""
    if ledger.empty:
        return pd.DataFrame(columns=["month", "bets", "staked", "profit", "roi"])

    frame = ledger.copy()
    frame["month"] = pd.to_datetime(frame["race_date"], errors="coerce").dt.to_period("M").astype(str)
    grouped = frame.groupby("month", sort=True).agg(
        bets=("stake", "size"),
        staked=("stake", "sum"),
        profit=("profit_loss", "sum"),
        wins=("won", "sum"),
    )
    grouped["roi"] = np.where(grouped["staked"] > 0, grouped["profit"] / grouped["staked"], 0.0)
    return grouped.reset_index()


def odds_distribution(
    ledger: pd.DataFrame, *, bins: tuple[float, ...] = (1, 2, 3, 5, 8, 13, 21, 1000)
) -> pd.DataFrame:
    """Bets, strike rate and ROI by price band.

    The most diagnostic table in the report. A strategy that makes all its money
    in one band is usually exploiting a calibration artefact in that band, not a
    general edge — and profits concentrated at long prices are the classic
    signature of an over-confident tail.
    """
    if ledger.empty:
        return pd.DataFrame(columns=["band", "bets", "strike_rate", "staked", "profit", "roi"])

    frame = ledger.copy()
    odds = pd.to_numeric(frame["odds"], errors="coerce")
    labels = [f"{bins[i]:g}-{bins[i + 1]:g}" for i in range(len(bins) - 1)]
    frame["band"] = pd.cut(odds, bins=list(bins), labels=labels, right=False)

    grouped = frame.groupby("band", observed=True, sort=True).agg(
        bets=("stake", "size"),
        wins=("won", "sum"),
        staked=("stake", "sum"),
        profit=("profit_loss", "sum"),
    )
    grouped["strike_rate"] = grouped["wins"] / grouped["bets"]
    grouped["roi"] = np.where(grouped["staked"] > 0, grouped["profit"] / grouped["staked"], 0.0)
    return grouped.reset_index()


def calibration_on_bets(ledger: pd.DataFrame, *, bins: int = 5) -> pd.DataFrame:
    """Did the model's confidence hold up on the bets it actually placed?

    Overall calibration can look fine while calibration *on selected bets* is
    poor — selection concentrates on exactly the runners the model is most
    optimistic about, which is where its errors live.
    """
    if ledger.empty or "model_probability" not in ledger:
        return pd.DataFrame(columns=["band", "bets", "mean_probability", "strike_rate", "ratio"])

    frame = ledger.copy()
    probability = pd.to_numeric(frame["model_probability"], errors="coerce")
    try:
        frame["band"] = pd.qcut(probability, q=bins, duplicates="drop")
    except ValueError:  # pragma: no cover - too few distinct values
        return pd.DataFrame(columns=["band", "bets", "mean_probability", "strike_rate", "ratio"])

    grouped = frame.groupby("band", observed=True, sort=True).agg(
        bets=("won", "size"), mean_probability=("model_probability", "mean"), wins=("won", "sum")
    )
    grouped["strike_rate"] = grouped["wins"] / grouped["bets"]
    grouped["ratio"] = grouped["strike_rate"] / grouped["mean_probability"]
    return grouped.reset_index().astype({"band": str})


def drawdown_periods(ledger: pd.DataFrame, starting_bankroll: float, *, top: int = 5) -> pd.DataFrame:
    """The worst peak-to-trough episodes, longest first."""
    curve = equity_curve(ledger, starting_bankroll)
    if curve.empty:
        return pd.DataFrame(columns=["start", "trough", "depth_pct", "bets"])

    episodes: list[dict[str, Any]] = []
    in_drawdown = False
    start_index = 0
    drawdowns = curve["drawdown_pct"].to_numpy()
    for index in range(len(curve)):
        if drawdowns[index] > 0 and not in_drawdown:
            in_drawdown, start_index = True, index
        elif drawdowns[index] == 0 and in_drawdown:
            window = curve.iloc[start_index : index + 1]
            episodes.append(
                {
                    "start": window["race_date"].iloc[0],
                    "trough": window["race_date"].to_numpy()[int(window["drawdown_pct"].to_numpy().argmax())],
                    "depth_pct": float(window["drawdown_pct"].max()),
                    "bets": len(window),
                }
            )
            in_drawdown = False
    if in_drawdown:
        window = curve.iloc[start_index:]
        episodes.append(
            {
                "start": window["race_date"].iloc[0],
                "trough": window["race_date"].to_numpy()[int(window["drawdown_pct"].to_numpy().argmax())],
                "depth_pct": float(window["drawdown_pct"].max()),
                "bets": len(window),
            }
        )

    frame = pd.DataFrame(episodes)
    return frame.nlargest(top, "depth_pct").reset_index(drop=True) if not frame.empty else frame


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class StrategyReport:
    """A full backtest report for one strategy."""

    metrics: StrategyMetrics
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_odds: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdowns: pd.DataFrame = field(default_factory=pd.DataFrame)
    rejections: dict[str, int] = field(default_factory=dict)
    period: tuple[date | None, date | None] = (None, None)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.as_dict(),
            "period": [str(self.period[0]), str(self.period[1])],
            "monthly": self.monthly.to_dict(orient="records"),
            "by_odds": self.by_odds.to_dict(orient="records"),
            "calibration_on_bets": self.calibration.to_dict(orient="records"),
            "rejections": self.rejections,
            "notes": self.notes,
        }

    def render(self, *, sparkline_width: int = 60) -> str:
        m = self.metrics
        lines = [
            f"Strategy: {m.strategy}",
            "=" * 72,
            f"  Period            {self.period[0]} → {self.period[1]}",
            f"  Races considered  {m.races_considered}",
            f"  Bets              {m.bets}  ({m.bet_rate:.1%} of races)",
            f"  Winners           {m.wins}",
            f"  Strike rate       {m.strike_rate:.2%}",
            f"  Average odds      {m.average_odds:.2f}",
            f"  Average stake     {m.average_stake:,.2f}",
            "",
            f"  Staked            {m.staked:,.2f}",
            f"  Profit / loss     {m.profit:+,.2f}",
            f"  ROI               {m.roi:+.2%}",
            f"  Profit factor     {m.profit_factor:.3f}",
            "",
            f"  Bankroll          {m.starting_bankroll:,.0f} → {m.final_bankroll:,.2f}",
            f"  Peak              {m.peak_bankroll:,.2f}",
            f"  Max drawdown      {m.max_drawdown:,.2f} ({m.max_drawdown_pct:.2%})",
            f"  Longest losing    {m.longest_losing_streak}",
            "",
            f"  Sharpe (per bet)  {m.sharpe_per_bet:+.4f}",
            f"  ROI std error     {m.roi_standard_error:.4f}",
            f"  t-statistic       {m.t_statistic:+.2f}",
            f"  Model calibration {m.calibration_on_bets:.3f}  (strike rate / mean predicted)",
            "",
            f"  VERDICT           {m.verdict}",
        ]

        if not self.equity.empty:
            lines += ["", "  Equity curve", "  " + _sparkline(self.equity["bankroll"], sparkline_width)]
        if not self.by_odds.empty:
            lines += ["", "  By price band", "    band        bets   strike     staked      profit      roi"]
            for row in self.by_odds.itertuples(index=False):
                lines.append(
                    f"    {row.band:<10} {row.bets:>5}  {row.strike_rate:>6.1%}  "
                    f"{row.staked:>10,.0f}  {row.profit:>+10,.0f}  {row.roi:>+7.1%}"
                )
        if not self.calibration.empty:
            lines += ["", "  Calibration on placed bets", "    predicted   actual   ratio   bets"]
            for row in self.calibration.itertuples(index=False):
                lines.append(
                    f"    {row.mean_probability:>9.1%}  {row.strike_rate:>7.1%}  "
                    f"{row.ratio:>6.2f}  {row.bets:>5}"
                )
        if self.rejections:
            lines += ["", "  Why runners were not backed"]
            for reason, count in list(self.rejections.items())[:8]:
                lines.append(f"    {reason:<26} {count:>8}")
        for note in self.notes:
            lines.append(f"\n  note: {note}")
        return "\n".join(lines)


def _sparkline(series: pd.Series, width: int = 60) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if values.size < 2:
        return "(not enough data)"
    blocks = "▁▂▃▄▅▆▇█"
    step = max(1, values.size // width)
    sampled = values[::step][:width]
    low, high = sampled.min(), sampled.max()
    if high <= low:
        return blocks[0] * len(sampled)
    scaled = ((sampled - low) / (high - low) * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[index] for index in scaled) + f"  [{low:,.0f} … {high:,.0f}]"


def build_report(run: Any) -> StrategyReport:
    """Assemble the full report from a :class:`~backend.strategy.backtester.StrategyRun`."""
    ledger = run.ledger
    bankroll = run.config.starting_bankroll
    metrics = compute_metrics(
        ledger,
        starting_bankroll=bankroll,
        races_considered=run.races_considered,
        strategy=run.config.strategy.name,
    )
    dates = (
        pd.to_datetime(ledger["race_date"], errors="coerce").dropna()
        if not ledger.empty
        else pd.Series(dtype="datetime64[ns]")
    )
    period = (dates.min().date(), dates.max().date()) if not dates.empty else (None, None)
    return StrategyReport(
        metrics=metrics,
        equity=equity_curve(ledger, bankroll),
        monthly=monthly_returns(ledger),
        by_odds=odds_distribution(ledger),
        calibration=calibration_on_bets(ledger),
        drawdowns=drawdown_periods(ledger, bankroll),
        rejections=run.rejections,
        period=period,
        notes=list(run.notes),
    )


def comparison_table(reports: list[StrategyReport]) -> pd.DataFrame:
    """Side-by-side comparison, best ROI first."""
    rows = [
        {
            "strategy": report.metrics.strategy,
            "bets": report.metrics.bets,
            "strike_rate": round(report.metrics.strike_rate, 4),
            "avg_odds": round(report.metrics.average_odds, 2),
            "roi": round(report.metrics.roi, 4),
            "profit": round(report.metrics.profit, 2),
            "profit_factor": round(report.metrics.profit_factor, 3),
            "max_dd": round(report.metrics.max_drawdown_pct, 4),
            "sharpe": round(report.metrics.sharpe_per_bet, 4),
            "t_stat": round(report.metrics.t_statistic, 2),
            "verdict": report.metrics.verdict,
        }
        for report in reports
    ]
    return pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)


__all__ = [
    "MIN_BETS_FOR_INFERENCE",
    "StrategyMetrics",
    "StrategyReport",
    "build_report",
    "calibration_on_bets",
    "comparison_table",
    "compute_metrics",
    "drawdown_periods",
    "equity_curve",
    "monthly_returns",
    "odds_distribution",
]
