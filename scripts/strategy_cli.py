#!/usr/bin/env python
"""Value-betting strategy CLI.

    python -m scripts.strategy_cli predictions --from 2020-01-01 --to 2026-06-30
    python -m scripts.strategy_cli backtest --strategy A_ev5
    python -m scripts.strategy_cli compare
    python -m scripts.strategy_cli market-dependency
    python -m scripts.strategy_cli sensitivity
    python -m scripts.strategy_cli signals --date 2026-06-30

``predictions`` runs the expensive walk-forward retraining once and caches the
result; everything else replays that cache, so strategies are compared against
identical model output rather than against separately retrained models.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.database.session import session_scope
from backend.strategy import (
    STRATEGY_LIBRARY,
    BacktestConfig,
    ExecutionConfig,
    StakingConfig,
    StrategyBacktester,
    WalkForwardPredictor,
    build_report,
    comparison_table,
)
from backend.strategy.filters import STRATEGY_A
from backend.strategy.odds import build_market_frame
from backend.utils.config import get_settings
from backend.utils.logging import configure_logging

app = typer.Typer(
    add_completion=False,
    # Typer renders local variables in tracebacks by default, which prints the
    # unwrapped database password and API credentials straight to the console.
    # SecretStr masks repr(); it cannot mask a frame local.
    pretty_exceptions_show_locals=False,
    help="Value betting: signals, backtests, analysis",
)
console = Console()

MARKET_PREFIXES = ("mkt_", "market_")
CACHE_NAME = "walk_forward_predictions.parquet"
NO_MARKET_CACHE = "walk_forward_predictions_no_market.parquet"


@app.callback()
def _setup() -> None:
    configure_logging()


def _cache_path(name: str = CACHE_NAME) -> Path:
    return get_settings().data_dir / "processed" / name


def _load_cache(name: str = CACHE_NAME) -> pd.DataFrame:
    path = _cache_path(name)
    if not path.exists():
        console.print(f"[red]no cached predictions at {path}[/red]")
        console.print("run: [bold]python -m scripts.strategy_cli predictions[/bold] first")
        raise typer.Exit(code=1)
    return pd.read_parquet(path)


def _parse(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


@app.command()
def predictions(
    history_start: str = typer.Option("2018-01-01", "--history-start"),
    date_from: str = typer.Option("2020-01-01", "--from", help="First date to predict"),
    date_to: str = typer.Option("2026-06-30", "--to"),
    retrain_months: int = typer.Option(12, help="How often to refit"),
    model: str = typer.Option("lightgbm"),
    no_market: bool = typer.Option(False, "--no-market", help="Train without market features"),
) -> None:
    """Generate walk-forward predictions and cache them."""
    with session_scope() as session:
        predictor = WalkForwardPredictor(
            session,
            model_type=model,
            drop_feature_prefixes=MARKET_PREFIXES if no_market else (),
        )
        frame = predictor.run(
            history_start=_parse(history_start),
            test_start=_parse(date_from),
            test_end=_parse(date_to),
            retrain_months=retrain_months,
        )

    for window in predictor.windows:
        console.print(f"  {window.describe()}")

    path = _cache_path(NO_MARKET_CACHE if no_market else CACHE_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    console.print(f"\n[green]{len(frame)} runners / {frame['race_id'].nunique()} races → {path}[/green]")


def _config(
    strategy_name: str, staking: str, stake: float, slippage: float, commission: float
) -> BacktestConfig:
    if strategy_name not in STRATEGY_LIBRARY:
        console.print(f"[red]unknown strategy {strategy_name!r}; have {sorted(STRATEGY_LIBRARY)}[/red]")
        raise typer.Exit(code=2)
    return BacktestConfig(
        strategy=STRATEGY_LIBRARY[strategy_name],
        staking=StakingConfig(method=staking, flat_stake=stake),  # type: ignore[arg-type]
        execution=ExecutionConfig(slippage=slippage, commission=commission),
        starting_bankroll=get_settings().starting_bankroll,
    )


@app.command()
def backtest(
    strategy: str = typer.Option("A_ev5"),
    staking: str = typer.Option("flat", help="flat | kelly | fixed_fraction"),
    stake: float = typer.Option(10.0),
    slippage: float = typer.Option(0.02),
    commission: float = typer.Option(0.0),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Backtest one strategy over the cached predictions."""
    run = StrategyBacktester(_config(strategy, staking, stake, slippage, commission)).run(_load_cache())
    report = build_report(run)

    if as_json:
        console.print_json(json.dumps(report.as_dict()))
    else:
        console.print(report.render())


@app.command()
def compare(
    staking: str = typer.Option("flat"),
    stake: float = typer.Option(10.0),
    slippage: float = typer.Option(0.02),
    commission: float = typer.Option(0.0),
) -> None:
    """Compare every strategy in the library over identical predictions."""
    frame = _load_cache()
    reports = [
        build_report(StrategyBacktester(_config(name, staking, stake, slippage, commission)).run(frame))
        for name in STRATEGY_LIBRARY
    ]
    console.print(comparison_table(reports).to_string(index=False))


@app.command("market-dependency")
def market_dependency(
    slippage: float = typer.Option(0.02),
    commission: float = typer.Option(0.0),
) -> None:
    """Does the model add anything the market does not already know?

    Three experiments. The market-only baseline is the negative control: it bets
    the market's own fair probability, so it must find nothing.
    """
    full = _load_cache()
    config = lambda: BacktestConfig(  # noqa: E731 - a tiny factory reads better here
        strategy=STRATEGY_A,
        staking=StakingConfig(method="flat", flat_stake=10.0),
        execution=ExecutionConfig(slippage=slippage, commission=commission),
    )

    experiments: list[tuple[str, pd.DataFrame]] = [("1_full_model", full)]

    no_market_path = _cache_path(NO_MARKET_CACHE)
    if no_market_path.exists():
        experiments.append(("2_no_market_model", pd.read_parquet(no_market_path)))
    else:
        console.print(
            "[yellow]no market-free predictions cached — run "
            "`strategy_cli predictions --no-market` for experiment 2[/yellow]\n"
        )

    market_only = build_market_frame(full).copy()
    market_only["model_probability"] = market_only["market_probability"]
    experiments.append(("3_market_only", market_only))

    reports = []
    for label, frame in experiments:
        report = build_report(StrategyBacktester(config()).run(frame))
        report.metrics.strategy = label
        reports.append(report)

    console.print(comparison_table(reports).to_string(index=False))
    console.print(
        "\n[bold]Reading this table:[/bold] experiment 3 must show no profitable bets — "
        "it is the negative control. If experiment 2 loses while experiment 1 wins, the "
        "edge depends on having the market as an input rather than on independent alpha."
    )


@app.command()
def sensitivity(strategy: str = typer.Option("A_ev5")) -> None:
    """How fast does the edge disappear under realistic costs?"""
    frame = _load_cache()
    rows = []
    for slip, comm in ((0.0, 0.0), (0.02, 0.0), (0.05, 0.0), (0.02, 0.05), (0.05, 0.05), (0.10, 0.05)):
        metrics = build_report(
            StrategyBacktester(_config(strategy, "flat", 10.0, slip, comm)).run(frame)
        ).metrics
        rows.append(
            {
                "slippage": f"{slip:.0%}",
                "commission": f"{comm:.0%}",
                "bets": metrics.bets,
                "roi": round(metrics.roi, 4),
                "profit": round(metrics.profit, 0),
                "t_stat": round(metrics.t_statistic, 2),
                "verdict": metrics.verdict,
            }
        )
    console.print(pd.DataFrame(rows).to_string(index=False))


@app.command()
def signals(
    race_date: str = typer.Option(None, "--date", help="YYYY-MM-DD"),
    race_id: str = typer.Option(None, "--race-id"),
    strategy: str = typer.Option("A_ev5"),
) -> None:
    """Show the value-bet signals for a race or a day, with reasons."""
    from backend.strategy.filters import select_bets

    frame = _load_cache()
    if race_id:
        frame = frame[frame["race_id"] == race_id]
    elif race_date:
        frame = frame[pd.to_datetime(frame["race_date"]).dt.date == _parse(race_date)]
    else:
        console.print("[red]give either --date or --race-id[/red]")
        raise typer.Exit(code=2)

    if frame.empty:
        console.print("[yellow]nothing found in the cached predictions[/yellow]")
        raise typer.Exit(code=1)

    prepared = StrategyBacktester(_config(strategy, "flat", 10.0, 0.02, 0.0)).prepare(frame)
    for current_race, group in prepared.groupby("race_id", sort=False):
        console.print(f"\n[bold]{current_race}[/bold]  {group['race_date'].iloc[0]}")
        for signal in select_bets(group, STRATEGY_LIBRARY[strategy]).signals:
            console.print(signal.render())


if __name__ == "__main__":
    app()
