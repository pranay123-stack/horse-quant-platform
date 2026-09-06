#!/usr/bin/env python
"""The product: train, predict, serve.

    python -m scripts.predict_cli train           # historical data -> production model
    python -m scripts.predict_cli today           # score today and print the card
    python -m scripts.predict_cli daily           # the morning job (fetch, score, store)
    python -m scripts.predict_cli performance     # the settled record
    python -m scripts.predict_cli replay          # live == backtest?
    python -m scripts.predict_cli status          # which model is live

``daily`` is the one that runs on a schedule. Everything else is for a human.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_pipeline.importer import import_racecards_for_date
from backend.database.session import session_scope
from backend.models.predictions import Recommendation
from backend.prediction_service import PredictionService, store_predictions
from backend.prediction_service.daily import run_daily_job
from backend.prediction_service.performance import summarise_performance
from backend.prediction_service.replay import replay_predictions
from backend.prediction_service.rules import RULES
from backend.prediction_service.training import load_manifest, train_production_model
from backend.services.racing_api import RacingAPIClient
from backend.utils.config import get_settings
from backend.utils.exceptions import PlatformError
from backend.utils.logging import configure_logging
from backend.utils.timeutils import today_uk

app = typer.Typer(
    add_completion=False,
    # Typer prints frame locals on an unhandled exception, which would put the
    # database password on the console. SecretStr cannot mask a frame local.
    pretty_exceptions_show_locals=False,
    help="Horse racing predictions: train, predict, serve",
)
console = Console()


@app.callback()
def _setup() -> None:
    configure_logging()


def _render_day(day, *, bets_only: bool = False) -> None:
    if not day.races:
        console.print(f"[yellow]{day.note or 'no races to score'}[/yellow]")
        return

    table = Table(title=f"Predictions — {day.race_date}", header_style="bold")
    table.add_column("Race")
    table.add_column("Horse")
    table.add_column("Prob", justify="right")
    table.add_column("Odds", justify="right")
    table.add_column("EV", justify="right")
    table.add_column("Recommendation")

    shown = 0
    for race in day.races:
        runners = race.bets if bets_only else race.runners
        for runner in runners:
            is_bet = runner.recommendation == Recommendation.BET
            table.add_row(
                race.race,
                runner.horse or runner.horse_id,
                f"{runner.model_probability:.0%}",
                f"{runner.odds:.2f}" if runner.odds else "—",
                f"{runner.expected_value:+.0%}" if runner.expected_value is not None else "—",
                "[bold green]BET[/bold green]" if is_bet else "[dim]NO_BET[/dim]",
            )
            shown += 1

    if shown == 0:
        console.print("[yellow]no runners met the betting rules today[/yellow]")
        console.print(f"[dim]{RULES.describe()}[/dim]")
        return

    console.print(table)
    console.print(
        f"\n[bold]{len(day.bets)}[/bold] recommended bet(s) from "
        f"{day.total_runners} runners across {len(day.races)} races"
    )


@app.command()
def train(
    train_start: str = typer.Option("2018-01-01"),
    train_end: str = typer.Option("2023-12-31"),
    test_start: str = typer.Option("2024-01-01"),
    test_end: str = typer.Option(None, help="Defaults to yesterday"),
) -> None:
    """Train the production model from historical data and promote the best."""
    settings = get_settings()
    end = date.fromisoformat(test_end) if test_end else today_uk() - timedelta(days=1)

    with session_scope() as session:
        try:
            model = train_production_model(
                session,
                train_start=date.fromisoformat(train_start),
                train_end=date.fromisoformat(train_end),
                test_start=date.fromisoformat(test_start),
                test_end=end,
                models_dir=settings.models_dir,
            )
        except PlatformError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    console.print(model.render())
    console.print(f"\n[green]manifest written to {model.path}[/green]")


@app.command()
def today(
    race_date: str = typer.Option(None, help="Defaults to today (UK)"),
    bets_only: bool = typer.Option(False, "--bets-only"),
    store: bool = typer.Option(False, "--store", help="Persist the result"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Score a day's races and print the card."""
    target = date.fromisoformat(race_date) if race_date else None

    with session_scope() as session:
        service = PredictionService(session)
        day = service.signals_for_date(target)
        if store and day.races:
            store_predictions(session, day)

        if as_json:
            console.print_json(json.dumps(day.as_dict()))
        else:
            _render_day(day, bets_only=bets_only)


@app.command()
def daily(
    race_date: str = typer.Option(None, help="Defaults to today (UK)"),
    fetch: bool = typer.Option(True, "--fetch/--no-fetch", help="Import today's racecards first"),
) -> None:
    """The morning job: fetch today's card, score it, store the signals."""
    target = date.fromisoformat(race_date) if race_date else today_uk()

    def do_fetch() -> int:
        async def run() -> int:
            async with RacingAPIClient() as client, session_scope() as fetch_session:
                summary = await import_racecards_for_date(
                    fetch_session, client, race_date=target.isoformat(), tier="pro"
                )
                return summary.races_created + summary.races_updated

        return asyncio.run(run())

    with session_scope() as session:
        result, day = run_daily_job(
            session,
            race_date=target,
            import_racecards=do_fetch if fetch else None,
        )

    console.print(result.render())
    console.print()
    _render_day(day, bets_only=True)
    raise typer.Exit(code=0 if result.succeeded else 1)


@app.command()
def performance(
    date_from: str = typer.Option(None),
    date_to: str = typer.Option(None),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """How the recommendations actually did, on settled bets only."""
    with session_scope() as session:
        summary = summarise_performance(
            session,
            date_from=date.fromisoformat(date_from) if date_from else None,
            date_to=date.fromisoformat(date_to) if date_to else None,
        )

    if as_json:
        console.print_json(json.dumps(summary.as_dict()))
        return

    console.print(f"[bold]{summary.headline}[/bold]\n")
    table = Table(header_style="bold")
    table.add_column("Measure")
    table.add_column("Value", justify="right")
    for label, value in (
        ("Bets recommended", f"{summary.bets:,}"),
        ("Settled", f"{summary.settled:,}"),
        ("Awaiting result", f"{summary.pending:,}"),
        ("Wins", f"{summary.wins:,}"),
        ("Win rate", f"{summary.strike_rate:.1%}" if summary.settled else "—"),
        ("Average price", f"{summary.average_odds:.2f}" if summary.settled else "—"),
        ("Staked", f"{summary.staked:,.0f}"),
        ("Profit / loss", f"{summary.profit:+,.0f}"),
        ("ROI", f"{summary.roi:+.2%}" if summary.settled else "—"),
        ("Max drawdown", f"{summary.max_drawdown:,.0f}"),
        ("t-statistic", f"{summary.t_statistic:+.2f}"),
    ):
        table.add_row(label, value)
    console.print(table)


@app.command()
def replay(race_date: str = typer.Argument(..., help="A past date to re-score")) -> None:
    """Does the live path produce the same numbers as the backtest path?"""
    with session_scope() as session:
        result = replay_predictions(session, date.fromisoformat(race_date))

    console.print(result.render())
    if result.runners_checked == 0:
        console.print("[yellow]nothing to compare — no races on that date[/yellow]")
        raise typer.Exit(code=1)
    if result.consistent:
        console.print("\n[green]CONSISTENT — the product scores exactly as the backtest did[/green]")
        raise typer.Exit(code=0)
    console.print("\n[red]DIVERGED — the backtest is no longer evidence about the product[/red]")
    raise typer.Exit(code=1)


@app.command()
def status() -> None:
    """Which model is live, and what the betting rules are."""
    settings = get_settings()
    manifest = load_manifest(settings.models_dir)

    if manifest is None:
        console.print("[yellow]no production model — run `predict_cli train` first[/yellow]")
        console.print(f"[dim]{RULES.describe()}[/dim]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Model[/bold]        {manifest['model']}:{manifest['version']}")
    console.print(f"[bold]Trained[/bold]      {manifest['trained_at'][:19]}")
    console.print(f"[bold]Features[/bold]     {manifest['feature_version']}")
    console.print(f"[bold]Dataset[/bold]      {manifest['dataset_hash']}")
    console.print(f"[bold]Window[/bold]       {manifest['training_window']}")
    console.print(f"[bold]Rows[/bold]         {manifest['rows']:,}")
    console.print(f"\n[bold]Rules[/bold]        {RULES.describe()}")


if __name__ == "__main__":
    app()
