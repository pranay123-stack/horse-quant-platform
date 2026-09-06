#!/usr/bin/env python
"""Operational CLI for the Racing API integration.

    python -m scripts.racing_cli check                       # verify credentials
    python -m scripts.racing_cli racecards --date 2026-08-10 # import today's card
    python -m scripts.racing_cli results --start 2026-08-01 --end 2026-08-07
    python -m scripts.racing_cli backfill --start 2025-08-01 --end 2026-08-01

``check`` is the first thing to run after setting credentials: it distinguishes
"wrong password" from "subscription inactive", which the raw 401 does not.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_pipeline.importer import (
    backfill_day_by_day,
    import_racecards_for_date,
    import_results_for_range,
)
from backend.database.session import session_scope
from backend.services.racing_api import RacingAPIClient
from backend.services.racing_api.authentication import credentials_from_settings
from backend.utils.config import get_settings
from backend.utils.logging import configure_logging
from backend.utils.timeutils import format_date, today_uk

app = typer.Typer(
    add_completion=False,
    # Typer renders local variables in tracebacks by default, which prints the
    # unwrapped database password and API credentials straight to the console.
    # SecretStr masks repr(); it cannot mask a frame local.
    pretty_exceptions_show_locals=False,
    help="Racing API operations",
)
console = Console()


@app.callback()
def _setup() -> None:
    configure_logging()


@app.command()
def check() -> None:
    """Verify connectivity and report precisely why it does or does not work."""
    settings = get_settings()
    credentials = credentials_from_settings(settings)

    console.print("[bold]Racing API check[/bold]")
    console.print(f"  base url : {settings.racing_api_base_url}")
    console.print(f"  username : {credentials.masked_username}")

    if not credentials.is_complete:
        console.print("[red]  credentials are not configured -- set them in .env[/red]")
        raise typer.Exit(code=2)

    async def run() -> dict:
        async with RacingAPIClient(settings) as client:
            return await client.check_connectivity()

    verdict = asyncio.run(run())

    if verdict["authorised"]:
        console.print("[green]  OK -- credentials valid and subscription active[/green]")
        raise typer.Exit(code=0)

    console.print(f"[red]  FAILED -- {verdict['message']}[/red]")
    if verdict.get("error_code") == "racing_api_subscription_inactive":
        console.print(
            "[yellow]  The credentials are recognised; the plan is not active.\n"
            "  Activate a subscription at https://www.theracingapi.com[/yellow]"
        )
    for key, value in verdict.get("details", {}).items():
        console.print(f"    {key}: {value}")
    raise typer.Exit(code=1)


def _render(summary) -> None:
    table = Table(title="Import summary", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    data = summary.as_dict()
    table.add_row("Races seen", str(data["races"]["seen"]))
    table.add_row("Imported", str(data["races"]["created"]))
    table.add_row("Updated", str(data["races"]["updated"]))
    table.add_row("Skipped (duplicates)", str(data["races"]["skipped_duplicates"]))
    table.add_row("Runners", f"{data['runners']['created']} new")
    table.add_row("Results", f"{data['results']['created']} new")
    table.add_row("Odds quotes", f"{data['odds']['created']} new")
    table.add_row("API requests", str(data["api"]["requests"]))
    table.add_row("Failed", str(data["failed"]))
    console.print(table)

    for failure in summary.failures[:10]:
        console.print(f"[red]  {failure.record_id}: {failure.error_type} — {failure.message[:120]}[/red]")


@app.command()
def racecards(
    date: str = typer.Option(None, help="YYYY-MM-DD (defaults to today, UK time)"),
    tier: str = typer.Option("standard", help="free | basic | standard | pro (pro includes odds)"),
    regions: str = typer.Option("gb,ire", help="Comma-separated region codes"),
    as_json: bool = typer.Option(False, "--json", help="Emit the summary as JSON"),
) -> None:
    """Import racecards for one day."""
    race_date = date or format_date(today_uk())
    region_list = [r.strip() for r in regions.split(",") if r.strip()]

    async def run():
        async with RacingAPIClient() as client, session_scope() as session:
            return await import_racecards_for_date(
                session, client, race_date=race_date, regions=region_list, tier=tier
            )

    summary = asyncio.run(run())
    console.print_json(json.dumps(summary.as_dict())) if as_json else _render(summary)
    raise typer.Exit(code=1 if summary.failed else 0)


@app.command()
def results(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    regions: str = typer.Option("gb,ire"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Import finished races for a date range."""
    region_list = [r.strip() for r in regions.split(",") if r.strip()]

    async def run():
        async with RacingAPIClient() as client, session_scope() as session:
            return await import_results_for_range(
                session, client, start_date=start, end_date=end, regions=region_list
            )

    summary = asyncio.run(run())
    console.print_json(json.dumps(summary.as_dict())) if as_json else _render(summary)
    raise typer.Exit(code=1 if summary.failed else 0)


@app.command()
def backfill(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    regions: str = typer.Option("gb,ire"),
) -> None:
    """Backfill results one day at a time (resilient; slower than --results)."""
    region_list = [r.strip() for r in regions.split(",") if r.strip()]

    async def run():
        async with RacingAPIClient() as client, session_scope() as session:
            return await backfill_day_by_day(
                session, client, start_date=start, end_date=end, regions=region_list
            )

    summary = asyncio.run(run())
    _render(summary)
    raise typer.Exit(code=1 if summary.failed else 0)


if __name__ == "__main__":
    app()
