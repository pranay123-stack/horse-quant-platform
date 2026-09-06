#!/usr/bin/env python
"""Phase 6: real-data validation.

The whole phase, in the order it must be run:

    python -m scripts.phase6_cli protocol          # show the frozen analysis plan
    python -m scripts.phase6_cli api-status        # auth, subscription, endpoints, range
    python -m scripts.phase6_cli phase6-preflight  # READY FOR VALIDATION / BLOCKED
    python -m scripts.phase6_cli phase6-run        # the whole workflow, resumable
    python -m scripts.phase6_cli experiments       # the research run ledger

``phase6-run`` is the one button: ingest → audit → features → train → predict →
backtest → report, checkpointed after every step so an interruption resumes
rather than restarting. It stops at the first pre-flight gate that fails —
inactive subscription, insufficient history, synthetic data, an edited protocol,
missing odds coverage, post-off timestamps, or detected leakage.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_pipeline.backfill import (
    CHECKPOINT_FILENAME,
    backfill_history,
    load_progress,
    month_windows,
    recent_runs,
)
from backend.database.session import session_scope
from backend.research.audit import audit_dataset
from backend.research.audit import write_report as write_audit_report
from backend.research.client_report import REPORT_FILENAME as CLIENT_REPORT_FILENAME
from backend.research.client_report import generate_client_report
from backend.research.execution import CHECKPOINT_FILENAME as EXECUTION_CHECKPOINT
from backend.research.execution import (
    ExecutionResult,
    protocol_window,
    recent_research_runs,
    run_workflow,
)
from backend.research.preflight import full_preflight
from backend.research.protocol import PROTOCOL
from backend.research.readiness import REPORT_FILENAME as READINESS_FILENAME
from backend.research.readiness import assess as assess_readiness
from backend.research.readiness import write_report as write_readiness_report
from backend.research.validation_report import (
    REPORT_FILENAME,
    VERDICT_LABELS,
    render_markdown,
    run_validation,
    write_report,
)
from backend.services.racing_api import RacingAPIClient
from backend.services.racing_api.authentication import credentials_from_settings
from backend.services.racing_api.capabilities import ApiCapabilities, probe_capabilities
from backend.utils.config import get_settings
from backend.utils.exceptions import PipelineError
from backend.utils.logging import configure_logging

app = typer.Typer(
    add_completion=False,
    # Typer renders local variables in tracebacks by default, which prints the
    # unwrapped database password and API credentials straight to the console.
    # SecretStr masks repr(); it cannot mask a frame local.
    pretty_exceptions_show_locals=False,
    help="Phase 6 — real data validation",
)
console = Console()

AUDIT_FILENAME = "REAL_DATA_AUDIT.md"

#: The fingerprint the analysis plan was registered with. Pinned here so an
#: edit to the protocol fails the run loudly instead of silently changing what
#: the study tests.
REGISTERED_FINGERPRINT = "76e18ddb6f11f6ad"


@app.callback()
def _setup() -> None:
    configure_logging()


def _reports_dir() -> Path:
    return get_settings().reports_dir


@app.command()
def protocol() -> None:
    """Print the pre-registered analysis plan."""
    console.print(PROTOCOL.render())
    console.print(
        "\n[dim]This plan was fixed before any real data existed. Editing it after "
        "results are seen changes the fingerprint and invalidates the run.[/dim]"
    )


@app.command(name="api-status")
def api_status(
    as_json: bool = typer.Option(False, "--json"),
    history_from: str = typer.Option(
        None, help="Probe history availability from this date (default: the protocol's train start)"
    ),
) -> None:
    """Authentication, subscription and per-endpoint access.

    Exits 0 only when the plan can actually support the backfill: subscription
    active, results readable, and history reaching the protocol's start.
    """
    settings = get_settings()
    credentials = credentials_from_settings(settings)

    if not credentials.is_complete:
        console.print("[red]credentials are not configured — set them in .env[/red]")
        raise typer.Exit(code=2)

    async def probe() -> ApiCapabilities:
        kwargs = {"history_start": date.fromisoformat(history_from)} if history_from else {}
        async with RacingAPIClient(settings) as client:
            return await probe_capabilities(client, **kwargs)

    capabilities = asyncio.run(probe())

    if as_json:
        console.print_json(json.dumps(capabilities.as_dict()))
    else:
        console.print(capabilities.render())
        if not capabilities.subscription_active:
            console.print(
                "\n[yellow]The credentials are valid; the plan is not active.\n"
                "Phase 6 cannot start until a subscription is enabled at\n"
                "https://www.theracingapi.com[/yellow]"
            )

    raise typer.Exit(code=0 if capabilities.ready_for_backfill else 1)


@app.command()
def check() -> None:
    """Deprecated alias for ``api-status``."""
    console.print("[dim]'check' is now 'api-status'[/dim]")
    api_status()


@app.command()
def ingest(
    start: str = typer.Option("2018-01-01", help="First month to import"),
    end: str = typer.Option(None, help="Last month (default: today)"),
    regions: str = typer.Option("gb,ire"),
    resume: bool = typer.Option(True, help="Continue from the checkpoint"),
) -> None:
    """Backfill historical results. Resumable — safe to interrupt and re-run."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end) if end else date.today()
    region_list = tuple(part.strip() for part in regions.split(",") if part.strip())

    checkpoint = get_settings().data_dir / CHECKPOINT_FILENAME
    if not resume and checkpoint.exists():
        checkpoint.unlink()

    windows = month_windows(start_date, end_date)
    console.print(f"[bold]Backfill[/bold] {start_date} → {end_date}  ({len(windows)} months)")
    console.print(f"  checkpoint: {checkpoint}")

    async def run() -> None:
        async with RacingAPIClient() as client, session_scope() as session:
            progress = await backfill_history(
                session,
                client,
                start=start_date,
                end=end_date,
                checkpoint=checkpoint,
                regions=region_list,
            )
            console.print("\n" + progress.render(len(windows)))

    try:
        asyncio.run(run())
    except PipelineError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def progress() -> None:
    """How far the backfill got."""
    checkpoint = get_settings().data_dir / CHECKPOINT_FILENAME
    if not checkpoint.exists():
        console.print("[yellow]no backfill has been started[/yellow]")
        raise typer.Exit(code=1)

    state = load_progress(checkpoint)
    console.print(state.render(len(state.completed) + len(state.failed)))
    if state.failed:
        console.print("\n[red]failed months:[/red]")
        for month, error in list(state.failed.items())[:10]:
            console.print(f"  {month}  {error}")


@app.command()
def runs(limit: int = typer.Option(10, help="How many runs to show")) -> None:
    """The backfill audit trail from ``backfill_runs``."""
    with session_scope() as session:
        rows = [run.as_dict() for run in recent_runs(session, limit=limit)]

    if not rows:
        console.print("[yellow]no backfill has been recorded[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Backfill runs", header_style="bold")
    for column in ("run id", "window", "status", "records", "failed", "months", "started"):
        table.add_column(column, overflow="fold")
    for row in rows:
        status = row["status"]
        colour = {"completed": "green", "partial": "yellow"}.get(status, "red")
        table.add_row(
            row["run_id"],
            f"{row['start_date']} → {row['end_date']}",
            f"[{colour}]{status}[/{colour}]",
            f"{row['records_processed']:,}",
            f"{row['failed_records']:,}",
            row["months"],
            (row["started_at"] or "")[:19],
        )
    console.print(table)


@app.command()
def audit(
    out: str = typer.Option(None, help=f"Output path (default: reports/{AUDIT_FILENAME})"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Audit the dataset and write ``REAL_DATA_AUDIT.md``."""
    with session_scope() as session:
        result = audit_dataset(session)

    path = Path(out) if out else _reports_dir() / AUDIT_FILENAME
    write_audit_report(result, path)

    if as_json:
        console.print_json(json.dumps(result.as_dict()))
    else:
        console.print(f"[bold]Source[/bold]: {result.source}")
        console.print(
            f"[bold]Races[/bold]: {result.races:,}   [bold]Span[/bold]: {result.date_from} → {result.date_to}"
        )
        console.print(f"[bold]Odds coverage[/bold]: {result.odds_coverage:.1%}")
        status = "[green]READY[/green]" if result.readiness.ready else "[red]NOT READY[/red]"
        console.print(f"[bold]Readiness[/bold]: {status}")
        for reason in result.readiness.blocking:
            console.print(f"  [red]blocking:[/red] {reason}")
        for warning in result.readiness.warnings:
            console.print(f"  [yellow]warning :[/yellow] {warning}")

    console.print(f"\n[green]wrote {path}[/green]")
    raise typer.Exit(code=0 if result.readiness.ready else 1)


async def _probe() -> ApiCapabilities | None:
    """Probe the API, or return None if credentials are not configured."""
    settings = get_settings()
    if not credentials_from_settings(settings).is_complete:
        return None
    async with RacingAPIClient(settings) as client:
        return await probe_capabilities(client)


@app.command(name="phase6-preflight")
def phase6_preflight(
    offline: bool = typer.Option(False, "--offline", help="Skip the API probe"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Is everything in place to run the validation?

    Prints READY FOR VALIDATION, or BLOCKED with the single reason that stopped
    it. Exits 0 only when ready, so it composes into a shell chain.
    """
    capabilities = None if offline else asyncio.run(_probe())

    with session_scope() as session:
        audit = audit_dataset(session)

    gates = full_preflight(
        audit,
        PROTOCOL,
        capabilities=capabilities,
        expected_fingerprint=REGISTERED_FINGERPRINT,
    )

    if as_json:
        console.print_json(json.dumps(gates.as_dict()))
    else:
        console.print(gates.render())
        console.print()
        if gates.passed:
            console.print("[bold green]READY FOR VALIDATION[/bold green]")
        else:
            console.print("[bold red]BLOCKED[/bold red]")
            console.print(f"  Reason: {gates.reason}")

    raise typer.Exit(code=0 if gates.passed else 1)


@app.command(name="phase6-run")
def phase6_run(
    skip_ingest: bool = typer.Option(False, "--skip-ingest", help="Use the data already in the database"),
    resume: bool = typer.Option(True, "--resume/--restart", help="Continue from the checkpoint"),
    allow_unready: bool = typer.Option(
        False, "--allow-unready", help="Exercise the workflow on data that failed a gate"
    ),
    pin_protocol: bool = typer.Option(True, "--pin-protocol/--no-pin-protocol"),
    retrain_months: int = typer.Option(12),
    regions: str = typer.Option("gb,ire"),
) -> None:
    """The one button: real data in, unbiased validation report out.

    Seven steps, each checkpointed. Safe to interrupt and re-run — it picks up
    at the step that stopped rather than repeating an eight-hour download.
    """
    settings = get_settings()
    workspace = settings.data_dir
    reports = _reports_dir()
    start, end = protocol_window(PROTOCOL)
    region_list = tuple(part.strip() for part in regions.split(",") if part.strip())

    def do_ingest() -> str:
        if skip_ingest:
            return "skipped (--skip-ingest)"

        async def run() -> str:
            checkpoint = workspace / CHECKPOINT_FILENAME
            async with RacingAPIClient() as client, session_scope() as ingest_session:
                progress = await backfill_history(
                    ingest_session,
                    client,
                    start=start,
                    end=min(end, date.today()),
                    checkpoint=checkpoint,
                    regions=region_list,
                )
                return (
                    f"{progress.races_imported:,} races over "
                    f"{len(progress.completed)} months ({len(progress.failed)} failed)"
                )

        return asyncio.run(run())

    def extra_reports(result: ExecutionResult) -> str:
        """The client PDF and the readiness checklist, once a verdict exists."""
        validation = result.validation
        if validation is None:
            return "no additional reports"

        # Always probe, even when ingest was skipped: the readiness checklist
        # reports on the API as a component, and "not probed" is a worse answer
        # than one cheap call.
        capabilities = asyncio.run(_probe())
        client_report = generate_client_report(validation, reports / CLIENT_REPORT_FILENAME)
        readiness = assess_readiness(validation, capabilities=capabilities)
        write_readiness_report(readiness, PROTOCOL, reports / READINESS_FILENAME)
        return f"{client_report.path.name}, {READINESS_FILENAME}"

    console.print(PROTOCOL.render())
    console.print()

    with session_scope() as session:
        try:
            result = run_workflow(
                session,
                PROTOCOL,
                workspace=workspace,
                reports_dir=reports,
                ingest=do_ingest,
                allow_unready=allow_unready,
                expected_fingerprint=REGISTERED_FINGERPRINT if pin_protocol else None,
                retrain_months=retrain_months,
                resume=resume,
                extra_reports=extra_reports,
            )
        except PipelineError as exc:
            console.print(f"[red]BLOCKED: {exc}[/red]")
            console.print("\n[dim]Re-run after fixing; completed steps are not repeated.[/dim]")
            raise typer.Exit(code=1) from exc

    console.print(result.state.render())
    console.print()
    if result.validation and result.validation.verdict:
        console.print(f"[bold]VERDICT:[/bold] [bold cyan]{result.verdict_label}[/bold cyan]")
        console.print(f"  {result.validation.verdict.headline}")
    for note in result.validation.notes if result.validation else []:
        console.print(f"[yellow]  {note}[/yellow]")
    console.print(f"\n[green]reports written to {reports}[/green]")


@app.command()
def experiments(limit: int = typer.Option(10)) -> None:
    """The research run ledger — every validation, and whether it is reproducible."""
    with session_scope() as session:
        rows = [run.as_dict() for run in recent_research_runs(session, limit=limit)]

    if not rows:
        console.print("[yellow]no research runs recorded[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Research runs", header_style="bold")
    for column in ("run id", "protocol", "dataset", "features", "status", "verdict", "repro"):
        table.add_column(column, overflow="fold")
    for row in rows:
        colour = {"completed": "green", "blocked": "yellow"}.get(row["status"], "red")
        table.add_row(
            row["run_id"],
            (row["protocol_hash"] or "-")[:12],
            (row["dataset_hash"] or "-")[:12],
            row["feature_version"] or "-",
            f"[{colour}]{row['status']}[/{colour}]",
            row["final_verdict"] or "-",
            "yes" if row["reproducible"] else "no",
        )
    console.print(table)


@app.command()
def steps() -> None:
    """How far the last workflow run got."""
    from backend.research.execution import load_state

    state = load_state(get_settings().data_dir / EXECUTION_CHECKPOINT)
    if not state.run_id:
        console.print("[yellow]no workflow run has been started[/yellow]")
        raise typer.Exit(code=1)
    console.print(state.render())


@app.command(name="phase6-validate")
def phase6_validate(
    out: str = typer.Option(None, help=f"Output path (default: reports/{REPORT_FILENAME})"),
    allow_unready: bool = typer.Option(
        False,
        "--allow-unready",
        help="Exercise the machinery on data that failed a gate (NOT a validation)",
    ),
    pin_protocol: bool = typer.Option(
        True,
        "--pin-protocol/--no-pin-protocol",
        help="Fail if the frozen analysis plan has been edited since registration",
    ),
    retrain_months: int = typer.Option(12),
    print_report: bool = typer.Option(False, "--print"),
) -> None:
    """The one button: real data in, unbiased validation report out.

    Runs audit → pre-flight gates → walk-forward prediction → baselines and
    encompassing → calibration → backtests → segments → verdict, and stops at
    the first gate that fails.
    """
    console.print(PROTOCOL.render())
    console.print()

    expected = REGISTERED_FINGERPRINT if pin_protocol else None
    with session_scope() as session:
        try:
            run = run_validation(
                session,
                PROTOCOL,
                allow_unready=allow_unready,
                retrain_months=retrain_months,
                expected_fingerprint=expected,
            )
        except PipelineError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    path = Path(out) if out else _reports_dir() / REPORT_FILENAME
    write_report(run, path)

    console.print(run.preflight.render())
    console.print()

    if print_report:
        console.print(render_markdown(run))
    elif run.verdict:
        label = VERDICT_LABELS.get(run.verdict.verdict, "INCONCLUSIVE")
        console.print(f"[bold]VERDICT:[/bold] [bold cyan]{label}[/bold cyan]")
        console.print(f"  {run.verdict.headline}")
        for reason in run.verdict.reasons:
            console.print(f"  - {reason}")
    for note in run.notes:
        console.print(f"[yellow]  {note}[/yellow]")

    console.print(f"\n[green]wrote {path}[/green]")


@app.command()
def validate(
    out: str = typer.Option(None),
    allow_unready: bool = typer.Option(False, "--allow-unready"),
    retrain_months: int = typer.Option(12),
    print_report: bool = typer.Option(False, "--print"),
) -> None:
    """Deprecated alias for ``phase6-validate``."""
    console.print("[dim]'validate' is now 'phase6-validate'[/dim]")
    phase6_validate(
        out=out,
        allow_unready=allow_unready,
        pin_protocol=True,
        retrain_months=retrain_months,
        print_report=print_report,
    )


if __name__ == "__main__":
    app()
