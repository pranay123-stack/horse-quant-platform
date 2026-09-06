#!/usr/bin/env python
"""Research operations CLI.

python -m scripts.research_cli quality                       # data quality report
python -m scripts.research_cli features --store              # build (and persist) features
python -m scripts.research_cli dataset --out data/processed/train.parquet
python -m scripts.research_cli backtest --predictor market
python -m scripts.research_cli synthetic --days 180          # DEV ONLY: fake data
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.backtesting import (
    BacktestConfig,
    BacktestEngine,
    FormScorePredictor,
    MarketPredictor,
    UniformPredictor,
    compare_predictors,
)
from backend.data_quality import DataValidator
from backend.database.session import session_scope
from backend.features import FeaturePipeline, feature_manifest, save_features
from backend.research import ResearchDatasetBuilder
from backend.research.synthetic import SyntheticConfig, generate_synthetic_data
from backend.utils.config import get_settings
from backend.utils.logging import configure_logging

app = typer.Typer(
    add_completion=False,
    # Typer renders local variables in tracebacks by default, which prints the
    # unwrapped database password and API credentials straight to the console.
    # SecretStr masks repr(); it cannot mask a frame local.
    pretty_exceptions_show_locals=False,
    help="Quantitative research operations",
)
console = Console()

PREDICTORS = {"market": MarketPredictor, "uniform": UniformPredictor, "form": FormScorePredictor}


@app.callback()
def _setup() -> None:
    configure_logging()


def _parse(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


@app.command()
def quality(
    date_from: str = typer.Option(None, "--from", help="YYYY-MM-DD"),
    date_to: str = typer.Option(None, "--to", help="YYYY-MM-DD"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Audit stored racing data and report which races are usable."""
    with session_scope() as session:
        report = DataValidator(session).validate_database(
            date_from=_parse(date_from), date_to=_parse(date_to)
        )
    if as_json:
        console.print_json(json.dumps(report.as_dict()))
    else:
        console.print(report.render())
    raise typer.Exit(code=1 if report.error_count else 0)


@app.command()
def features(
    date_from: str = typer.Option(None, "--from"),
    date_to: str = typer.Option(None, "--to"),
    store: bool = typer.Option(False, "--store", help="Persist to the race_features table"),
    manifest: bool = typer.Option(False, "--manifest", help="Print per-feature coverage"),
) -> None:
    """Build point-in-time features."""
    with session_scope() as session:
        pipeline = FeaturePipeline(session)
        frame = pipeline.build(date_from=_parse(date_from), date_to=_parse(date_to))

        console.print(f"[bold]Features built[/bold]: {len(frame)} rows")
        for key, value in pipeline.stats.as_dict().items():
            console.print(f"  {key}: {value}")

        if manifest and not frame.empty:
            table = Table(title="Feature manifest")
            for column in ("feature", "group", "coverage", "mean", "std"):
                table.add_column(column)
            for row in feature_manifest(frame).itertuples(index=False):
                table.add_row(row.feature, row.group, f"{row.coverage:.2f}", str(row.mean), str(row.std))
            console.print(table)

        if store and not frame.empty:
            written = save_features(session, frame)
            console.print(f"[green]stored {written} feature rows[/green]")


@app.command()
def dataset(
    date_from: str = typer.Option(None, "--from"),
    date_to: str = typer.Option(None, "--to"),
    out: str = typer.Option("data/processed/dataset.parquet", "--out"),
    min_runs: int = typer.Option(0, help="Drop runners with fewer prior runs"),
    require_market: bool = typer.Option(False, help="Keep only runners with captured odds"),
    train_end: str = typer.Option(None, help="Also write a time-based train/test split"),
) -> None:
    """Build the research dataset and write it to disk."""
    with session_scope() as session:
        built = ResearchDatasetBuilder(session).build(
            date_from=_parse(date_from),
            date_to=_parse(date_to),
            min_runs=min_runs,
            require_market=require_market,
        )
    console.print(built.render())
    if built.rows == 0:
        console.print("[yellow]nothing to write[/yellow]")
        raise typer.Exit(code=1)

    path = Path(out)
    written = built.to_parquet(path) if path.suffix == ".parquet" else built.to_csv(path)
    console.print(f"[green]wrote {written}[/green]")

    if train_end:
        split = built.split(train_end=_parse(train_end), embargo_days=7)
        console.print(f"  train rows: {split.train_rows}  test rows: {split.test_rows}")
        base = path.with_suffix("")
        split.train.to_parquet(f"{base}_train.parquet", index=False)
        split.test.to_parquet(f"{base}_test.parquet", index=False)
        console.print(f"[green]wrote {base}_train.parquet and {base}_test.parquet[/green]")


@app.command()
def backtest(
    predictor: str = typer.Option("market", help=f"one of {', '.join(PREDICTORS)}, or 'all'"),
    date_from: str = typer.Option(None, "--from"),
    date_to: str = typer.Option(None, "--to"),
    min_ev: float = typer.Option(0.05, help="Minimum expected value to bet"),
    staking: str = typer.Option("flat", help="flat | kelly"),
    stake: float = typer.Option(10.0, help="Flat stake size"),
) -> None:
    """Simulate a strategy over historical races."""
    settings = get_settings()
    config = BacktestConfig(
        starting_bankroll=settings.starting_bankroll,
        min_expected_value=min_ev,
        staking=staking,
        flat_stake=stake,
        kelly_fraction_multiplier=settings.kelly_fraction,
        max_stake_fraction=settings.max_stake_fraction,
    )

    with session_scope() as session:
        built = ResearchDatasetBuilder(session).build(
            date_from=_parse(date_from), date_to=_parse(date_to), require_market=True
        )

    if built.rows == 0:
        console.print("[yellow]no data to backtest[/yellow]")
        raise typer.Exit(code=1)

    if predictor == "all":
        table = compare_predictors(
            built.frame, [MarketPredictor(), UniformPredictor(), FormScorePredictor()], config
        )
        console.print(table.to_string(index=False))
        return

    if predictor not in PREDICTORS:
        console.print(f"[red]unknown predictor {predictor!r}[/red]")
        raise typer.Exit(code=2)

    result = BacktestEngine(config).run(built.frame, PREDICTORS[predictor]())
    console.print(result.render())


@app.command()
def synthetic(
    days: int = typer.Option(180, help="Number of days to generate"),
    races_per_day: int = typer.Option(4),
    seed: int = typer.Option(42),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt"),
) -> None:
    """DEV ONLY — populate the database with synthetic races.

    Never run this against a database holding real ingested data: the synthetic
    rows are indistinguishable from real ones to every downstream consumer.
    """
    console.print("[bold yellow]This inserts FAKE racing data into the configured database.[/bold yellow]")
    console.print(f"  target: {get_settings().safe_database_url}")
    if not yes:
        typer.confirm("Continue?", abort=True)

    with session_scope() as session:
        summary = generate_synthetic_data(
            session, SyntheticConfig(n_days=days, races_per_day=races_per_day, seed=seed)
        )
    console.print(summary.render())


if __name__ == "__main__":
    app()
