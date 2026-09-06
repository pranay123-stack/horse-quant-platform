#!/usr/bin/env python
"""Model training and inference CLI.

python -m scripts.ml_cli train                       # train, calibrate, register
python -m scripts.ml_cli train --no-market           # market-free control
python -m scripts.ml_cli compare                     # registry comparison table
python -m scripts.ml_cli predict --race-id rac_18f2a
python -m scripts.ml_cli predict --date 2026-08-10
python -m scripts.ml_cli importance --model lightgbm
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.database.session import session_scope
from backend.ml import (
    ModelRegistry,
    ModelTrainer,
    RacePredictor,
    SplitConfig,
    TrainingDatasetBuilder,
)
from backend.ml.train import TrainingConfig
from backend.utils.config import get_settings
from backend.utils.logging import configure_logging

app = typer.Typer(
    add_completion=False,
    # Typer renders local variables in tracebacks by default, which prints the
    # unwrapped database password and API credentials straight to the console.
    # SecretStr masks repr(); it cannot mask a frame local.
    pretty_exceptions_show_locals=False,
    help="Probability engine: training and inference",
)
console = Console()

MARKET_PREFIXES = ("mkt_", "market_")


@app.callback()
def _setup() -> None:
    configure_logging()


def _parse(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


@app.command()
def train(
    train_start: str = typer.Option(None, help="YYYY-MM-DD (default 2018-01-01)"),
    train_end: str = typer.Option(None, help="default 2023-12-31"),
    valid_start: str = typer.Option(None, help="default 2024-01-01"),
    valid_end: str = typer.Option(None, help="default 2024-12-31"),
    test_start: str = typer.Option(None, help="default 2025-01-01"),
    test_end: str = typer.Option(None, help="default 2026-12-31"),
    embargo: int = typer.Option(14, help="Days dropped after the training window"),
    models: str = typer.Option("logistic_regression,xgboost,lightgbm"),
    calibrations: str = typer.Option("none,sigmoid,isotonic"),
    min_runs: int = typer.Option(0, help="Drop runners with fewer prior runs"),
    no_market: bool = typer.Option(False, "--no-market", help="Exclude market features"),
    adaptive: bool = typer.Option(
        False, "--adaptive", help="Fall back to proportional windows if the dates are empty"
    ),
    detail: bool = typer.Option(False, "--detail", help="Print the champion's full report"),
) -> None:
    """Build the dataset, train every model, calibrate, evaluate and register."""
    defaults = SplitConfig()
    split = SplitConfig(
        train_start=_parse(train_start) or defaults.train_start,
        train_end=_parse(train_end) or defaults.train_end,
        valid_start=_parse(valid_start) or defaults.valid_start,
        valid_end=_parse(valid_end) or defaults.valid_end,
        test_start=_parse(test_start) or defaults.test_start,
        test_end=_parse(test_end) or defaults.test_end,
        embargo_days=embargo,
    )

    with session_scope() as session:
        data = TrainingDatasetBuilder(session).build(
            split,
            min_runs=min_runs,
            adaptive_if_empty=adaptive,
            drop_feature_prefixes=MARKET_PREFIXES if no_market else (),
        )
        console.print(data.render())
        console.print()

        config = TrainingConfig(
            models=tuple(name.strip() for name in models.split(",") if name.strip()),
            calibrations=tuple(name.strip() for name in calibrations.split(",") if name.strip()),  # type: ignore[arg-type]
        )
        run = ModelTrainer(config).train(data)

    console.print(run.render())
    if detail and run.champion is not None:
        console.print()
        console.print(run.report_for(run.champion.model).render())


@app.command()
def compare(as_json: bool = typer.Option(False, "--json")) -> None:
    """Show every registered model version and its metrics."""
    rows = ModelRegistry().comparison_table()
    if not rows:
        console.print("[yellow]no models registered — run `ml_cli train` first[/yellow]")
        raise typer.Exit(code=1)

    if as_json:
        console.print_json(json.dumps(rows))
        return

    console.print(
        f"  {'model':<22}{'ver':<6}{'calib':<10}{'logloss':>9}{'brier':>9}{'auc':>8}{'race_ll':>9}  champion"
    )
    console.print("  " + "-" * 82)
    for row in rows:
        console.print(
            f"  {row['model']:<22}{row['version']:<6}{row['calibration']:<10}"
            f"{row['log_loss'] or 0:>9.5f}{row['brier'] or 0:>9.5f}"
            f"{row['roc_auc'] or 0:>8.4f}{row['race_log_loss'] or 0:>9.5f}"
            f"  {'*' if row['champion'] else ''}"
        )


@app.command()
def registry() -> None:
    """Describe the champion model's provenance."""
    record = ModelRegistry().champion()
    if record is None:
        console.print("[yellow]no champion promoted[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[bold]{record.name} {record.version}[/bold]")
    console.print(f"  trained at      : {record.created_at}")
    console.print(f"  feature version : {record.feature_version}")
    console.print(f"  schema          : {record.schema_fingerprint}")
    console.print(f"  calibration     : {record.calibration_method}")
    console.print(f"  training rows   : {record.training_rows}")
    console.print(f"  split           : {record.split}")
    console.print(f"  libraries       : {record.libraries}")


@app.command()
def predict(
    race_id: str = typer.Option(None, "--race-id"),
    race_date: str = typer.Option(None, "--date", help="Score every race that day"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Score a race, or a whole day's card."""
    if not race_id and not race_date:
        console.print("[red]give either --race-id or --date[/red]")
        raise typer.Exit(code=2)

    with session_scope() as session:
        predictor = RacePredictor(session)
        predictions = [predictor.predict_race(race_id)] if race_id else predictor.predict_date(race_date)

    if not predictions:
        console.print("[yellow]nothing to predict[/yellow]")
        raise typer.Exit(code=1)

    if as_json:
        console.print_json(json.dumps([p.as_dict() for p in predictions]))
        return

    for prediction in predictions:
        console.print(prediction.render())
        console.print()


@app.command()
def importance(
    model: str = typer.Option(None, help="Model name (default: the champion)"),
    top: int = typer.Option(25),
) -> None:
    """Show which features the model actually uses."""
    registry_instance = ModelRegistry()
    loaded = registry_instance.load(model) if model else registry_instance.load_champion()

    frame = loaded.feature_importance(top=top)
    if frame.empty:
        console.print("[yellow]this model exposes no feature importance[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[bold]{loaded.name}[/bold] — top {top} features")
    for row in frame.itertuples(index=False):
        console.print(f"  {row.importance_pct:>7.2%}  {row.feature}")

    market_share = frame[frame["feature"].str.startswith(MARKET_PREFIXES)]["importance_pct"].sum()
    console.print(f"\n  market-derived share of the top {top}: {market_share:.1%}")
    if market_share > 0.4:
        console.print(
            "[yellow]  A model leaning this hard on market features is largely re-deriving\n"
            "  the bookmakers' price, and cannot be expected to beat it. Compare with\n"
            "  `ml_cli train --no-market`.[/yellow]"
        )


@app.command()
def settings() -> None:
    """Where models are stored."""
    console.print(f"  models_dir      : {get_settings().models_dir}")
    console.print(f"  active version  : {get_settings().active_model_version}")


if __name__ == "__main__":
    app()
