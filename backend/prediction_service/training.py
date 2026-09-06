"""Train the production model and promote it.

    historical data → features → dataset → LR / LightGBM / XGBoost
                    → calibration → registry → promoted champion

Three models are trained because the comparison is the point. Logistic
regression is the baseline that says whether the gradient boosters are earning
their complexity; if LightGBM cannot beat it on race log loss, the honest
conclusion is to ship the simpler model.

Selection is by **race log loss**, not accuracy and not AUC. A race has exactly
one winner, so per-runner accuracy is dominated by correctly predicting the
seven horses that lose, and a model that says "no" to everything scores well on
it. Race log loss asks the only question that matters: how much probability was
placed on the horse that actually won.

The promoted artefact is written to ``models/production/`` alongside the
metadata needed to reproduce it — feature version, training date, dataset hash,
and the metrics it was promoted on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from backend.ml.dataset import SplitConfig, TrainingDatasetBuilder
from backend.ml.models.registry import ModelRegistry
from backend.ml.train import ModelTrainer, TrainingConfig
from backend.models.features import FEATURE_VERSION
from backend.research.audit import audit_dataset
from backend.research.execution import dataset_hash
from backend.utils.exceptions import ModelError
from backend.utils.logging import get_logger, safe_extra
from backend.utils.timeutils import utcnow

logger = get_logger(__name__, channel="model")

PRODUCTION_DIRNAME = "production"
MANIFEST_FILENAME = "production_model.json"

#: The three models, in the order they are reported. Logistic regression first
#: because it is the bar the others have to clear.
MODEL_TYPES: tuple[str, ...] = ("logistic_regression", "lightgbm", "xgboost")

#: Months taken from the end of the training window for validation. Half
#: stops training early, half fits the calibrator.
DEFAULT_VALID_MONTHS = 12


@dataclass(slots=True)
class ProductionModel:
    """What was promoted, and everything needed to reproduce it."""

    name: str
    version: str
    trained_at: str
    feature_version: str
    dataset_hash: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    rows: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    comparison: list[dict[str, Any]] = field(default_factory=list)
    path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "version": self.version,
            "trained_at": self.trained_at,
            "feature_version": self.feature_version,
            "dataset_hash": self.dataset_hash,
            "training_window": f"{self.train_start} → {self.train_end}",
            "test_window": f"{self.test_start} → {self.test_end}",
            "rows": self.rows,
            "metrics": self.metrics,
            "comparison": self.comparison,
            "path": self.path,
        }

    def render(self) -> str:
        lines = [
            "Production model",
            "----------------",
            f"  Model            {self.name}:{self.version}",
            f"  Trained          {self.trained_at[:19]}",
            f"  Features         {self.feature_version}",
            f"  Dataset hash     {self.dataset_hash}",
            f"  Training window  {self.train_start} → {self.train_end}",
            f"  Rows             {self.rows:,}",
            "",
            "  Candidates (race log loss, lower is better)",
        ]
        for row in self.comparison:
            marker = "*" if row.get("model") == self.name else " "
            lines.append(f"   {marker} {row.get('model')!s:<12} {row.get('race_log_loss', float('nan')):.5f}")
        lines += ["", "  * promoted"]
        return "\n".join(lines)


def build_split(
    *,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    valid_months: int = DEFAULT_VALID_MONTHS,
    embargo_days: int = 14,
) -> SplitConfig:
    """Carve four strictly-ordered windows out of three dates.

    The validation window is taken from the *end* of the training period rather
    than squeezed between training and test. Two windows come out of it — one
    for early stopping, one for fitting the calibrator — and both must sit
    before the test period, or the calibrator would be fitted on races the model
    is then scored against.
    """
    valid_start = train_end - timedelta(days=int(valid_months * 30.44))
    if valid_start <= train_start:
        raise ModelError(
            f"training window {train_start}..{train_end} is too short to hold a "
            f"{valid_months}-month validation period"
        )

    if not train_end < test_start:
        raise ModelError(
            f"the validation window ends {train_end} but testing starts {test_start}; "
            "they must not overlap, or the calibrator is fitted on races the model "
            "is then scored against"
        )

    return SplitConfig(
        train_start=train_start,
        train_end=valid_start - timedelta(days=1),
        valid_start=valid_start,
        valid_end=train_end,
        test_start=test_start,
        test_end=test_end,
        embargo_days=embargo_days,
    )


def train_production_model(
    session: Session,
    *,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    models_dir: Path,
    model_types: tuple[str, ...] = MODEL_TYPES,
    valid_months: int = DEFAULT_VALID_MONTHS,
    embargo_days: int = 14,
    model_params: dict[str, Any] | None = None,
) -> ProductionModel:
    """Build the dataset, train every candidate, promote the best.

    Splits are by date and never at random: a random split lets the model learn
    from races that had not been run yet, and the resulting metrics are fiction.
    """
    split = build_split(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        valid_months=valid_months,
        embargo_days=embargo_days,
    )

    builder = TrainingDatasetBuilder(session)
    data = builder.build(split)
    if data.train.empty:
        raise ModelError("no training rows in the requested window — import historical results first")

    registry = ModelRegistry(models_dir)
    trainer = ModelTrainer(
        TrainingConfig(models=tuple(model_types), params=model_params or {}),
        registry=registry,
    )
    trainer.train(data)

    champion = registry.champion()
    if champion is None:
        raise ModelError("training completed but no model was promoted")

    audit = audit_dataset(session)
    production = ProductionModel(
        name=champion.name,
        version=champion.version,
        trained_at=utcnow().isoformat(),
        feature_version=FEATURE_VERSION,
        dataset_hash=dataset_hash(audit),
        train_start=str(split.train_start),
        train_end=str(split.train_end),
        test_start=str(test_start),
        test_end=str(test_end),
        rows=len(data.train),
        metrics=champion.headline(),
        comparison=registry.comparison_table(),
    )

    production.path = str(_write_manifest(production, models_dir))
    logger.info("production model promoted", extra=safe_extra(production.as_dict()))
    return production


def _write_manifest(model: ProductionModel, models_dir: Path) -> Path:
    """Record what is live, next to the artefacts it points at."""
    target = Path(models_dir) / PRODUCTION_DIRNAME
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / MANIFEST_FILENAME
    manifest.write_text(json.dumps(model.as_dict(), indent=2, default=str), encoding="utf-8")
    return manifest


def load_manifest(models_dir: Path) -> dict[str, Any] | None:
    """Read the production manifest, or ``None`` if nothing is promoted."""
    manifest = Path(models_dir) / PRODUCTION_DIRNAME / MANIFEST_FILENAME
    if not manifest.exists():
        return None
    try:
        payload: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("unreadable production manifest", extra={"error": str(exc)[:200]})
        return None
    return payload


__all__ = [
    "DEFAULT_VALID_MONTHS",
    "MANIFEST_FILENAME",
    "MODEL_TYPES",
    "PRODUCTION_DIRNAME",
    "ProductionModel",
    "build_split",
    "load_manifest",
    "train_production_model",
]
