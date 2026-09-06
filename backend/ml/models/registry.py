"""Model registry.

A model that cannot be traced back to the data, code and features that produced
it is not reproducible, and a betting result from an untraceable model is an
anecdote. Every registered version records:

* what it is — model type, version, parameters, seed
* what it was built from — feature version, schema fingerprint, split windows
* how good it was — the full metric set at registration time
* when, and with which library versions

Layout on disk::

    models/
      registry.json                 index of every version, plus the champion
      xgboost/
        v1/
          estimator.joblib
          preprocessor.joblib
          feature_schema.json
          calibrator.joblib
          model.json
          metrics.json

The champion is a pointer, not a copy. Promoting a new model is a one-line
change to the index and is instantly reversible — which is what you want at
03:00 when the new model is behaving oddly.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.ml.models.calibrated import CalibratedRacingModel
from backend.utils.config import get_settings
from backend.utils.exceptions import ModelError
from backend.utils.logging import get_logger
from backend.utils.timeutils import utcnow

logger = get_logger(__name__, channel="model")

REGISTRY_FILENAME = "registry.json"
METRICS_FILENAME = "metrics.json"


def library_versions() -> dict[str, str]:
    """Versions of everything that can change a prediction."""
    versions = {"python": platform.python_version()}
    for module_name in ("sklearn", "xgboost", "lightgbm", "numpy", "pandas"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:  # pragma: no cover - optional at runtime
            versions[module_name] = "not installed"
    return versions


@dataclass(slots=True)
class ModelRecord:
    """One registered model version."""

    name: str
    version: str
    path: str
    created_at: str
    feature_version: str
    schema_fingerprint: str
    calibration_method: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    split: dict[str, str] = field(default_factory=dict)
    training_rows: int = 0
    libraries: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.name}/{self.version}"

    def headline(self) -> dict[str, float]:
        """The three numbers a comparison table is built from."""
        return {
            "log_loss": self.metrics.get("classification", {}).get("log_loss", float("nan")),
            "brier": self.metrics.get("classification", {}).get("brier", float("nan")),
            "roc_auc": self.metrics.get("classification", {}).get("roc_auc", float("nan")),
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """Versioned storage for trained models."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else get_settings().models_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / REGISTRY_FILENAME

    # ------------------------------------------------------------------
    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"models": {}, "champion": None, "updated_at": None}
        index: dict[str, Any] = json.loads(self.index_path.read_text(encoding="utf-8"))
        return index

    def _write_index(self, index: dict[str, Any]) -> None:
        index["updated_at"] = utcnow().isoformat()
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def next_version(self, name: str) -> str:
        """``v1``, ``v2``, … per model name."""
        existing = self._read_index()["models"].get(name, {})
        numbers = [int(key[1:]) for key in existing if key.startswith("v") and key[1:].isdigit()]
        return f"v{max(numbers, default=0) + 1}"

    # ------------------------------------------------------------------
    def register(
        self,
        model: CalibratedRacingModel,
        *,
        metrics: dict[str, Any] | None = None,
        split: dict[str, str] | None = None,
        version: str | None = None,
        notes: str = "",
    ) -> ModelRecord:
        """Save a model and add it to the index."""
        name = model.model.name
        version = version or self.next_version(name)
        directory = self.root / name / version

        model.save(directory)
        metadata = model.metadata()
        (directory / METRICS_FILENAME).write_text(json.dumps(metrics or {}, indent=2), encoding="utf-8")

        record = ModelRecord(
            name=name,
            version=version,
            path=str(directory.relative_to(self.root)),
            created_at=utcnow().isoformat(),
            feature_version=metadata.get("feature_version", "unknown"),
            schema_fingerprint=metadata.get("schema_fingerprint", ""),
            calibration_method=metadata.get("calibration_method", "none"),
            params=metadata.get("params", {}),
            metrics=metrics or {},
            split=split or {},
            training_rows=metadata.get("training_rows", 0),
            libraries=library_versions(),
            notes=notes,
        )

        index = self._read_index()
        index["models"].setdefault(name, {})[version] = record.as_dict()
        if index.get("champion") is None:
            index["champion"] = record.key
        self._write_index(index)

        logger.info(
            "model registered",
            extra={"model": name, "version": version, "path": str(directory)},
        )
        return record

    # ------------------------------------------------------------------
    def records(self) -> list[ModelRecord]:
        index = self._read_index()
        return [
            ModelRecord(**payload) for versions in index["models"].values() for payload in versions.values()
        ]

    def get(self, name: str, version: str | None = None) -> ModelRecord:
        index = self._read_index()
        versions = index["models"].get(name)
        if not versions:
            raise ModelError(f"no registered model named {name!r}")
        if version is None:
            version = max(versions, key=lambda key: versions[key]["created_at"])
        if version not in versions:
            raise ModelError(f"{name} has no version {version!r}; have {sorted(versions)}")
        return ModelRecord(**versions[version])

    def load(self, name: str, version: str | None = None) -> CalibratedRacingModel:
        record = self.get(name, version)
        return CalibratedRacingModel.load(self.root / record.path)

    # ------------------------------------------------------------------
    def promote(self, name: str, version: str | None = None) -> ModelRecord:
        """Make a version the champion — the model inference will load."""
        record = self.get(name, version)
        index = self._read_index()
        index["champion"] = record.key
        self._write_index(index)
        logger.info("champion promoted", extra={"model": name, "version": record.version})
        return record

    def champion(self) -> ModelRecord | None:
        key = self._read_index().get("champion")
        if not key:
            return None
        name, version = key.split("/", 1)
        try:
            return self.get(name, version)
        except ModelError:  # pragma: no cover - index pointing at a deleted model
            return None

    def load_champion(self) -> CalibratedRacingModel:
        record = self.champion()
        if record is None:
            raise ModelError("no champion model has been promoted")
        return CalibratedRacingModel.load(self.root / record.path)

    # ------------------------------------------------------------------
    def comparison_table(self) -> list[dict[str, Any]]:
        """Every registered version with its headline metrics, best first."""
        champion = self.champion()
        rows = []
        for record in self.records():
            classification = record.metrics.get("classification", {})
            racing = record.metrics.get("racing", {})
            rows.append(
                {
                    "model": record.name,
                    "version": record.version,
                    "calibration": record.calibration_method,
                    "log_loss": classification.get("log_loss"),
                    "brier": classification.get("brier"),
                    "roc_auc": classification.get("roc_auc"),
                    "ece": classification.get("expected_calibration_error"),
                    "race_log_loss": racing.get("race_log_loss"),
                    "top1_hit_rate": racing.get("top1_hit_rate"),
                    "champion": champion is not None and record.key == champion.key,
                }
            )
        return sorted(rows, key=lambda row: (row["log_loss"] is None, row["log_loss"]))


__all__ = ["METRICS_FILENAME", "REGISTRY_FILENAME", "ModelRecord", "ModelRegistry", "library_versions"]
