"""Feature schema: the contract between training and inference.

The single most common way a working model breaks in production is a silent
mismatch between the columns it was trained on and the columns it is scored
with — a renamed feature, a reordered matrix, a category that never appeared in
training. None of these raise. They just make the predictions wrong.

The schema pins the exact feature names, their order and their types, and hashes
them. Inference refuses to run against a matrix whose hash does not match.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from backend.utils.timeutils import utcnow

SCHEMA_FILENAME = "feature_schema.json"


@dataclass(slots=True)
class FeatureSchema:
    """Names, order and types of everything the model consumes."""

    numeric: list[str]
    categorical: list[str]
    feature_version: str = "v1"
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    #: Categories seen during training, per categorical column. Anything unseen
    #: at inference is mapped to the "unknown" bucket rather than exploding.
    categories: dict[str, list[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def input_columns(self) -> list[str]:
        """Raw columns the preprocessor expects, in order."""
        return [*self.numeric, *self.categorical]

    @property
    def fingerprint(self) -> str:
        """Stable hash of the input contract."""
        payload = json.dumps(
            {"numeric": self.numeric, "categorical": self.categorical, "version": self.feature_version},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        numeric: list[str],
        categorical: list[str],
        *,
        feature_version: str = "v1",
    ) -> FeatureSchema:
        categories = {
            column: sorted(str(value) for value in frame[column].dropna().unique())
            for column in categorical
            if column in frame.columns
        }
        return cls(
            numeric=list(numeric),
            categorical=list(categorical),
            feature_version=feature_version,
            categories=categories,
        )

    def validate(self, frame: pd.DataFrame) -> None:
        """Raise if ``frame`` cannot be scored by a model built on this schema."""
        missing = [column for column in self.input_columns if column not in frame.columns]
        if missing:
            raise FeatureSchemaError(
                f"{len(missing)} feature(s) missing from the input frame: {missing[:10]}"
            )

    def align(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the frame's columns in exactly the trained order.

        Extra columns are dropped rather than passed through: a preprocessor
        fitted on N columns cannot accept N+1, and dropping is the behaviour that
        lets a richer feature build score an older model.
        """
        self.validate(frame)
        return frame[self.input_columns]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint
        return payload

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / SCHEMA_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> FeatureSchema:
        path = Path(directory) / SCHEMA_FILENAME
        if not path.exists():
            raise FeatureSchemaError(f"no feature schema at {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("fingerprint", None)
        return cls(**payload)


class FeatureSchemaError(ValueError):
    """The input frame does not match the schema the model was trained on."""


__all__ = ["SCHEMA_FILENAME", "FeatureSchema", "FeatureSchemaError"]
