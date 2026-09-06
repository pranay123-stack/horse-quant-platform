"""Feature preprocessing: raw feature frame -> model matrix.

.. note::
   Two classes in this codebase are called something like "feature pipeline",
   and they do different jobs:

   * :class:`backend.features.FeaturePipeline` — reads the database and builds
     **point-in-time features** (Phase 3).
   * :class:`FeaturePreprocessor` here — turns that feature frame into a numeric
     **model matrix** (Phase 4). Exported as ``FeaturePipeline`` too, for the
     name the specification uses.

Choices that matter
===================

**Missing stays informative.** Racing data is missing for reasons: a debutant has
no form, a runner with no market had no price captured. Median imputation alone
would erase that signal, so every imputed numeric column is paired with a
``*_was_missing`` indicator and the tree models can learn from the absence.

**Scaling only where it changes the answer.** Logistic regression needs
standardised inputs to converge sensibly and to make its coefficients
comparable. Gradient-boosted trees are invariant to monotone rescaling, so
scaling them costs time and buys nothing — the preprocessor takes a ``scale``
flag rather than pretending one setting suits both.

**Unseen categories do not explode.** ``handle_unknown="ignore"`` means a course
that never appeared in training encodes as all-zeros instead of raising, which is
the correct behaviour for a system that must score a card at 13:00 whatever the
data does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.ml.preprocessing.schema import FeatureSchema, FeatureSchemaError
from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="model")

PREPROCESSOR_FILENAME = "preprocessor.joblib"
MISSING_SUFFIX = "_was_missing"


class FeaturePreprocessor:
    """Imputes, optionally scales, and one-hot encodes."""

    def __init__(
        self, schema: FeatureSchema, *, scale: bool = True, add_missing_indicators: bool = True
    ) -> None:
        self.schema = schema
        self.scale = scale
        self.add_missing_indicators = add_missing_indicators
        self._transformer: ColumnTransformer | None = None
        self._output_names: list[str] = []

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self._transformer is not None

    @property
    def output_names(self) -> list[str]:
        """Column names of the transformed matrix, in order."""
        if not self.is_fitted:
            raise FeatureSchemaError("preprocessor is not fitted")
        return list(self._output_names)

    # ------------------------------------------------------------------
    def _build(self) -> ColumnTransformer:
        numeric_steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="median", add_indicator=self.add_missing_indicators))
        ]
        if self.scale:
            numeric_steps.append(("scale", StandardScaler()))

        return ColumnTransformer(
            transformers=[
                ("numeric", Pipeline(numeric_steps), self.schema.numeric),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                            (
                                "encode",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                    min_frequency=0.001,  # fold rare courses into one bucket
                                ),
                            ),
                        ]
                    ),
                    self.schema.categorical,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def fit(self, frame: pd.DataFrame) -> FeaturePreprocessor:
        aligned = self.schema.align(frame)
        # Categoricals arrive as mixed object columns; force string so the
        # encoder does not treat NaN and "nan" as different categories.
        aligned = self._stringify_categoricals(aligned)

        self._transformer = self._build()
        self._transformer.fit(aligned)
        self._output_names = self._resolve_output_names()

        logger.info(
            "preprocessor fitted",
            extra={
                "input_columns": len(self.schema.input_columns),
                "output_columns": len(self._output_names),
                "scaled": self.scale,
            },
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted or self._transformer is None:
            raise FeatureSchemaError("preprocessor must be fitted before transform")
        aligned = self._stringify_categoricals(self.schema.align(frame))
        matrix = self._transformer.transform(aligned)
        return pd.DataFrame(matrix, columns=self._output_names, index=frame.index)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    # ------------------------------------------------------------------
    def _stringify_categoricals(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.schema.categorical:
            return frame
        result = frame.copy()
        for column in self.schema.categorical:
            result[column] = result[column].astype("object").where(result[column].notna(), None)
            result[column] = result[column].map(lambda value: str(value) if value is not None else None)
        return result

    def _resolve_output_names(self) -> list[str]:
        assert self._transformer is not None
        try:
            return [str(name) for name in self._transformer.get_feature_names_out()]
        except Exception:  # pragma: no cover - defensive, sklearn version drift
            width = self._transformer.transform(
                pd.DataFrame([[np.nan] * len(self.schema.input_columns)], columns=self.schema.input_columns)
            ).shape[1]
            return [f"f{index}" for index in range(width)]

    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / PREPROCESSOR_FILENAME
        joblib.dump(
            {
                "transformer": self._transformer,
                "output_names": self._output_names,
                "scale": self.scale,
                "add_missing_indicators": self.add_missing_indicators,
            },
            path,
        )
        self.schema.save(target)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> FeaturePreprocessor:
        target = Path(directory)
        path = target / PREPROCESSOR_FILENAME
        if not path.exists():
            raise FeatureSchemaError(f"no preprocessor at {path}")

        payload = joblib.load(path)
        preprocessor = cls(
            FeatureSchema.load(target),
            scale=payload["scale"],
            add_missing_indicators=payload["add_missing_indicators"],
        )
        preprocessor._transformer = payload["transformer"]
        preprocessor._output_names = payload["output_names"]
        return preprocessor


#: The name used in the Phase 4 specification.
FeaturePipeline = FeaturePreprocessor


__all__ = [
    "MISSING_SUFFIX",
    "PREPROCESSOR_FILENAME",
    "FeaturePipeline",
    "FeaturePreprocessor",
]
