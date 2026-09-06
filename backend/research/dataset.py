"""Research dataset construction.

Turns the feature frame plus outcomes into a model-ready table, and writes it to
Parquet or CSV.

The label join is the last thing that happens and it happens here, in one place,
explicitly. Features are built by :mod:`backend.features` without ever loading
an outcome; :func:`~backend.features.base.load_labels` reads them separately.
Keeping the two paths apart means a leak requires deliberately editing this
function, not merely forgetting a filter somewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from sqlalchemy.orm import Session

from backend.data_quality import DataValidator
from backend.features.base import EVENT_TIME, LABEL_COLUMNS, FeatureConfig, load_labels
from backend.features.feature_pipeline import FeaturePipeline, feature_columns, feature_manifest
from backend.research.splits import DatasetSplit, time_split, walk_forward_splits
from backend.utils.logging import get_logger, safe_extra

logger = get_logger(__name__, channel="model")

TARGET_COLUMN: Literal["won"] = "won"
#: Columns kept beside the features for joining, auditing and backtesting.
IDENTITY_COLUMNS: tuple[str, ...] = ("race_id", "horse_id", "race_date", EVENT_TIME)
#: Outcome columns. Present for evaluation, never inputs to a model. Sourced
#: from :data:`backend.features.base.LABEL_COLUMNS` so it cannot fall behind
#: what the label query actually returns.
OUTCOME_COLUMNS: tuple[str, ...] = LABEL_COLUMNS
#: Declaration-time context that is categorical rather than numeric. Excluded
#: from the numeric feature list, but the ML layer one-hot encodes them, so the
#: dataset can be asked to carry them through.
CATEGORICAL_COLUMNS: tuple[str, ...] = ("course_id", "going_band", "race_type", "distance_band")


@dataclass(slots=True)
class ResearchDataset:
    """A model-ready dataset plus the metadata needed to use it correctly."""

    frame: pd.DataFrame
    feature_names: list[str]
    target: str = TARGET_COLUMN
    excluded_races: int = 0
    build_stats: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def rows(self) -> int:
        return len(self.frame)

    @property
    def races(self) -> int:
        return int(self.frame["race_id"].nunique()) if not self.frame.empty else 0

    @property
    def date_range(self) -> tuple[date | None, date | None]:
        if self.frame.empty or "race_date" not in self.frame:
            return None, None
        dates = pd.to_datetime(self.frame["race_date"], errors="coerce").dt.date.dropna()
        return (dates.min(), dates.max()) if not dates.empty else (None, None)

    @property
    def base_rate(self) -> float:
        """Fraction of runners that won — the accuracy a constant model achieves."""
        if self.frame.empty or self.target not in self.frame:
            return 0.0
        return float(self.frame[self.target].mean())

    def X(self) -> pd.DataFrame:  # noqa: N802 - conventional name in ML code
        return self.frame[self.feature_names]

    def y(self) -> pd.Series:
        return self.frame[self.target]

    # ------------------------------------------------------------------
    def to_parquet(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(target, index=False)
        logger.info("dataset written", extra={"path": str(target), "rows": self.rows, "format": "parquet"})
        return target

    def to_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(target, index=False)
        logger.info("dataset written", extra={"path": str(target), "rows": self.rows, "format": "csv"})
        return target

    def split(self, *, train_end: date, embargo_days: int = 0) -> DatasetSplit:
        return time_split(self.frame, train_end=train_end, embargo_days=embargo_days)

    def walk_forward(self, **kwargs: Any) -> list[DatasetSplit]:
        return walk_forward_splits(self.frame, **kwargs)

    def manifest(self) -> pd.DataFrame:
        return feature_manifest(self.frame)

    def summary(self) -> dict[str, Any]:
        start, end = self.date_range
        return {
            "rows": self.rows,
            "races": self.races,
            "features": len(self.feature_names),
            "target": self.target,
            "base_rate": round(self.base_rate, 6),
            "date_from": start.isoformat() if start else None,
            "date_to": end.isoformat() if end else None,
            "excluded_races": self.excluded_races,
        }

    def render(self) -> str:
        start, end = self.date_range
        return "\n".join(
            [
                "Research dataset",
                "----------------",
                f"  Rows            : {self.rows}",
                f"  Races           : {self.races}",
                f"  Features        : {len(self.feature_names)}",
                f"  Target          : {self.target} (base rate {self.base_rate:.3%})",
                f"  Date range      : {start} → {end}",
                f"  Excluded races  : {self.excluded_races} (failed data quality)",
            ]
        )


class ResearchDatasetBuilder:
    """Builds a leak-free supervised dataset from the warehouse."""

    def __init__(
        self,
        session: Session,
        config: FeatureConfig | None = None,
        *,
        apply_data_quality: bool = True,
    ) -> None:
        self.session = session
        self.config = config or FeatureConfig()
        self.apply_data_quality = apply_data_quality

    def build(
        self,
        *,
        date_from: date | None = None,
        date_to: date | None = None,
        target: Literal["won", "placed"] = TARGET_COLUMN,
        min_runs: int = 0,
        require_market: bool = False,
        drop_incomplete: bool = True,
        include_categoricals: bool = False,
    ) -> ResearchDataset:
        """Build the dataset.

        Parameters
        ----------
        min_runs:
            Drop runners with fewer than this many prior runs. Debutants have no
            form, so their feature vector is mostly priors — useful to exclude
            when training, but note that doing so changes the population the
            model is valid for.
        require_market:
            Keep only runners with captured odds. Necessary for anything that
            compares a model probability against a price.
        """
        pipeline = FeaturePipeline(self.session, self.config)
        # ``require_result=True``: supervised learning needs an outcome.
        features = pipeline.build(date_from=date_from, date_to=date_to, require_result=True)

        if features.empty:
            logger.warning("dataset build produced no rows")
            return ResearchDataset(frame=features, feature_names=[], target=target)

        excluded = 0
        if self.apply_data_quality:
            report = DataValidator(self.session).validate_database(date_from=date_from, date_to=date_to)
            rejected = report.rejected_race_ids
            if rejected:
                before = len(features)
                features = features[~features["race_id"].isin(rejected)]
                excluded = len(rejected)
                logger.info(
                    "dropped races failing data quality",
                    extra=safe_extra({"races": excluded, "rows": before - len(features)}),
                )

        labels = load_labels(self.session, race_ids=features["race_id"].unique().tolist())
        merged = features.merge(labels, on=["race_id", "horse_id"], how="inner", suffixes=("", "_label"))

        if drop_incomplete:
            merged = merged[merged[target].notna()]
        if min_runs > 0 and "horse_runs_before" in merged:
            merged = merged[merged["horse_runs_before"] >= min_runs]
        if require_market and "mkt_has_market" in merged:
            merged = merged[merged["mkt_has_market"] == 1]

        merged = merged.sort_values([EVENT_TIME, "race_id", "horse_id"], kind="stable").reset_index(drop=True)

        names = [column for column in feature_columns(merged) if column not in OUTCOME_COLUMNS]
        categoricals = (
            [column for column in CATEGORICAL_COLUMNS if column in merged.columns]
            if include_categoricals
            else []
        )
        ordered = [
            *[column for column in IDENTITY_COLUMNS if column in merged.columns],
            *categoricals,
            *names,
            *[column for column in OUTCOME_COLUMNS if column in merged.columns],
        ]
        merged = merged[ordered]

        dataset = ResearchDataset(
            frame=merged,
            feature_names=names,
            target=target,
            excluded_races=excluded,
            build_stats=pipeline.stats.as_dict(),
        )
        logger.info("dataset built", extra=safe_extra(dataset.summary()))
        return dataset


__all__ = [
    "CATEGORICAL_COLUMNS",
    "IDENTITY_COLUMNS",
    "OUTCOME_COLUMNS",
    "TARGET_COLUMN",
    "ResearchDataset",
    "ResearchDatasetBuilder",
]
