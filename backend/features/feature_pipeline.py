"""The feature pipeline: raw tables in, model-ready frame out.

    from backend.features import FeaturePipeline

    frame = FeaturePipeline(session).build(date_from=date(2024, 1, 1))

Order matters, and not only for correctness:

1. **History** and **targets** are loaded separately — outcomes and declarations
   never share a code path.
2. Horse, jockey and trainer features are attached by point-in-time as-of joins.
3. Race and market features are computed cross-sectionally within each race.
4. Composite scores are derived last, from features that already exist.

Labels are never loaded here. :class:`~backend.research.dataset.ResearchDatasetBuilder`
joins them on at the very end, as a separate, explicit step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from backend.features.base import (
    EVENT_TIME,
    FeatureConfig,
    load_history,
    load_odds,
    load_targets,
)
from backend.features.horse_features import build_horse_features
from backend.features.jockey_features import build_jockey_features
from backend.features.market_features import MARKET_FEATURE_COLUMNS, build_market_features
from backend.features.race_features import RACE_FEATURE_COLUMNS, build_race_features
from backend.features.scoring import SCORE_COLUMNS, build_scores
from backend.features.trainer_features import build_trainer_features
from backend.utils.logging import get_logger, safe_extra

logger = get_logger(__name__, channel="model")

#: Identity columns kept alongside the features for joining and auditing.
KEY_COLUMNS: tuple[str, ...] = ("race_id", "horse_id", "race_date", EVENT_TIME)

#: Columns that exist only to carry context; excluded from the model matrix.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "race_id",
        "horse_id",
        "jockey_id",
        "trainer_id",
        "course_id",
        "race_date",
        "off_time",
        EVENT_TIME,
        "going",
        "going_band",
        "distance_band",
        "race_type",
        "region",
        "form",
        "has_result",
        "saddle_cloth",
        "field_size",
        "prize_money",
        "distance_yards",
        "race_class",
        "age",
        "weight_lbs",
        "official_rating",
        "rpr",
        "topspeed",
        "draw",
        "days_since_last_run",
        "going_value",
        "distance_furlongs",
        "race_name",
    }
)


@dataclass(slots=True)
class FeatureBuildStats:
    """What the build actually did — surfaced in logs and the CLI."""

    targets: int = 0
    history_rows: int = 0
    odds_rows: int = 0
    feature_columns: int = 0
    debutants: int = 0
    with_market: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "targets": self.targets,
            "history_rows": self.history_rows,
            "odds_rows": self.odds_rows,
            "feature_columns": self.feature_columns,
            "debutants": self.debutants,
            "with_market": self.with_market,
            "warnings": self.warnings,
        }


class FeaturePipeline:
    """Builds the point-in-time feature frame."""

    def __init__(self, session: Session, config: FeatureConfig | None = None) -> None:
        self.session = session
        self.config = config or FeatureConfig()
        self.stats = FeatureBuildStats()

    # ------------------------------------------------------------------
    def build(
        self,
        *,
        date_from: date | None = None,
        date_to: date | None = None,
        race_ids: list[str] | None = None,
        require_result: bool | None = None,
        history_from: date | None = None,
    ) -> pd.DataFrame:
        """Build features for every runner in the window.

        ``history_from`` bounds how far back form is computed. Leave it unset for
        full history; set it to trade a little feature depth for speed. Note the
        history window is deliberately *not* bounded above by ``date_from``:
        runners early in the window still need the form that preceded it.
        """
        targets = load_targets(
            self.session,
            date_from=date_from,
            date_to=date_to,
            race_ids=race_ids,
            require_result=require_result,
        )
        history = load_history(self.session, date_from=history_from, date_to=date_to)
        odds = load_odds(self.session, race_ids=race_ids, date_from=date_from, date_to=date_to)

        self.stats.targets = len(targets)
        self.stats.history_rows = len(history)
        self.stats.odds_rows = len(odds)

        if targets.empty:
            logger.warning("feature build found no target runners")
            self.stats.warnings.append("no target runners in window")
            return targets

        frame = build_horse_features(targets, history, self.config)
        frame = build_jockey_features(frame, history, self.config)
        frame = build_trainer_features(frame, history, self.config)
        frame = build_race_features(frame)
        frame = build_market_features(frame, odds)
        frame = build_scores(frame, self.config)

        frame = frame.sort_values([EVENT_TIME, "race_id", "horse_id"], kind="stable").reset_index(drop=True)

        self.stats.feature_columns = len(feature_columns(frame))
        self.stats.debutants = int(frame.get("horse_is_debutant", pd.Series(dtype=int)).sum())
        self.stats.with_market = int(frame.get("mkt_has_market", pd.Series(dtype=int)).sum())

        if self.stats.with_market == 0:
            self.stats.warnings.append(
                "no market data: odds features are null and market_score falls back to a uniform prior"
            )
        if self.stats.history_rows == 0:
            self.stats.warnings.append(
                "no result history: every form feature is at its prior; features are near-useless"
            )

        logger.info("feature build complete", extra=safe_extra(self.stats.as_dict()))
        return frame

    def build_for_race(self, race_id: str) -> pd.DataFrame:
        """Features for a single race — the live prediction path."""
        return self.build(race_ids=[race_id])


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric model-input columns, in a stable order.

    Stability matters: a model trained on one column order must be scored with
    the same one, and silent reordering between train and inference is a classic
    production failure.
    """
    candidates = [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS
        and not column.startswith("_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return sorted(candidates)


def feature_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: coverage, mean, spread — a build-time sanity check."""
    columns = feature_columns(frame)
    if not columns:
        return pd.DataFrame(columns=["feature", "group", "non_null", "coverage", "mean", "std", "min", "max"])

    rows = []
    total = len(frame)
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "group": _feature_group(column),
                "non_null": int(series.notna().sum()),
                "coverage": round(float(series.notna().mean()), 4) if total else 0.0,
                "mean": round(float(series.mean()), 4) if series.notna().any() else None,
                "std": round(float(series.std()), 4) if series.notna().any() else None,
                "min": round(float(series.min()), 4) if series.notna().any() else None,
                "max": round(float(series.max()), 4) if series.notna().any() else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["group", "feature"]).reset_index(drop=True)


def _feature_group(column: str) -> str:
    if column in SCORE_COLUMNS:
        return "score"
    if column.startswith("horse_"):
        return "horse"
    if column.startswith("jky_"):
        return "jockey"
    if column.startswith("trn_"):
        return "trainer"
    if column.startswith("mkt_"):
        return "market"
    if column.startswith(("race_", "runner_")) or column in RACE_FEATURE_COLUMNS:
        return "race"
    if column in MARKET_FEATURE_COLUMNS:
        return "market"
    return "other"


__all__ = [
    "KEY_COLUMNS",
    "NON_FEATURE_COLUMNS",
    "FeatureBuildStats",
    "FeaturePipeline",
    "feature_columns",
    "feature_manifest",
]
