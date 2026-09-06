"""Point-in-time feature engineering.

    from backend.features import FeaturePipeline

    frame = FeaturePipeline(session).build(date_from=date(2024, 1, 1))

Every feature is computed as of the moment before its race starts. The mechanism
is :func:`backend.features.base.as_of_join` -- a backward ``merge_asof`` with
``allow_exact_matches=False`` -- which makes look-ahead structurally impossible
rather than merely discouraged. See :mod:`backend.features.base` for the full
argument.

Modules:

* ``base``            -- loaders, encodings and the point-in-time join
* ``horse_features``  -- form, speed, class, distance, going, rest
* ``jockey_features`` -- rolling strike rate, ROI, course record
* ``trainer_features``-- 30-day and 100-runner form, trainer/jockey pairing
* ``race_features``   -- race context and within-race relative measures
* ``market_features`` -- odds level, drift, implied probability, market rank
* ``scoring``         -- composite scores
* ``feature_pipeline``-- orchestration
* ``store``           -- persistence to ``race_features``
"""

from backend.features.base import FeatureConfig, ScoreWeights, as_of_join
from backend.features.feature_pipeline import (
    FeatureBuildStats,
    FeaturePipeline,
    feature_columns,
    feature_manifest,
)
from backend.features.scoring import SCORE_COLUMNS, build_scores
from backend.features.store import load_features, save_features

__all__ = [
    "SCORE_COLUMNS",
    "FeatureBuildStats",
    "FeatureConfig",
    "FeaturePipeline",
    "ScoreWeights",
    "as_of_join",
    "build_scores",
    "feature_columns",
    "feature_manifest",
    "load_features",
    "save_features",
]
