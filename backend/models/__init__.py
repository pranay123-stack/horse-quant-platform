"""SQLAlchemy ORM entities.

Every model **must** be imported here. ``Base.metadata`` is what Alembic
autogenerate inspects, and a model that is never imported is invisible to it —
the migration would silently omit its table.

Schema (Phase 2)::

    courses ──< races ──< race_runners >── horses
                  │  └──< race_results >── horses / jockeys / trainers
                  └──< odds_history  >── horses

Phase 3 adds ``race_features``; Phase 5 adds ``bet_ledger``; Phase 6 adds
``backfill_runs`` and ``research_runs``; Phase 7 adds ``predictions``.
"""

from backend.database.base import Base, TimestampMixin
from backend.models.betting import Bet
from backend.models.entities import ID_LENGTH, Course, Horse, Jockey, Trainer
from backend.models.features import FEATURE_VERSION, RaceFeatures
from backend.models.operations import (
    BackfillRun,
    BackfillStatus,
    ResearchRun,
    ResearchStatus,
)
from backend.models.predictions import Prediction, Recommendation
from backend.models.racing import MONEY, ODDS, OddsHistory, Race, RaceResult, RaceRunner

__all__ = [
    "FEATURE_VERSION",
    "ID_LENGTH",
    "MONEY",
    "ODDS",
    "BackfillRun",
    "BackfillStatus",
    "Base",
    "Bet",
    "Course",
    "Horse",
    "Jockey",
    "OddsHistory",
    "Prediction",
    "Race",
    "RaceFeatures",
    "RaceResult",
    "RaceRunner",
    "Recommendation",
    "ResearchRun",
    "ResearchStatus",
    "TimestampMixin",
    "Trainer",
]
