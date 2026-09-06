"""Quantitative research layer.

    from backend.research import ResearchDatasetBuilder

    dataset = ResearchDatasetBuilder(session).build(date_from=date(2025, 1, 1))
    dataset.to_parquet("data/processed/train.parquet")
    split = dataset.split(train_end=date(2025, 9, 30), embargo_days=7)

* ``dataset``   -- leak-free supervised dataset construction
* ``splits``    -- date-based train/test splits and walk-forward folds
* ``synthetic`` -- **development only** synthetic race generator
* ``audit``     -- dataset readiness gate (Phase 6)
* ``protocol``  -- the pre-registered validation plan (Phase 6)
* ``validation_report`` -- the real-money validation report (Phase 6)
"""

from backend.research.audit import DatasetAudit, audit_dataset
from backend.research.dataset import ResearchDataset, ResearchDatasetBuilder
from backend.research.protocol import PROTOCOL, ValidationProtocol, Verdict
from backend.research.splits import (
    DatasetSplit,
    TemporalLeakageError,
    assert_no_leakage_between,
    iter_race_groups,
    time_split,
    walk_forward_splits,
)

__all__ = [
    "PROTOCOL",
    "DatasetAudit",
    "DatasetSplit",
    "ResearchDataset",
    "ResearchDatasetBuilder",
    "TemporalLeakageError",
    "ValidationProtocol",
    "Verdict",
    "assert_no_leakage_between",
    "audit_dataset",
    "iter_race_groups",
    "time_split",
    "walk_forward_splits",
]
