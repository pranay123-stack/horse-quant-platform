"""ETL pipeline: API -> validation -> parser -> PostgreSQL.

**Phase 2** delivers ingestion of racecards, results and odds:

* :class:`~backend.data_pipeline.importer.RaceDataImporter` — idempotent upserts
  with per-race savepoints and failure capture
* :class:`~backend.data_pipeline.report.ImportSummary` — what landed, what was
  already there, what broke

**Phase 4** adds scheduling, incremental watermarks and bulk loading.
"""

from backend.data_pipeline.importer import (
    RaceDataImporter,
    backfill_day_by_day,
    import_racecards_for_date,
    import_results_for_range,
)
from backend.data_pipeline.report import FailedRecord, ImportSummary

__all__ = [
    "FailedRecord",
    "ImportSummary",
    "RaceDataImporter",
    "backfill_day_by_day",
    "import_racecards_for_date",
    "import_results_for_range",
]
