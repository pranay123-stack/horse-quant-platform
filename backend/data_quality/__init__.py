"""Data quality: validate stored racing data before it reaches research.

    from backend.data_quality import DataValidator

    report = DataValidator(session).validate_database(date_from=date(2025, 1, 1))
    print(report.render())
    usable = set(all_race_ids) - report.rejected_race_ids

Rules live in :mod:`backend.data_quality.rules` — one pure function per check, so
every threshold is independently testable and independently arguable.
"""

from backend.data_quality.report import DataQualityReport
from backend.data_quality.rules import Severity, ValidationIssue
from backend.data_quality.validator import DataValidator, usable_race_ids

__all__ = [
    "DataQualityReport",
    "DataValidator",
    "Severity",
    "ValidationIssue",
    "usable_race_ids",
]
