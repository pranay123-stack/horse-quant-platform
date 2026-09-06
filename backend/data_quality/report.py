"""Data-quality reporting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from backend.data_quality.rules import Severity, ValidationIssue
from backend.utils.timeutils import utcnow


@dataclass(slots=True)
class DataQualityReport:
    """What was checked, what passed, and precisely what failed."""

    generated_at: str = field(default_factory=lambda: utcnow().isoformat())
    date_from: str | None = None
    date_to: str | None = None

    total_races: int = 0
    total_runners: int = 0
    total_results: int = 0
    total_odds: int = 0

    #: Race ids excluded from research because they broke an ERROR rule.
    rejected_race_ids: set[str] = field(default_factory=set)
    issues: list[ValidationIssue] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def rejected_races(self) -> int:
        return len(self.rejected_race_ids)

    @property
    def valid_races(self) -> int:
        return max(0, self.total_races - self.rejected_races)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.is_error)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if not issue.is_error)

    @property
    def pass_rate(self) -> float:
        return (self.valid_races / self.total_races) if self.total_races else 1.0

    @property
    def is_clean(self) -> bool:
        return self.error_count == 0

    # ------------------------------------------------------------------
    def add(self, issues: list[ValidationIssue], *, race_id: str | None = None) -> None:
        """Record issues; an ERROR on a race excludes that race from research."""
        for issue in issues:
            self.issues.append(issue)
            if issue.is_error:
                target = race_id or (issue.entity_id if issue.entity == "race" else None)
                if target:
                    self.rejected_race_ids.add(target)

    def counts_by_rule(self) -> dict[str, int]:
        return dict(Counter(issue.rule for issue in self.issues).most_common())

    def counts_by_severity(self) -> dict[str, int]:
        return dict(Counter(str(issue.severity) for issue in self.issues))

    def issues_for(self, severity: Severity) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is severity]

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "window": {"from": self.date_from, "to": self.date_to},
            "totals": {
                "races": self.total_races,
                "runners": self.total_runners,
                "results": self.total_results,
                "odds": self.total_odds,
            },
            "races": {
                "valid": self.valid_races,
                "rejected": self.rejected_races,
                "pass_rate": round(self.pass_rate, 6),
            },
            "issues": {
                "errors": self.error_count,
                "warnings": self.warning_count,
                "by_rule": self.counts_by_rule(),
            },
            "sample_issues": [issue.as_dict() for issue in self.issues[:50]],
        }

    def render(self) -> str:
        window = ""
        if self.date_from or self.date_to:
            window = f"  Window          : {self.date_from or 'start'} → {self.date_to or 'now'}\n"

        lines = [
            "Data quality report",
            "-------------------",
            window.rstrip("\n") if window else None,
            f"  Total races     : {self.total_races}",
            f"  Valid           : {self.valid_races}",
            f"  Rejected        : {self.rejected_races}",
            f"  Pass rate       : {self.pass_rate:.2%}",
            "",
            f"  Runners checked : {self.total_runners}",
            f"  Results checked : {self.total_results}",
            f"  Odds checked    : {self.total_odds}",
            "",
            f"  Errors          : {self.error_count}",
            f"  Warnings        : {self.warning_count}",
        ]
        rendered = [line for line in lines if line is not None]

        by_rule = self.counts_by_rule()
        if by_rule:
            rendered.append("")
            rendered.append("  Breakdown by rule:")
            for rule, count in by_rule.items():
                severity = next(str(i.severity) for i in self.issues if i.rule == rule)
                rendered.append(f"    {severity:<8} {rule:<28} {count}")

        errors = self.issues_for(Severity.ERROR)
        if errors:
            rendered.append("")
            rendered.append("  First errors:")
            for issue in errors[:10]:
                rendered.append(f"    {issue.entity_id:<20} {issue.rule:<28} {issue.message}")
            if len(errors) > 10:
                rendered.append(f"    ... and {len(errors) - 10} more")

        return "\n".join(rendered)


__all__ = ["DataQualityReport"]
