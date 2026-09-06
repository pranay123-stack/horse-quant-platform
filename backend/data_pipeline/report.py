"""Import reporting.

An ingestion run that says only "done" is useless in production. Every run
produces an :class:`ImportSummary` that answers: what landed, what was already
there, what broke, and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.utils.timeutils import utcnow


@dataclass(slots=True)
class FailedRecord:
    """One record that could not be ingested, kept with enough context to retry."""

    record_id: str
    stage: str
    error_type: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "record_id": self.record_id,
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(slots=True)
class ImportSummary:
    """Outcome of one ingestion run."""

    source: str = "racing_api"
    started_at: str = field(default_factory=lambda: utcnow().isoformat())
    finished_at: str | None = None

    races_seen: int = 0
    races_created: int = 0
    races_updated: int = 0
    races_skipped: int = 0

    runners_created: int = 0
    runners_updated: int = 0
    results_created: int = 0
    results_updated: int = 0
    odds_created: int = 0
    odds_skipped: int = 0

    courses_upserted: int = 0
    horses_upserted: int = 0
    jockeys_upserted: int = 0
    trainers_upserted: int = 0

    api_requests: int = 0
    api_retries: int = 0

    failures: list[FailedRecord] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def succeeded(self) -> int:
        return self.races_created + self.races_updated

    @property
    def is_clean(self) -> bool:
        return not self.failures

    def record_failure(self, record_id: str, stage: str, error: BaseException) -> None:
        self.failures.append(
            FailedRecord(
                record_id=record_id,
                stage=stage,
                error_type=type(error).__name__,
                message=str(error)[:500],
            )
        )

    def finish(self) -> ImportSummary:
        self.finished_at = utcnow().isoformat()
        return self

    def merge(self, other: ImportSummary) -> ImportSummary:
        """Fold another summary into this one (used when batching by date)."""
        for name in (
            "races_seen",
            "races_created",
            "races_updated",
            "races_skipped",
            "runners_created",
            "runners_updated",
            "results_created",
            "results_updated",
            "odds_created",
            "odds_skipped",
            "courses_upserted",
            "horses_upserted",
            "jockeys_upserted",
            "trainers_upserted",
            "api_requests",
            "api_retries",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.failures.extend(other.failures)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "races": {
                "seen": self.races_seen,
                "created": self.races_created,
                "updated": self.races_updated,
                "skipped_duplicates": self.races_skipped,
            },
            "runners": {"created": self.runners_created, "updated": self.runners_updated},
            "results": {"created": self.results_created, "updated": self.results_updated},
            "odds": {"created": self.odds_created, "skipped_duplicates": self.odds_skipped},
            "entities": {
                "courses": self.courses_upserted,
                "horses": self.horses_upserted,
                "jockeys": self.jockeys_upserted,
                "trainers": self.trainers_upserted,
            },
            "api": {"requests": self.api_requests, "retries": self.api_retries},
            "failed": self.failed,
            "failures": [failure.as_dict() for failure in self.failures[:50]],
        }

    def render(self) -> str:
        """Human-readable block for logs and CLI output."""
        lines = [
            "Import summary",
            "--------------",
            f"  Races seen      : {self.races_seen}",
            f"  Imported        : {self.races_created}",
            f"  Updated         : {self.races_updated}",
            f"  Skipped (dupes) : {self.races_skipped}",
            f"  Runners         : {self.runners_created} new / {self.runners_updated} updated",
            f"  Results         : {self.results_created} new / {self.results_updated} updated",
            f"  Odds quotes     : {self.odds_created} new / {self.odds_skipped} unchanged",
            f"  Entities        : {self.horses_upserted} horses, {self.jockeys_upserted} jockeys, "
            f"{self.trainers_upserted} trainers, {self.courses_upserted} courses",
            f"  API             : {self.api_requests} requests, {self.api_retries} retries",
            f"  Failed          : {self.failed}",
        ]
        for failure in self.failures[:10]:
            lines.append(
                f"      - {failure.record_id} [{failure.stage}] {failure.error_type}: {failure.message[:120]}"
            )
        if self.failed > 10:
            lines.append(f"      ... and {self.failed - 10} more")
        return "\n".join(lines)


__all__ = ["FailedRecord", "ImportSummary"]
