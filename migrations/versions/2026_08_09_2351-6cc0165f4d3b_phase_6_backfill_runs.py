"""phase 6 backfill runs

Audit trail for the historical backfill: one row per invocation, recording the
window requested, how far it got, what it wrote and why it stopped.

This is deliberately separate from the on-disk JSON checkpoint. The checkpoint
makes a run resumable; this table makes it accountable. When the data audit
later reports a thin month, the first question is whether that month was ever
successfully imported, and only a durable record can answer it.

Revision ID: 6cc0165f4d3b
Revises: 6df1c06be46b
Create Date: 2026-08-09 23:51:29.579240+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "6cc0165f4d3b"
down_revision: str | None = "6df1c06be46b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "backfill_runs",
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("regions", sa.String(length=120), nullable=True),
        sa.Column("status", sa.String(length=20), server_default="running", nullable=False),
        sa.Column("records_processed", sa.Integer(), nullable=False),
        sa.Column("failed_records", sa.Integer(), nullable=False),
        sa.Column("months_total", sa.Integer(), nullable=False),
        sa.Column("months_completed", sa.Integer(), nullable=False),
        sa.Column("months_failed", sa.Integer(), nullable=False),
        sa.Column("api_requests", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_backfill_runs")),
    )
    op.create_index("ix_backfill_runs_status", "backfill_runs", ["status"], unique=False)
    op.create_index("ix_backfill_runs_window", "backfill_runs", ["start_date", "end_date"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_backfill_runs_window", table_name="backfill_runs")
    op.drop_index("ix_backfill_runs_status", table_name="backfill_runs")
    op.drop_table("backfill_runs")
