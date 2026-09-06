"""phase 6 research runs

The experiment ledger: one row per validation run, recording which analysis
plan was applied, to which rows, with which feature and model versions, and
what conclusion came out.

Reproducibility needs three hashes rather than one. The protocol hash says
which hypothesis was tested; the dataset hash says what it was tested on; the
feature version says how those rows became inputs. A verdict carrying all
three can be re-derived. A verdict carrying none of them is an anecdote.

Revision ID: 9b5802faf234
Revises: 6cc0165f4d3b
Create Date: 2026-08-10 01:16:35.348712+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "9b5802faf234"
down_revision: str | None = "6cc0165f4d3b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_runs",
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("protocol_hash", sa.String(length=64), nullable=False),
        sa.Column("dataset_hash", sa.String(length=64), nullable=True),
        sa.Column("feature_version", sa.String(length=40), nullable=True),
        sa.Column("model_version", sa.String(length=60), nullable=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=20), server_default="running", nullable=False),
        sa.Column("final_verdict", sa.String(length=60), nullable=True),
        sa.Column("steps_completed", sa.Integer(), nullable=False),
        sa.Column("steps_total", sa.Integer(), nullable=False),
        sa.Column("blocked_by", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_research_runs")),
    )
    op.create_index("ix_research_runs_started", "research_runs", ["start_time"], unique=False)
    op.create_index("ix_research_runs_status", "research_runs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_research_runs_status", table_name="research_runs")
    op.drop_index("ix_research_runs_started", table_name="research_runs")
    op.drop_table("research_runs")
