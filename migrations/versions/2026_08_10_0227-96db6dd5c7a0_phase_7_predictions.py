"""phase 7 predictions

The product's output: one row per runner per upcoming race, written by the
daily job and read by the API and the dashboard.

It stores the recommendation *and the inputs that justified it* — probability,
the price it was compared against, and the model version. Storing only 'BET'
would make this a list of opinions; storing the reasoning makes it a record
that can be settled and scored, which is the only way the performance page
means anything.

Revision ID: 96db6dd5c7a0
Revises: 9b5802faf234
Create Date: 2026-08-10 02:27:59.604611+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "96db6dd5c7a0"
down_revision: str | None = "9b5802faf234"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("race_id", sa.String(length=40), nullable=False),
        sa.Column("horse_id", sa.String(length=40), nullable=False),
        sa.Column("race_date", sa.Date(), nullable=True),
        sa.Column("off_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("course", sa.String(length=120), nullable=True),
        sa.Column("horse_name", sa.String(length=160), nullable=True),
        sa.Column("model_probability", sa.Float(), nullable=False),
        sa.Column("market_probability", sa.Float(), nullable=True),
        sa.Column("odds", sa.Numeric(precision=10, scale=3), nullable=True),
        sa.Column("expected_value", sa.Float(), nullable=True),
        sa.Column("edge", sa.Float(), nullable=True),
        sa.Column("recommendation", sa.String(length=10), nullable=False),
        sa.Column("rejection_reason", sa.String(length=60), nullable=True),
        sa.Column("model_name", sa.String(length=60), nullable=False),
        sa.Column("model_version", sa.String(length=40), nullable=False),
        sa.Column("feature_version", sa.String(length=40), nullable=True),
        sa.Column("settled", sa.Boolean(), nullable=False),
        sa.Column("won", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["horse_id"], ["horses.horse_id"], name=op.f("fk_predictions_horse_id_horses"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["race_id"], ["races.race_id"], name=op.f("fk_predictions_race_id_races"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
    )
    op.create_index("ix_predictions_race", "predictions", ["race_id"], unique=False)
    op.create_index("ix_predictions_race_date", "predictions", ["race_date"], unique=False)
    op.create_index(
        "uq_predictions_runner", "predictions", ["race_id", "horse_id", "model_version"], unique=True
    )


def downgrade() -> None:
    op.drop_index("uq_predictions_runner", table_name="predictions")
    op.drop_index("ix_predictions_race_date", table_name="predictions")
    op.drop_index("ix_predictions_race", table_name="predictions")
    op.drop_table("predictions")
