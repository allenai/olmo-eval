"""Add duration metrics columns for experiment and task timing.

Adds columns to track:
- experiment_duration_seconds: Total time for the experiment run
- provider_init_seconds: Time to initialize each inference provider (JSONB)
- duration_seconds: Time for each individual task

These metrics enable performance analysis and resource planning.

Revision ID: add_duration_metrics
Revises: add_agent_metrics
Create Date: 2026-02-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, JSONB

from alembic import op

revision: str = "add_duration_metrics"
down_revision: str | None = "add_agent_metrics"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add experiment-level duration columns
    op.add_column(
        "experiments",
        sa.Column(
            "experiment_duration_seconds",
            DOUBLE_PRECISION,
            nullable=True,
            comment="Total time for the experiment run in seconds",
        ),
    )
    op.add_column(
        "experiments",
        sa.Column(
            "provider_init_seconds",
            JSONB,
            nullable=True,
            comment="Time to initialize each inference provider (model_name -> seconds)",
        ),
    )

    # Add task-level duration column
    op.add_column(
        "task_results",
        sa.Column(
            "duration_seconds",
            DOUBLE_PRECISION,
            nullable=True,
            comment="Time to complete this task in seconds",
        ),
    )


def downgrade() -> None:
    op.drop_column("task_results", "duration_seconds")
    op.drop_column("experiments", "provider_init_seconds")
    op.drop_column("experiments", "experiment_duration_seconds")
