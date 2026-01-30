"""Add agent_metrics column for agent evaluation storage.

Adds agent_metrics JSONB column to task_results table for storing
nested agent evaluation metrics (tool accuracy, abstention, trajectory,
reliability, execution, and judge metrics).

Revision ID: add_agent_metrics
Revises: add_experiment_group
Create Date: 2026-01-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "add_agent_metrics"
down_revision: str | None = "add_experiment_group"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add agent_metrics JSONB column (nullable - only populated for agent tasks)
    op.add_column(
        "task_results",
        sa.Column(
            "agent_metrics",
            JSONB,
            nullable=True,
            comment="Nested agent evaluation metrics (tool, abstention, trajectory, etc.)",
        ),
    )


def downgrade() -> None:
    op.drop_column("task_results", "agent_metrics")
