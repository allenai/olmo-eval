"""Add per-task generation counts.

Adds a column to task_results recording how the task's saved generations ended,
so a score over truncated or empty answers is distinguishable from one over
complete answers:
- generation_counts: {"generations", "cap_hit", "empty", "unclosed_think",
  "finish_reason_unknown"}; NULL for tasks that generate nothing

Revision ID: add_generation_counts
Revises: add_instance_failure_accounting
Create Date: 2026-09-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "add_generation_counts"
down_revision: str | None = "add_instance_failure_accounting"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "task_results",
        sa.Column(
            "generation_counts",
            JSONB,
            nullable=True,
            comment="How the saved generations ended (cap-hit, empty, unclosed think)",
        ),
    )


def downgrade() -> None:
    op.drop_column("task_results", "generation_counts")
