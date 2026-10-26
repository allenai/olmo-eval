"""Scoring times: judge, sandbox and compute seconds spent scoring a task.

Expand-only: every new column is nullable, so the previous release keeps working.

Revision ID: 0008
Revises: 0007
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = ("scoring_judge_seconds", "scoring_sandbox_seconds", "scoring_compute_seconds")


def upgrade() -> None:
    for column in _COLUMNS:
        op.add_column("task_results", sa.Column(column, sa.Double(), nullable=True))


def downgrade() -> None:
    for column in reversed(_COLUMNS):
        op.drop_column("task_results", column)
