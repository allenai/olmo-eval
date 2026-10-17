"""Per-task scoring time.

Expand-only: the new column is nullable, so the previous release keeps working.

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


def upgrade() -> None:
    op.add_column("task_results", sa.Column("scoring_seconds", sa.Double(), nullable=True))


def downgrade() -> None:
    op.drop_column("task_results", "scoring_seconds")
