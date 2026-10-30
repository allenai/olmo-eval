"""Interrupted run status: a stopped run that a resumed job continues under the same run ID.

Expand-only: the constraint allows one more value, so the previous release keeps working.

Revision ID: 0009
Revises: 0008
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("ck_runs_status", "runs", type_="check")
    op.create_check_constraint(
        "ck_runs_status",
        "runs",
        "status IN ('running', 'interrupted', 'complete', 'partial', 'failed')",
    )


def downgrade() -> None:
    # The previous release reported a stopped run as failed.
    op.execute("UPDATE runs SET status = 'failed' WHERE status = 'interrupted'")
    op.drop_constraint("ck_runs_status", "runs", type_="check")
    op.create_check_constraint(
        "ck_runs_status", "runs", "status IN ('running', 'complete', 'partial', 'failed')"
    )
