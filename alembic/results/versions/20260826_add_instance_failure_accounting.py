"""Add per-task instance failure accounting columns.

Adds columns to task_results so a failed task is stored as a row that says what
happened rather than a row with empty metrics:
- instances_processed: instances the runner saw (saved plus hard-failed)
- instances_failed: instances that returned an error and no outputs
- error_summary: human-readable summary of the underlying instance errors

Revision ID: add_instance_failure_accounting
Revises: add_results_viewer_perf_indexes
Create Date: 2026-08-26
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "add_instance_failure_accounting"
down_revision: str | None = "add_results_viewer_perf_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "task_results",
        sa.Column(
            "instances_processed",
            sa.Integer,
            nullable=True,
            comment="Instances the runner processed (saved plus hard-failed)",
        ),
    )
    op.add_column(
        "task_results",
        sa.Column(
            "instances_failed",
            sa.Integer,
            nullable=True,
            comment="Instances that returned an error and no outputs",
        ),
    )
    op.add_column(
        "task_results",
        sa.Column(
            "error_summary",
            sa.Text,
            nullable=True,
            comment="Summary of the instance errors behind a failed task",
        ),
    )


def downgrade() -> None:
    op.drop_column("task_results", "error_summary")
    op.drop_column("task_results", "instances_failed")
    op.drop_column("task_results", "instances_processed")
