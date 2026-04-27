"""Add composite indexes for results viewer latest/suite performance.

Revision ID: add_results_viewer_perf_indexes
Revises: nested_metrics
Create Date: 2026-04-27
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "add_results_viewer_perf_indexes"
down_revision: str | None = "nested_metrics"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "idx_experiments_group_model_hash_ts",
        "experiments",
        ["experiment_group", "model_hash", sa.text("timestamp DESC")],
    )
    op.create_index(
        "idx_instance_exp_task_hash_native",
        "instance_predictions",
        ["experiment_pk", "task_hash", "native_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_instance_exp_task_hash_native", table_name="instance_predictions")
    op.drop_index("idx_experiments_group_model_hash_ts", table_name="experiments")
