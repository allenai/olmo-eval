"""Add experiment_group column for cross-model analysis.

Adds experiment_group to experiments and instance_predictions tables.
This enables fast queries for grouping related experiments (e.g., from
a single benchmark run) for cross-model statistical analysis.

The column is denormalized in instance_predictions for query performance
at 100M+ row scale.

Revision ID: add_experiment_group
Revises: add_keyset_pagination_index
Create Date: 2026-01-28
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "add_experiment_group"
down_revision: str | None = "add_keyset_pagination_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Step 1: Add nullable columns first
    op.add_column(
        "experiments",
        sa.Column(
            "experiment_group",
            sa.String(255),
            nullable=True,
            comment="Groups related experiments for analysis",
        ),
    )
    op.add_column(
        "instance_predictions",
        sa.Column(
            "experiment_group",
            sa.String(255),
            nullable=True,
            comment="Denormalized from experiment for fast filtering",
        ),
    )

    # Step 2: Backfill experiment_group from experiment_name (or experiment_id as fallback)
    op.execute(
        """
        UPDATE experiments
        SET experiment_group = COALESCE(experiment_name, experiment_id)
        WHERE experiment_group IS NULL
        """
    )

    # Step 3: Backfill instance_predictions.experiment_group from experiments
    op.execute(
        """
        UPDATE instance_predictions ip
        SET experiment_group = e.experiment_group
        FROM experiments e
        WHERE ip.experiment_pk = e.id AND ip.experiment_group IS NULL
        """
    )

    # Step 4: Add NOT NULL constraint after backfill
    op.alter_column("experiments", "experiment_group", nullable=False)
    op.alter_column("instance_predictions", "experiment_group", nullable=False)

    # Step 5: Add indexes for efficient queries
    # Index on experiments.experiment_group for filtering by group
    op.create_index(
        "ix_experiments_experiment_group",
        "experiments",
        ["experiment_group"],
    )

    # Composite index for the primary cross-model query pattern:
    # Get all predictions for an experiment_group, filtered by task
    op.create_index(
        "idx_instance_group_task_exp",
        "instance_predictions",
        ["experiment_group", "task_hash", "experiment_pk"],
    )

    # Composite index for keyset pagination within an experiment group
    op.create_index(
        "idx_instance_group_id",
        "instance_predictions",
        ["experiment_group", "id"],
    )


def downgrade() -> None:
    # Drop indexes first
    op.drop_index("idx_instance_group_id", table_name="instance_predictions")
    op.drop_index("idx_instance_group_task_exp", table_name="instance_predictions")
    op.drop_index("ix_experiments_experiment_group", table_name="experiments")

    # Drop columns
    op.drop_column("instance_predictions", "experiment_group")
    op.drop_column("experiments", "experiment_group")
