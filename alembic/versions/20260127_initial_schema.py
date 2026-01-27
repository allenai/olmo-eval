"""Initial schema for evaluation storage.

Revision ID: initial_schema
Revises:
Create Date: 2026-01-27
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ==========================================================================
    # EXPERIMENTS TABLE
    # ==========================================================================
    op.create_table(
        "experiments",
        # Primary key - auto-increment ID (experiment_id is NOT unique)
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        # Experiment ID (can have duplicates - one launch can run multiple models)
        sa.Column("experiment_id", sa.String(64), nullable=False),
        # Model identification
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("model_hash", sa.String(64), nullable=False),
        sa.Column("model_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("backend_name", sa.String(50), nullable=False),
        # Timestamp
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        # Experiment metadata
        sa.Column("experiment_name", sa.String(255), nullable=False),
        sa.Column("workspace", sa.String(255), nullable=False),
        sa.Column("author", sa.String(100), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=True),
        # Version tracking
        sa.Column("git_ref", sa.String(100), nullable=False),
        sa.Column("revision", sa.String(255), nullable=False),
        # S3 reference
        sa.Column("s3_location", sa.String(512), nullable=True),
        # Flexible metadata
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        # Audit timestamp
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Experiments indexes
    op.create_index("idx_experiments_experiment_id", "experiments", ["experiment_id"])
    op.create_index("idx_experiments_model_hash", "experiments", ["model_hash"])
    op.create_index("idx_experiments_model_name", "experiments", ["model_name"])
    op.create_index(
        "idx_experiments_model_name_ts",
        "experiments",
        ["model_name", sa.text("timestamp DESC")],
    )
    op.create_index("ix_experiments_author", "experiments", ["author"])
    op.create_index("ix_experiments_timestamp", "experiments", ["timestamp"])
    op.create_index("ix_experiments_workspace", "experiments", ["workspace"])

    # ==========================================================================
    # TASK_RESULTS TABLE
    # ==========================================================================
    op.create_table(
        "task_results",
        # Primary key
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        # Foreign key to experiments.id (NOT experiment_id)
        sa.Column("experiment_pk", sa.Integer(), nullable=False),
        # Denormalized model_hash for query convenience
        sa.Column("model_hash", sa.String(64), nullable=False),
        # Task identification
        sa.Column("task_name", sa.String(255), nullable=False),
        sa.Column("task_hash", sa.String(64), nullable=False),
        sa.Column("task_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        # Aggregated metrics
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("num_instances", sa.Integer(), nullable=True),
        sa.Column("primary_metric", sa.String(100), nullable=True),
        sa.Column("primary_score", postgresql.DOUBLE_PRECISION(), nullable=True),
        # S3 keys for detailed task data
        sa.Column("s3_metrics_key", sa.String(512), nullable=True),
        sa.Column("s3_predictions_key", sa.String(512), nullable=True),
        sa.Column("s3_requests_key", sa.String(512), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["experiment_pk"],
            ["experiments.id"],
            ondelete="CASCADE",
        ),
    )

    # Task results indexes
    op.create_index("ix_task_results_experiment_pk", "task_results", ["experiment_pk"])
    op.create_index("ix_task_results_model_hash", "task_results", ["model_hash"])
    op.create_index("ix_task_results_task_name", "task_results", ["task_name"])
    op.create_index("ix_task_results_primary_score", "task_results", ["primary_score"])
    op.create_index("ix_task_results_task_hash", "task_results", ["task_hash"])
    op.create_index(
        "idx_task_results_exp_task", "task_results", ["experiment_pk", "task_name"]
    )
    op.create_index(
        "idx_task_results_model_task", "task_results", ["model_hash", "task_name"]
    )
    op.create_index(
        "idx_task_results_score_desc",
        "task_results",
        [sa.text("primary_score DESC")],
    )

    # ==========================================================================
    # INSTANCE_PREDICTIONS TABLE
    # ==========================================================================
    op.create_table(
        "instance_predictions",
        # Primary key
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        # Foreign key to experiments.id
        sa.Column("experiment_pk", sa.Integer(), nullable=False),
        # Task hash for joining to task_results
        sa.Column("task_hash", sa.String(64), nullable=False),
        # Instance identification
        sa.Column("native_id", sa.String(255), nullable=False),
        # Instance-level metrics
        sa.Column(
            "instance_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["experiment_pk"],
            ["experiments.id"],
            ondelete="CASCADE",
        ),
    )

    # Instance predictions indexes
    op.create_index(
        "ix_instance_predictions_experiment_pk", "instance_predictions", ["experiment_pk"]
    )
    op.create_index("ix_instance_predictions_task_hash", "instance_predictions", ["task_hash"])
    op.create_index("ix_instance_predictions_native_id", "instance_predictions", ["native_id"])
    op.create_index(
        "idx_instance_exp_task_hash",
        "instance_predictions",
        ["experiment_pk", "task_hash"],
    )
    op.create_index(
        "idx_instance_task_hash_native",
        "instance_predictions",
        ["task_hash", "native_id"],
    )


def downgrade() -> None:
    # Drop instance_predictions
    op.drop_index("idx_instance_task_hash_native", table_name="instance_predictions")
    op.drop_index("idx_instance_exp_task_hash", table_name="instance_predictions")
    op.drop_index("ix_instance_predictions_native_id", table_name="instance_predictions")
    op.drop_index("ix_instance_predictions_task_hash", table_name="instance_predictions")
    op.drop_index("ix_instance_predictions_experiment_pk", table_name="instance_predictions")
    op.drop_table("instance_predictions")

    # Drop task_results
    op.drop_index("idx_task_results_score_desc", table_name="task_results")
    op.drop_index("idx_task_results_model_task", table_name="task_results")
    op.drop_index("idx_task_results_exp_task", table_name="task_results")
    op.drop_index("ix_task_results_task_hash", table_name="task_results")
    op.drop_index("ix_task_results_primary_score", table_name="task_results")
    op.drop_index("ix_task_results_task_name", table_name="task_results")
    op.drop_index("ix_task_results_model_hash", table_name="task_results")
    op.drop_index("ix_task_results_experiment_pk", table_name="task_results")
    op.drop_table("task_results")

    # Drop experiments
    op.drop_index("ix_experiments_workspace", table_name="experiments")
    op.drop_index("ix_experiments_timestamp", table_name="experiments")
    op.drop_index("ix_experiments_author", table_name="experiments")
    op.drop_index("idx_experiments_model_name_ts", table_name="experiments")
    op.drop_index("idx_experiments_model_name", table_name="experiments")
    op.drop_index("idx_experiments_model_hash", table_name="experiments")
    op.drop_index("idx_experiments_experiment_id", table_name="experiments")
    op.drop_table("experiments")
