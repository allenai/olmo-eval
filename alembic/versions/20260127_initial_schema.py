"""Initial schema for evaluation storage.

Revision ID: initial_schema
Revises:
Create Date: 2026-01-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiment_id", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("model_hash", sa.String(64), nullable=False),
        sa.Column("model_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("backend_name", sa.String(50), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("experiment_name", sa.String(255), nullable=False),
        sa.Column("workspace", sa.String(255), nullable=False),
        sa.Column("author", sa.String(100), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("git_ref", sa.String(100), nullable=False),
        sa.Column("revision", sa.String(255), nullable=False),
        sa.Column("s3_location", sa.String(512), nullable=True),
        sa.Column("model_path", sa.String(512), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )

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

    op.create_table(
        "task_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiment_pk", sa.Integer(), nullable=False),
        sa.Column("model_hash", sa.String(64), nullable=False),
        sa.Column("task_name", sa.String(255), nullable=False),
        sa.Column("task_hash", sa.String(64), nullable=False),
        sa.Column("task_config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("num_instances", sa.Integer(), nullable=True),
        sa.Column("primary_metric", sa.String(100), nullable=True),
        sa.Column("primary_score", postgresql.DOUBLE_PRECISION(), nullable=True),
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

    op.create_table(
        "instance_predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiment_pk", sa.Integer(), nullable=False),
        sa.Column("task_hash", sa.String(64), nullable=False),
        sa.Column("native_id", sa.String(255), nullable=False),
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
    op.drop_index("idx_instance_task_hash_native", table_name="instance_predictions")
    op.drop_index("idx_instance_exp_task_hash", table_name="instance_predictions")
    op.drop_index("ix_instance_predictions_native_id", table_name="instance_predictions")
    op.drop_index("ix_instance_predictions_task_hash", table_name="instance_predictions")
    op.drop_index("ix_instance_predictions_experiment_pk", table_name="instance_predictions")
    op.drop_table("instance_predictions")

    op.drop_index("idx_task_results_score_desc", table_name="task_results")
    op.drop_index("idx_task_results_model_task", table_name="task_results")
    op.drop_index("idx_task_results_exp_task", table_name="task_results")
    op.drop_index("ix_task_results_task_hash", table_name="task_results")
    op.drop_index("ix_task_results_primary_score", table_name="task_results")
    op.drop_index("ix_task_results_task_name", table_name="task_results")
    op.drop_index("ix_task_results_model_hash", table_name="task_results")
    op.drop_index("ix_task_results_experiment_pk", table_name="task_results")
    op.drop_table("task_results")

    op.drop_index("ix_experiments_workspace", table_name="experiments")
    op.drop_index("ix_experiments_timestamp", table_name="experiments")
    op.drop_index("ix_experiments_author", table_name="experiments")
    op.drop_index("idx_experiments_model_name_ts", table_name="experiments")
    op.drop_index("idx_experiments_model_name", table_name="experiments")
    op.drop_index("idx_experiments_model_hash", table_name="experiments")
    op.drop_index("idx_experiments_experiment_id", table_name="experiments")
    op.drop_table("experiments")
