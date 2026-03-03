"""Rename inference_runs to inference_samples and drop request metrics.

The inference_request_metrics table was never used, and "samples" better
describes what the table contains (batch-level metrics samples).

Revision ID: rename_to_samples
Revises: drop_percentile_latencies
Create Date: 2026-02-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "rename_to_samples"
down_revision: str = "drop_percentile_latencies"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop inference_request_metrics table (never used)
    op.drop_index("idx_request_metrics_run_ts", table_name="inference_request_metrics")
    op.drop_index("ix_request_metrics_timestamp", table_name="inference_request_metrics")
    op.drop_index("ix_request_metrics_request_id", table_name="inference_request_metrics")
    op.drop_index("ix_request_metrics_inference_run_id", table_name="inference_request_metrics")
    op.drop_table("inference_request_metrics")

    # Rename inference_runs to inference_samples
    op.rename_table("inference_runs", "inference_samples")

    # Rename indexes to match new table name
    op.drop_index("ix_inference_runs_experiment_id", table_name="inference_samples")
    op.drop_index("ix_inference_runs_experiment_group", table_name="inference_samples")
    op.drop_index("ix_inference_runs_model_hash", table_name="inference_samples")
    op.drop_index("ix_inference_runs_model_name", table_name="inference_samples")
    op.drop_index("ix_inference_runs_task_name", table_name="inference_samples")
    op.drop_index("ix_inference_runs_task_hash", table_name="inference_samples")
    op.drop_index("ix_inference_runs_provider_kind", table_name="inference_samples")
    op.drop_index("ix_inference_runs_author", table_name="inference_samples")
    op.drop_index("ix_inference_runs_timestamp", table_name="inference_samples")
    op.drop_index("idx_inference_runs_provider_ts", table_name="inference_samples")
    op.drop_index("idx_inference_runs_model_ts", table_name="inference_samples")
    op.drop_index("idx_inference_runs_exp_task", table_name="inference_samples")

    # Create new indexes with updated names
    op.create_index("ix_inference_samples_experiment_id", "inference_samples", ["experiment_id"])
    op.create_index(
        "ix_inference_samples_experiment_group", "inference_samples", ["experiment_group"]
    )
    op.create_index("ix_inference_samples_model_hash", "inference_samples", ["model_hash"])
    op.create_index("ix_inference_samples_model_name", "inference_samples", ["model_name"])
    op.create_index("ix_inference_samples_task_name", "inference_samples", ["task_name"])
    op.create_index("ix_inference_samples_task_hash", "inference_samples", ["task_hash"])
    op.create_index("ix_inference_samples_provider_kind", "inference_samples", ["provider_kind"])
    op.create_index("ix_inference_samples_author", "inference_samples", ["author"])
    op.create_index("ix_inference_samples_timestamp", "inference_samples", ["timestamp"])
    op.create_index(
        "idx_inference_samples_provider_ts",
        "inference_samples",
        ["provider_kind", sa.text("timestamp DESC")],
    )
    op.create_index(
        "idx_inference_samples_model_ts",
        "inference_samples",
        ["model_name", sa.text("timestamp DESC")],
    )
    op.create_index(
        "idx_inference_samples_exp_task",
        "inference_samples",
        ["experiment_id", "task_hash"],
    )


def downgrade() -> None:
    # Rename indexes back
    op.drop_index("idx_inference_samples_exp_task", table_name="inference_samples")
    op.drop_index("idx_inference_samples_model_ts", table_name="inference_samples")
    op.drop_index("idx_inference_samples_provider_ts", table_name="inference_samples")
    op.drop_index("ix_inference_samples_timestamp", table_name="inference_samples")
    op.drop_index("ix_inference_samples_author", table_name="inference_samples")
    op.drop_index("ix_inference_samples_provider_kind", table_name="inference_samples")
    op.drop_index("ix_inference_samples_task_hash", table_name="inference_samples")
    op.drop_index("ix_inference_samples_task_name", table_name="inference_samples")
    op.drop_index("ix_inference_samples_model_name", table_name="inference_samples")
    op.drop_index("ix_inference_samples_model_hash", table_name="inference_samples")
    op.drop_index("ix_inference_samples_experiment_group", table_name="inference_samples")
    op.drop_index("ix_inference_samples_experiment_id", table_name="inference_samples")

    # Rename table back
    op.rename_table("inference_samples", "inference_runs")

    # Recreate old indexes
    op.create_index("ix_inference_runs_experiment_id", "inference_runs", ["experiment_id"])
    op.create_index("ix_inference_runs_experiment_group", "inference_runs", ["experiment_group"])
    op.create_index("ix_inference_runs_model_hash", "inference_runs", ["model_hash"])
    op.create_index("ix_inference_runs_model_name", "inference_runs", ["model_name"])
    op.create_index("ix_inference_runs_task_name", "inference_runs", ["task_name"])
    op.create_index("ix_inference_runs_task_hash", "inference_runs", ["task_hash"])
    op.create_index("ix_inference_runs_provider_kind", "inference_runs", ["provider_kind"])
    op.create_index("ix_inference_runs_author", "inference_runs", ["author"])
    op.create_index("ix_inference_runs_timestamp", "inference_runs", ["timestamp"])
    op.create_index(
        "idx_inference_runs_provider_ts",
        "inference_runs",
        ["provider_kind", sa.text("timestamp DESC")],
    )
    op.create_index(
        "idx_inference_runs_model_ts",
        "inference_runs",
        ["model_name", sa.text("timestamp DESC")],
    )
    op.create_index(
        "idx_inference_runs_exp_task",
        "inference_runs",
        ["experiment_id", "task_hash"],
    )

    # Recreate inference_request_metrics table
    op.create_table(
        "inference_request_metrics",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("inference_run_id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(64), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("end_to_end_latency_s", postgresql.DOUBLE_PRECISION(), nullable=False),
        sa.Column("tokens_per_second", postgresql.DOUBLE_PRECISION(), nullable=False),
        sa.Column("time_to_first_token_s", postgresql.DOUBLE_PRECISION(), nullable=True),
        sa.Column("time_per_output_token_s", postgresql.DOUBLE_PRECISION(), nullable=True),
        sa.Column("finish_reason", sa.String(50), nullable=True),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["inference_run_id"],
            ["inference_runs.id"],
            ondelete="CASCADE",
        ),
    )

    op.create_index(
        "ix_request_metrics_inference_run_id",
        "inference_request_metrics",
        ["inference_run_id"],
    )
    op.create_index(
        "ix_request_metrics_request_id",
        "inference_request_metrics",
        ["request_id"],
    )
    op.create_index(
        "ix_request_metrics_timestamp",
        "inference_request_metrics",
        ["timestamp"],
    )
    op.create_index(
        "idx_request_metrics_run_ts",
        "inference_request_metrics",
        ["inference_run_id", "timestamp"],
    )
