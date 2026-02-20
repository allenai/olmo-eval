"""Initial metrics schema.

Revision ID: initial_schema
Revises:
Create Date: 2026-02-19
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
        "inference_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiment_id", sa.String(64), nullable=True),
        sa.Column("experiment_name", sa.String(255), nullable=True),
        sa.Column("experiment_group", sa.String(255), nullable=True),
        sa.Column("model_name", sa.String(255), nullable=True),
        sa.Column("model_hash", sa.String(64), nullable=True),
        sa.Column("task_name", sa.String(255), nullable=True),
        sa.Column("task_hash", sa.String(64), nullable=True),
        sa.Column("workspace", sa.String(255), nullable=True),
        sa.Column("author", sa.String(100), nullable=True),
        sa.Column("provider_kind", sa.String(50), nullable=True),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("total_requests", sa.Integer(), nullable=False),
        sa.Column("successful_requests", sa.Integer(), nullable=False),
        sa.Column("failed_requests", sa.Integer(), nullable=False),
        sa.Column("total_prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("total_completion_tokens", sa.Integer(), nullable=False),
        sa.Column("wall_clock_time_s", postgresql.DOUBLE_PRECISION(), nullable=False),
        sa.Column("output_tokens_per_second", postgresql.DOUBLE_PRECISION(), nullable=False),
        sa.Column("mean_latency_s", postgresql.DOUBLE_PRECISION(), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

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


def downgrade() -> None:
    op.drop_index("idx_request_metrics_run_ts", table_name="inference_request_metrics")
    op.drop_index("ix_request_metrics_timestamp", table_name="inference_request_metrics")
    op.drop_index("ix_request_metrics_request_id", table_name="inference_request_metrics")
    op.drop_index("ix_request_metrics_inference_run_id", table_name="inference_request_metrics")
    op.drop_table("inference_request_metrics")

    op.drop_index("idx_inference_runs_exp_task", table_name="inference_runs")
    op.drop_index("idx_inference_runs_model_ts", table_name="inference_runs")
    op.drop_index("idx_inference_runs_provider_ts", table_name="inference_runs")
    op.drop_index("ix_inference_runs_timestamp", table_name="inference_runs")
    op.drop_index("ix_inference_runs_author", table_name="inference_runs")
    op.drop_index("ix_inference_runs_provider_kind", table_name="inference_runs")
    op.drop_index("ix_inference_runs_task_hash", table_name="inference_runs")
    op.drop_index("ix_inference_runs_task_name", table_name="inference_runs")
    op.drop_index("ix_inference_runs_model_name", table_name="inference_runs")
    op.drop_index("ix_inference_runs_model_hash", table_name="inference_runs")
    op.drop_index("ix_inference_runs_experiment_group", table_name="inference_runs")
    op.drop_index("ix_inference_runs_experiment_id", table_name="inference_runs")
    op.drop_table("inference_runs")
