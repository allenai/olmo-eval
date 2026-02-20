"""Restructure metrics to nested format and remove redundant columns.

This migration:
1. Removes agent_metrics column (metrics now unified in nested metrics structure)
2. Removes primary_score column (can be derived from metrics + primary_metric)
3. Drops idx_task_results_score_desc index (primary_score column removed)

The metrics column is now expected to contain nested structure:
    {metric_name: {scorer_name: value}}

IMPORTANT: This migration requires existing data to be migrated or truncated.
Run `TRUNCATE TABLE experiments CASCADE;` before applying if backwards
compatibility with old data is not needed.

Revision ID: nested_metrics
Revises: add_duration_metrics
Create Date: 2026-02-01
"""

from collections.abc import Sequence

from alembic import op

revision: str = "nested_metrics"
down_revision: str | None = "add_duration_metrics"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop the index on primary_score first
    op.drop_index("idx_task_results_score_desc", table_name="task_results")

    # Drop agent_metrics column (now part of unified metrics structure)
    op.drop_column("task_results", "agent_metrics")

    # Drop primary_score column (derived from metrics[metric][scorer])
    op.drop_column("task_results", "primary_score")


def downgrade() -> None:
    import sqlalchemy as sa
    from sqlalchemy.dialects.postgresql import JSONB

    # Re-add primary_score column
    op.add_column(
        "task_results",
        sa.Column("primary_score", sa.DOUBLE_PRECISION(), nullable=True),
    )

    # Re-add agent_metrics column
    op.add_column(
        "task_results",
        sa.Column(
            "agent_metrics",
            JSONB,
            nullable=True,
            comment="Nested agent evaluation metrics (tool, abstention, trajectory, etc.)",
        ),
    )

    # Re-create the index on primary_score
    op.create_index(
        "idx_task_results_score_desc",
        "task_results",
        [sa.text("primary_score DESC")],
    )
