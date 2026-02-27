"""Drop percentile latency columns.

These columns are not useful because we only take one snapshot per batch,
so percentile calculations don't make sense.

Revision ID: drop_percentile_latencies
Revises: initial_schema
Create Date: 2026-02-20
"""

from collections.abc import Sequence

from alembic import op

revision: str = "drop_percentile_latencies"
down_revision: str = "initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop the percentile latency columns that were added manually
    op.drop_column("inference_runs", "p50_latency_s")
    op.drop_column("inference_runs", "p95_latency_s")
    op.drop_column("inference_runs", "p99_latency_s")


def downgrade() -> None:
    # Re-add the columns if needed (nullable to avoid issues)
    from sqlalchemy import Column
    from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION

    op.add_column(
        "inference_runs",
        Column("p50_latency_s", DOUBLE_PRECISION(), nullable=True),
    )
    op.add_column(
        "inference_runs",
        Column("p95_latency_s", DOUBLE_PRECISION(), nullable=True),
    )
    op.add_column(
        "inference_runs",
        Column("p99_latency_s", DOUBLE_PRECISION(), nullable=True),
    )
