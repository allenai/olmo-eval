"""Add GIN index on metadata for GPU summary queries.

Supports efficient queries on gpu_summary fields like:
- WHERE metadata_->'gpu_summary' IS NOT NULL
- WHERE metadata_ @> '{"gpu_summary": {"device_count": 1}}'

Revision ID: add_gpu_summary_index
Revises: rename_to_samples
Create Date: 2026-02-20
"""

from collections.abc import Sequence

from alembic import op

revision: str = "add_gpu_summary_index"
down_revision: str = "rename_to_samples"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add GIN index on metadata for efficient JSONB queries
    # Uses jsonb_path_ops for containment queries (@>, ?)
    op.create_index(
        "idx_inference_samples_metadata_gin",
        "inference_samples",
        ["metadata"],
        postgresql_using="gin",
        postgresql_ops={"metadata": "jsonb_path_ops"},
    )


def downgrade() -> None:
    op.drop_index("idx_inference_samples_metadata_gin", table_name="inference_samples")
