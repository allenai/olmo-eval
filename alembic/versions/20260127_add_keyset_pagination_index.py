"""Add composite index for keyset pagination.

Revision ID: add_keyset_pagination_index
Revises: initial_schema
Create Date: 2026-01-27
"""

from collections.abc import Sequence

from alembic import op

revision: str = "add_keyset_pagination_index"
down_revision: str | None = "initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add composite index for efficient keyset pagination within experiments.
    # This index supports queries like:
    #   WHERE experiment_pk = ? AND id > ? ORDER BY id
    # which are much faster than OFFSET-based pagination for large datasets.
    op.create_index(
        "idx_instance_exp_id",
        "instance_predictions",
        ["experiment_pk", "id"],
    )


def downgrade() -> None:
    op.drop_index("idx_instance_exp_id", table_name="instance_predictions")
