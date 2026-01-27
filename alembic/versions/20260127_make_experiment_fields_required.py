"""Make experiment fields non-nullable (only tags can be null)

Revision ID: 002_required_fields
Revises: 001_initial
Create Date: 2026-01-27

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002_required_fields'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # First, update workspace column length from 100 to 255
    op.alter_column('experiments', 'workspace',
                    type_=sa.String(length=255),
                    existing_type=sa.String(length=100))

    # Make columns non-nullable (only tags should remain nullable)
    op.alter_column('experiments', 'model_hash',
                    existing_type=sa.String(length=64),
                    nullable=False)
    op.alter_column('experiments', 'experiment_name',
                    existing_type=sa.String(length=255),
                    nullable=False)
    op.alter_column('experiments', 'workspace',
                    existing_type=sa.String(length=255),
                    nullable=False)
    op.alter_column('experiments', 'author',
                    existing_type=sa.String(length=100),
                    nullable=False)
    op.alter_column('experiments', 'git_ref',
                    existing_type=sa.String(length=100),
                    nullable=False)
    op.alter_column('experiments', 'revision',
                    existing_type=sa.String(length=255),
                    nullable=False)


def downgrade() -> None:
    # Revert columns to nullable
    op.alter_column('experiments', 'revision',
                    existing_type=sa.String(length=255),
                    nullable=True)
    op.alter_column('experiments', 'git_ref',
                    existing_type=sa.String(length=100),
                    nullable=True)
    op.alter_column('experiments', 'author',
                    existing_type=sa.String(length=100),
                    nullable=True)
    op.alter_column('experiments', 'workspace',
                    existing_type=sa.String(length=255),
                    nullable=True)
    op.alter_column('experiments', 'experiment_name',
                    existing_type=sa.String(length=255),
                    nullable=True)
    op.alter_column('experiments', 'model_hash',
                    existing_type=sa.String(length=64),
                    nullable=True)

    # Revert workspace column length from 255 to 100
    op.alter_column('experiments', 'workspace',
                    type_=sa.String(length=100),
                    existing_type=sa.String(length=255))
