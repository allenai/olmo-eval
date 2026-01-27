"""Rename config to model_config and add task_hash to instance_predictions

Revision ID: 003_rename_config_add_task_hash
Revises: 002_required_fields
Create Date: 2026-01-27

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003_rename_config_add_task_hash'
down_revision = '002_required_fields'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename config column to model_config in experiments table
    op.alter_column('experiments', 'config',
                    new_column_name='model_config',
                    existing_type=postgresql.JSONB())

    # Make task_hash required in task_results table and add index
    op.alter_column('task_results', 'task_hash',
                    existing_type=sa.String(length=64),
                    nullable=False)
    op.create_index('ix_task_results_task_hash',
                    'task_results', ['task_hash'])

    # Add task_hash column to instance_predictions table
    op.add_column('instance_predictions',
                  sa.Column('task_hash', sa.String(64), nullable=False))

    # Make model_hash required in instance_predictions
    op.alter_column('instance_predictions', 'model_hash',
                    existing_type=sa.String(length=64),
                    nullable=False)

    # Add index on task_hash in instance_predictions
    op.create_index('ix_instance_predictions_task_hash',
                    'instance_predictions', ['task_hash'])


def downgrade() -> None:
    # Drop index on task_hash in instance_predictions
    op.drop_index('ix_instance_predictions_task_hash', table_name='instance_predictions')

    # Make model_hash nullable again
    op.alter_column('instance_predictions', 'model_hash',
                    existing_type=sa.String(length=64),
                    nullable=True)

    # Drop task_hash column from instance_predictions
    op.drop_column('instance_predictions', 'task_hash')

    # Drop index on task_hash in task_results and make nullable again
    op.drop_index('ix_task_results_task_hash', table_name='task_results')
    op.alter_column('task_results', 'task_hash',
                    existing_type=sa.String(length=64),
                    nullable=True)

    # Rename model_config back to config
    op.alter_column('experiments', 'model_config',
                    new_column_name='config',
                    existing_type=postgresql.JSONB())
