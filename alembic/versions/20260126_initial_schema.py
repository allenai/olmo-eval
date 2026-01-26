"""Initial schema with experiments, task_results, and instance_predictions tables

Revision ID: 001_initial
Revises:
Create Date: 2026-01-26 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create experiments table
    op.create_table(
        'experiments',
        sa.Column('experiment_id', sa.String(length=64), nullable=False),
        sa.Column('model_name', sa.String(length=255), nullable=False),
        sa.Column('model_hash', sa.String(length=64), nullable=True),
        sa.Column('backend_name', sa.String(length=50), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('experiment_name', sa.String(length=255), nullable=True),
        sa.Column('workspace', sa.String(length=100), nullable=True),
        sa.Column('author', sa.String(length=100), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column('git_ref', sa.String(length=100), nullable=True),
        sa.Column('revision', sa.String(length=255), nullable=True),
        sa.Column('s3_location', sa.String(length=512), nullable=True),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.PrimaryKeyConstraint('experiment_id')
    )
    # Indexes for experiments
    op.create_index('idx_experiments_model_hash', 'experiments', ['model_hash'])
    op.create_index('idx_experiments_model_name', 'experiments', ['model_name'])
    op.create_index('idx_experiments_model_name_ts', 'experiments', ['model_name', sa.text('timestamp DESC')])
    op.create_index(op.f('ix_experiments_author'), 'experiments', ['author'])
    op.create_index(op.f('ix_experiments_timestamp'), 'experiments', ['timestamp'])
    op.create_index(op.f('ix_experiments_workspace'), 'experiments', ['workspace'])

    # Create task_results table
    op.create_table(
        'task_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('experiment_id', sa.String(length=64), nullable=False),
        sa.Column('task_name', sa.String(length=255), nullable=False),
        sa.Column('task_hash', sa.String(length=64), nullable=True),
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('num_instances', sa.Integer(), nullable=True),
        sa.Column('primary_metric', sa.String(length=100), nullable=True),
        sa.Column('primary_score', postgresql.DOUBLE_PRECISION(), nullable=True),
        sa.Column('s3_metrics_key', sa.String(length=512), nullable=True),
        sa.Column('s3_predictions_key', sa.String(length=512), nullable=True),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.experiment_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    # Indexes for task_results
    op.create_index('idx_task_results_exp_task', 'task_results', ['experiment_id', 'task_name'])
    op.create_index('idx_task_results_score_desc', 'task_results', [sa.text('primary_score DESC')])
    op.create_index(op.f('ix_task_results_experiment_id'), 'task_results', ['experiment_id'])
    op.create_index(op.f('ix_task_results_primary_score'), 'task_results', ['primary_score'])
    op.create_index(op.f('ix_task_results_task_name'), 'task_results', ['task_name'])

    # Create instance_predictions table
    op.create_table(
        'instance_predictions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('experiment_id', sa.String(length=64), nullable=False),
        sa.Column('model_hash', sa.String(length=64), nullable=True),
        sa.Column('task_name', sa.String(length=255), nullable=False),
        sa.Column('native_id', sa.String(length=255), nullable=False),
        sa.Column('doc_id', sa.Integer(), nullable=False),
        sa.Column('instance_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('s3_prediction_key', sa.String(length=512), nullable=True),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.experiment_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    # Indexes for instance_predictions
    op.create_index('idx_instance_exp_task', 'instance_predictions', ['experiment_id', 'task_name'])
    op.create_index('idx_instance_model_task', 'instance_predictions', ['model_hash', 'task_name'])
    op.create_index('idx_instance_task_native', 'instance_predictions', ['task_name', 'native_id'])
    op.create_index(op.f('ix_instance_predictions_experiment_id'), 'instance_predictions', ['experiment_id'])
    op.create_index(op.f('ix_instance_predictions_model_hash'), 'instance_predictions', ['model_hash'])
    op.create_index(op.f('ix_instance_predictions_native_id'), 'instance_predictions', ['native_id'])
    op.create_index(op.f('ix_instance_predictions_task_name'), 'instance_predictions', ['task_name'])


def downgrade() -> None:
    op.drop_table('instance_predictions')
    op.drop_table('task_results')
    op.drop_table('experiments')
