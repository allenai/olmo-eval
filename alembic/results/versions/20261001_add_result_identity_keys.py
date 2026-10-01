"""Add unique keys so saving a result twice updates it instead of duplicating it.

- experiments: unique on (experiment_id, model_name, model_hash)
- task_results: unique on (experiment_pk, task_name)

Rows that already break these keys are removed first, keeping the most recently
inserted row of each set. Deleting a duplicate experiment also deletes its task
results and instance predictions through the existing cascading foreign keys.

Revision ID: add_result_identity_keys
Revises: add_instance_failure_accounting
Create Date: 2026-10-01
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "add_result_identity_keys"
down_revision: str | None = "add_instance_failure_accounting"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            DELETE FROM experiments e
            USING experiments newer
            WHERE e.experiment_id = newer.experiment_id
              AND e.model_name = newer.model_name
              AND e.model_hash = newer.model_hash
              AND e.id < newer.id
            """
        )
    )
    op.execute(
        sa.text(
            """
            DELETE FROM task_results t
            USING task_results newer
            WHERE t.experiment_pk = newer.experiment_pk
              AND t.task_name = newer.task_name
              AND t.id < newer.id
            """
        )
    )
    op.create_unique_constraint(
        "uq_experiments_identity",
        "experiments",
        ["experiment_id", "model_name", "model_hash"],
    )
    op.create_unique_constraint(
        "uq_task_results_experiment_task",
        "task_results",
        ["experiment_pk", "task_name"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_task_results_experiment_task", "task_results", type_="unique")
    op.drop_constraint("uq_experiments_identity", "experiments", type_="unique")
