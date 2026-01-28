"""SQLAlchemy ORM models for evaluation storage."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import ARRAY, TIMESTAMP, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Experiment(Base):
    """ORM model for evaluation experiments.

    Note: experiment_id is NOT unique - a single experiment launch can run
    multiple models, each as a separate row sharing the same experiment_id.
    The auto-increment `id` is the true primary key.
    """

    __tablename__ = "experiments"

    # Primary key - auto-increment ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Experiment ID (NOT unique - can have duplicates when running multiple models)
    experiment_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Model identification
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    backend_name: Mapped[str] = mapped_column(String(50), nullable=False)

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, index=True
    )

    # Experiment metadata
    experiment_name: Mapped[str] = mapped_column(String(255), nullable=False)
    workspace: Mapped[str] = mapped_column(String(255), nullable=False)
    author: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    # Version tracking
    git_ref: Mapped[str] = mapped_column(String(100), nullable=False)
    revision: Mapped[str] = mapped_column(String(255), nullable=False)

    # S3 reference for full evaluation data
    s3_location: Mapped[str | None] = mapped_column(String(512))

    # Original model path (when alias is used, model_name is the alias)
    model_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Flexible storage (JSONB for efficient querying)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)

    # Audit timestamp
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="NOW()"
    )

    # Relationships
    task_results: Mapped[list[TaskResult]] = relationship(
        "TaskResult",
        back_populates="experiment",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    instance_predictions: Mapped[list[InstancePrediction]] = relationship(
        "InstancePrediction",
        back_populates="experiment",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:
        return (
            f"<Experiment(id={self.id}, experiment_id={self.experiment_id!r}, "
            f"model={self.model_name!r}, timestamp={self.timestamp})>"
        )


class TaskResult(Base):
    """ORM model for task-level aggregated results.

    Contains task config/metadata, top-level task metrics, and file paths.
    """

    __tablename__ = "task_results"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to experiments.id (NOT experiment_id)
    experiment_pk: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Denormalized model_hash for query convenience
    model_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Task identification
    task_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    task_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    task_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # Aggregated metrics (task-level)
    metrics: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False)
    num_instances: Mapped[int | None] = mapped_column(Integer)
    primary_metric: Mapped[str | None] = mapped_column(String(100))
    primary_score: Mapped[float | None] = mapped_column(DOUBLE_PRECISION, index=True)

    # S3 keys for detailed task data
    s3_metrics_key: Mapped[str | None] = mapped_column(String(512))
    s3_predictions_key: Mapped[str | None] = mapped_column(String(512))
    s3_requests_key: Mapped[str | None] = mapped_column(String(512))

    # Relationships
    experiment: Mapped[Experiment] = relationship("Experiment", back_populates="task_results")

    def __repr__(self) -> str:
        return (
            f"<TaskResult(id={self.id}, experiment_pk={self.experiment_pk}, "
            f"task={self.task_name!r}, score={self.primary_score})>"
        )


class InstancePrediction(Base):
    """ORM model for instance-level predictions.

    Join against task_results and experiments for aggregates/configs.
    Removed: model_hash, task_name, s3_prediction_key, doc_id - get via JOIN when needed.
    """

    __tablename__ = "instance_predictions"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to experiments.id
    experiment_pk: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Task hash for joining to task_results
    task_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Instance identification
    native_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Instance-level metrics
    instance_metrics: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False)

    # Relationships
    experiment: Mapped[Experiment] = relationship(
        "Experiment", back_populates="instance_predictions"
    )

    def __repr__(self) -> str:
        return (
            f"<InstancePrediction(id={self.id}, experiment_pk={self.experiment_pk}, "
            f"task_hash={self.task_hash!r}, native_id={self.native_id!r})>"
        )


# ==============================================================================
# INDEXES
# ==============================================================================

# Experiments Indexes
Index("idx_experiments_experiment_id", Experiment.experiment_id)
Index("idx_experiments_model_hash", Experiment.model_hash)
Index("idx_experiments_model_name", Experiment.model_name)
Index("idx_experiments_model_name_ts", Experiment.model_name, Experiment.timestamp.desc())

# Task Results Indexes
Index("idx_task_results_exp_task", TaskResult.experiment_pk, TaskResult.task_name)
Index("idx_task_results_model_task", TaskResult.model_hash, TaskResult.task_name)
Index("idx_task_results_task_hash", TaskResult.task_hash)
Index("idx_task_results_score_desc", TaskResult.primary_score.desc())

# Instance Predictions Indexes
Index(
    "idx_instance_exp_task_hash",
    InstancePrediction.experiment_pk,
    InstancePrediction.task_hash,
)
Index(
    "idx_instance_task_hash_native",
    InstancePrediction.task_hash,
    InstancePrediction.native_id,
)
# Composite index for efficient keyset pagination within experiments
Index(
    "idx_instance_exp_id",
    InstancePrediction.experiment_pk,
    InstancePrediction.id,
)
