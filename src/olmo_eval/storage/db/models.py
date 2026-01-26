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
    """ORM model for evaluation experiments."""

    __tablename__ = "experiments"

    # Primary key
    experiment_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Model identification
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_hash: Mapped[str | None] = mapped_column(String(64), index=True)  # Hash of model config
    backend_name: Mapped[str] = mapped_column(String(50), nullable=False)

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, index=True
    )

    # Experiment metadata
    experiment_name: Mapped[str | None] = mapped_column(String(255))
    workspace: Mapped[str | None] = mapped_column(String(100), index=True)
    author: Mapped[str | None] = mapped_column(String(100), index=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    # Version tracking
    git_ref: Mapped[str | None] = mapped_column(String(100))
    revision: Mapped[str | None] = mapped_column(String(255))

    # S3 reference for full evaluation data
    s3_location: Mapped[str | None] = mapped_column(String(512))

    # Flexible storage (JSONB for efficient querying)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
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
        lazy="selectin",  # Eager load for dashboard queries
    )
    instance_predictions: Mapped[list[InstancePrediction]] = relationship(
        "InstancePrediction",
        back_populates="experiment",
        cascade="all, delete-orphan",
        lazy="select",  # Don't eager load instances (can be large)
    )

    def __repr__(self) -> str:
        return (
            f"<Experiment(id={self.experiment_id!r}, model={self.model_name!r}, "
            f"timestamp={self.timestamp})>"
        )


class TaskResult(Base):
    """ORM model for task-level aggregated results."""

    __tablename__ = "task_results"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key with cascade delete
    experiment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("experiments.experiment_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Task metadata
    task_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    task_hash: Mapped[str | None] = mapped_column(String(64))

    # Aggregated metrics (task-level)
    metrics: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False)
    num_instances: Mapped[int | None] = mapped_column(Integer)
    primary_metric: Mapped[str | None] = mapped_column(String(100))
    primary_score: Mapped[float | None] = mapped_column(DOUBLE_PRECISION, index=True)

    # S3 keys for detailed task data
    s3_metrics_key: Mapped[str | None] = mapped_column(String(512))
    s3_predictions_key: Mapped[str | None] = mapped_column(String(512))

    # Relationships
    experiment: Mapped[Experiment] = relationship("Experiment", back_populates="task_results")

    def __repr__(self) -> str:
        return (
            f"<TaskResult(id={self.id}, experiment_id={self.experiment_id!r}, "
            f"task={self.task_name!r}, score={self.primary_score})>"
        )


class InstancePrediction(Base):
    """ORM model for instance-level predictions."""

    __tablename__ = "instance_predictions"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key with cascade delete
    experiment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("experiments.experiment_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    # Task identification
    task_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Instance identification (for matching across experiments)
    native_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )  # Original dataset ID
    doc_id: Mapped[int] = mapped_column(Integer, nullable=False)  # Sequential ID within task

    # Instance-level metrics
    instance_metrics: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False)

    # S3 key for full prediction (generation, logprobs, etc.)
    # Only loaded when user needs to inspect actual generations
    s3_prediction_key: Mapped[str | None] = mapped_column(String(512))

    # Relationships
    experiment: Mapped[Experiment] = relationship(
        "Experiment", back_populates="instance_predictions"
    )

    def __repr__(self) -> str:
        return (
            f"<InstancePrediction(id={self.id}, experiment_id={self.experiment_id!r}, "
            f"task={self.task_name!r}, native_id={self.native_id!r})>"
        )


# ==============================================================================
# INDEXES
# ==============================================================================

# Task Results Indexes
Index("idx_task_results_exp_task", TaskResult.experiment_id, TaskResult.task_name)
Index("idx_task_results_score_desc", TaskResult.primary_score.desc())

# Instance Predictions Indexes
Index(
    "idx_instance_model_task",
    InstancePrediction.model_hash,
    InstancePrediction.task_name,
)
Index(
    "idx_instance_task_native",
    InstancePrediction.task_name,
    InstancePrediction.native_id,
)
Index(
    "idx_instance_exp_task",
    InstancePrediction.experiment_id,
    InstancePrediction.task_name,
)

# Experiment Indexes
Index("idx_experiments_model_hash", Experiment.model_hash)
Index("idx_experiments_model_name", Experiment.model_name)
Index("idx_experiments_model_name_ts", Experiment.model_name, Experiment.timestamp.desc())
