"""SQLAlchemy ORM models for inference metrics storage."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import ARRAY, TIMESTAMP, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class MetricsBase(DeclarativeBase):
    """Base class for metrics ORM models.

    Separate from the evaluation Base to support a separate database.
    """

    pass


class InferenceSample(MetricsBase):
    """ORM model for batch-level inference metrics.

    Stores aggregate statistics for a batch of inference requests, along with
    metadata fields that mirror the evaluation schema for join-ability.

    The experiment_id field can be used to join with the experiments table
    for cross-referencing evaluation results with inference performance.
    """

    __tablename__ = "inference_samples"

    # Primary key - auto-increment ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Core metadata (mirrors evaluation schema for joins)
    experiment_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    experiment_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    experiment_group: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    model_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    task_name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    task_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    workspace: Mapped[str | None] = mapped_column(String(255), nullable=True)
    author: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Provider identification
    provider_kind: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)

    # Timestamps
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default="NOW()"
    )

    # Batch aggregate statistics
    total_requests: Mapped[int] = mapped_column(Integer, nullable=False)
    successful_requests: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_requests: Mapped[int] = mapped_column(Integer, nullable=False)
    total_prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)

    # Timing metrics
    wall_clock_time_s: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)
    output_tokens_per_second: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)

    # Latency statistics
    mean_latency_s: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)

    # User-defined tags (for special filtering beyond core metadata)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    # Flexible storage for additional data (e.g., GPU snapshots)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)

    def __repr__(self) -> str:
        return (
            f"<InferenceSample(id={self.id}, experiment_id={self.experiment_id!r}, "
            f"model={self.model_name!r}, provider={self.provider_kind!r}, "
            f"requests={self.total_requests}, timestamp={self.timestamp})>"
        )


# ==============================================================================
# INDEXES
# ==============================================================================

Index("idx_inference_samples_experiment_id", InferenceSample.experiment_id)
Index("idx_inference_samples_experiment_group", InferenceSample.experiment_group)
Index("idx_inference_samples_model_hash", InferenceSample.model_hash)
Index("idx_inference_samples_model_name", InferenceSample.model_name)
Index(
    "idx_inference_samples_provider_ts",
    InferenceSample.provider_kind,
    InferenceSample.timestamp.desc(),
)
Index(
    "idx_inference_samples_model_ts",
    InferenceSample.model_name,
    InferenceSample.timestamp.desc(),
)

# Composite index for experiment + task filtering
Index(
    "idx_inference_samples_exp_task",
    InferenceSample.experiment_id,
    InferenceSample.task_hash,
)
