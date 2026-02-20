"""SQLAlchemy ORM models for inference metrics storage."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import ARRAY, TIMESTAMP, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class MetricsBase(DeclarativeBase):
    """Base class for metrics ORM models.

    Separate from the evaluation Base to support a separate database.
    """

    pass


class InferenceRun(MetricsBase):
    """ORM model for batch-level inference metrics.

    Stores aggregate statistics for a batch of inference requests, along with
    metadata fields that mirror the evaluation schema for join-ability.

    The experiment_id field can be used to join with the experiments table
    for cross-referencing evaluation results with inference performance.
    """

    __tablename__ = "inference_runs"

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

    # Relationships
    request_metrics: Mapped[list[InferenceRequestMetric]] = relationship(
        "InferenceRequestMetric",
        back_populates="inference_run",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:
        return (
            f"<InferenceRun(id={self.id}, experiment_id={self.experiment_id!r}, "
            f"model={self.model_name!r}, provider={self.provider_kind!r}, "
            f"requests={self.total_requests}, timestamp={self.timestamp})>"
        )


class InferenceRequestMetric(MetricsBase):
    """ORM model for per-request inference metrics.

    Stores detailed timing and token metrics for individual inference requests,
    linked to an InferenceRun for batch-level context.
    """

    __tablename__ = "inference_request_metrics"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to inference_runs
    inference_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("inference_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Request identification
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Token counts
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)

    # Timing metrics
    end_to_end_latency_s: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)
    tokens_per_second: Mapped[float] = mapped_column(DOUBLE_PRECISION, nullable=False)
    time_to_first_token_s: Mapped[float | None] = mapped_column(DOUBLE_PRECISION, nullable=True)
    time_per_output_token_s: Mapped[float | None] = mapped_column(DOUBLE_PRECISION, nullable=True)

    # Request outcome
    finish_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False)

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, index=True
    )

    # Relationships
    inference_run: Mapped[InferenceRun] = relationship(
        "InferenceRun", back_populates="request_metrics"
    )

    def __repr__(self) -> str:
        return (
            f"<InferenceRequestMetric(id={self.id}, request_id={self.request_id!r}, "
            f"prompt_tokens={self.prompt_tokens}, completion_tokens={self.completion_tokens}, "
            f"latency={self.end_to_end_latency_s:.3f}s)>"
        )


# ==============================================================================
# INDEXES
# ==============================================================================

# InferenceRun Indexes
Index("idx_inference_runs_experiment_id", InferenceRun.experiment_id)
Index("idx_inference_runs_experiment_group", InferenceRun.experiment_group)
Index("idx_inference_runs_model_hash", InferenceRun.model_hash)
Index("idx_inference_runs_model_name", InferenceRun.model_name)
Index("idx_inference_runs_provider_ts", InferenceRun.provider_kind, InferenceRun.timestamp.desc())
Index("idx_inference_runs_model_ts", InferenceRun.model_name, InferenceRun.timestamp.desc())

# Composite index for experiment + task filtering
Index(
    "idx_inference_runs_exp_task",
    InferenceRun.experiment_id,
    InferenceRun.task_hash,
)

# InferenceRequestMetric Indexes
Index("idx_request_metrics_run_id", InferenceRequestMetric.inference_run_id)
Index("idx_request_metrics_request_id", InferenceRequestMetric.request_id)
Index(
    "idx_request_metrics_run_ts",
    InferenceRequestMetric.inference_run_id,
    InferenceRequestMetric.timestamp,
)
