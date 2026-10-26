"""Carry observed worker settings into result identity before releasing queued work."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from olmo_eval.common.types import compute_model_hash
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.runners.asynq.batching import BatchConfig


def resolved_worker_config(
    config: HarnessConfig,
    batch: BatchConfig,
    *,
    profile: dict[str, Any] | None = None,
    packages: dict[str, str] | None = None,
) -> HarnessConfig:
    """Preserve requested provider options, adding only stable observed identity.

    Free memory, device ordinals and worker IDs stay in diagnostic artifacts;
    they must not create a new model hash on every launch. Engine settings and
    package revisions do participate. Replaying a config refreshes this field.
    """
    provider = config.provider
    if str(provider.kind) == "sglang":
        if profile is None:
            raise RuntimeError("SGLang worker did not report its resolved profile")
        runtime = {
            "profile": profile["selected"],
            "engine_options": profile["effective_options"],
            "packages": packages or {},
            "logprobs": "original",
            "arithmetic_environment": dict(profile.get("arithmetic_environment", {})),
        }
        if "effective_max_running_requests" in profile:
            runtime["effective_max_running_requests"] = profile["effective_max_running_requests"]
        provider = replace(provider, resolved_runtime=runtime)
    resolved = replace(config, provider=provider, batching=batch)
    if str(provider.kind) == "sglang" and resolved.metrics is not None and resolved.metrics.enabled:
        resolved = resolved.with_metrics(
            resolved.metrics.with_metadata(model_hash=compute_model_hash(provider.to_dict()))
        )
    return resolved


def apply_worker_configs(
    requested: HarnessConfig, workers: dict[str, dict[str, Any]]
) -> HarnessConfig:
    """Require one runtime identity across replicas before any requests run."""
    if not workers:
        raise RuntimeError("No resolved inference worker configurations received")
    configs = [HarnessConfig.from_dict(value) for value in workers.values()]
    first = configs[0]
    for other in configs[1:]:
        if (
            other.provider.resolved_runtime != first.provider.resolved_runtime
            or other.batching != first.batching
        ):
            raise RuntimeError(
                "Inference workers resolved different runtime profiles or batch sizes; "
                "use homogeneous GPUs and explicit settings or evaluate them separately."
            )
    if str(requested.provider.kind) == "sglang" and not first.provider.resolved_runtime:
        raise RuntimeError("SGLang workers did not report resolved runtime identity")
    # Do not copy worker-local GPU allocation, log paths or metrics destinations.
    return replace(
        requested,
        provider=replace(requested.provider, resolved_runtime=first.provider.resolved_runtime),
        batching=first.batching,
    )
