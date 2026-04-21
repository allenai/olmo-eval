"""Async evaluation runner with instance-level queuing."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import random
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from rich.console import Console

from olmo_eval.common.configs import expand_tasks
from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.common.logging import configure_worker_logging, get_logger
from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.runners.asynq.monitoring import (
    terminate_workers,
    wait_for_init_times,
    wait_for_workers_ready,
)
from olmo_eval.runners.asynq.preparation import (
    build_requests_from_items,
    prepare_task_items,
)
from olmo_eval.runners.asynq.results import aggregate_results, process_results
from olmo_eval.runners.asynq.types import QueueItem, TaskTracker
from olmo_eval.runners.common.base import BaseEvalRunner
from olmo_eval.runners.common.mixins import RunnerResultsMixin
from olmo_eval.runners.common.models import S3Config
from olmo_eval.runners.processing.utils import compute_task_hash, generate_experiment_id
from olmo_eval.storage import StorageBackend

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox import SandboxConfig

logger = get_logger(__name__)
runner_logger = configure_worker_logging("runner")
console = Console(force_terminal=True, width=120)

_DEFAULT_SANDBOX_ENV = "__default__"


@dataclass(frozen=True)
class _SandboxPlan:
    """Resolved sandbox configs and allocation data for relevant sandbox envs."""

    sandboxes: list[SandboxConfig]
    env_demand: dict[str, int]
    env_task_count: dict[str, int]
    allocated: dict[str, int]
    budget: int
    needs_default: bool


def _materialize_sandbox_instances(sandboxes: Sequence[SandboxConfig]) -> list[SandboxConfig]:
    """Replace auto-managed sandbox instance counts with their runtime defaults."""
    return [
        replace(cfg, instances=cfg.resolved_instances) if cfg.instances is None else cfg
        for cfg in sandboxes
    ]


def _collect_sandbox_demand(
    trackers: Mapping[str, TaskTracker],
) -> tuple[dict[str, int], dict[str, int]]:
    """Collect scoring demand and task counts per sandbox environment."""
    env_demand: dict[str, int] = {}
    env_task_count: dict[str, int] = {}
    for tracker in trackers.values():
        if tracker.task is None:
            continue
        tcfg = tracker.task.config
        num_samples = tcfg.sampling_params.num_samples if tcfg.sampling_params else 1
        scoring_items = tracker.total_instances * num_samples
        env_key = tcfg.sandbox_env.name if tcfg.sandbox_env else _DEFAULT_SANDBOX_ENV
        env_demand[env_key] = env_demand.get(env_key, 0) + scoring_items
        env_task_count[env_key] = env_task_count.get(env_key, 0) + 1
    return env_demand, env_task_count


def _allocate_auto_sandbox_instances(
    auto_env_keys: Sequence[str],
    env_demand: Mapping[str, int],
    sandbox_pool_instances: int | None,
) -> dict[str, int]:
    """Allocate the shared sandbox pool across auto-managed environments."""
    if not auto_env_keys:
        return {}
    if sandbox_pool_instances is None:
        return {env_key: 1 for env_key in auto_env_keys}

    budget = sandbox_pool_instances
    min_budget = len(auto_env_keys)
    if budget < min_budget:
        logger.warning(
            f"Sandbox pool budget ({budget}) is less than the number of "
            f"auto-allocated environments ({min_budget}); clamping to {min_budget}"
        )
        budget = min_budget

    total_demand = sum(env_demand.get(env_key, 0) for env_key in auto_env_keys)
    if total_demand <= 0:
        base = budget // len(auto_env_keys)
        remainder = budget % len(auto_env_keys)
        allocated: dict[str, int] = {}
        for i, env_key in enumerate(auto_env_keys):
            allocated[env_key] = base + (1 if i < remainder else 0)
        return allocated

    distributable = budget - len(auto_env_keys)
    allocated = {}
    for env_key in auto_env_keys:
        demand = env_demand.get(env_key, 0)
        extra = max(0, round(distributable * demand / total_demand))
        allocated[env_key] = 1 + extra

    diff = budget - sum(allocated.values())
    if diff != 0:
        top = max(auto_env_keys, key=lambda key: env_demand.get(key, 0))
        allocated[top] = max(1, allocated[top] + diff)
    return allocated


def _plan_sandbox_configs(
    base_sandboxes: Sequence[SandboxConfig],
    expanded_tasks: Sequence[str],
    trackers: Mapping[str, TaskTracker],
    sandbox_pool_instances: int | None,
) -> _SandboxPlan | None:
    """Resolve relevant sandbox configs and concrete executor counts."""
    from olmo_eval.evals.tasks.common import get_sandbox_envs, get_task
    from olmo_eval.harness.sandbox.image import dependencies_to_dockerfile_extra

    sandbox_envs = get_sandbox_envs(list(expanded_tasks))
    needs_default = any(get_task(spec).config.sandbox_env is None for spec in expanded_tasks)
    if not sandbox_envs and not needs_default:
        return None

    template = base_sandboxes[0]
    used_caps = {senv.capability for senv in sandbox_envs}
    sandboxes = [
        cfg
        for i, cfg in enumerate(base_sandboxes)
        if (i == 0 and needs_default) or (i != 0 and cfg.capabilities in used_caps)
    ]
    env_demand, env_task_count = _collect_sandbox_demand(trackers)

    if not needs_default:
        env_demand.pop(_DEFAULT_SANDBOX_ENV, None)
        env_task_count.pop(_DEFAULT_SANDBOX_ENV, None)

    env_to_index: dict[str, int] = {}
    if needs_default and sandboxes:
        env_to_index[_DEFAULT_SANDBOX_ENV] = 0

    for senv in sandbox_envs:
        extra = dependencies_to_dockerfile_extra(senv.dependencies)
        match_idx = next(
            (i for i, cfg in enumerate(sandboxes) if cfg.capabilities == senv.capability),
            None,
        )
        if match_idx is not None:
            matched = sandboxes[match_idx]
            sandboxes[match_idx] = replace(
                matched,
                dockerfile_extra=matched.dockerfile_extra + extra + senv.dockerfile_extra,
            )
            env_to_index[senv.name] = match_idx
        else:
            sandboxes.append(
                replace(
                    template,
                    capabilities=senv.capability,
                    dockerfile_extra=template.dockerfile_extra + extra + senv.dockerfile_extra,
                    inject_swerex=True,
                    instances=None,
                )
            )
            env_to_index[senv.name] = len(sandboxes) - 1

    explicit_allocations: dict[str, int] = {}
    for env_key, idx in env_to_index.items():
        instances = sandboxes[idx].instances
        if instances is not None:
            explicit_allocations[env_key] = instances
    auto_env_keys = [env_key for env_key in env_to_index if env_key not in explicit_allocations]
    auto_allocations = _allocate_auto_sandbox_instances(
        auto_env_keys,
        env_demand,
        sandbox_pool_instances,
    )
    allocated: dict[str, int] = dict(explicit_allocations)
    allocated.update(auto_allocations)

    materialized_sandboxes = list(sandboxes)
    for env_key, idx in env_to_index.items():
        materialized_sandboxes[idx] = replace(
            materialized_sandboxes[idx],
            instances=allocated[env_key],
        )

    return _SandboxPlan(
        sandboxes=materialized_sandboxes,
        env_demand=env_demand,
        env_task_count=env_task_count,
        allocated=allocated,
        budget=sum(allocated.values()),
        needs_default=needs_default,
    )


@dataclass
class AsyncEvalRunner(RunnerResultsMixin, BaseEvalRunner):
    """Async evaluation runner with instance-level queuing.

    Uses a single model with instance-level queuing where instances from all
    tasks are mixed together, enabling better GPU utilization and early
    completion reporting.
    """

    # Harness configuration (includes provider, tools, system prompt)
    harness_config: HarnessConfig = field(default_factory=lambda: HarnessConfig(name="default"))

    # Task configuration
    task_specs: list[str] = field(default_factory=list)
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Output configuration
    output_dir: str = BEAKER_RESULT_DIR
    storages: list[StorageBackend] = field(default_factory=list)

    # vLLM-specific configuration
    attention_backend: str | None = None

    # S3 upload configuration (optional)
    s3_config: S3Config | None = None

    # Experiment metadata
    experiment_name: str | None = None
    experiment_group: str | None = None

    # Output persistence options
    save_predictions: bool = True
    save_requests: bool = True

    # Shuffle seed for deterministic instance ordering (enables future checkpointing)
    shuffle_seed: int = 42

    # Instance inspection options
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_response: bool = False
    inspect_request: bool = False

    # Configuration for print_config display
    _mode_name: str = "Async Mode"
    _mode_description: str = "Async (All-at-once)"

    @property
    def provider_config(self) -> ProviderConfig:
        """Get the provider config from harness config."""
        return self.harness_config.provider

    @property
    def model_name(self) -> str:
        """Get the model name from provider config."""
        return self.provider_config.model

    def validate(self) -> None:
        """Validate runner configuration."""
        from olmo_eval.runners.common.constants import ValidationError

        if not self.provider_config.model:
            raise ValidationError("provider_config.model is required")

        if not self.task_specs:
            raise ValidationError("task_specs is required")

        # Validate task specs
        errors = self._validate_task_specs()
        if errors:
            raise ValidationError("\n".join(errors))

    def print_config(self) -> None:
        """Print runner configuration."""
        from rich.table import Table

        from olmo_eval.common.configs import expand_tasks

        table = Table(title=f"Run Configuration ({self._mode_name})")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Model", self.model_name)
        table.add_row("Provider", self.provider_config.get_provider_name())
        table.add_row("Mode", self._mode_description)
        table.add_row("Output Dir", self.output_dir)

        console.print(table)

        expanded = expand_tasks(self.task_specs)
        console.print(f"\n[bold]Tasks:[/bold] {len(expanded)}")
        for spec in expanded:
            console.print(f"  - {spec}")

    def run(self) -> dict[str, Any]:
        """Sync wrapper for async execution."""
        return asyncio.run(self.run_async())

    async def run_async(self) -> dict[str, Any]:
        """Execute evaluations using instance-level queuing."""
        # Track experiment start time
        experiment_start = time.time()

        # Generate experiment ID early so metrics can include it
        experiment_id = generate_experiment_id()

        # Prepare tasks
        expanded_tasks, trackers, items = self._prepare_tasks()
        total_instances = len(items)
        runner_logger.info(f"Total instances: {total_instances}")

        # Update harness metrics config with experiment metadata before starting workers
        self._update_metrics_config(experiment_id)

        # Setup multiprocessing
        ctx = mp.get_context("spawn")
        item_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue()
        total_gpus = self._get_gpu_count()

        # Queue for workers to report init times (replaces mp.Manager dict to
        # avoid a long-lived server process that can zombie and hang on shutdown)
        init_queue: mp.Queue = ctx.Queue()

        # Shuffle with seed for deterministic ordering (enables future checkpointing)
        rng = random.Random(self.shuffle_seed)
        rng.shuffle(items)
        for item in items:
            item_queue.put(item)

        # Start workers
        workers: list[mp.process.BaseProcess] = []
        sandbox_manager = None
        inference_manager = None

        try:
            from olmo_eval.inference.gpu_planner import GPUPlanner
            from olmo_eval.inference.provider_manager import (
                ProviderManager,
                validate_inference_workers,
            )

            # Use GPU planner to allocate GPUs for main workers and auxiliary providers
            planner = GPUPlanner.from_harness_config(self.harness_config, total_gpus)
            gpu_plan = planner.plan()

            # Get main worker GPUs (flattened list from all worker allocations)
            main_gpu_ids: list[int] = []
            for alloc in gpu_plan.main_workers:
                main_gpu_ids.extend(alloc.gpu_ids)

            # Number of inference workers is determined by provider.num_instances
            num_inference_workers = self.harness_config.provider.num_instances

            # Validate configuration
            validate_inference_workers(num_inference_workers, len(main_gpu_ids))

            provider_manager = ProviderManager(
                harness_config=self.harness_config,
                num_inference_workers=num_inference_workers,
                gpu_ids=main_gpu_ids,
                item_queue=item_queue,
                result_queue=result_queue,
                output_dir=self.output_dir,
            )

            # Add poison pills (one per worker)
            provider_manager.add_poison_pills()

            # Start workers
            workers = provider_manager.start(ctx, total_instances, init_queue)

            # Start auxiliary inference servers if configured
            registry_config: dict[str, list[dict[str, Any]]] | None = None
            if self.harness_config.auxiliary_providers:
                from olmo_eval.inference.manager import InferenceManager

                auxiliary_gpus = gpu_plan.get_auxiliary_gpus()

                inference_manager = InferenceManager(
                    configs=dict(self.harness_config.auxiliary_providers),
                    available_gpu_ids=auxiliary_gpus,
                )
                registry_config = inference_manager.start()
                runner_logger.info(f"Auxiliary providers ready: {list(registry_config.keys())}")

            # Prepare sandbox configs for scoring worker if configured
            sandbox_configs_list = None
            if self.harness_config.sandboxes:
                sandboxes = list(self.harness_config.sandboxes)

                sandbox_plan = _plan_sandbox_configs(
                    self.harness_config.sandboxes,
                    expanded_tasks,
                    trackers,
                    self.harness_config.sandbox_pool_instances,
                )
                if sandbox_plan is not None:
                    sandboxes = sandbox_plan.sandboxes

                    for env_key in sorted(sandbox_plan.env_demand):
                        if env_key == _DEFAULT_SANDBOX_ENV:
                            continue
                        runner_logger.info(
                            f"Sandbox env '{env_key}' "
                            f"({sandbox_plan.allocated.get(env_key, 1)} executors, "
                            f"{sandbox_plan.env_demand.get(env_key, 0)} scoring items)"
                        )
                    if sandbox_plan.needs_default:
                        runner_logger.info(
                            f"Sandbox env 'default' "
                            f"({sandbox_plan.allocated.get(_DEFAULT_SANDBOX_ENV, 1)} executors, "
                            f"{sandbox_plan.env_demand.get(_DEFAULT_SANDBOX_ENV, 0)} scoring items)"
                        )

                    # Print sandbox distribution summary
                    from rich.table import Table

                    dist_table = Table(title="Sandbox Distribution")
                    dist_table.add_column("Env", style="cyan")
                    dist_table.add_column("Tasks", justify="right")
                    dist_table.add_column("Scoring Items", justify="right")
                    dist_table.add_column("Executors", justify="right")
                    dist_table.add_column("Share", justify="right")

                    for env_key in sorted(sandbox_plan.env_demand):
                        display_name = "default" if env_key == _DEFAULT_SANDBOX_ENV else env_key
                        executors = sandbox_plan.allocated.get(env_key, 0)
                        share = (
                            f"{executors / sandbox_plan.budget * 100:.1f}%"
                            if sandbox_plan.budget > 0
                            else "0.0%"
                        )
                        dist_table.add_row(
                            display_name,
                            str(sandbox_plan.env_task_count.get(env_key, 0)),
                            str(sandbox_plan.env_demand.get(env_key, 0)),
                            str(executors),
                            share,
                        )

                    console.print(dist_table)
                else:
                    sandboxes = _materialize_sandbox_instances(sandboxes)

                # Pre-build unique sandbox images in parallel to avoid
                # duplicate builds when multiple executors share the same image
                from concurrent.futures import ThreadPoolExecutor

                from olmo_eval.harness.sandbox.image import get_swerex_image

                # Dedupe by (image, dockerfile_extra) — configs sharing these produce the same image
                unique_builds: dict[tuple[str, tuple[str, ...]], Any] = {}
                for cfg in sandboxes:
                    if cfg.inject_swerex:
                        key = (cfg.image, cfg.dockerfile_extra)
                        if key not in unique_builds:
                            unique_builds[key] = cfg

                if unique_builds:
                    runner_logger.info(
                        f"Pre-building {len(unique_builds)} unique sandbox image(s)..."
                    )

                    def _build(cfg: Any) -> None:
                        require_registry = cfg.mode.value == "modal"
                        get_swerex_image(
                            cfg.image,
                            cfg.container_runtime,
                            cfg.dockerfile_extra,
                            require_registry=require_registry,
                        )

                    with ThreadPoolExecutor(max_workers=len(unique_builds)) as pool:
                        list(pool.map(_build, unique_builds.values()))

                sandbox_configs_list = [s.to_dict() for s in sandboxes]

            # Determine scoring concurrency
            if self.harness_config.scoring_concurrency:
                scoring_concurrency = self.harness_config.scoring_concurrency
            elif sandbox_configs_list is not None:
                # Match sandbox pool size (sum of instances across all configs)
                sandbox_pool_size = sum((cfg.get("instances") or 1) for cfg in sandbox_configs_list)
                scoring_concurrency = max(1, sandbox_pool_size)
            else:
                # CPU-bound scoring - use available cores
                import os

                cpu_count = os.cpu_count() or 4
                scoring_concurrency = max(1, int(cpu_count * 0.75))

            runner_logger.info(f"Scoring concurrency: {scoring_concurrency}")

            # Initialize sandbox manager inline (no separate scorer process)
            sandbox_manager = None
            if sandbox_configs_list is not None:
                from olmo_eval.harness.sandbox import SandboxConfig, SandboxManager

                sandbox_configs = [SandboxConfig.from_dict(d) for d in sandbox_configs_list]
                runner_logger.info(
                    f"Initializing sandbox manager with {len(sandbox_configs)} config(s)..."
                )
                sandbox_manager = SandboxManager(sandbox_configs, owner="scorer")
                await sandbox_manager.start()
                runner_logger.info("Sandbox manager ready")

            # Create provider registry for auxiliary providers
            provider_registry = None
            if registry_config:
                from olmo_eval.inference.registry import ProviderRegistry

                provider_registry = ProviderRegistry.from_serialized(registry_config)
                if provider_registry:
                    runner_logger.info(
                        f"Provider registry ready with providers: {provider_registry.names}"
                    )

            from olmo_eval.common.execution import ScoringContext

            scoring_context = ScoringContext(
                execution_env=sandbox_manager,
                scoring_concurrency=scoring_concurrency,
                inference_pool=provider_registry,
            )

            # Ensure inference workers are ready before dispatching
            wait_for_workers_ready(workers, result_queue, startup_timeout=60.0)

            # Wait for workers to report their init times (also checks for crashes)
            provider_init_seconds = wait_for_init_times(
                init_queue, num_inference_workers, workers=workers, result_queue=result_queue
            )

            # Reset tracker start times now that workers are ready
            # This ensures task duration only measures actual processing time
            processing_start = time.time()
            for tracker in trackers.values():
                tracker.start_time = processing_start

            # Process results
            results = await self._process_results(
                trackers,
                result_queue,
                workers,
                scoring_context,
                scoring_concurrency,
                len(expanded_tasks),
                total_instances,
            )

            # Wait for all workers
            for worker in workers:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

            # Optionally inspect first response of each task
            if self.inspect_response:
                from olmo_eval.common.inspection import inspect_response

                for spec, tracker in trackers.items():
                    if tracker.responses:
                        first_response = next(iter(tracker.responses.values()))
                        console.print()
                        inspect_response(
                            first_response,
                            console=console,
                            task_name=spec,
                        )
                        break  # Only show first task's first response

            # Compute experiment duration
            experiment_duration_seconds = time.time() - experiment_start

            # Aggregate and save results
            results_dict = self._aggregate_results(results, expanded_tasks)
            return self._finalize_and_save(
                results_dict,
                experiment_id=experiment_id,
                experiment_duration_seconds=experiment_duration_seconds,
                provider_init_seconds=provider_init_seconds,
            )
        finally:
            terminate_workers(workers)
            if sandbox_manager is not None:
                runner_logger.info("Stopping sandbox manager...")
                with contextlib.suppress(Exception):
                    await sandbox_manager.stop()
            if inference_manager is not None:
                inference_manager.shutdown()
            for q in [item_queue, result_queue, init_queue]:
                q.cancel_join_thread()

    def _prepare_tasks(
        self,
    ) -> tuple[list[str], dict[str, TaskTracker], list[QueueItem]]:
        """Prepare all tasks and return tracking data structures."""
        expanded_tasks = expand_tasks(self.task_specs)

        if not expanded_tasks:
            raise ValueError("No tasks to run after expansion. Check task_specs configuration.")

        trackers: dict[str, TaskTracker] = {}
        items: list[QueueItem] = []

        def prepare_one(spec: str) -> tuple[str, TaskTracker, list[QueueItem]]:
            try:
                overrides, sampling_overrides = self._build_task_overrides(spec)
                task, task_items = prepare_task_items(
                    spec,
                    self.model_name,
                    overrides or None,
                    sampling_overrides=sampling_overrides or None,
                )
                tracker = TaskTracker(
                    model_name=self.model_name,
                    spec=spec,
                    task=task,
                    total_instances=len(task_items),
                )
                return (spec, tracker, task_items)
            except Exception as e:
                tracker = TaskTracker(
                    model_name=self.model_name,
                    spec=spec,
                    task=None,
                    total_instances=0,
                    error=str(e),
                )
                return (spec, tracker, [])

        from rich.table import Table

        # Collect prepared tasks in parallel, but accumulate results for deterministic ordering
        prepared_results: dict[str, tuple[TaskTracker, list[QueueItem]]] = {}
        with ThreadPoolExecutor(max_workers=min(32, len(expanded_tasks))) as executor:
            futures = {executor.submit(prepare_one, spec): spec for spec in expanded_tasks}
            for future in as_completed(futures):
                spec, tracker, task_items = future.result()
                trackers[spec] = tracker
                prepared_results[spec] = (tracker, task_items)

        # Add items in deterministic task spec order (not completion order)
        # This ensures shuffle produces identical results across runs
        for spec in expanded_tasks:
            tracker, task_items = prepared_results[spec]
            items.extend(task_items)
            if not tracker.error and self.save_requests and task_items and tracker.task:
                request_objects = build_requests_from_items(task_items, tracker.task.config.name)
                task_hash = compute_task_hash(tracker.task.config.to_dict())
                self._write_requests(self.model_name, spec, request_objects, task_hash)

        # Print task preparation summary table
        table = Table(title="Tasks")
        table.add_column("Task", style="cyan")
        table.add_column("Instances", justify="right")
        table.add_column("Status")

        for spec in expanded_tasks:
            tracker = trackers[spec]
            if tracker.error:
                table.add_row(spec, "-", f"[red]ERROR: {tracker.error}[/red]")
            else:
                table.add_row(spec, str(tracker.total_instances), "[green]Ready[/green]")

        console.print(table)

        # Fail fast if any tasks failed to prepare
        failed_tasks = [spec for spec, tracker in trackers.items() if tracker.error]
        if failed_tasks:
            error_details = "\n".join(
                f"  - {spec}: {trackers[spec].error}" for spec in failed_tasks
            )
            raise RuntimeError(f"Failed to prepare {len(failed_tasks)} task(s):\n{error_details}")

        # Optionally inspect first instance of each task
        if (
            self.inspect_instance
            or self.inspect_formatted
            or self.inspect_tokens
            or self.inspect_request
        ):
            from olmo_eval.common.inspection import inspect_task_instances

            inspect_task_instances(
                trackers,
                self.provider_config,
                inspect_instance_flag=self.inspect_instance,
                inspect_formatted=self.inspect_formatted,
                inspect_tokens_flag=self.inspect_tokens,
                inspect_request_flag=self.inspect_request,
                console=console,
            )

        return expanded_tasks, trackers, items

    def _build_task_overrides(self, spec: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build task and sampling overrides for a given task spec.

        Returns:
            Tuple of (task_overrides, sampling_overrides)
        """
        from dataclasses import fields

        from olmo_eval.common.types import SamplingParams
        from olmo_eval.evals.tasks.common.base import TaskConfig

        task_overrides: dict[str, Any] = {}
        sampling_overrides: dict[str, Any] = {}

        # Get field names from dataclasses
        task_fields = {f.name for f in fields(TaskConfig)}
        sampling_fields = {f.name for f in fields(SamplingParams)}

        # Apply per-task overrides
        per_task = self.task_overrides.get(spec, {})
        for key, value in per_task.items():
            if key in task_fields:
                task_overrides[key] = value
            elif key in sampling_fields:
                sampling_overrides[key] = value

        return task_overrides, sampling_overrides

    def _get_gpu_count(self) -> int:
        """Get total number of available GPUs."""
        try:
            import torch

            return torch.cuda.device_count()
        except ImportError:
            return 0

    def _update_metrics_config(self, experiment_id: str) -> None:
        """Update harness metrics config with experiment metadata.

        This must be called before starting workers so the serialized config
        includes all metadata for metrics reporting.
        """
        from olmo_eval.common.types import compute_model_hash

        if self.harness_config.metrics is None or not self.harness_config.metrics.enabled:
            return

        # Compute model hash from provider config
        model_hash = compute_model_hash(self.harness_config.provider.to_dict())

        # Update metrics config with all available metadata
        updated_metrics = self.harness_config.metrics.with_metadata(
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            model_name=self.model_name,
            model_hash=model_hash,
        )

        # Replace harness config with updated metrics
        self.harness_config = self.harness_config.with_metrics(updated_metrics)

    async def _process_results(
        self,
        trackers: dict[str, TaskTracker],
        result_queue: mp.Queue,
        workers: list[mp.process.BaseProcess],
        scoring_context: Any,
        scoring_concurrency: int,
        total_tasks: int,
        total_instances: int,
    ) -> dict[str, Any]:
        """Process results from workers with inline async scoring."""
        return await process_results(
            trackers=trackers,
            result_queue=result_queue,
            workers=workers,
            scoring_context=scoring_context,
            scoring_concurrency=scoring_concurrency,
            total_tasks=total_tasks,
            total_instances=total_instances,
            model_name=self.model_name,
            save_predictions=self.save_predictions,
            write_predictions_fn=self._write_predictions,
        )

    def _aggregate_results(
        self,
        results: dict[str, Any],
        expanded_tasks: list[str],
    ) -> dict[str, Any]:
        """Aggregate results and prepare final output."""
        return aggregate_results(
            results=results,
            expanded_tasks=expanded_tasks,
            task_specs=self.task_specs,
            provider_config=self.provider_config,
            attention_backend=self.attention_backend,
            harness_config=self.harness_config,
        )

    def _finalize_and_save(
        self,
        results_dict: dict[str, Any],
        experiment_id: str,
        experiment_duration_seconds: float | None = None,
        provider_init_seconds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Log summary, write metrics, upload to S3, and save results."""
        from olmo_eval.common.types import compute_model_hash

        self._log_summary(results_dict)

        model_hash = compute_model_hash(results_dict.get("model_config", {}))
        results_dict["_model_hash"] = model_hash

        self._write_metrics_json(
            results=results_dict,
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            model_hash=model_hash,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

        s3_location: str | None = None
        if self.s3_config and model_hash:
            s3_location = self._upload_to_s3(
                model_name=self.model_name,
                model_hash=model_hash,
                experiment_id=experiment_id,
            )

        results_dict["_experiment_id"] = experiment_id
        results_dict["_s3_location"] = s3_location

        self._save_results(
            results=results_dict,
            experiment_id=experiment_id,
            model_hash=model_hash,
            s3_location=s3_location,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

        return results_dict
