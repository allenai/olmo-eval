"""Task execution functions for running evaluations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from olmo_eval.core.logging import get_logger
from olmo_eval.core.types import Response, SamplingParams
from olmo_eval.evals.tasks import AgentTask, get_task
from olmo_eval.inference import InferenceProvider
from olmo_eval.runners.builders import build_predictions, build_requests
from olmo_eval.runners.common import get_metric_metadata
from olmo_eval.runners.types import TaskResult

logger = get_logger("runners.execution")


def run_agent_task_impl(
    task: AgentTask,
    spec: str,
    model_name: str,
    model_overrides: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    num_gpus: int = 1,
) -> TaskResult:
    """Execute an agent task using async agent loop.

    This function handles the special execution path for agent tasks, which
    use multi-turn agent interactions instead of single-turn inference.

    The model can be either:
    1. A HuggingFace model (starts local vLLM server)
    2. A model with model_url in config/overrides (uses API directly)

    Args:
        task: The AgentTask instance to execute.
        spec: The original task specification string.
        model_name: Model name/path (required). Can be:
            - HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            - Model preset name (e.g., "gpt-4o", "llama3.1-8b-instruct")
            - Local checkpoint path
        model_overrides: Optional overrides for model configuration. Can include:
            - tokenizer: Custom tokenizer path
            - model_url: API endpoint (skips vLLM, uses API directly)
            - max_model_len: Maximum sequence length
        overrides: Optional config overrides (limit, etc.).
        progress_callback: Optional callback for progress messages.
        num_gpus: Number of GPUs for tensor parallelism when using vLLM.

    Returns:
        TaskResult with metrics and metadata.

    Raises:
        ValueError: If no model is specified.
    """
    import asyncio
    import time

    from olmo_eval.core.configs import get_model_config

    start_time = time.time()

    try:
        # Model is required for agent tasks
        if not model_name:
            raise ValueError(
                f"Agent task '{spec}' requires a model. Specify with -m/--model. "
                "Examples:\n"
                "  olmo-eval run -m llama3.1-8b-instruct -t simpleqa_agent\n"
                "  olmo-eval run -m gpt-4o -t simpleqa_agent"
            )

        # Apply overrides
        if overrides:
            task.config = replace(task.config, **overrides)

        # Get model configuration (handles presets and inline overrides)
        model_config = get_model_config(model_name, **(model_overrides or {}))

        # Collect instances
        instances = list(task.instances)
        if task.config.limit:
            instances = instances[: task.config.limit]

        if progress_callback:
            progress_callback(f"Running agent on {len(instances)} instances...")

        # Get agent settings from task config
        system_prompt = getattr(task.config, "system_prompt", "") or None
        max_turns = getattr(task.config, "max_turns", 10)
        max_concurrency = getattr(task.config, "max_concurrency", 1)

        # Temperature and max_tokens come from sampling_params
        sampling_params = task.config.sampling_params
        temperature = sampling_params.temperature if sampling_params else 0.0

        # Determine execution mode based on model_url
        # If model has a URL, use it directly (API mode)
        # Otherwise, start vLLM server (local mode)
        if model_config.model_url:
            # API mode: use the model_url directly
            effective_model = model_config.model
            effective_url = model_config.model_url

            if progress_callback:
                progress_callback(f"Using API endpoint: {effective_url}")

            # Validate secrets for API access
            task._validate_secrets()

            from tqdm import tqdm

            with tqdm(total=len(instances), desc="Processing instances", unit="inst") as pbar:
                results = asyncio.run(
                    task._run_agent_loop(
                        instances=instances,
                        model=effective_model,
                        model_url=effective_url,
                        system_prompt=system_prompt,
                        max_turns=max_turns,
                        max_concurrency=max_concurrency,
                        temperature=temperature,
                        on_instance_complete=lambda: pbar.update(1),
                    )
                )
        else:
            # vLLM mode: start local server
            results, effective_model, effective_url = _run_agent_with_vllm_server(
                task=task,
                instances=instances,
                model_name=model_config.model,
                model_overrides=model_overrides or {},
                system_prompt=system_prompt,
                max_turns=max_turns,
                max_concurrency=max_concurrency,
                temperature=temperature,
                num_gpus=num_gpus,
                progress_callback=progress_callback,
            )

        # Build responses from results
        responses = task._build_responses(instances, results)

        # Score responses and compute metrics using standard flow
        from tqdm import tqdm

        with tqdm(total=len(responses), desc="Scoring instances", unit="inst") as pbar:
            scored = task.score_responses(responses)
            pbar.update(len(responses))
        metrics = task.compute_metrics(scored)

        # Build predictions for per-instance inspection
        predictions = build_predictions(scored)

        # Build requests (for consistency with standard tasks)
        requests_list = [task.format_request(inst) for inst in instances]
        request_objects = build_requests(instances, requests_list, task.config.name)

        duration = time.time() - start_time

        # Extract metric metadata (returns "metric:scorer" format)
        primary_metric = get_metric_metadata(task)

        return TaskResult(
            spec=spec,
            config=task.config.to_dict(),
            num_instances=len(instances),
            metrics=metrics,
            duration_seconds=duration,
            predictions=predictions,
            requests=request_objects,
            primary_metric=primary_metric,
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.exception(f"Agent task {spec} failed: {e}")
        return TaskResult(
            spec=spec,
            config={},
            num_instances=0,
            metrics={},
            error=str(e),
            duration_seconds=duration,
        )


def _run_agent_with_vllm_server(
    task: AgentTask,
    instances: list,
    model_name: str,
    model_overrides: dict[str, Any],
    system_prompt: str | None,
    max_turns: int,
    max_concurrency: int,
    temperature: float,
    num_gpus: int,
    progress_callback: Callable[[str], None] | None,
) -> tuple[list, str, str]:
    """Run agent task with a local vLLM server.

    Starts a vLLM server, runs the agent loop, and returns results.

    Args:
        task: The agent task to run.
        instances: List of instances to process.
        model_name: Model name/path to serve.
        model_overrides: Model configuration overrides.
        system_prompt: System prompt for the agent.
        max_turns: Maximum agent turns.
        max_concurrency: Maximum concurrent executions.
        temperature: Sampling temperature.
        num_gpus: Number of GPUs for tensor parallelism.
        progress_callback: Optional progress callback.

    Returns:
        Tuple of (results, effective_model, effective_url).
    """
    import asyncio

    from olmo_eval.inference.vllm_server import vllm_server_context

    if progress_callback:
        progress_callback(f"Starting vLLM server for {model_name}...")

    # Extract vLLM-specific overrides
    tokenizer = model_overrides.get("tokenizer")
    max_model_len = model_overrides.get("max_model_len")

    # Determine tool call parser based on model name
    # Users can override via model_overrides["tool_call_parser"]
    tool_call_parser = model_overrides.get("tool_call_parser")
    if tool_call_parser is None:
        model_lower = model_name.lower()
        if "llama" in model_lower:
            tool_call_parser = "llama3_json"
        elif "mistral" in model_lower:
            tool_call_parser = "mistral"
        else:
            # Default for Qwen, OLMo, and other models
            tool_call_parser = "hermes"

    from tqdm import tqdm

    with vllm_server_context(
        model_name=model_name,
        tensor_parallel_size=num_gpus,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        enable_auto_tool_choice=True,
        tool_call_parser=tool_call_parser,
    ) as server_url:
        if progress_callback:
            progress_callback(f"vLLM server ready at {server_url}")

        with tqdm(total=len(instances), desc="Processing instances", unit="inst") as pbar:
            results = asyncio.run(
                task._run_agent_loop(
                    instances=instances,
                    model=model_name,
                    model_url=server_url,
                    system_prompt=system_prompt,
                    max_turns=max_turns,
                    max_concurrency=max_concurrency,
                    temperature=temperature,
                    on_instance_complete=lambda: pbar.update(1),
                )
            )

        return results, model_name, server_url


def run_task_impl(
    spec: str,
    provider: InferenceProvider,
    overrides: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    temperature: float | None = None,
    sampling_overrides: dict[str, Any] | None = None,
    requests_callback: Callable[[list[dict]], None] | None = None,
    response_callback: Callable[[Response], None] | None = None,
) -> TaskResult:
    """Execute a single task and return results.

    This is the core task execution logic shared by both EvalRunner and AsyncEvalRunner.

    Args:
        spec: Task specification (e.g., "mmlu_history" or "mmlu_history:olmes")
        provider: InferenceProvider instance to use for generation
        overrides: Optional overrides for task config (num_fewshot, limit, fewshot_seed)
        progress_callback: Optional callback for progress messages
        temperature: Optional temperature for sampling (deprecated, use sampling_overrides)
        sampling_overrides: Optional overrides for sampling params (temperature, max_tokens, etc.)
        requests_callback: Optional callback to receive requests early (before generation).
            Called with the list of request dicts immediately after they're built.
            Use this to write requests.jsonl before waiting for generation to complete.
        response_callback: Optional callback to receive the first scored response.
            Called after scoring with the first Response object. Useful for inspection/debugging.

    Returns:
        TaskResult with metrics and metadata

    Raises:
        Exception: Any error during task execution (should be caught by caller)
    """
    import time

    start_time = time.time()

    try:
        # Get task
        task = get_task(spec)

        # Agent tasks must be run through run_agent_task_impl directly
        if isinstance(task, AgentTask):
            raise ValueError(
                f"Agent task '{spec}' cannot be run through run_task_impl. "
                "Use run_agent_task_impl with a model specified via CLI instead."
            )

        # Apply overrides
        if overrides:
            task.config = replace(task.config, **overrides)

        # Build sampling params from overrides
        # Priority: sampling_overrides > temperature > task default
        existing_params = task.config.sampling_params or SamplingParams()

        # Apply legacy temperature parameter (deprecated)
        if temperature is not None:
            existing_params = replace(existing_params, temperature=temperature)

        # Apply sampling_overrides (highest priority)
        if sampling_overrides:
            for key, value in sampling_overrides.items():
                if hasattr(existing_params, key):
                    existing_params = replace(existing_params, **{key: value})

        # Update task config with final sampling params
        if temperature is not None or sampling_overrides:
            task.config = replace(task.config, sampling_params=existing_params)

        # Collect instances
        instances = list(task.instances)
        if task.config.limit:
            instances = instances[: task.config.limit]

        if progress_callback:
            progress_callback(f"Evaluating {len(instances)} instances...")

        # Format requests
        requests = [task.format_request(inst) for inst in instances]

        # Build requests in oe-eval compatible format (for debugging what model saw)
        # We do this early since we know the requests upfront - no need to wait for generation
        request_objects = build_requests(instances, requests, task.config.name, existing_params)

        # Call the requests callback early (before generation) if provided
        # This allows writing requests.jsonl without waiting for generation to complete
        if requests_callback:
            requests_callback(request_objects)

        # Generate outputs - use logprobs for LOGLIKELIHOOD requests
        from olmo_eval.core.types import RequestType

        if requests and requests[0].request_type == RequestType.LOGLIKELIHOOD:
            outputs = provider.logprobs(requests)
        else:
            outputs = provider.generate(requests, task.config.sampling_params)

        # Build responses
        responses = [
            Response(instance=inst, request=req, outputs=out)
            for inst, req, out in zip(instances, requests, outputs, strict=True)
        ]

        # Score and compute metrics
        scored = task.score_responses(responses)
        metrics = task.compute_metrics(scored)

        # Call the response callback with first scored response if provided
        if response_callback and scored:
            response_callback(scored[0])

        # Build predictions for per-instance inspection
        predictions = build_predictions(scored)

        duration = time.time() - start_time

        # Extract metric metadata (returns "metric:scorer" format)
        primary_metric = get_metric_metadata(task)

        return TaskResult(
            spec=spec,
            config=task.config.to_dict(),
            num_instances=len(instances),
            metrics=metrics,
            duration_seconds=duration,
            predictions=predictions,
            requests=request_objects,
            primary_metric=primary_metric,
        )

    except Exception as e:
        duration = time.time() - start_time
        return TaskResult(
            spec=spec,
            config={},
            num_instances=0,
            metrics={},
            error=str(e),
            duration_seconds=duration,
        )
