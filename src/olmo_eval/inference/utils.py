"""Inference provider utilities."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any, cast


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run async code from a sync context, handling nested event loops.

    Unlike asyncio.run(), this helper detects when there's already a running
    event loop (e.g., in Jupyter notebooks or async applications) and runs
    the coroutine in a dedicated thread to avoid RuntimeError.

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    try:
        asyncio.get_running_loop()
        # Already in an async context - run in a thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return cast(T, executor.submit(asyncio.run, coro).result())
    except RuntimeError:
        # No running loop - use asyncio.run directly
        return asyncio.run(coro)


def patch_openai_agents_for_vllm() -> None:
    """Patch openai-agents SDK to omit 'strict' field for vLLM compatibility.

    vLLM doesn't support the 'strict' field in tool schemas.
    The openai-agents SDK always includes it, even when strict_mode=False.
    This patch makes the SDK omit the field when strict_json_schema is False.

    See: https://github.com/vllm-project/vllm/issues/27746

    Call this once before creating any agents that will talk to vLLM.
    Safe to call multiple times (idempotent).
    """
    from agents import FunctionTool  # type: ignore[ty:unresolved-import]
    from agents.models.chatcmpl_converter import Converter  # type: ignore[ty:unresolved-import]

    # Check if already patched
    if getattr(Converter, "_vllm_patched", False):
        return

    _original_tool_to_openai = Converter.tool_to_openai

    @classmethod
    def _patched_tool_to_openai(cls, tool):
        result = _original_tool_to_openai(tool)
        # Remove 'strict' field if False (vLLM doesn't support it)
        if isinstance(tool, FunctionTool) and not tool.strict_json_schema:
            result.get("function", {}).pop("strict", None)
        return result

    Converter.tool_to_openai = _patched_tool_to_openai
    Converter._vllm_patched = True


def patch_openai_agents_for_litellm_usage() -> None:
    """Give InputTokensDetails.cache_write_tokens a default, for litellm compatibility.

    openai>=2.8's InputTokensDetails made `cache_write_tokens` a required field
    (added for prompt-caching support), but openai-agents 0.7.0's own usage
    normalization (agents/usage.py) still constructs it as
    InputTokensDetails(cached_tokens=0), predating that change. litellm's usage
    translation hits this path (it never sets cache_write_tokens either), so
    every openai_agents-scaffolded run through a litellm provider fails with
    "1 validation error for InputTokensDetails: cache_write_tokens Field required".

    Call this once before creating any agents that will talk through litellm.
    Safe to call multiple times (idempotent); a no-op once either package
    upgrades to fix the skew upstream (checked via the field's own default).
    """
    from openai.types.responses.response_usage import InputTokensDetails

    field = InputTokensDetails.model_fields["cache_write_tokens"]
    if not field.is_required():
        return

    field.default = 0
    InputTokensDetails.model_rebuild(force=True)
