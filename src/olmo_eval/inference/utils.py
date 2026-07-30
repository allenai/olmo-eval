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


def _normalize_vllm_function_name(name: str, allowed_names: set[str]) -> str:
    """Remove a leaked Harmony channel suffix from a known function name.

    The correction is deliberately schema-bound: arbitrary or ambiguous names
    remain unchanged so genuine model tool-name errors are still surfaced.
    """
    if name in allowed_names:
        return name

    for allowed_name in sorted(allowed_names, key=len, reverse=True):
        if not name.startswith(allowed_name):
            continue
        suffix = name[len(allowed_name) :]
        if suffix in {"analysis", "commentary"} or suffix.startswith("<|"):
            return allowed_name
    return name


def patch_openai_agents_for_vllm() -> None:
    """Patch openai-agents SDK for vLLM compatibility.

    vLLM doesn't support the 'strict' field in tool schemas.
    The openai-agents SDK always includes it, even when strict_mode=False.
    This patch makes the SDK omit the field when strict_json_schema is False.

    Some GPT-OSS Harmony generations also append a channel marker or channel
    name to an otherwise exact function name. Normalize only names that match
    a configured function tool plus that known protocol artifact.

    See: https://github.com/vllm-project/vllm/issues/27746

    Call this once before creating any agents that will talk to vLLM.
    Safe to call multiple times (idempotent).
    """
    from agents import FunctionTool  # type: ignore[ty:unresolved-import]
    from agents._run_impl import RunImpl  # type: ignore[ty:unresolved-import]
    from agents.models.chatcmpl_converter import Converter  # type: ignore[ty:unresolved-import]
    from openai.types.responses import ResponseFunctionToolCall

    if not getattr(Converter, "_vllm_patched", False):
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

    if not getattr(RunImpl, "_vllm_tool_name_patched", False):
        _original_process_model_response = RunImpl.process_model_response

        @classmethod
        def _patched_process_model_response(cls, *, all_tools, response, **kwargs):
            allowed_names = {tool.name for tool in all_tools if isinstance(tool, FunctionTool)}
            for output in response.output:
                if isinstance(output, ResponseFunctionToolCall):
                    output.name = _normalize_vllm_function_name(output.name, allowed_names)
            return _original_process_model_response(
                all_tools=all_tools,
                response=response,
                **kwargs,
            )

        RunImpl.process_model_response = _patched_process_model_response
        RunImpl._vllm_tool_name_patched = True
