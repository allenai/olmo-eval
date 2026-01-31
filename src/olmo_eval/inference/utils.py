"""Inference provider utilities."""


def patch_openai_agents_for_vllm() -> None:
    """Patch openai-agents SDK to omit 'strict' field for vLLM compatibility.

    vLLM doesn't support the 'strict' field in tool schemas.
    The openai-agents SDK always includes it, even when strict_mode=False.
    This patch makes the SDK omit the field when strict_json_schema is False.

    See: https://github.com/vllm-project/vllm/issues/27746

    Call this once before creating any agents that will talk to vLLM.
    Safe to call multiple times (idempotent).
    """
    from agents import FunctionTool  # type: ignore[import-not-found]
    from agents.models.chatcmpl_converter import Converter  # type: ignore[import-not-found]

    # Check if already patched
    if getattr(Converter, "_vllm_patched", False):
        return

    _original_tool_to_openai = Converter.tool_to_openai

    @classmethod  # type: ignore[misc]
    def _patched_tool_to_openai(cls, tool):  # type: ignore[no-untyped-def]
        result = _original_tool_to_openai(tool)
        # Remove 'strict' field if False (vLLM doesn't support it)
        if isinstance(tool, FunctionTool) and not tool.strict_json_schema:
            result.get("function", {}).pop("strict", None)
        return result

    Converter.tool_to_openai = _patched_tool_to_openai
    Converter._vllm_patched = True
