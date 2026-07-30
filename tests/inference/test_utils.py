"""Tests for inference provider compatibility helpers."""

from types import SimpleNamespace

import pytest

from olmo_eval.inference.utils import (
    _normalize_vllm_function_name,
    patch_openai_agents_for_vllm,
)


def test_normalize_vllm_function_name_is_schema_bound():
    allowed = {"semantic_scholar_snippet_search", "search"}

    assert (
        _normalize_vllm_function_name(
            "semantic_scholar_snippet_search<|channel|>commentary", allowed
        )
        == "semantic_scholar_snippet_search"
    )
    assert (
        _normalize_vllm_function_name("semantic_scholar_snippet_searchcommentary", allowed)
        == "semantic_scholar_snippet_search"
    )
    assert _normalize_vllm_function_name("search", allowed) == "search"
    assert _normalize_vllm_function_name("search_web", allowed) == "search_web"
    assert _normalize_vllm_function_name("invented_toolcommentary", allowed) == (
        "invented_toolcommentary"
    )


def test_agents_patch_normalizes_response_before_tool_lookup(monkeypatch):
    pytest.importorskip("agents")

    from agents import function_tool
    from agents._run_impl import RunImpl
    from agents.models.chatcmpl_converter import Converter
    from openai.types.responses import ResponseFunctionToolCall

    @function_tool(strict_mode=False)
    def semantic_scholar_snippet_search(query: str) -> str:
        return query

    seen_names = []

    @classmethod
    def fake_process_model_response(cls, *, response, **kwargs):
        seen_names.extend(output.name for output in response.output)
        return "processed"

    monkeypatch.setattr(Converter, "_vllm_patched", True, raising=False)
    monkeypatch.setattr(RunImpl, "_vllm_tool_name_patched", False, raising=False)
    monkeypatch.setattr(RunImpl, "process_model_response", fake_process_model_response)

    patch_openai_agents_for_vllm()
    response = SimpleNamespace(
        output=[
            ResponseFunctionToolCall(
                arguments='{"query":"test"}',
                call_id="call_test",
                name="semantic_scholar_snippet_search<|channel|>commentary",
                type="function_call",
            )
        ]
    )

    result = RunImpl.process_model_response(
        agent=SimpleNamespace(),
        all_tools=[semantic_scholar_snippet_search],
        response=response,
        output_schema=None,
        handoffs=[],
    )

    assert result == "processed"
    assert seen_names == ["semantic_scholar_snippet_search"]
