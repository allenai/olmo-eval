"""Tests for fitting a generation budget into the model's context window."""

from __future__ import annotations

import pytest

from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

MESSAGES = [{"role": "user", "content": "hello"}]


class FittingProvider:
    """The fitting logic with the server calls it depends on stubbed out."""

    _CONTEXT_MARGIN_TOKENS = VLLMServerProvider._CONTEXT_MARGIN_TOKENS
    _NO_ROOM_MAX_TOKENS = VLLMServerProvider._NO_ROOM_MAX_TOKENS
    _fit_max_tokens = VLLMServerProvider._fit_max_tokens

    def __init__(self, max_length: int | None, prompt_tokens: int | None) -> None:
        self.max_length = max_length
        self._prompt_tokens = prompt_tokens

    async def _count_chat_tokens(self, messages, tools):  # noqa: ANN001, ANN202
        return self._prompt_tokens


async def fit(max_tokens: int, *, context: int | None, prompt: int | None) -> int:
    return await FittingProvider(context, prompt)._fit_max_tokens(max_tokens, MESSAGES, None)


@pytest.mark.anyio
async def test_a_prompt_that_leaves_room_keeps_the_whole_budget() -> None:
    assert await fit(4096, context=40960, prompt=100) == 4096


@pytest.mark.anyio
async def test_a_long_prompt_shrinks_the_budget_to_what_is_left() -> None:
    # The case that made the server reject whole long-context requests.
    assert await fit(1024, context=40960, prompt=39937) == 1021


@pytest.mark.anyio
async def test_a_prompt_that_fills_the_window_gives_up() -> None:
    assert await fit(4096, context=4096, prompt=4096) == FittingProvider._NO_ROOM_MAX_TOKENS


@pytest.mark.anyio
async def test_an_unknown_context_window_leaves_the_budget_alone() -> None:
    assert await fit(4096, context=None, prompt=100) == 4096


@pytest.mark.anyio
async def test_a_prompt_that_cannot_be_counted_leaves_the_budget_alone() -> None:
    assert await fit(4096, context=40960, prompt=None) == 4096
