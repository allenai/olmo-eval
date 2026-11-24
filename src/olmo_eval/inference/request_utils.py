"""Helpers for interpreting ``LMRequest`` fields consistently across providers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmo_eval.common.types import LMRequest

logger = logging.getLogger(__name__)
_DROPPED_SYSTEM_WARNED: list[bool] = []


def prompt_opens_think(rendered_prompt: str) -> bool:
    """Whether a rendered prompt ends by opening a ``<think>`` trace.

    A chat template may end its generation prompt with ``<think>``. The generation then
    starts inside the trace and has no opening tag, so providers record this on each
    output under ``PROMPT_OPENS_THINK_KEY`` for scoring to tell an unfinished trace from
    an answer.
    """
    return rendered_prompt.rstrip().endswith("<think>")


THINK_CLOSE = "</think>"


def think_close_token_id(tokenizer: Any) -> int | None:
    """The id of ``</think>`` when ``tokenizer`` holds the tag as one token, else None.

    A single-token tag can be missing from decoded text (a decoder that skips special
    tokens drops it), so only the generated ids show whether it was produced. A tag that
    is ordinary text spans several tokens and always decodes into the text.
    """
    for vocabulary in ("get_added_vocab", "get_vocab"):
        getter = getattr(tokenizer, vocabulary, None)
        if not callable(getter):
            continue
        try:
            token_id = getter().get(THINK_CLOSE)
        except (AttributeError, TypeError):
            continue
        if isinstance(token_id, int) and not isinstance(token_id, bool):
            return token_id
    return None


def think_closed(
    text: str, token_ids: Sequence[int] | None, close_token_id: int | None
) -> bool | None:
    """Whether a generation that started inside a trace closed it, or None without evidence.

    ``</think>`` in the text is evidence that it closed. Without it, the generated ids
    decide when the tag is one token. When it is not one token it is plain text that
    decoding cannot drop, so its absence from the text is evidence that the trace stayed
    open.
    """
    if THINK_CLOSE in text:
        return True
    if close_token_id is None:
        return False
    if token_ids is None:
        return None
    return close_token_id in token_ids


def chat_messages_for_request(request: LMRequest) -> tuple[dict, ...]:
    """The request's chat turns, preserving a completion prompt as a user turn.

    A completion-style request carries its question in ``prompt`` with empty
    ``messages``; the harness may then inject a system message into ``messages``,
    which must not displace the question. The prompt is appended as a user turn
    whenever no user turn exists yet.
    """
    messages = tuple(request.messages or ())
    if request.prompt is not None and not any(m.get("role") == "user" for m in messages):
        messages = (*messages, {"role": "user", "content": request.prompt})
    return messages


def chat_with_images(
    messages: tuple[dict, ...], image_parts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Chat turns as content-part lists, with ``image_parts`` attached in order.

    A turn whose ``content`` is a list of parts places an image at each ``{"type": "image"}``
    placeholder, so text and images can interleave as a benchmark specifies. Otherwise every
    image goes at the start of the first user turn (before its text), or into a leading user
    turn when there is none.

    :raises ValueError: When the placeholders do not match the number of images.
    """
    placed = any(isinstance(m.get("content"), list) for m in messages)
    remaining = list(image_parts)
    chat: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for part in content:
                if part.get("type") == "image":
                    if not remaining:
                        raise ValueError("more image placeholders than images in the request")
                    parts.append(remaining.pop(0))
                else:
                    parts.append(part)
        else:
            parts = [{"type": "text", "text": content}]
            if not placed and role == "user" and remaining:
                parts = [*remaining, *parts]
                remaining = []
        chat.append({"role": role, "content": parts})
    if remaining:
        if placed:
            raise ValueError("fewer image placeholders than images in the request")
        chat.insert(0, {"role": "user", "content": remaining})
    return chat


def drop_system_turns(chat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation without its system turns; a warning is logged once."""
    kept = [m for m in chat if m["role"] != "system"]
    if len(kept) != len(chat) and not _DROPPED_SYSTEM_WARNED:
        _DROPPED_SYSTEM_WARNED.append(True)
        logger.warning(
            "The chat template rejected the system turn; sending the request without it."
        )
    return kept


def render_chat[T](
    apply_template: Callable[[list[dict[str, Any]]], T], chat: list[dict[str, Any]]
) -> T:
    """Render ``chat`` with ``apply_template``, keeping its system turn when the template can.

    A chat template that rejects a system turn (Molmo2's raises: its roles must alternate
    from a user turn) gets the conversation without it. Moving the system text into the
    user turn instead would put text the model never trained on in front of the question.
    Templates that accept a system turn render it as before.
    """
    from jinja2.exceptions import TemplateError

    try:
        return apply_template(chat)
    except TemplateError:
        if not any(m["role"] == "system" for m in chat):
            raise
        return apply_template(drop_system_turns(chat))
