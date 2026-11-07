"""Tokenizer utilities for provider-agnostic BOS handling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def get_bos_token_ids(tokenizer: Any, *, fallback_to_eos: bool = True) -> list[int]:
    """Get BOS token ID, optionally falling back to EOS.

    Args:
        tokenizer: A tokenizer object with bos_token_id and optionally eos_token_id.
        fallback_to_eos: If True and BOS is None, use EOS token ID instead.

    Returns:
        A list containing the BOS token ID, or empty list if none available.
    """
    bos_id = tokenizer.bos_token_id
    if bos_id is None and fallback_to_eos:
        bos_id = tokenizer.eos_token_id
    return [bos_id] if bos_id is not None else []


def has_bos_token(tokenizer: Any) -> bool:
    """Check if tokenizer has BOS defined.

    Args:
        tokenizer: A tokenizer object with bos_token_id attribute.

    Returns:
        True if the tokenizer has a BOS token defined.
    """
    return tokenizer.bos_token_id is not None


def default_prefix_ids(tokenizer: Any) -> list[int]:
    """Return the special tokens the tokenizer puts in front of a text by default.

    This is the one rule for BOS: providers start every generation prompt and every
    non-empty log-likelihood context with these tokens. They are measured, not read from
    a flag: a probe is encoded with and without special tokens, and whatever comes before
    the probe's own tokens is the prefix. This is the BOS for tokenizers that add it
    through their post-processor template (Llama 3, Gemma 3 and 4, LFM2.5, Nemotron-H),
    whose ``add_bos_token`` attribute reads False under transformers 5, and it works the
    same for a tokenizer that encodes on a server. Special tokens appended after the text
    (an EOS) are not part of the prefix.

    Args:
        tokenizer: A tokenizer object with an encode method.

    Returns:
        The prefix token IDs; empty for tokenizers that add nothing in front.
    """
    probe = "a"
    with_special = tokenizer.encode(probe, add_special_tokens=True)
    without_special = tokenizer.encode(probe, add_special_tokens=False)
    if not without_special:
        return []
    width = len(without_special)
    for start in range(len(with_special) - width + 1):
        if with_special[start : start + width] == without_special:
            return list(with_special[:start])
    return []


ADD_BOS_TOKEN_REMOVED = (
    "add_bos_token was removed: inputs start with whatever the tokenizer puts in front of "
    "a text when encoding it. If that is wrong for the model, use a tokenizer whose "
    "defaults match how the model was trained (provider tokenizer=...)."
)


def reject_add_bos_token(kwargs: Mapping[str, Any]) -> None:
    """Fail on the removed ``add_bos_token`` provider setting instead of ignoring it."""
    if "add_bos_token" in kwargs:
        raise ValueError(ADD_BOS_TOKEN_REMOVED)


def encode_prompt(tokenizer: Any, text: str, prefix_ids: list[int]) -> list[int]:
    """Encode a generation prompt with the tokenizer's defaults, without a doubled prefix.

    A chat template that writes BOS into the text would otherwise get a second one from
    the tokenizer (``[bos, bos, ...]``); the duplicate is dropped.

    Args:
        tokenizer: A tokenizer object with an encode method.
        text: The prompt text.
        prefix_ids: The tokenizer's prefix (see ``default_prefix_ids``).

    Returns:
        The prompt's token IDs.
    """
    ids = tokenizer.encode(text, add_special_tokens=True)
    n = len(prefix_ids)
    if n and ids[:n] == prefix_ids and ids[n : 2 * n] == prefix_ids:
        return ids[n:]
    return ids


def get_context_token_ids(
    tokenizer: Any,
    context: str,
    *,
    use_bos_for_empty: bool = True,
    fallback_to_eos: bool = True,
) -> list[int]:
    """Tokenize context, using BOS for empty contexts.

    Args:
        tokenizer: A tokenizer object with encode method.
        context: The context string to tokenize.
        use_bos_for_empty: If True and context is empty, return BOS token(s).
        fallback_to_eos: If True and BOS is None, use EOS token ID instead.

    Returns:
        List of token IDs for the context.
    """
    if context == "" and use_bos_for_empty:
        return get_bos_token_ids(tokenizer, fallback_to_eos=fallback_to_eos)
    return tokenizer.encode(context, add_special_tokens=False)


def encode_context_and_continuation(
    tokenizer: Any,
    context: str,
    continuation: str,
    *,
    use_bos_for_empty: bool = True,
    fallback_to_eos: bool = True,
    prefix_ids: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Encode context/continuation pair with proper BOS handling and trailing space logic.

    This matches lm_eval behavior: trailing spaces from context are moved to continuation
    before tokenization to ensure consistent token boundaries.

    For empty contexts, BOS token is used as the context (optionally falling back to EOS).

    Args:
        tokenizer: A tokenizer object with encode method.
        context: The context/prompt string.
        continuation: The continuation string to evaluate.
        use_bos_for_empty: If True and context is empty, use BOS token as context.
        fallback_to_eos: If True and BOS is None, use EOS token ID instead.
        prefix_ids: The tokens a non-empty context starts with (``default_prefix_ids``).
            Providers pass the prefix they computed once; None measures it here.

    Returns:
        Tuple of (context_token_ids, continuation_token_ids).
    """
    # Handle empty context: use BOS token
    if context == "" and use_bos_for_empty:
        context_enc = get_bos_token_ids(tokenizer, fallback_to_eos=fallback_to_eos)
        continuation_enc = tokenizer.encode(continuation, add_special_tokens=False)
        return context_enc, continuation_enc

    # Match lm_eval behavior: move trailing spaces from context to continuation
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]

    # Start the context the way generation prompts start (encode_prompt)
    if prefix_ids is None:
        prefix_ids = default_prefix_ids(tokenizer)

    # Encode the full sequence and extract continuation tokens
    whole_enc = tokenizer.encode(context + continuation, add_special_tokens=False)
    context_enc = tokenizer.encode(context, add_special_tokens=False)
    continuation_enc = whole_enc[len(context_enc) :]

    if prefix_ids and context_enc[: len(prefix_ids)] != prefix_ids:
        context_enc = [*prefix_ids, *context_enc]
    return context_enc, continuation_enc
