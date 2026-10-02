"""Tokenizer utilities for provider-agnostic BOS handling."""

from __future__ import annotations

from typing import Any, Literal


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


def resolve_prompt_truncation_limit(
    truncate_prompt_tokens: int | None,
    *,
    max_input_tokens: int | None = None,
) -> int | None:
    """Resolve a prompt-token limit, including vLLM's ``-1`` convention."""
    if truncate_prompt_tokens is None:
        return None
    if truncate_prompt_tokens == -1:
        if max_input_tokens is None or max_input_tokens <= 0:
            raise ValueError("truncate_prompt_tokens=-1 requires a positive max_input_tokens")
        return max_input_tokens
    if truncate_prompt_tokens <= 0:
        raise ValueError("truncate_prompt_tokens must be a positive integer, -1, or None")
    return truncate_prompt_tokens


def resolve_truncation_side(
    truncation_side: Literal["left", "right"] | None,
    *,
    tokenizer: Any | None = None,
    default: Literal["left", "right"] = "right",
) -> Literal["left", "right"]:
    """Resolve the requested side, falling back to the tokenizer's default."""
    if truncation_side is not None:
        return truncation_side
    tokenizer_side = getattr(tokenizer, "truncation_side", None)
    if tokenizer_side in ("left", "right"):
        return tokenizer_side
    return default


def truncate_token_ids(
    token_ids: list[int],
    truncate_prompt_tokens: int | None,
    truncation_side: Literal["left", "right"] | None = None,
    *,
    tokenizer: Any | None = None,
    max_input_tokens: int | None = None,
    default_side: Literal["left", "right"] = "right",
) -> list[int]:
    """Truncate prompt token IDs using vLLM-compatible side semantics.

    ``left`` removes tokens from the beginning and keeps the last N tokens.
    ``right`` removes tokens from the end and keeps the first N tokens.
    """
    limit = resolve_prompt_truncation_limit(
        truncate_prompt_tokens,
        max_input_tokens=max_input_tokens,
    )
    if limit is None or len(token_ids) <= limit:
        return list(token_ids)

    side = resolve_truncation_side(
        truncation_side,
        tokenizer=tokenizer,
        default=default_side,
    )
    if side == "left":
        return list(token_ids[-limit:])
    return list(token_ids[:limit])


def encode_context_and_continuation(
    tokenizer: Any,
    context: str,
    continuation: str,
    *,
    use_bos_for_empty: bool = True,
    fallback_to_eos: bool = True,
) -> tuple[list[int], list[int]]:
    """Encode context/continuation pair with proper BOS handling and trailing space logic.

    This matches lm_eval behavior: trailing spaces from context are moved to continuation
    before tokenization to ensure consistent token boundaries.

    For empty contexts, BOS token is used as the context (optionally falling back to EOS).

    Args:
        tokenizer: A tokenizer object with encode method.
        context: The context string to tokenize.
        continuation: The continuation string to evaluate.
        use_bos_for_empty: If True and context is empty, use BOS token as context.
        fallback_to_eos: If True and BOS is None, use EOS token ID instead.

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

    # Match lm_eval behavior: add BOS token if the tokenizer is configured for it
    # (lm_eval's tok_encode uses add_special_tokens=tokenizer.add_bos_token)
    add_bos = getattr(tokenizer, "add_bos_token", False)
    bos_ids = get_bos_token_ids(tokenizer, fallback_to_eos=fallback_to_eos) if add_bos else []

    # Encode the full sequence and extract continuation tokens
    whole_enc = tokenizer.encode(context + continuation, add_special_tokens=False)
    context_enc = tokenizer.encode(context, add_special_tokens=False)
    continuation_enc = whole_enc[len(context_enc) :]

    return bos_ids + context_enc, continuation_enc