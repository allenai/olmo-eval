"""Tokenizer utilities for provider-agnostic BOS handling."""

from __future__ import annotations

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


def rolling_token_windows(
    prefix: list[int], tokens: list[int], max_len: int
) -> list[tuple[list[int], int]]:
    """Cut ``prefix + tokens`` into inputs of at most ``max_len`` tokens that score
    every token of ``tokens`` exactly once.

    The first input is ``prefix`` followed by as many tokens as fit. Each later
    input starts with the last token the previous one scored, as context only,
    followed by the next ``max_len - 1`` tokens. Inputs share no scored tokens,
    so their summed logprobs are the logprob of the whole text, the way
    lm-eval-harness's ``loglikelihood_rolling`` scores with a context length of 1.

    Args:
        prefix: Context tokens before the text, usually just BOS. Not scored.
        tokens: Tokens of the text to score.
        max_len: Most tokens one input may hold.

    Returns:
        ``(input_ids, ctxlen)`` pairs; ``input_ids[ctxlen:]`` are the tokens scored.
    """
    if not prefix:
        raise ValueError(
            "rolling_token_windows needs a prefix token (BOS) to score the first token"
        )
    if max_len <= len(prefix):
        raise ValueError(f"max_len {max_len} leaves no room after a {len(prefix)}-token prefix")
    if max_len < 2:
        raise ValueError(f"max_len must be at least 2, got {max_len}")
    first = max_len - len(prefix)
    windows = [(prefix + tokens[:first], len(prefix))]
    start = first
    while start < len(tokens):
        windows.append((tokens[start - 1 : start + max_len - 1], 1))
        start += max_len - 1
    return windows
