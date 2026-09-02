"""Reasoning-trace handling shared by the inference providers."""

from __future__ import annotations

THINK_END = "</think>"


def split_reasoning(text: str) -> tuple[str, str]:
    """Split a reasoning model's output into ``(reasoning, answer)`` on ``</think>``.

    Qwen3-VL-Thinking (and Qwen3 generally) prefill ``<think>\\n`` in the generation
    prompt, so the generated text *begins* inside the reasoning block and closes it with
    ``</think>`` before the user-facing answer. None of the image-QA scorers strip that
    block: ``MmmuScorer`` runs ``clean_prediction``, which splits on the *first*
    ``"Answer:"`` -- frequently one the model wrote mid-thought -- and the CharXiv GPT judge
    would be handed the whole trace. Splitting on the **last** ``</think>`` restores the
    answer-only text those scorers expect while keeping the trace for error analysis.

    Returns ``("", text)`` unchanged when no ``</think>`` is present, so this is a no-op
    for non-reasoning models and for traces truncated before they close.
    """
    idx = text.rfind(THINK_END)
    if idx == -1:
        return "", text
    return text[:idx].strip(), text[idx + len(THINK_END) :].strip()
