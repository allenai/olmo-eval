"""Shared formatting utilities for QA tasks.

Tasks that follow the standard ``Question: ... / Answer:`` template can use
these directly.  Tasks with custom templates (e.g. PiQA, HellaSwag) should
define their own ``format_mc`` / ``format_rc`` functions.
"""

from __future__ import annotations


def format_mc(question: str, choices: tuple[str, ...], answer: str | None = None) -> str:
    """Format a multiple-choice prompt.

    Example output::

        Question: What color is the sky?
         A. Red
         B. Blue
        Answer: B
    """
    choices_text = "\n".join(f" {chr(ord('A') + i)}. {c}" for i, c in enumerate(choices))
    prompt = f"Question: {question}\n{choices_text}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt


def format_rc(question: str, answer: str | None = None) -> str:
    """Format a reading-comprehension (open-ended) prompt.

    Example output::

        Question: What color is the sky?
        Answer: Blue
    """
    prompt = f"Question: {question}\nAnswer:"
    if answer:
        prompt += f" {answer}"
    return prompt
