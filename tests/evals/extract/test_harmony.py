"""Harmony parsing requires reserved message delimiters."""

import pytest

from olmo_eval.evals.extract.reasoning import extract_harmony_answer


@pytest.mark.parametrize(
    "text, expected",
    [
        ("plain assistantfinal text", "plain assistantfinal text"),
        ("analysis of assistantfinal in prose", "analysis of assistantfinal in prose"),
        ("", ""),
        ("<|channel|>analysis", ""),
        ("<|channel|>final", ""),
        ("<|start|>assistant", ""),
        ("<|start|>assistant<|channel|>anal", ""),
        ("<|start|>assistant<|channel|>final", ""),
        ("<|channel|>final to=python<|message|>tool payload", ""),
        ("<|channel|>analysis<|message|>unfinished", ""),
        ("<|channel|>final<|message|>\n\n 4 <|return|>", "\n\n 4 "),
        ("<|channel|>final<|message|>4<|endoftext|>", "4"),
        (
            "<|channel|>analysis<|message|>wrong<|end|>"
            "<|start|>assistant<|channel|>final<|message|>right<|return|>",
            "right",
        ),
        (
            "<|start|>assistant to=python<|channel|>analysis<|message|>1+1<|call|>",
            "",
        ),
        (
            "<|channel|>analysis<|message|>mentions <|channel|>final<|message|>not a header",
            "",
        ),
        (
            "<|channel|>final<|message|>assistantfinal is literal\nanswer",
            "assistantfinal is literal\nanswer",
        ),
        (
            "<|channel|>analysis<|message|>wrong<|end|>"
            "<|start|>assistant<|channel|>final<|message|>first<|end|>"
            "<|start|>user<|message|>another turn",
            "first",
        ),
        (
            "<|im_start|>assistant<|meta_sep|>final<|im_sep|>chatml text<|im_end|>",
            "<|im_start|>assistant<|meta_sep|>final<|im_sep|>chatml text<|im_end|>",
        ),
    ],
)
def test_harmony_final_body(text: str, expected: str) -> None:
    assert extract_harmony_answer(text) == expected
