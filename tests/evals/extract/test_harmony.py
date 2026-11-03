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
        ("<|meta_sep|>final", ""),
        ("<|start|>assistant", ""),
        ("<|start|>assistant<|channel|>anal", ""),
        ("<|im_start|>assistant<|meta_sep|>final", ""),
        ("<|channel|>final to=python<|message|>tool payload", ""),
        ("<|meta_sep|>analysis<|im_sep|>unfinished", ""),
        ("<|channel|>analysis<|message|>unfinished", ""),
        ("<|meta_sep|>final<|im_sep|>\n\n 4 <|fim_suffix|>", "\n\n 4 "),
        ("<|channel|>final<|message|>4<|endoftext|>", "4"),
        (
            "<|channel|>analysis<|message|>wrong<|end|>"
            "<|start|>assistant<|channel|>final<|message|>right<|return|>",
            "right",
        ),
        (
            "<|meta_sep|>analysis<|im_sep|>wrong<|im_end|>"
            "<|im_start|>assistant<|meta_sep|>final<|im_sep|>right<|fim_suffix|>",
            "right",
        ),
        (
            "<|im_start|>assistant to=python<|meta_sep|>analysis<|im_sep|>1+1<|ghissue|>",
            "",
        ),
        (
            "<|meta_sep|>analysis<|im_sep|>mentions <|meta_sep|>final<|im_sep|>not a header",
            "",
        ),
        (
            "<|meta_sep|>final<|im_sep|>assistantfinal is literal\nanswer",
            "assistantfinal is literal\nanswer",
        ),
        (
            "<|meta_sep|>analysis<|im_sep|>wrong<|im_end|>"
            "<|im_start|>assistant<|meta_sep|>final<|im_sep|>first<|im_end|>"
            "<|im_start|>user<|im_sep|>another turn",
            "first",
        ),
    ],
)
def test_harmony_final_body(text: str, expected: str) -> None:
    assert extract_harmony_answer(text) == expected
