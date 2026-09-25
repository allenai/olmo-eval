"""Per-sample scoring for CC-OCR's multi-scene OCR track, vendored from the official evaluator.

Source: ``Benchmarks/CC-OCR/evaluation/evaluator/ocr_evaluator.py`` of
https://github.com/AlibabaResearch/AdvancedLiterateMachinery (MIT). Behavior is preserved
exactly; the one mechanical change is that a sample is scored on its own and returns its
sufficient statistics, so the harness can score responses as they arrive. The dataset-level
reductions live in :mod:`olmo_eval.common.scorers.cc_ocr`.

A prediction is compared with the reference as a multiset of basic units: characters for the
Chinese sub-datasets (those whose name contains ``zh``), lower-cased alphanumeric-only words
for the rest. ``###`` / ``***`` markers and whitespace layout are ignored.
"""

from __future__ import annotations

import re
from collections import Counter


def ocr_eval_config(dataset: str) -> dict[str, bool]:
    """The official tokenization switches for a multi-scene sub-dataset."""
    is_word_level = "zh" not in dataset
    return {"word_level": is_word_level, "alphanum_only": is_word_level, "lowercase": True}


def _token_normalize(token_text: str, is_lower: bool, is_alphanum_only: bool) -> str:
    if is_lower:
        token_text = token_text.lower()
    if is_alphanum_only:
        token_text = re.sub("[^A-Za-z0-9]+", "", token_text)
    return token_text


def text_normalize_and_tokenize(
    text: str, is_keep_blank: bool = True, is_lower: bool = True, is_alphanum_only: bool = False
) -> list[str]:
    text = text.replace("\t", " ").replace("\n", " ").replace("###", "").replace("***", "")
    text = re.sub(r"\s+", " ", text)
    if not is_keep_blank:
        text = text.replace(" ", "")
    text_tokens = text.split(" ") if is_keep_blank else list(text)
    normalized = [_token_normalize(t, is_lower, is_alphanum_only) for t in text_tokens]
    return [x for x in normalized if len(x) > 0]


def score_ocr_sample(prediction: str, answer: str, dataset: str) -> dict[str, float]:
    """Multiset unit overlap for one image, plus its per-image recall/precision/f1."""
    config = ocr_eval_config(dataset)
    args = (config["word_level"], config["lowercase"], config["alphanum_only"])
    preds = text_normalize_and_tokenize(str(prediction).strip(), *args)
    gts = text_normalize_and_tokenize(str(answer).strip(), *args)

    pred_counter = Counter(preds)
    right_num = sum(min(count, pred_counter.get(token, 0)) for token, count in Counter(gts).items())
    recall = right_num / (len(gts) + 1e-9)
    precision = right_num / (len(preds) + 1e-9)
    return {
        "right_num": right_num,
        "gt_num": len(gts),
        "pred_num": len(preds),
        "recall": recall,
        "precision": precision,
        "f1": 2 * recall * precision / (recall + precision + 1e-9),
    }
