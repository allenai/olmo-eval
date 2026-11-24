"""HELMET Multi-LexSum (summarization) data loading.

Ports HELMET's `load_multi_lexsum` (https://github.com/princeton-nlp/HELMET,
data.py at aeadacc6): the filings from a civil rights lawsuit, summarized in a
paragraph and graded by an LLM judge.

`load_dataset("allenai/multi_lexsum")` no longer works: `datasets` 4.0 dropped
script-based loading and the Hub's parquet conversion has no `v20230518` config.
The pieces that loader assembled come from two pinned sources instead:

- `allenai/helmet-plus` hosts HELMET's `multi_lexsum_val.jsonl`: the validation
  cases with their joined sources, both summary lengths, and the pre-extracted
  key points the judge needs.
- The `v20230518` release files in `allenai/multi_lexsum` give the validation
  split's row order (`dev.json`), which seeded sampling depends on, and the train
  split's short summaries (`train.json`), which the demos are drawn from. Neither
  needs the multi-gigabyte `sources.json`.

Selection then follows upstream step for step: drop cases without a short
summary, keep validation cases whose documents span at least
`MIN_SOURCE_TOKENS` reference tokens, sample with `Dataset.shuffle(seed)`, and
draw each instance's demos from the train split with its own `demo_seed`.
"""

import hashlib
import json
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import silent_tqdm

from olmo_eval.data.helmet_infbench_loader import (
    REFERENCE_TOKENIZER,
    TRUNCATION_POSTFIX,
    _load_reference_tokenizer,
    _truncate_context,
)
from olmo_eval.data.helmet_loader import download_helmet_plus_file

MULTI_LEXSUM_FILE = "multi_lexsum/multi_lexsum_val.jsonl"

MULTI_LEXSUM_REPO = "allenai/multi_lexsum"
MULTI_LEXSUM_REVISION = "055f5fa5300a5fdefb5b691c4d6c1e58f485b379"
MULTI_LEXSUM_RELEASE = "releases/v20230518"

# Upstream's `filter_length(test_data, 65536, "context")`: only validation cases
# whose joined sources reach this many reference tokens are evaluated, at every
# context length.
MIN_SOURCE_TOKENS = 65536

_USER_TEMPLATE = (
    "You are given the legal documents in a civil rights lawsuit, and you are tasked "
    "to summarize the case. Write a concise summary of one paragraph (200 to 250 "
    "words). The summary should contain a short description of the background, the "
    "parties involved, and the outcomes of the case.\n\n{demo}Legal documents:\n"
    "{context}\n\nNow please summarize the case."
)
_SYSTEM_TEMPLATE = "Summary:"
_DEMO_TEMPLATE = "Summary: {summary}"
_DEMO_PREFIX = "Example summaries:\n\n"
# between documents, and between demos
_SEPARATOR = "\n\n"
_DEMO_SUFFIX = "\n\nNow, write a summary of the following legal documents.\n"

# Upstream's prompt_template puts a blank line before the answer prefix here,
# where most HELMET loaders use a single newline.
_ANSWER_PREFIX_SEPARATOR = "\n\n"

#: Every piece of text and every selection threshold this loader builds prompts
#: from. HELMET's task settings list it, so changing one moves the task hash.
MEASURED_CONSTANTS = {
    "user_template": _USER_TEMPLATE,
    "system_template": _SYSTEM_TEMPLATE,
    "demo_template": _DEMO_TEMPLATE,
    "demo_prefix": _DEMO_PREFIX,
    "separator": _SEPARATOR,
    "demo_suffix": _DEMO_SUFFIX,
    "truncation_postfix": TRUNCATION_POSTFIX,
    "min_source_tokens": MIN_SOURCE_TOKENS,
    "answer_prefix_separator": _ANSWER_PREFIX_SEPARATOR,
}


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _release_cases(filename: str) -> list[dict[str, Any]]:
    """The cases of one `v20230518` release split, in file order."""
    path = hf_hub_download(  # ty: ignore[no-matching-overload]
        repo_id=MULTI_LEXSUM_REPO,
        filename=f"{MULTI_LEXSUM_RELEASE}/{filename}",
        repo_type="dataset",
        revision=MULTI_LEXSUM_REVISION,
        tqdm_class=silent_tqdm,
    )
    return _read_jsonl(path)


def demo_seed(key: str, seed: int) -> int:
    """Upstream's per-instance demo seed: deterministic in the key and the run seed."""
    return (int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) + seed) % 2**31


def shuffled_indices(length: int, seed: int) -> list[int]:
    """The row order `datasets.Dataset.shuffle(seed=seed)` produces for `length` rows."""
    return np.random.default_rng(seed).permutation(length).tolist()


def _spans_min_tokens(context: str, tokenizer) -> bool:
    """Whether `context` reaches `MIN_SOURCE_TOKENS` reference tokens, BOS included.

    Every Llama-2 token covers at least one UTF-8 byte except the BOS token and the
    leading word-boundary marker, so text this short in bytes cannot reach the
    threshold and is not tokenized.
    """
    if len(context.encode("utf-8")) + 2 < MIN_SOURCE_TOKENS:
        return False
    return len(tokenizer(context)["input_ids"]) >= MIN_SOURCE_TOKENS


def load_multi_lexsum_dataset(
    max_context_tokens: int,
    shots: int = 2,
    max_samples: int | None = None,
    seed: int = 42,
    reference_tokenizer: str = REFERENCE_TOKENIZER,
) -> dict[str, Any]:
    """Load Multi-LexSum truncated to a target context length.

    Args:
        max_context_tokens: Token budget for the documents, measured with the
            reference tokenizer. HELMET reserves more of the tier here than for
            LongQA (300 tokens rather than 200), so this is computed by the
            caller from the task config.
        shots: Number of example summaries to prepend, drawn from the train split
            per instance. They carry no source documents, so they cost little
            against the budget.
        max_samples: Cap on the number of instances.
        seed: Seed for instance sampling and demo selection.
        reference_tokenizer: Tokenizer used for the length filter and truncation.

    Returns:
        Dictionary with `data` (processed records) and the HELMET prompt templates.

    Raises:
        ValueError: If a validation case is missing from the hosted file, or a
            selected case has no key points for the judge.
    """
    hosted = {row["id"]: row for row in _read_jsonl(download_helmet_plus_file(MULTI_LEXSUM_FILE))}

    # HELMET keeps only cases with a short gold summary, in every split
    validation_ids = [
        case["case_id"] for case in _release_cases("dev.json") if case["summary/short"] is not None
    ]
    absent = [case_id for case_id in validation_ids if case_id not in hosted]
    if absent:
        raise ValueError(
            f"{len(absent)} Multi-LexSum validation cases are missing from "
            f"{MULTI_LEXSUM_FILE}: {absent[:5]}"
        )

    tokenizer = _load_reference_tokenizer(reference_tokenizer)
    rows = []
    for case_id in validation_ids:
        row = hosted[case_id]
        context = _SEPARATOR.join(row["sources"])
        if _spans_min_tokens(context, tokenizer):
            rows.append((row, context))

    if max_samples is not None and len(rows) > max_samples:
        rows = [rows[i] for i in shuffled_indices(len(rows), seed)[:max_samples]]

    without_key_points = [row["id"] for row, _ in rows if not row.get("summary/short_keypoints")]
    if without_key_points:
        # the judge cannot grade a summary without them
        raise ValueError(
            f"{len(without_key_points)} Multi-LexSum cases have no key points in "
            f"{MULTI_LEXSUM_FILE}: {without_key_points[:5]}"
        )

    train_summaries = [
        case["summary/short"]
        for case in _release_cases("train.json")
        if case["summary/short"] is not None
    ]
    separator_length = len(tokenizer(TRUNCATION_POSTFIX)["input_ids"])

    data = []
    for row, context in rows:
        demo_text = ""
        if shots > 0:
            picks = shuffled_indices(len(train_summaries), demo_seed(row["id"], seed))[:shots]
            rendered = _SEPARATOR.join(
                _DEMO_TEMPLATE.format(summary=train_summaries[i]) for i in picks
            )
            demo_text = f"{_DEMO_PREFIX}{rendered}{_DEMO_SUFFIX}"
        data.append(
            {
                "context": _truncate_context(
                    context, max_context_tokens, tokenizer, separator_length
                ),
                "question": "",
                "demo": demo_text,
                "answer": [row["summary/short"]],
                "keypoints": row["summary/short_keypoints"],
                # the precision rubric grades the model's sentences against the
                # expert *long* summary, not the short one it is asked to match
                "expert_summary": row["summary/long"],
            }
        )

    return {
        "data": data,
        "user_template": _USER_TEMPLATE,
        "system_template": _SYSTEM_TEMPLATE,
    }
