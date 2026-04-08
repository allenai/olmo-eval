"""Example: serialize a HuggingFace dataset to the JSONL format used by SerializedTask.

This standalone script does not depend on oe-eval.  It loads the SciQ
dataset (a simple science multiple-choice QA benchmark), formats each
item as a 0-shot loglikelihood request, and writes a JSONL file that
can be consumed directly by ``SerializedTask`` in olmo-eval.

Usage:
    pip install datasets
    python examples/serialize_task_example.py --output /tmp/sciq_serialized.jsonl

After producing the file you can register it in olmo-eval:

    register_variant(
        "serialized",
        "sciq_mc",
        data_source=DataSource(path="/tmp/sciq_serialized.jsonl"),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )

See docs/serialized_tasks.md for the full guide.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any


def load_sciq_test() -> list[dict[str, Any]]:
    """Load the SciQ test split from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("allenai/sciq", split="test")
    return list(ds)


def make_prompt(doc: dict[str, Any], choices: list[str]) -> str:
    """Build a simple 0-shot MC prompt."""
    labels = "ABCD"
    lines = [doc["question"]]
    for i, choice in enumerate(choices):
        lines.append(f"  {labels[i]}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def serialize_doc(doc_id: int, doc: dict[str, Any]) -> dict[str, Any]:
    """Convert one SciQ document into a serialized record dict.

    SciQ has three distractor fields and one correct_answer field.
    We assemble the four choices, shuffle them, and record the gold
    index in metadata.
    """
    choices = [doc["distractor1"], doc["distractor2"], doc["distractor3"], doc["correct_answer"]]
    gold_idx = 3  # correct_answer is last before shuffle

    rng = random.Random(doc_id)
    order = list(range(4))
    rng.shuffle(order)
    choices = [choices[i] for i in order]
    gold_idx = order.index(3)

    prompt = make_prompt(doc, choices)

    # Each continuation is one answer choice; the model scores each via
    # loglikelihood and the highest-scoring continuation is the prediction.
    labels = "ABCD"
    continuations = [f" {labels[i]}" for i in range(len(choices))]

    return {
        # -- Instance-level fields (for scoring) --
        "doc_id": doc_id,
        "question": doc["question"],
        "gold_answers": [doc["correct_answer"]],
        "choices": choices,
        "metadata": {
            "task_name": "sciq_serialized",
            "gold_idx": gold_idx,
            "support": doc.get("support", ""),
        },
        # -- LMRequest-level fields (for inference) --
        "request_type": "loglikelihood",
        "prompt": prompt,
        "messages": None,
        "continuations": continuations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=str, required=True, help="Path for the output JSONL file")
    args = parser.parse_args()

    docs = load_sciq_test()
    print(f"Loaded {len(docs)} SciQ test examples")

    with open(args.output, "w") as f:
        for doc_id, doc in enumerate(docs):
            record = serialize_doc(doc_id, doc)
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
