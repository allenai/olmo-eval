"""CTC suite: registration, formatting, and scoring round-trips on synthetic examples.

No network and no dataset downloads: examples are written to tmp_path as JSONL and loaded through
``CTC_SUITE_DATA_ROOT``, the suite's local-tree escape hatch.
"""

from __future__ import annotations

import json

import pytest

from olmo_eval.common.types import LMOutput
from olmo_eval.evals.tasks.common.registry import get_task, list_tasks, list_variants
from olmo_eval.evals.tasks.ctc_suite import ROSTER, RUNG_TOKENS


def test_all_22_rows_register() -> None:
    names = {n for n in list_tasks() if n.startswith("ctc_")}
    assert names == set(ROSTER)
    assert len(names) == 22


def test_every_row_has_its_rung_variants() -> None:
    for name, row in ROSTER.items():
        variants = set(list_variants(name).get(name, []))
        assert set(row.rungs) <= variants, f"{name}: missing rung variants"


def test_sub500_rungs_are_flagged() -> None:
    for name, row in ROSTER.items():
        for rung in ("r256k", "r512k", "r1m"):
            if rung in row.rungs:
                assert row.eval_size.get(rung, 500) < 500, (
                    f"{name}:{rung} must carry an eval_size flag -- xlong rungs are subsampled"
                )


def _write_ladder(tmp_path, subset: str, rung_tokens: int, example: dict) -> None:
    d = tmp_path / subset
    d.mkdir(parents=True, exist_ok=True)
    (d / f"rung_{rung_tokens}.jsonl").write_text(json.dumps(example) + "\n")


#: Minimal but structurally faithful examples for three representative families.
RETRIEVAL_EXAMPLE = {
    "source": "nq",
    "queries": ["who wrote the paper"],
    "answers": ["ada"],
    "documents": [{"text": f"filler document number {i}"} for i in range(4)]
    + [{"text": "the paper was written by ada"}],
    "gold_doc_indices": [4],  # 0-based; the prompt shows documents 1-based
    "hard_neg_indices": [0],
}

PAIR_EXAMPLE = {
    "source": "contradiction",
    "queries": [],
    "answers": [],
    "documents": [
        {"text": "the sky is blue"},
        {"text": "water boils at 100C"},
        {"text": "the sky is not blue"},
    ],
    "gold_doc_indices": [[1, 3]],  # 1-based pairs for the pair family
}

QDMATCH_EXAMPLE = {
    "source": "qdmatch_nq",
    "queries": [],
    "answers": [],
    "num_queries": 2,
    "num_docs": 2,
    "num_relevant": 1,
    "layout": "separate",
    "documents": [
        {"type": "query", "text": "capital of france?"},
        {"type": "query", "text": "tallest mountain?"},
        {"type": "document", "text": "paris is the capital of france"},
        {"type": "document", "text": "unrelated filler text"},
    ],
    "gold_pairs": [[1, 3]],  # 1-based, ordered (query, document)
    "gold_doc_indices": [],
}


@pytest.mark.parametrize(
    ("task_name", "example", "gold_text", "wrong_text"),
    [
        ("ctc_nq", RETRIEVAL_EXAMPLE, "[5]", "[2]"),
        ("ctc_contradiction", PAIR_EXAMPLE, "[[1, 3]]", "[[1, 2]]"),
        ("ctc_qdmatch_nq", QDMATCH_EXAMPLE, "[[1, 3]]", "[[2, 3]]"),
    ],
)
def test_round_trip(tmp_path, monkeypatch, task_name, example, gold_text, wrong_text) -> None:
    row = ROSTER[task_name]
    rung = row.rungs[0]
    tokens = row.rung_alias.get(rung, RUNG_TOKENS[rung])
    _write_ladder(tmp_path, row.subset, tokens, example)
    monkeypatch.setenv("CTC_SUITE_DATA_ROOT", str(tmp_path))

    task = get_task(f"{task_name}:{rung}")
    instances = list(task.instances)
    assert len(instances) == 1
    request = task.format_request(instances[0])
    # every document's text must appear in the rendered prompt
    for doc in example["documents"]:
        assert doc["text"] in request.prompt

    scorer = task.config.metrics[0].scorer()
    assert scorer.score(instances[0], LMOutput(text=gold_text)) == 1.0
    assert scorer.score(instances[0], LMOutput(text=wrong_text)) < 1.0
    assert scorer.score(instances[0], LMOutput(text="no ids at all")) == 0.0


def test_gold_index_base_is_the_graders(tmp_path, monkeypatch) -> None:
    """The 0-vs-1-based split is per-family and has produced silent zero-scores before:
    retrieval gold is stored 0-based and shifted by the grader; pair gold is stored 1-based."""
    row = ROSTER["ctc_nq"]
    _write_ladder(tmp_path, row.subset, RUNG_TOKENS[row.rungs[0]], RETRIEVAL_EXAMPLE)
    monkeypatch.setenv("CTC_SUITE_DATA_ROOT", str(tmp_path))
    task = get_task(f"ctc_nq:{row.rungs[0]}")
    inst = list(task.instances)[0]
    scorer = task.config.metrics[0].scorer()
    # answering with the raw stored index (0-based "4") instead of the prompt's 1-based "5"
    # must NOT get credit
    assert scorer.score(inst, LMOutput(text="[4]")) == 0.0
