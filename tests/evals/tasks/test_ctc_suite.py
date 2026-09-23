"""CTC suite: registration, formatting, and scoring round-trips on synthetic examples.

No network and no dataset downloads: examples are written to tmp_path as JSONL and loaded through
``CTC_SUITE_DATA_ROOT``, the suite's local-tree escape hatch.
"""

from __future__ import annotations

import json

import pytest

from olmo_eval.common.types import LMOutput
from olmo_eval.evals.suites import get_suite
from olmo_eval.evals.tasks.common.registry import get_task, list_tasks, list_variants
from olmo_eval.evals.tasks.ctc_suite import OOD_ROSTER, ROSTER, RUNG_TOKENS, CTCClass


def test_all_22_rows_register() -> None:
    names = {n for n in list_tasks() if n.startswith("ctc_")}
    # the 22 in-distribution rows, plus the held-out OOD rows that register alongside them
    assert names == set(ROSTER) | set(OOD_ROSTER)
    assert len(ROSTER) == 22
    assert not set(ROSTER) & set(OOD_ROSTER)


def test_every_roster_spec_is_registered() -> None:
    # ctc_grouping named a spec that was never vendored, so the row could not be graded at all
    from olmo_eval.evals.tasks.ctc_suite import _resolve_spec

    for row in (*ROSTER.values(), *OOD_ROSTER.values()):
        _resolve_spec(row.spec)


def test_grouping_all_singleton_partition_scores_one() -> None:
    from olmo_eval.evals.tasks.ctc_suite import _resolve_spec

    spec = _resolve_spec("grouping")
    assert spec.score([[1], [2], [3]], [[0], [1], [2]])["pairwise_f1"] == 1.0
    assert spec.score([[1], [2], [3]], [[0, 1], [2]])["pairwise_f1"] == 0.0


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


#: The 10 rows whose answer is a relation over documents rather than a document to retrieve.
#: Pinned here, not derived from the roster, so that flipping a row's class is a two-file change
#: someone has to mean -- ``ctc:high`` is a number people report against.
#: ``ctc_absence`` is deliberately NOT here: its two versions are aligned and in order, so the
#: deletions come out of one pass. Only ``ctc_xabsence`` needs the all-pairs comparison.
_HIGH_CTC = {
    "ctc_qdmatch_fiqa",
    "ctc_qdmatch_nq",
    "ctc_qdmatch_hpqa",
    "ctc_outlier",
    "ctc_grouping",
    "ctc_xabsence",
    "ctc_reorder",
    "ctc_contradiction",
    "ctc_strmatch",
    "ctc_textgroups",
}


def test_ctc_class_split_is_the_declared_one() -> None:
    high = {n for n, row in ROSTER.items() if row.ctc_class is CTCClass.HIGH}
    low = {n for n, row in ROSTER.items() if row.ctc_class is CTCClass.LOW}
    assert high == _HIGH_CTC
    assert low == set(ROSTER) - _HIGH_CTC
    assert len(high) == 10
    assert len(low) == 12
    # The coarse split must agree with the per-row complexity notation the README table prints.
    for name, row in ROSTER.items():
        expected = CTCClass.LOW if row.complexity == "O(N)" else CTCClass.HIGH
        assert row.ctc_class is expected, f"{name}: {row.complexity} vs {row.ctc_class}"


def test_low_and_high_suites_partition_the_suite() -> None:
    low = set(get_suite("ctc:low").expand())
    high = set(get_suite("ctc:high").expand())
    assert not (low & high), "a task:rung cannot be both low- and high-CTC"
    assert low | high == set(get_suite("ctc").expand())
    for cls in ("low", "high"):
        halves = set(get_suite(f"ctc:{cls}:figure").expand()) | set(
            get_suite(f"ctc:{cls}:xlong").expand()
        )
        assert halves == set(get_suite(f"ctc:{cls}").expand()), (
            f"ctc:{cls}:figure + ctc:{cls}:xlong must cover ctc:{cls}"
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


def test_every_row_reports_parse_rate_next_to_its_primary_metric():
    """A parse-rate collapse is a decoding regression wearing an accuracy drop's clothes, and the
    suite README tells readers to check it. It previously lived only on ``output.metadata``, which
    the predictions writer drops, so the advice was unfollowable from the shipped files."""
    from olmo_eval.evals.tasks.common import get_task

    for name in ROSTER:
        metrics = get_task(name).config.metrics
        assert "parse_rate" in {m.name for m in metrics}, f"{name} lost its parse-rate metric"
        assert get_task(name).config.primary_metric.name != "parse_rate"


def test_each_rung_pins_the_single_parquet_it_needs():
    """Without ``data_files`` the loader builds every split in the config first, so one rung
    downloads the whole 2k-to-1M ladder (~900MB for nq) to read a few hundred short rows."""
    from olmo_eval.evals.tasks.common import get_task

    for name, row in ROSTER.items():
        for rung in row.rungs:
            source = get_task(f"{name}:{rung}").config.data_source
            assert source.data_files == {rung: f"data/{row.subset}/{rung}.parquet"}


@pytest.mark.parametrize(
    ("task_name", "example"),
    [
        ("ctc_nq", RETRIEVAL_EXAMPLE),
        ("ctc_contradiction", PAIR_EXAMPLE),
        ("ctc_qdmatch_nq", QDMATCH_EXAMPLE),
    ],
)
def test_chat_prompt_format_is_the_sft_rendering(tmp_path, monkeypatch, task_name, example) -> None:
    from olmo_eval.common.types import RequestType
    from olmo_eval.evals.tasks.ctc_suite import PROMPT_FORMAT_ENV, QUERY_POSITION

    row = ROSTER[task_name]
    rung = row.rungs[0]
    _write_ladder(tmp_path, row.subset, row.rung_alias.get(rung, RUNG_TOKENS[rung]), example)
    monkeypatch.setenv("CTC_SUITE_DATA_ROOT", str(tmp_path))
    task = get_task(f"{task_name}:{rung}")
    instance = list(task.instances)[0]

    default = task.format_request(instance)  # unset -> alpaca, the published setting
    assert default.request_type == RequestType.COMPLETION
    assert default.prompt.startswith("Below is an instruction")

    monkeypatch.setenv(PROMPT_FORMAT_ENV, "chat")
    chat = task.format_request(instance)
    assert chat.request_type == RequestType.CHAT
    assert len(chat.messages) == 1 and chat.messages[0]["role"] == "user"
    body = chat.messages[0]["content"]
    assert "Below is an instruction" not in body and "### Instruction" not in body
    assert body == task.spec.build_prompt(example, query_position=QUERY_POSITION, use_alpaca=False)
    for doc in example["documents"]:
        assert doc["text"] in body

    monkeypatch.setenv(PROMPT_FORMAT_ENV, "nonsense")
    with pytest.raises(ValueError):
        task.format_request(instance)


def test_rerank_decode_cap_is_opt_in_and_score_preserving(monkeypatch) -> None:
    from olmo_eval.evals.tasks.ctc_suite import RERANK_DECODE_ENV, _resolve_spec

    monkeypatch.delenv(RERANK_DECODE_ENV, raising=False)
    task = get_task("ctc_rerank:r2k")
    assert task.get_sampling_params(None).max_tokens == 512
    monkeypatch.setenv(RERANK_DECODE_ENV, "160")
    assert task.get_sampling_params(None).max_tokens == 160

    # the score only reads the first 10 distinct ids, so a ranking cut after them scores the same
    spec = _resolve_spec("rerank")
    example = {
        "gold_doc_indices": [2],
        "ce_scores": [5.0, None, 3.0] + [-9.0] * 27,
        "documents": [{"text": str(i)} for i in range(30)],
    }
    full = ", ".join(f"[{i}]" for i in [3, 1, 7, 9, 11, 2, 4, 5, 6, 8, 10, 12, 13, 14, 15])
    cut = ", ".join(f"[{i}]" for i in [3, 1, 7, 9, 11, 2, 4, 5, 6, 8])
    assert spec.score(spec.parse(full, 30), example) == spec.score(spec.parse(cut, 30), example)


def test_shards_partition_the_limited_sample(tmp_path, monkeypatch) -> None:
    from dataclasses import replace as dc_replace

    from olmo_eval.evals.tasks.ctc_suite import SHARDS_ENV

    row = ROSTER["ctc_nq"]
    rung = row.rungs[0]
    d = tmp_path / row.subset
    d.mkdir(parents=True)
    rows = [dict(RETRIEVAL_EXAMPLE, queries=[f"q{i}"]) for i in range(40)]
    (d / f"rung_{RUNG_TOKENS[rung]}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    monkeypatch.setenv("CTC_SUITE_DATA_ROOT", str(tmp_path))

    def questions(shard):
        if shard:
            monkeypatch.setenv(SHARDS_ENV, json.dumps({f"{row.subset}:{rung}": shard}))
        else:
            monkeypatch.delenv(SHARDS_ENV, raising=False)
        task = get_task(f"ctc_nq:{rung}")
        task.config = dc_replace(task.config, limit=10)
        got = [i.question for i in task.instances]
        if not shard:  # the runner's own sampling, as preparation.py does it
            import random

            got = [
                i.question for i in random.Random(task.config.seed).sample(list(task.instances), 10)
            ]
        return got

    full = set(questions(None))
    parts = [questions(f"{i}/3") for i in range(3)]
    assert sum(len(p) for p in parts) == 10
    assert set().union(*map(set, parts)) == full
