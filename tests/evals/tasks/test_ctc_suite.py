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
from olmo_eval.evals.tasks.ctc_suite import ROSTER, RUNG_TOKENS, CTCClass


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

#: A rerank row with no cross-encoder-positive document -- the shape 14 of 500 real rows have at
#: both r2k and r32k. ``ce_pos_recall`` has no reference set to recall, so it is 0 for every
#: possible answer, the perfect one included.
RERANK_EXAMPLE_NO_CE = {
    "source": "msmarco",
    "queries": ["what is the capital of france"],
    "answers": ["paris"],
    "documents": [{"text": "paris is the capital of france"}, {"text": "unrelated filler"}],
    "gold_doc_indices": [0],  # 0-based
    "ce_scores": [-1.5, -4.0],  # nothing above 0
}

#: A plain-grouping row: unlabeled clusters over short abstracts, with the requested K in the query.
GROUPING_EXAMPLE = {
    "source": "openalex",
    "queries": ["Partition the 4 documents into 2 groups."],
    "answers": [],
    "documents": [
        {"text": "a study of protein folding"},
        {"text": "further results on protein folding"},
        {"text": "a survey of graph algorithms"},
        {"text": "new bounds for graph algorithms"},
    ],
    "gold_doc_indices": [[0, 1], [2, 3]],
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


# ── Task identity ───────────────────────────────────────────────────────────────────────────────


def test_every_rung_pins_the_dataset_revision():
    """A dataset rebuild must not land under an existing task hash.

    ``PrasannSinghal/ctc-suite-eval`` has been rebuilt during this suite's life (the xabsence
    exact-copy rebuild, the rerank repricing), and each rebuild changes what an already-stored
    number meant. ``DataSource.revision`` is serialized into ``TaskConfig.to_dict()``, so pinning
    it is what makes a rebuild show up as a new hash rather than as a silent reinterpretation.
    """
    from olmo_eval.evals.tasks.common import get_task
    from olmo_eval.evals.tasks.ctc_suite import HF_REVISION

    for name, row in ROSTER.items():
        for rung in row.rungs:
            source = get_task(f"{name}:{rung}").config.data_source
            assert source.revision == HF_REVISION, f"{name}:{rung} is not pinned"
        assert get_task(name).config.data_source.revision == HF_REVISION


def test_a_local_data_root_changes_the_task_hash(tmp_path, monkeypatch):
    """A run over an unvalidated local ladder must not be stored as a run over the published data.

    ``CTC_SUITE_DATA_ROOT`` used to swap the rows at load time without touching the config, so both
    runs produced the same ``TaskConfig.to_dict()`` hash and the same artifact name.
    """
    from olmo_eval.common.types import compute_task_hash
    from olmo_eval.evals.tasks.common import get_task

    row = ROSTER["ctc_nq"]
    _write_ladder(tmp_path, row.subset, RUNG_TOKENS["r2k"], RETRIEVAL_EXAMPLE)

    published = compute_task_hash(get_task("ctc_nq:r2k").config.to_dict())
    monkeypatch.setenv("CTC_SUITE_DATA_ROOT", str(tmp_path))
    local = compute_task_hash(get_task("ctc_nq:r2k").config.to_dict())

    assert published and local
    assert published != local, "a local ladder must not share the published data's task hash"
    assert str(tmp_path) in get_task("ctc_nq:r2k").config.data_source.path


# ── Scoring robustness ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("task_name", "example", "generation"),
    [
        # parse_pairs converted pair elements outside its try:
        ("ctc_contradiction", PAIR_EXAMPLE, '[["Doc 1", "Doc 3"]]'),  # ValueError from int(...)
        ("ctc_contradiction", PAIR_EXAMPLE, "[[1, null]]"),  # TypeError from int(None)
        # _first_groups_object iterated obj["groups"] without checking it was iterable:
        ("ctc_grouping", GROUPING_EXAMPLE, '{"groups": 5}'),  # TypeError: not iterable
        ("ctc_grouping", GROUPING_EXAMPLE, '{"groups": [["a", "b"]]}'),  # ValueError from int("a")
    ],
)
def test_one_malformed_generation_does_not_abort_the_task(
    tmp_path, monkeypatch, task_name, example, generation
) -> None:
    """The synchronous scoring path has no per-output exception handling, so a parser that raises
    takes the whole task's results with it. A generation nobody can parse must score 0, not throw.
    """
    row = ROSTER[task_name]
    rung = row.rungs[0]
    _write_ladder(tmp_path, row.subset, row.rung_alias.get(rung, RUNG_TOKENS[rung]), example)
    monkeypatch.setenv("CTC_SUITE_DATA_ROOT", str(tmp_path))

    task = get_task(f"{task_name}:{rung}")
    instance = list(task.instances)[0]
    for metric in task.config.metrics:
        assert metric.scorer().score(instance, LMOutput(text=generation)) == 0.0


@pytest.mark.parametrize(
    ("parser", "generation"),
    [
        ("pairs", '[["Doc 1", "Doc 3"]]'),
        ("pairs", "[[1, null]]"),
        ("partition", '{"groups": 5}'),
        ("partition", '{"groups": [["a", "b"]]}'),
    ],
)
def test_the_vendored_parsers_return_rather_than_raise(parser, generation) -> None:
    """The same four inputs against the parsers directly, so a future re-vendor that reintroduces
    the bug fails here and not only through a task."""
    from olmo_eval.evals.tasks.ctc_suite._vendor.ctc.format import parsing

    if parser == "pairs":
        assert parsing.parse_pairs(generation) in (None, [])
    else:
        assert parsing.parse_partition(generation, 4) is None


def test_an_all_singleton_partition_scores_one_when_it_is_exactly_right() -> None:
    """48 of 500 grouping examples at r2k have k == n. With no co-clustered pairs on either side,
    pairwise precision/recall/f1 were all 0 even for an exact match, capping the r2k mean at 0.904.
    """
    from olmo_eval.evals.tasks.ctc_suite._vendor.ctc.format.metrics import pairwise_metrics

    singletons = list(range(10))
    scored = pairwise_metrics(singletons, singletons)
    assert scored["pairwise_precision"] == 1.0
    assert scored["pairwise_recall"] == 1.0
    assert scored["pairwise_f1"] == 1.0

    # One-sided empty is still 0.0: predicting one big cluster against an all-singleton gold agrees
    # on no pair that either side asserts, and must not inherit the both-empty credit.
    assert pairwise_metrics([0] * 10, singletons)["pairwise_f1"] == 0.0


def test_a_truncated_reasoning_trace_is_a_parse_failure(tmp_path, monkeypatch) -> None:
    """Ids inside an unclosed ``<think>`` are the ones the model was still weighing. Scoring them
    credits a model for thinking out loud, and at 64-512 token budgets almost every reasoning-model
    generation is an unfinished trace."""
    row = ROSTER["ctc_contradiction"]
    rung = row.rungs[0]
    _write_ladder(tmp_path, row.subset, row.rung_alias.get(rung, RUNG_TOKENS[rung]), PAIR_EXAMPLE)
    monkeypatch.setenv("CTC_SUITE_DATA_ROOT", str(tmp_path))

    task = get_task(f"ctc_contradiction:{rung}")
    instance = list(task.instances)[0]
    scorers = {m.scorer().name: m.scorer() for m in task.config.metrics}
    truncated = "<think>Candidates [[1, 3]] and [[1, 2]], let me check"

    assert scorers["ctc_parse_ok"].score(instance, LMOutput(text=truncated)) == 0.0
    assert scorers["ctc"].score(instance, LMOutput(text=truncated)) == 0.0
    # A closed trace is still graded on what follows it.
    closed = "<think>weighing [[1, 2]]</think>[[1, 3]]"
    assert scorers["ctc"].score(instance, LMOutput(text=closed)) == 1.0


def test_rerank_rows_without_a_ce_reference_leave_the_mean(tmp_path, monkeypatch) -> None:
    """14 of 500 rerank rows at r2k and at r32k have no document with CE > 0. ``ce_pos_recall`` is
    undefined there -- a perfect qrel-first ranking still scores 0 -- so averaging them in as zeros
    put the ceiling a perfect model could reach at 0.972."""
    from olmo_eval.common.types import Response

    row = ROSTER["ctc_rerank"]
    rung = row.rungs[0]
    _write_ladder(tmp_path, row.subset, RUNG_TOKENS[rung], RERANK_EXAMPLE_NO_CE)
    monkeypatch.setenv("CTC_SUITE_DATA_ROOT", str(tmp_path))

    task = get_task(f"ctc_rerank:{rung}")
    instance = list(task.instances)[0]
    primary = task.config.primary_metric
    scorer = primary.scorer()

    output = LMOutput(text="[2], [1]")
    response = Response(instance=instance, request=task.format_request(instance), outputs=[output])
    response.scores[scorer.name] = scorer.score(instance, output)

    assert output.metadata["ctc_all_metrics"]["ce_ref_available"] == 0.0
    # No countable row -> the mean is not a 0.0 that reads as a model failure.
    assert primary.compute([response]) == 0.0
    assert primary.skip_when_zero == "ce_ref_available"
    assert "ce_ref_available" in {m.name for m in task.config.metrics}


# ── Decode budgets ──────────────────────────────────────────────────────────────────────────────


def test_obliq_gets_a_budget_its_own_gold_fits() -> None:
    """obliq is in the retrieval family but unlike the rest of it (at most 3 gold ids) its gold
    sets reach 64 ids. Under the family's 64-token budget the longest perfect answer -- 311 Qwen3
    tokens at r32k, 404 at r1m -- was truncated into partial credit for 27/126 rows at r32k."""
    from olmo_eval.evals.tasks.common import get_task

    obliq = get_task("ctc_obliq:r32k")
    assert obliq.decode_budget() == 512
    # The rest of the retrieval family keeps the spec's own budget.
    assert get_task("ctc_nq:r32k").decode_budget() == 64
    assert obliq.get_sampling_params(None).max_tokens == 512


def test_an_explicit_sampling_override_reaches_the_request() -> None:
    """``config.sampling_params`` was ignored, so an override changed the task hash and nothing
    else -- and a reasoning model could not be given room to close its trace."""
    from olmo_eval.common.types import SamplingParams
    from olmo_eval.evals.tasks.common import get_task

    task = get_task(
        "ctc_nq:r2k",
        config_overrides={"sampling_params": SamplingParams(max_tokens=4096, temperature=0.7)},
    )
    params = task.get_sampling_params(None)
    assert params.max_tokens == 4096
    assert params.temperature == 0.7
    # A field the caller left at its default still gets the suite's setting.
    partial = get_task(
        "ctc_nq:r2k", config_overrides={"sampling_params": SamplingParams(temperature=0.7)}
    )
    assert partial.get_sampling_params(None).max_tokens == 64


# ── Roster bookkeeping ──────────────────────────────────────────────────────────────────────────


def test_rungs_that_are_not_500_examples_are_declared() -> None:
    """Measured against the published dataset on 2026-09-28. The README tells readers "everything
    else is 500", which is only true if every exception is written down -- including the rungs that
    are *larger* than the default, which an "under 500" rule would never catch."""
    assert ROSTER["ctc_absence"].eval_size["r16k"] == 148
    assert ROSTER["ctc_oolong"].eval_size["r64k"] == 668
    assert ROSTER["ctc_oolong"].eval_size["r128k"] == 669
    assert ROSTER["ctc_obliq"].eval_size["r2k"] == 123
    assert ROSTER["ctc_scifact"].eval_size["r32k"] == 300


def test_the_grouping_prompt_states_its_instruction_once(tmp_path, monkeypatch) -> None:
    """The factory already puts GROUPING_INSTRUCTION in the header, so a query builder that
    prepends it again printed the whole instruction twice in every plain-grouping prompt."""
    from olmo_eval.evals.tasks.ctc_suite._vendor.ctc.format.prompts import GROUPING_INSTRUCTION

    row = ROSTER["ctc_grouping"]
    rung = row.rungs[0]
    _write_ladder(tmp_path, row.subset, RUNG_TOKENS[rung], GROUPING_EXAMPLE)
    monkeypatch.setenv("CTC_SUITE_DATA_ROOT", str(tmp_path))

    task = get_task(f"ctc_grouping:{rung}")
    prompt = task.format_request(list(task.instances)[0]).prompt
    assert prompt.count(GROUPING_INSTRUCTION) == 1


def test_every_row_is_a_real_registered_class() -> None:
    """The rows were built with ``type()``, which leaves 22 classes that cannot be imported,
    grepped or pickled under their own names."""
    import olmo_eval.evals.tasks.ctc_suite as suite
    from olmo_eval.evals.tasks.common import get_task

    for name in ROSTER:
        cls = type(get_task(name))
        assert cls.row_name == name
        assert getattr(suite, cls.__name__, None) is cls, f"{name} is not importable by name"


def test_the_run_counts_the_readme_quotes_are_the_real_ones() -> None:
    """The README's quickstart prints a run count next to each suite ("all 22 tasks, 2k-32k grid
    (108 runs)"). Those are the numbers someone budgets GPU time against, and nothing stopped them
    going stale as rungs were added -- so they are pinned here rather than in prose alone."""
    expected = {
        "ctc": 177,
        "ctc:figure": 108,
        "ctc:xlong": 69,
        "ctc:low": 106,
        "ctc:high": 71,
        "ctc:high:figure": 49,
        "ctc:low:xlong": 47,
    }
    actual = {name: len(get_suite(name).expand()) for name in expected}
    assert actual == expected, "a suite changed size; update the README's counts with it"


def test_the_sub_metrics_reach_the_predictions_file(tmp_path, monkeypatch) -> None:
    """``ctc_all_metrics`` (k_exact, coverage, recall@k, ce_ref_available ...) is what someone
    reading a low score needs, and it lived only on ``output.metadata``, which the predictions
    writer drops. ``score:``-prefixed metadata keys are persisted per output as ``sample_metrics``,
    so each sub-metric is written under one."""
    from olmo_eval.common.types import Response
    from olmo_eval.runners.io.builders import build_predictions

    row = ROSTER["ctc_nq"]
    rung = row.rungs[0]
    _write_ladder(tmp_path, row.subset, RUNG_TOKENS[rung], RETRIEVAL_EXAMPLE)
    monkeypatch.setenv("CTC_SUITE_DATA_ROOT", str(tmp_path))

    task = get_task(f"ctc_nq:{rung}")
    instance = list(task.instances)[0]
    output = LMOutput(text="[5]")
    scorer = task.config.primary_metric.scorer()
    response = Response(instance=instance, request=task.format_request(instance), outputs=[output])
    response.scores[scorer.name] = scorer.score(instance, output)

    sub = output.metadata["ctc_all_metrics"]
    assert {"exact_match", "recall", "precision", "f1"} <= set(sub)
    for key, value in sub.items():
        assert output.metadata[f"score:ctc_{key}"] == value

    persisted = build_predictions([response], "ctc_nq")[0]["model_output"][0]["sample_metrics"]
    assert persisted["ctc_f1"]["ctc_f1"] == 1.0
