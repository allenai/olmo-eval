"""Offline tests for the HELMET tasks, scorers, and suite structure.

Everything here runs without network or GPU: scorers and parsers are pure
functions, task/suite structure is import-time registration, the judged tasks
run against a scripted judge, and the loader tests write their own fixture
files. Reference expectations were derived
from HELMET's own implementations (utils.py, eval_alce.py,
scripts/eval_gpt4_*.py) -- where a case is labeled "HELMET gives X", it was
checked against upstream code.
"""

import json

import pytest

from olmo_eval.common.scorers import (
    NDCGScorer,
    RougeLF1Scorer,
    SubstringExactMatchScorer,
    SubstringRecallScorer,
    ndcg_at_k,
)
from olmo_eval.common.scorers.alce import AlceQampariRecTop5Scorer, AlceStrEmScorer
from olmo_eval.common.scorers.llm_judge import JUDGE_ATTEMPTS
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.data.helmet_icl_loader import _balance_labels
from olmo_eval.data.helmet_kilt_loader import _load_jsonl, _question_key
from olmo_eval.data.helmet_loader import sample_jsonl_by_key
from olmo_eval.data.helmet_msmarco_loader import parse_rankings
from olmo_eval.data.helmet_tasks import HELMET_TASKS, STANDARD_CONTEXT_SIZES
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.common.judge import JudgedTask
from olmo_eval.evals.tasks.helmet import (
    HelmetExactMatchScorer,
    InfbenchChoiceScorer,
    _parse_labeled_output,
)
from olmo_eval.evals.tasks.helmet_judge import (
    parse_judge_json,
    parse_longqa_answer,
    parse_longqa_verdict,
    summ_f1,
)
from olmo_eval.runners.io.builders import build_predictions
from tests.evals.tasks.judge_stubs import scripted_judge, use_judge


def _instance(gold=None, **metadata) -> Instance:
    return Instance(question="q", gold_answer=gold, metadata=metadata)


def _output(text: str, extracted=None) -> LMOutput:
    out = LMOutput(text=text)
    out.extracted_answer = extracted if extracted is not None else text
    return out


# ---------------------------------------------------------------------------
# answer parsing


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", "42"),
        (" 42\n", "42"),
        ("label: 42", "42"),
        ("Label: 42\nfoo", "42"),
        ("label:42", "42"),
        # the prefix match wins over the first line, as in HELMET's parse_output
        ("7\nlabel: 9", "9"),
        ("", ""),
        ("LABEL:  13  ", "13"),
    ],
)
def test_parse_labeled_output_matches_helmet(raw, expected):
    assert _parse_labeled_output(raw, prefix="label:") == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ranking: ID3 > ID1 > ID2", ["3", "1", "2"]),
        ("[ID: 8305152] > [ID: 12] > [ID: 7]", ["8305152", "12", "7"]),
        # duplicates keep their first (best) position
        ("Ranking: 5 > 3 > 5 > 9", ["5", "3", "9"]),
        ("The ranking is: 100 > 200 > 300", ["100", "200", "300"]),
        ("Ranking: ID10>ID20>ID30", ["10", "20", "30"]),
    ],
)
def test_parse_rankings_matches_helmet(raw, expected):
    assert parse_rankings(raw) == expected


def test_parse_rankings_unparseable_output_scores_zero():
    ranking = parse_rankings("no ranking here")
    assert ndcg_at_k(ranking, {"1": 3, "2": 1}, 10) == 0.0


# ---------------------------------------------------------------------------
# NDCG (reference values computed with pytrec_eval, trec_eval's linear gain)


def test_ndcg_matches_pytrec_eval_reference():
    relevance = {"d1": 3, "d2": 0, "d3": 2, "d4": 1, "d5": 0}
    ranking = ["d3", "d1", "d5", "d2", "d4"]
    assert ndcg_at_k(ranking, relevance, 3) == pytest.approx(0.817494, abs=1e-6)
    assert ndcg_at_k(ranking, relevance, 10) == pytest.approx(0.898733, abs=1e-6)


def test_ndcg_edge_cases():
    relevance = {"d1": 3, "d2": 1}
    # ideal ranking comes from all judged docs, so omissions cost score
    assert ndcg_at_k(["d2"], relevance, 10) < 1.0
    assert ndcg_at_k(["d1", "d2"], relevance, 10) == 1.0
    assert ndcg_at_k([], relevance, 10) == 0.0
    assert ndcg_at_k(["unjudged"], relevance, 10) == 0.0
    assert ndcg_at_k(["d1"], {}, 10) == 0.0


def test_ndcg_scorer_reads_qrel_pair_list():
    inst = _instance(qrel=[["d1", "3"], ["d2", "0"], ["d3", "2"]])
    out = LMOutput(text="")
    out.extracted_answer = ["d1", "d3", "d2"]
    assert NDCGScorer().score(inst, out) == 1.0
    out.extracted_answer = "not a list"
    assert NDCGScorer().score(inst, out) == 0.0


# ---------------------------------------------------------------------------
# substring / exact match


def test_substring_exact_match_is_max_over_aliases_not_fraction():
    golds = ["Barack Obama", "Obama", "President Obama"]
    inst = _instance(gold=golds, all_gold_answers=golds)
    out = _output("Answer: Barack Obama")
    # one alias found = fully correct; the recall scorer would give 1/3
    assert SubstringExactMatchScorer().score(inst, out) == 1.0
    assert SubstringRecallScorer().score(inst, out) == pytest.approx(2 / 3)


def test_substring_exact_match_uses_helmet_normalization():
    inst = _instance(gold=["United States"], all_gold_answers=["United States"])
    assert (
        SubstringExactMatchScorer().score(inst, _output("the answer is THE united states")) == 1.0
    )
    assert SubstringExactMatchScorer().score(inst, _output("I do not know")) == 0.0


def test_helmet_exact_match_normalizes_punctuation_and_articles():
    # HELMET's drqa-normalized EM scores these 1; the generic scorer scored 0
    inst = _instance(gold="42")
    assert HelmetExactMatchScorer().score(inst, _output("42.")) == 1.0
    assert HelmetExactMatchScorer().score(inst, _output('"42"')) == 1.0
    assert HelmetExactMatchScorer().score(inst, _output("42")) == 1.0
    assert HelmetExactMatchScorer().score(inst, _output("label 7")) == 0.0


# ---------------------------------------------------------------------------
# InfiniteBench multiple choice


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("B", 1.0),
        ("b", 1.0),
        ("B.", 1.0),
        ("B. Paris", 1.0),
        ("Answer: B", 1.0),
        ("The answer is B. Paris", 1.0),  # embedded long form counts
        ("B. Paris is correct", 1.0),
        ("C", 0.0),
        ("I think the answer is B", 0.0),  # bare letter in prose does NOT count
        ("", 0.0),
        ("Answer: C", 0.0),
    ],
)
def test_infbench_choice_scorer_matches_helmet(prediction, expected):
    golds = ["B", "B. Paris"]
    inst = _instance(gold=golds, all_gold_answers=golds)
    assert InfbenchChoiceScorer().score(inst, _output(prediction)) == expected


# ---------------------------------------------------------------------------
# ALCE


def test_alce_str_em_partial_credit_and_citation_stripping():
    qa_pairs = [
        {"short_answers": ["Daei", "Ali Daei"]},
        {"short_answers": ["Bican"]},
        {"short_answers": ["Pele"]},
    ]
    inst = _instance(qa_pairs=qa_pairs)
    scorer = AlceStrEmScorer()
    # citations must be stripped before matching; 2 of 3 sub-answers present
    assert scorer.score(
        inst, _output("Ali Daei scored most [1]. Bican also [2].")
    ) == pytest.approx(2 / 3)
    assert scorer.score(inst, _output("Nobody knows.")) == 0.0
    assert scorer.score(inst, _output("Daei, Bican, and Pele [1][2][3].")) == 1.0


def test_alce_qampari_rec_top5_caps_at_five():
    answers = [
        ["Heat"],
        ["Mai, the Psychic Girl"],
        ["Akira"],
        ["Nausicaa"],
        ["Ghost"],
        ["Lone Wolf"],
    ]
    inst = _instance(qampari_answers=answers)
    scorer = AlceQampariRecTop5Scorer()
    # 5 of 6 found, denominator capped at 5 -> full credit
    assert scorer.score(inst, _output("Heat, Akira, Ghost, Nausicaa, Lone Wolf, Extra")) == 1.0
    assert scorer.score(inst, _output("Heat, Akira")) == pytest.approx(2 / 5)
    assert scorer.score(inst, _output("nothing relevant")) == 0.0


# ---------------------------------------------------------------------------
# LLM-judge parsing and combination


def test_longqa_verdict_parsing_matches_helmet():
    assert parse_longqa_verdict('Reasoning. {"fluency": 1, "correctness": 3}') == (1.0, 3.0)
    # the LAST json object is the verdict
    assert parse_longqa_verdict('{"a": 1} then {"fluency": 0, "correctness": 2}') == (0.0, 2.0)
    # a reply without both scores gives no verdict, so the judge is asked again
    assert parse_longqa_verdict("no json at all") is None
    assert parse_longqa_verdict('{"fluency": 1}') is None
    assert parse_judge_json("{fluency: 1}") is None


def test_summ_parsing_falls_back_to_a_fenced_json_block():
    # the non-greedy brace match stops inside a nested object and fails to load;
    # eval_gpt4_summ.py then tries the last ```json block, eval_gpt4_longqa.py doesn't
    reply = 'Counting.\n```json\n{"precision": 2, "detail": {"x": 1}}\n```'
    assert parse_judge_json(reply) is None
    assert parse_judge_json(reply, fenced_fallback=True) == {"precision": 2, "detail": {"x": 1}}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Answer: the North Pole", "the North Pole"),
        # newlines become spaces, so everything after the prefix is kept
        ("Answer: the North\nPole", "the North Pole"),
        ("They head north.\nAnswer: Answer: the Pole", "the Pole"),
        ("no prefix here\nsecond line", "no prefix here second line"),
    ],
)
def test_longqa_judged_answer_matches_helmets_parse_output(raw, expected):
    assert parse_longqa_answer(raw) == expected


def test_summ_f1_matches_helmet_formula():
    rec, prec = 4 / 7, 3 / 4
    scores = summ_f1(fluency=1, found=4, supported=3, sentence_count=4, num_key_points=7)
    assert scores["gpt4_f1"] == pytest.approx(2 * rec * prec / (rec + prec))
    assert (scores["gpt4_recall"], scores["gpt4_precision"]) == (pytest.approx(rec), prec)
    # disfluency gates everything; zero denominators give zero, as upstream
    assert summ_f1(0, 7, 4, 4, 7)["gpt4_f1"] == 0.0
    assert summ_f1(1, 0, 3, 5, 10)["gpt4_f1"] == 0.0
    assert summ_f1(1, 1, 1, 0, 5)["gpt4_precision"] == 0.0


# ---------------------------------------------------------------------------
# ROUGE takes the best of raw and extracted, per HELMET's default_post_process


def test_rouge_scores_max_of_raw_and_extracted():
    inst = _instance(gold=["the cat sat on the mat"], all_gold_answers=["the cat sat on the mat"])
    out = LMOutput(text="the cat sat on the mat")
    out.extracted_answer = "unrelated words entirely"  # a bad parse must not zero the score
    assert RougeLF1Scorer().score(inst, out) == 1.0


# ---------------------------------------------------------------------------
# ICL demo balancing (pure function)


def test_balance_labels_is_deterministic_and_covers_labels():
    records = [{"label": i % 5, "text": f"t{i}"} for i in range(50)]
    first = _balance_labels(records, "label", 10, seed=7)
    second = _balance_labels(records, "label", 10, seed=7)
    assert first == second
    assert len(first) == 10
    # 10 shots over 5 labels -> every label appears exactly twice
    counts = {label: 0 for label in range(5)}
    for record in first:
        counts[record["label"]] += 1
    assert all(count == 2 for count in counts.values())


# ---------------------------------------------------------------------------
# KILT streaming sampler (fixture file; no network)


def test_sampled_rows_identical_to_full_load(tmp_path):
    rows = []
    for question in range(20):
        for depth in (0.0, 0.5, 0.95):  # dep-style repeats per question
            rows.append({"question": f"q{question}", "depth": depth, "s_pop": 10 ** (question % 6)})
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    import math
    import random

    def old_path(max_samples, keep=None):
        loaded = _load_jsonl(str(path))
        if keep is not None:
            loaded = [r for r in loaded if keep(r)]
        if max_samples is not None:
            keys = sorted({r["question"] for r in loaded})
            kept = set(random.Random(42).sample(keys, min(max_samples, len(keys))))
            loaded = [r for r in loaded if r["question"] in kept]
        return loaded

    def popularity(r):
        return math.log10(r["s_pop"]) < 3.0

    for cap in (3, 10, None):
        assert sample_jsonl_by_key(str(path), cap, 42, key=_question_key) == old_path(cap)
        assert sample_jsonl_by_key(
            str(path), cap, 42, key=_question_key, keep=popularity
        ) == old_path(cap, keep=popularity)
    # sampling caps questions, not rows: each question carries its 3 depths
    sampled = sample_jsonl_by_key(str(path), 3, 42, key=_question_key)
    assert len(sampled) == 9
    assert len({r["question"] for r in sampled}) == 3


def test_kilt_runner_keeps_every_depth_of_each_question(tmp_path, monkeypatch):
    """The runner's instance cap must not break up a question's depth sweep.

    Runs the real `prepare_task_items` against a stub 300-question x 6-depth
    file. The loader caps questions; if the task also set an instance
    `limit`, the runner would re-sample rows and queue a question at only
    some of its depths.
    """
    import olmo_eval.data.helmet_kilt_loader as kilt
    from olmo_eval.runners.asynq.preparation import prepare_task_items

    depths = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
    rows = [
        {
            "id": f"q{question}",
            "question": f"question {question}?",
            "answers": [f"answer {question}"],
            "depth": depth,
            "ctxs": [{"title": f"t{question}", "text": f"depth {depth}"}],
        }
        for question in range(300)
        for depth in depths
    ]
    test_file = tmp_path / "test.jsonl"
    test_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    demo_file = tmp_path / "demo.jsonl"
    demo_file.write_text("\n".join(json.dumps(r) for r in rows[:12]) + "\n")

    manifest = {"kilt_nq": {"4k": {"test_file": str(test_file), "demo_file": str(demo_file)}}}
    monkeypatch.setattr(kilt, "load_kilt_manifest", lambda: manifest)
    monkeypatch.setattr(kilt, "download_helmet_plus_file", lambda path: path)
    monkeypatch.setattr(kilt, "_load_reference_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(kilt, "_truncate_prompt_context", lambda row, *args: row)

    task, items = prepare_task_items("helmet_kilt_nq__4096", "stub-model", None)

    max_questions = HELMET_TASKS["kilt_nq__4096"]["max_questions"]
    assert task.config.limit is None
    assert len(items) == max_questions * len(depths)
    depths_by_question: dict[str, set[str]] = {}
    for item in items:
        question = item.instance.metadata["judge_question"]
        depths_by_question.setdefault(question, set()).add(item.instance.question)
    assert len(depths_by_question) == max_questions
    assert all(len(seen) == len(depths) for seen in depths_by_question.values())


# ---------------------------------------------------------------------------
# task registry and suite structure


def test_helmet_task_inventory():
    by_tag: dict[str, int] = {}
    for config in HELMET_TASKS.values():
        by_tag[config["tag"]] = by_tag.get(config["tag"], 0) + 1
    assert by_tag == {
        "recall": 24,
        "rag": 24,
        "rerank": 6,
        "longqa": 18,
        "summ": 12,
        "icl": 30,
        "cite": 24,
    }
    assert len(HELMET_TASKS) == 138


def test_helmet_context_budgets_match_helmet():
    # HELMET truncates to `size - reserve - generation`; these are its numbers
    assert HELMET_TASKS["narrativeqa__4096"]["max_context_tokens"] == 4096 - 200 - 100
    assert HELMET_TASKS["infbench_qa_eng__4096"]["max_context_tokens"] == 4096 - 200 - 10
    assert HELMET_TASKS["infbench_sum_eng__4096"]["max_context_tokens"] == 4096 - 200 - 1200
    # multi_lexsum reserves 300 where the others reserve 200
    assert HELMET_TASKS["multi_lexsum__4096"]["max_context_tokens"] == 4096 - 300 - 400


def test_helmet_task_config_details():
    # PopQA is restricted to long-tail entities (kilt_popqa_3 upstream)
    assert HELMET_TASKS["kilt_popqa__4096"]["popularity_threshold"] == 3.0
    assert "popularity_threshold" not in HELMET_TASKS["kilt_nq__4096"]
    # nocite ALCE variants run zero-shot with a larger budget, per HELMET
    assert HELMET_TASKS["alce_asqa_nocite__4096"]["shots"] == 0
    assert HELMET_TASKS["alce_asqa_nocite__4096"]["max_gen_toks"] == 600
    assert HELMET_TASKS["alce_asqa__4096"]["shots"] == 2
    # ICL shot counts are HELMET's own (comparability over recalibration)
    assert [HELMET_TASKS[f"icl_banking77__{size}"]["shots"] for size in STANDARD_CONTEXT_SIZES] == [
        180,
        360,
        720,
        1450,
        2900,
        5900,
    ]
    # judged tasks are flagged, so the nojudge suites can exclude them
    judged = {name for name, config in HELMET_TASKS.items() if config.get("judged")}
    assert judged == {
        f"{task}__{size}"
        for task in ("narrativeqa", "infbench_sum_eng", "multi_lexsum")
        for size in STANDARD_CONTEXT_SIZES
    }


def test_helmet_suite_structure():
    import olmo_eval.evals.suites.helmet  # noqa: F401 - triggers registration
    from olmo_eval.evals.suites.registry import get_suite

    assert len(get_suite("helmet_all__4096").expanded_tasks) == 21
    # HELMET's Recall average: json_kv plus three RULER NIAH tasks
    assert set(get_suite("helmet_recall__4096").expanded_tasks) == {
        "helmet_json_kv__4096",
        "helmet_ruler_niah_mk_2__4096",
        "helmet_ruler_niah_mk_3__4096",
        "helmet_ruler_niah_mv__4096",
    }
    # HELMET's Cite average leaves out the nocite ablations, which stay runnable
    assert set(get_suite("helmet_cite__4096").expanded_tasks) == {
        "helmet_alce_asqa__4096",
        "helmet_alce_qampari__4096",
    }
    assert HELMET_TASKS["alce_asqa_nocite__4096"]["ablation"] is True
    # nojudge excludes exactly the three judged tasks
    all_tasks = set(get_suite("helmet_all__4096").expanded_tasks)
    nojudge = set(get_suite("helmet_nojudge__4096").expanded_tasks)
    assert all_tasks - nojudge == {
        "helmet_narrativeqa__4096",
        "helmet_infbench_sum_eng__4096",
        "helmet_multi_lexsum__4096",
    }


# ---------------------------------------------------------------------------
# config serialization (regression: judge_fn closure crashed results
# aggregation after all instances were already scored)


def test_every_helmet_task_config_is_json_serializable():
    """`compute_task_hash` json.dumps's each task's config at aggregation time.

    The judged tasks' scorers carry a `judge_fn` closure, which leaked into
    the serialized config via dataclasses.asdict and crashed a full demo run
    at 100% scored. Instantiating a task touches no data, so this sweeps all
    registered helmet tasks cheaply.
    """
    from olmo_eval.evals.tasks.common.registry import get_task

    for task_name in HELMET_TASKS:
        task = get_task(f"helmet_{task_name}", {"limit": 1, "seed": 42})
        config = task.config.to_dict()
        assert json.dumps(config, sort_keys=True)


# ---------------------------------------------------------------------------
# judged tasks: graded through JudgedTask, with failures left unscored

LONGQA_SPEC = "helmet_narrativeqa__8192"
LONGQA_VERDICT = 'Step by step, it matches.\n{"fluency": 1, "correctness": 2}'


def _judged_response(spec: str, text: str, **metadata) -> tuple[JudgedTask, Response]:
    task = get_task(spec)
    assert isinstance(task, JudgedTask)
    instance = Instance(question="BOOK\n\nQuestion: Where?", metadata={"id": 7, **metadata})
    request = task.format_request(instance)
    return task, Response(instance=instance, request=request, outputs=[LMOutput(text=text)])


def _longqa_response(text: str = "Answer: the North Pole"):
    return _judged_response(
        LONGQA_SPEC, text, judge_question="Where?", all_gold_answers=["the North Pole"]
    )


@pytest.mark.usefixtures("judge_test_env")
class TestLongQAJudge:
    @pytest.mark.anyio
    async def test_scores_fluency_times_correctness_over_three(self, monkeypatch):
        task, response = _longqa_response()
        judge, prompts = scripted_judge(LONGQA_VERDICT)
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert response.scores["gpt4_score"] == pytest.approx(2 / 3)
        # the judge sees the bare question, the gold list and the parsed answer
        assert prompts[0].endswith(
            "Question: Where?\nCorrect answers: ['the North Pole']\nAnswer: the North Pole\n"
        )
        result = response.outputs[0].metadata["judge_result"]
        assert result["raw_judge_response"] == LONGQA_VERDICT
        assert (result["fluency"], result["correctness"], result["parse_error"]) == (1, 2, False)

    @pytest.mark.anyio
    async def test_an_unparseable_reply_is_retried_not_scored(self, monkeypatch):
        task, response = _longqa_response()
        judge, prompts = scripted_judge("I cannot decide.", LONGQA_VERDICT)
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert len(prompts) == 2
        assert response.scores["gpt4_score"] == pytest.approx(2 / 3)

    @pytest.mark.anyio
    async def test_a_reply_that_never_parses_leaves_the_output_unscored(self, monkeypatch):
        task, response = _longqa_response()
        judge, prompts = scripted_judge(*["no verdict"] * JUDGE_ATTEMPTS)
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert len(prompts) == JUDGE_ATTEMPTS
        assert "gpt4_score" not in response.scores
        output = response.outputs[0]
        assert "gpt4_score" in output.metadata["scoring_errors"]
        assert output.metadata["judge_result"]["parse_error"] is True
        (prediction,) = build_predictions([response])
        judge_result = prediction["model_output"][0]["judge_result"]
        assert judge_result["raw_judge_response"] == "no verdict"

    @pytest.mark.anyio
    async def test_judge_overrides_change_the_judge_that_runs(self, monkeypatch):
        built = {}
        judge, _ = scripted_judge(LONGQA_VERDICT)
        monkeypatch.setattr(
            "olmo_eval.common.scorers.llm_judge.build_openai_judge_fn",
            lambda **kwargs: built.update(kwargs) or judge,
        )
        task = get_task(LONGQA_SPEC, {"judge_model": "gpt-4.1-2025-04-14"})
        _, response = _longqa_response()

        await task.score_responses([response])

        assert built["model"] == "gpt-4.1-2025-04-14"
        assert built["max_tokens"] == 2048
        assert built["temperature"] == 0.1
        assert task.config.to_dict()["judge_model"] == "gpt-4.1-2025-04-14"


SUMM_REPLIES = {
    "fluency": '{"fluency": 1}',
    "recall": 'Two are covered. {"recall": 2}',
    "precision": '{"precision": 3, "sentence_count": 4}',
}


def _rubric_judge(replies: dict[str, list[str]]):
    """A judge that answers each rubric from its own queue, whatever the call order."""
    prompts: dict[str, list[str]] = {rubric: [] for rubric in replies}

    async def judge(prompt: str, *, system_prompt: str | None = None) -> str:
        if "evaluate the fluency" in prompt:
            rubric = "fluency"
        elif "EXPERT SUMMARY TEXT" in prompt:
            rubric = "precision"
        else:
            rubric = "recall"
        prompts[rubric].append(prompt)
        return replies[rubric].pop(0)

    return judge, prompts


def _summ_response(spec: str, **metadata):
    fields = {"keypoints": ["k1", "k2", "k3"], "expert_summary": "EXPERT SUMMARY TEXT"}
    return _judged_response(spec, "  A short summary.  ", **{**fields, **metadata})


@pytest.mark.usefixtures("judge_test_env")
class TestSummJudge:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("spec", "domain", "other"),
        [
            ("helmet_multi_lexsum__8192", "lawsuit", "novel"),
            ("helmet_infbench_sum_eng__8192", "novel", "lawsuit"),
        ],
    )
    async def test_three_rubrics_combine_into_helmets_f1(self, spec, domain, other, monkeypatch):
        task, response = _summ_response(spec)
        judge, prompts = _rubric_judge({k: [v] for k, v in SUMM_REPLIES.items()})
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        rec, prec = 2 / 3, 3 / 4
        assert response.scores["gpt4_f1"] == pytest.approx(2 * rec * prec / (rec + prec))
        assert response.scores["gpt4_recall"] == pytest.approx(rec)
        assert response.scores["gpt4_precision"] == prec
        assert response.scores["gpt4_fluency"] == 1.0
        recall_prompt = prompts["recall"][0]
        assert "1. k1\n2. k2\n3. k3" in recall_prompt
        assert domain in recall_prompt.lower() and other not in recall_prompt.lower()
        # the summary is judged stripped, as upstream's d["output"].strip()
        assert "A short summary." in prompts["fluency"][0]
        assert "  A short summary.  " not in prompts["fluency"][0]

    @pytest.mark.anyio
    async def test_one_rubric_that_never_parses_leaves_the_output_unscored(self, monkeypatch):
        task, response = _summ_response("helmet_multi_lexsum__8192")
        replies = {k: [v] for k, v in SUMM_REPLIES.items()}
        replies["precision"] = ["no count"] * JUDGE_ATTEMPTS
        judge, _ = _rubric_judge(replies)
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert "gpt4_f1" not in response.scores
        result = response.outputs[0].metadata["judge_result"]
        assert result["parse_error"] is True
        assert result["precision_judge_response"] == "no count"

    @pytest.mark.anyio
    async def test_an_instance_without_key_points_is_an_error_not_a_zero(self, monkeypatch):
        task, response = _summ_response("helmet_multi_lexsum__8192", keypoints=[])
        judge, prompts = _rubric_judge({k: [] for k in SUMM_REPLIES})
        use_judge(monkeypatch, judge)

        await task.score_responses([response])

        assert not any(prompts.values())
        assert "gpt4_f1" not in response.scores
        assert "gpt4_f1" in response.outputs[0].metadata["scoring_errors"]


# ---------------------------------------------------------------------------
# HELMET's own settings and source revisions reach the task hash


def _settings_hash(spec: str, **overrides) -> str:
    from olmo_eval.common.types.base import compute_task_hash

    return compute_task_hash(get_task(spec, overrides or None).config.to_dict())


def test_task_settings_record_what_the_run_measures():
    from olmo_eval.evals.tasks import helmet

    settings = get_task("helmet_icl_trec_coarse__4096").config.to_dict()["task_settings"]
    assert settings["upstream"] == "princeton-nlp/HELMET@aeadacc6"
    assert settings["helmet"] == {
        "length_name": "4k",
        "shots": 200,
        "icl_dataset": "trec_coarse",
    }
    assert settings["sources"]["icl"]["revision"] == "65752bf53af25bc935a0dce92fb5b6c930728450"
    assert settings["constants"]["system_template"] == "label:"
    assert settings["constants"]["answer_prefix_separator"] == "\n"

    lexsum = helmet._task_settings(HELMET_TASKS["multi_lexsum__8192"])
    assert lexsum["sources"]["multi_lexsum"].endswith(helmet.MULTI_LEXSUM_REVISION)
    assert lexsum["constants"]["min_source_tokens"] == 65536
    assert lexsum["helmet"]["max_context_tokens"] == 8192 - 300 - 400


def test_suite_flags_and_config_fields_stay_out_of_the_hash():
    from olmo_eval.evals.tasks import helmet

    for name in ("alce_asqa_nocite__4096", "narrativeqa__8192", "kilt_popqa__8192"):
        cfg = HELMET_TASKS[name]
        flagged = {**cfg, "tag": "other", "ablation": not cfg.get("ablation"), "judged": True}
        flagged.update(limit=5, max_gen_toks=1, stop_new_line=True, metrics_key="x")
        assert helmet._task_settings(flagged) == helmet._task_settings(cfg), name
    # `-o limit=5` is the limit field's alone, not repeated in the settings
    limited = get_task("helmet_alce_asqa_nocite__4096", {"limit": 5}).config.to_dict()
    assert "limit" not in limited["task_settings"]["helmet"]


@pytest.mark.parametrize(
    ("loader", "spec"),
    [
        ("helmet_infbench_loader", "infbench_qa_eng__8192"),
        ("helmet_narrativeqa_loader", "narrativeqa__8192"),
        ("helmet_multilexsum_loader", "multi_lexsum__8192"),
    ],
)
def test_prompt_constants_move_the_hash(loader, spec, monkeypatch):
    import importlib

    from olmo_eval.data.helmet_infbench_loader import TRUNCATION_POSTFIX
    from olmo_eval.evals.tasks import helmet

    module = importlib.import_module(f"olmo_eval.data.{loader}")
    # the listed constant is the one truncation appends
    assert module.MEASURED_CONSTANTS["truncation_postfix"] is TRUNCATION_POSTFIX
    before = helmet._task_settings(HELMET_TASKS[spec])
    monkeypatch.setitem(module.MEASURED_CONSTANTS, "truncation_postfix", " [cut]")
    assert helmet._task_settings(HELMET_TASKS[spec]) != before


def test_the_run_reads_its_settings_from_the_config(monkeypatch):
    from olmo_eval.evals.tasks import helmet

    spec = "helmet_kilt_nq__8192"
    settings = get_task(spec).config.task_settings
    changed = {**settings, "helmet": {**settings["helmet"], "shots": 5, "max_questions": 7}}
    seen = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        return {"data": [], "user_template": "", "system_template": ""}

    monkeypatch.setattr(helmet, "load_kilt_dataset", fake_load)
    task = get_task(spec, {"task_settings": changed})
    task._load_data()

    assert (seen["shots"], seen["max_samples"]) == (5, 7)
    assert _settings_hash(spec, task_settings=changed) != _settings_hash(spec)


@pytest.mark.parametrize(
    "override",
    [
        lambda s: {**s, "constants": {**s["constants"], "system_template": "A:"}},
        lambda s: {**s, "upstream": "princeton-nlp/HELMET@main"},
        lambda s: {**s, "helmet": {**s["helmet"], "unknown": 1}},
        lambda s: {**s, "helmet": {k: v for k, v in s["helmet"].items() if k != "shots"}},
    ],
)
def test_overrides_the_run_cannot_honor_are_refused(override):
    settings = get_task("helmet_kilt_nq__8192").config.task_settings
    with pytest.raises(ValueError, match="task_settings"):
        get_task("helmet_kilt_nq__8192", {"task_settings": override(settings)})


@pytest.mark.parametrize(
    "override", [{"num_fewshot": 3}, {"fewshot_seed": 1}, {"split": "validation"}]
)
def test_fewshot_and_split_overrides_are_refused(override):
    with pytest.raises(ValueError, match="cannot be overridden"):
        get_task("helmet_icl_banking77__4096", override)


def test_settings_that_arrive_through_json_are_accepted():
    settings = get_task("helmet_icl_banking77__4096").config.task_settings
    round_tripped = json.loads(json.dumps(settings))
    task = get_task("helmet_icl_banking77__4096", {"task_settings": round_tripped})
    assert task.settings == settings["helmet"]


def test_task_settings_are_omitted_when_unset():
    from olmo_eval.evals.tasks.common.registry import get_task

    assert "task_settings" not in get_task("hellaswag").config.to_dict()


def test_nlu_csv_is_parsed_like_its_original_loading_script(tmp_path, monkeypatch):
    from olmo_eval.data import helmet_icl_loader

    header = (
        "userid;answerid;scenario;intent;status;answer_annotation;notes;"
        "suggested_entities;answer_normalised;answer;question"
    )
    lines = [
        header,
        '1;1;alarm;set;;"wake me up at [time : five am] this week";;;x;x;q',
        '1;2;alarm;set;;"null";;;x;x;q',
        '1;3;weather;query;;"is it [weather_descriptor : raining]";;;x;x;q',
        '1;4;alarm;query;;"what alarms do i have";;;x;x;q',
    ]
    path = tmp_path / "nlu.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(helmet_icl_loader.DownloadManager, "download", lambda self, url: str(path))

    train = helmet_icl_loader._load_nlu_csv("unused")["train"]
    # the null row is dropped, entity markup stripped, and file order kept
    assert train["text"] == [
        "wake me up at five am this week",
        "is it raining",
        "what alarms do i have",
    ]
    # labels are `{scenario}_{intent}`, indexed in sorted order
    assert train.features["label"].names == ["alarm_query", "alarm_set", "weather_query"]
    assert train["label"] == [1, 2, 0]


def test_helmet_ruler_tasks_render_helmets_prompt(tmp_path, monkeypatch):
    """HELMET renders RULER from its templates with the answer prefix on its own
    line, not from the files' prebuilt `input`, and keeps HELMET's budgets."""
    from olmo_eval.data.ruler_tasks import RULER_TASKS
    from olmo_eval.evals.tasks import helmet
    from olmo_eval.evals.tasks.common.registry import get_task

    row = {
        "index": 0,
        "input": "prebuilt prompt that HELMET does not use",
        "outputs": ["111", "222"],
        "answer": ["111", "222"],
        "context": "HAYSTACK",
        "query": "blue-tick",
        "type_needle_v": "numbers",
        "length": 10,
    }
    path = tmp_path / RULER_TASKS["niah_mv__4096"]["data"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(helmet, "download_ruler_data", lambda: str(tmp_path))

    task = get_task("helmet_ruler_niah_mv__4096")
    (instance,) = list(task.instances)
    prompt = task.format_request(instance).prompt
    assert prompt.endswith(
        "What are all the special magic numbers for blue-tick mentioned in the provided text?"
        "\nThe special magic numbers for blue-tick mentioned in the provided text are"
    )
    assert "HAYSTACK" in prompt and "prebuilt" not in prompt
    assert instance.gold_answer == ["111", "222"]
    # ruler_recall is the fraction of gold values found
    assert task.config.get_primary_metric().scorer().score(instance, _output("111 and 333")) == 0.5

    budgets = {
        name: HELMET_TASKS[f"ruler_{name}__4096"]["max_gen_toks"]
        for name in ("niah_mk_2", "niah_mk_3", "niah_mv")
    }
    assert budgets == {"niah_mk_2": 50, "niah_mk_3": 100, "niah_mv": 50}
    sources = task.config.to_dict()["task_settings"]["sources"]
    assert sources["ruler_data"].startswith("allenai/ruler_data@")


@pytest.mark.parametrize(
    ("task_name", "chat"),
    [
        ("helmet_infbench_qa_eng__8192", True),
        ("helmet_narrativeqa__8192", True),
        ("helmet_alce_asqa__8192", True),
        ("helmet_multi_lexsum__8192", True),
        ("helmet_kilt_nq__8192", False),
        ("helmet_json_kv__8192", False),
    ],
)
def test_chat_template_tasks_send_a_single_user_turn(task_name, chat):
    """HELMET's chat-format categories go out as one user message for the server
    to wrap in the chat template, with no answer prefix; the rest stay completions
    with the prefix on its own line."""
    from olmo_eval.common.types import RequestType
    from olmo_eval.evals.tasks.common.registry import get_task

    task = get_task(task_name)
    prepend_text = "" if chat else "Answer:"
    instance = Instance(question="CONTEXT\nQuestion: q", metadata={"prepend_text": prepend_text})
    request = task.format_request(instance)

    if chat:
        assert request.request_type == RequestType.CHAT
        assert request.messages == ({"role": "user", "content": "CONTEXT\nQuestion: q"},)
        assert request.prompt == ""
    else:
        assert request.request_type == RequestType.COMPLETION
        assert request.messages == ()
        assert request.prompt == "CONTEXT\nQuestion: q\nAnswer:"


def test_judged_tasks_are_judged_tasks_and_require_the_openai_key():
    """Exactly the tasks flagged as judged grade through JudgedTask and need the key."""
    for name, config in HELMET_TASKS.items():
        task = get_task(f"helmet_{name}")
        judged = bool(config.get("judged"))
        assert isinstance(task, JudgedTask) == judged, name
        assert task.config.required_secrets == (("OPENAI_API_KEY",) if judged else ()), name


# ---------------------------------------------------------------------------
# Multi-LexSum selection follows upstream's load_multi_lexsum


class _WordTokenizer:
    """One token per whitespace-separated word, with character offsets."""

    def __call__(self, text: str, return_offsets_mapping: bool = False) -> dict:
        import re

        spans = [m.span() for m in re.finditer(r"\S+", text)]
        encoded: dict = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            encoded["offset_mapping"] = spans
        return encoded


def _multi_lexsum_fixture(tmp_path, monkeypatch, *, min_tokens: int = 6):
    """Hosted validation file, release files and tokenizer, with a small length floor.

    Validation cases with 4-12 words of sources; the floor keeps those with at least
    `min_tokens` words. The hosted file lists cases in reverse release order.
    """
    from olmo_eval.data import helmet_multilexsum_loader as loader

    dev = [
        {"case_id": f"V{i}", "summary/short": None if i == 3 else f"short {i}"} for i in range(10)
    ]
    hosted = [
        {
            "id": f"V{i}",
            "sources": ["word " * (4 + i), "tail"],
            "summary/short": f"short {i}",
            "summary/long": f"long {i}",
            "summary/short_keypoints": [f"point {i}"],
        }
        for i in reversed(range(10))
    ]
    train = [
        {"case_id": f"T{i}", "summary/short": None if i % 4 == 0 else f"train summary {i}"}
        for i in range(12)
    ]
    path = tmp_path / "multi_lexsum_val.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in hosted), encoding="utf-8")
    monkeypatch.setattr(loader, "download_helmet_plus_file", lambda name: str(path))
    monkeypatch.setattr(
        loader, "_release_cases", lambda name: {"dev.json": dev, "train.json": train}[name]
    )
    monkeypatch.setattr(loader, "_load_reference_tokenizer", lambda name: _WordTokenizer())
    monkeypatch.setattr(loader, "MIN_SOURCE_TOKENS", min_tokens)
    return loader, hosted, train


def test_multi_lexsum_selects_long_cases_in_release_order(tmp_path, monkeypatch):
    from datasets import Dataset

    loader, _, _ = _multi_lexsum_fixture(tmp_path, monkeypatch)

    data = loader.load_multi_lexsum_dataset(max_context_tokens=100, shots=0)["data"]

    # V0's 5 words fall below the floor, V3 has no short summary, and the rest
    # follow dev.json's order, not the hosted file's
    eligible = [f"short {i}" for i in (1, 2, 4, 5, 6, 7, 8, 9)]
    assert [row["answer"] for row in data] == [[s] for s in eligible]
    assert data[0]["keypoints"] == ["point 1"]
    assert data[0]["expert_summary"] == "long 1"

    sampled = loader.load_multi_lexsum_dataset(max_context_tokens=100, shots=0, max_samples=3)
    expected = Dataset.from_list([{"s": s} for s in eligible]).shuffle(seed=42).select(range(3))
    assert [row["answer"][0] for row in sampled["data"]] == expected["s"]


def test_multi_lexsum_draws_train_demos_per_instance(tmp_path, monkeypatch):
    from datasets import Dataset

    loader, _, train = _multi_lexsum_fixture(tmp_path, monkeypatch)
    pool = Dataset.from_list([c for c in train if c["summary/short"] is not None])

    data = loader.load_multi_lexsum_dataset(max_context_tokens=100, shots=2, seed=7)["data"]

    for case_id, row in zip(("V1", "V2"), data, strict=False):
        picks = pool.shuffle(seed=loader.demo_seed(case_id, 7)).select(range(2))
        rendered = "\n\n".join(f"Summary: {s}" for s in picks["summary/short"])
        assert row["demo"] == (
            f"Example summaries:\n\n{rendered}\n\n"
            "Now, write a summary of the following legal documents.\n"
        )
    assert len({row["demo"] for row in data}) > 1


def test_multi_lexsum_truncates_the_documents(tmp_path, monkeypatch):
    from olmo_eval.data.helmet_infbench_loader import TRUNCATION_POSTFIX

    loader, _, _ = _multi_lexsum_fixture(tmp_path, monkeypatch)
    budget = 10
    separator = len(_WordTokenizer()(TRUNCATION_POSTFIX)["input_ids"])

    data = loader.load_multi_lexsum_dataset(max_context_tokens=budget, shots=0)["data"]

    # V1's 6 words fit; V9's 14 are cut after word budget - separator, plus the notice
    assert data[0]["context"] == "word " * 5 + "\n\ntail"
    kept = data[-1]["context"].removesuffix(TRUNCATION_POSTFIX)
    assert data[-1]["context"].endswith(TRUNCATION_POSTFIX)
    assert kept.split() == ["word"] * (budget - separator + 1)


def test_multi_lexsum_refuses_incomplete_data(tmp_path, monkeypatch):
    loader, hosted, _ = _multi_lexsum_fixture(tmp_path, monkeypatch)
    path = tmp_path / "multi_lexsum_val.jsonl"

    without_points = [{**row, "summary/short_keypoints": []} for row in hosted]
    path.write_text("".join(json.dumps(r) + "\n" for r in without_points), encoding="utf-8")
    with pytest.raises(ValueError, match="no key points"):
        loader.load_multi_lexsum_dataset(max_context_tokens=100, shots=0)

    path.write_text("".join(json.dumps(r) + "\n" for r in hosted[1:]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing from"):
        loader.load_multi_lexsum_dataset(max_context_tokens=100, shots=0)
