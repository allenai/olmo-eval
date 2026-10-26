"""Tests for the Lost in Conversation multi-turn reliability tasks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from olmo_eval.common.execution import ExecutionResult, ScoringContext
from olmo_eval.common.scorers import SandboxRequiredError
from olmo_eval.common.types import LMOutput, Response
from olmo_eval.evals.tasks import lost_in_conversation as lic
from olmo_eval.evals.tasks.common import get_task, task_exists
from olmo_eval.evals.tasks.common.judge import JudgeParseError
from olmo_eval.evals.tasks.common.judge_panel import DEFAULT_JUDGE_PANEL
from olmo_eval.evals.tasks.constants import lost_in_conversation as prompts
from olmo_eval.evals.tasks.lost_in_conversation import (
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    TASK_NAMES,
    ActionsJudge,
    CodeJudge,
    Data2TextJudge,
    LastReplyCodeScorer,
    MathJudge,
    SummaryJudge,
    bullets_json,
    citations,
    evaluate_actions,
    evaluate_data2text,
    evaluate_math,
    extract_code,
    parse_function_calls,
    prepare_solution,
    summary_bullets,
    summary_scores,
)

# Git blob SHA-1s of the prompt files at c865793fe34a929d316119b0451d01bd9183bcfd
PROMPT_BLOB_SHAS = {
    "TURN_CATEGORIZATION_PROMPT": "af7b95189ce1693172fc0ed1fcc3530c591189aa",
    "ANSWER_EXTRACTION_PROMPT": "89c6719df33590dff8edd0c4b0941f829d8bc012",
    "SUMMARY_COVERAGE_PROMPT": "f19380dd7a219662812c9fe4e1f964e667c4fe9b",
    "MATH_SYSTEM_PROMPT": "762608b89e68c5ccc0ad6f211c0ee1e737e96c1a",
    "MATH_FULL_PROMPT": "3d49aaa5da3dfd46fd763431f0c7e8cb24fb280f",
    "CODE_SYSTEM_PROMPT": "8e738fe17b58cd392f1c3aac1970e8ac5c4238da",
    "LCB_FULL_PROMPT": "faad6478ab098b1fcfd1fefa64b08c460ec4f4ff",
    "HUMANEVAL_FULL_PROMPT": "d30504b1b5a9111aee28f9906a71df7efcbc45e0",
    "ACTIONS_SYSTEM_PROMPT": "4be6796c4149915be118e77e05587c42266f069a",
    "ACTIONS_FULL_PROMPT": "3368901e0e3baf5d7e8257586f49e60676c185c7",
    "DATA2TEXT_SYSTEM_PROMPT": "3cab001bfd11f644551349dd9ef1328b01a18561",
    "DATA2TEXT_FULL_PROMPT": "4d1237f382fe68af0ba5188cf91f1e97bb0868e3",
    "SUMMARY_SYSTEM_PROMPT": "674bb78e755d13ae4ebe0b0507eecc3bfa50e1b8",
    "SUMMARY_FULL_PROMPT_CONV": "fda19f6f690b329a277da8dc8a3fe50ee0c57021",
    "SUMMARY_FULL_PROMPT_NEWS": "8e2c8e5642d9976960ff46fd21726a03aea59fdf",
}

# Samples shaped like the dataset's rows, trimmed
MATH = {
    "task_id": "sharded-GSM8K/1246",
    "task": "math",
    "question": "Ara plays 40 games a year for four years, scoring 21 points a game. Total?",
    "answer": "40*4 = <<40*4=160>>160 games.\n160*21 = <<21*160=3360>>3360 points\n#### 3360",
    "shards": [
        {"shard_id": 1, "shard": "what's Ara's total basketball score over four years?"},
        {"shard_id": 2, "shard": "Ara has been on the school basketball team for four years now"},
        {"shard_id": 3, "shard": "Ara plays 40 games per year on average"},
        {"shard_id": 4, "shard": "she scores 21 points in each game"},
    ],
}
HUMANEVAL_PROMPT = (
    "from typing import List\n\n\ndef remove_duplicates(numbers: List[int]) -> List[int]:\n"
    '    """ From a list of integers, remove all elements that occur more than once."""\n'
)
CODE = {
    "task_id": "sharded-HumanEval/26",
    "task": "code",
    "source": "humaneval",
    "prompt": HUMANEVAL_PROMPT,
    "test": "def check(candidate):\n    assert candidate([]) == []\n",
    "public_test_cases": json.dumps(
        [
            {"input": "[]", "output": "[]", "testtype": "functional"},
            {"input": "[1, 2, 3, 4]", "output": "[1, 2, 3, 4]", "testtype": "functional"},
            {"input": "[1, 2, 3, 2, 4, 3, 5]", "output": "[1, 4, 5]", "testtype": "functional"},
        ]
    ),
    "metadata": {"func_name": "remove_duplicates"},
    "shards": [
        {"shard_id": 1, "shard": "Filter out duplicates from a list"},
        {"shard_id": 2, "shard": "The input is a list of numbers"},
        {"shard_id": 3, "shard": "duplicates must entirely be removed"},
    ],
}
LCB_CODE = {
    "task_id": "sharded-lcb-abc",
    "task": "code",
    "source": "lcb_easy",
    "question_content": "Return the number doubled.",
    "starter_code": "class Solution:\n    def double(self, x: int) -> int:\n        ",
    "public_test_cases": json.dumps([{"input": "2", "output": "4", "testtype": "functional"}]),
    "private_test_cases": "",
    "metadata": {"func_name": "double"},
    "shards": [
        {"shard_id": 1, "shard": "double a number"},
        {"shard_id": 2, "shard": "it's an int"},
    ],
}
FUNCTIONS = [
    {
        "name": "create_histogram",
        "description": "Create a histogram based on provided data.",
        "parameters": {
            "type": "dict",
            "properties": {
                "data": {"type": "array", "items": {"type": "integer"}, "description": "Data."},
                "bins": {"type": "integer", "description": "The number of bins."},
            },
            "required": ["data", "bins"],
        },
    }
]
ACTIONS = {
    "task_id": "sharded-BFCL/parallel_129",
    "task": "actions",
    "function": FUNCTIONS,
    "fully_specified_question": [
        [{"role": "user", "content": "Make histograms of [1, 2] and [3, 4] with 5 bins each."}]
    ],
    "language": "Python",
    "test_category": "parallel",
    "reference_answer": [
        {"create_histogram": {"data": [[1, 2]], "bins": [5]}},
        {"create_histogram": {"data": [[3, 4]], "bins": [5]}},
    ],
    "shards": [
        {"shard_id": 1, "shard": "Create two histograms"},
        {"shard_id": 2, "shard": "The first data set is [1, 2]"},
        {"shard_id": 3, "shard": "The second data set is [3, 4], 5 bins each"},
    ],
}
REFERENCE_SENTENCE = "AFC 3 was an event held on April 14, 2012, at Geelong Arena in Geelong."
DATA2TEXT = {
    "task_id": "sharded-totto_7401674541652223105",
    "task": "data2text",
    "table_highlighted_html": "<table><tr><td>AFC 3</td></tr></table>",
    "fewshot_descriptions": "One description.\nAnother description.",
    "metadata": {"table_page_title": "2012 in AFC", "table_section_title": "Events list"},
    "references": [REFERENCE_SENTENCE, REFERENCE_SENTENCE, REFERENCE_SENTENCE],
    "shards": [
        {"shard_id": 1, "shard": "Describe this table in one sentence.\n<table></table>"},
        {"shard_id": 2, "shard": "The table is from the page 2012 in AFC."},
    ],
}
SUMMARY = {
    "task_id": "sharded-summary_8",
    "task": "summary",
    "topic": "A meeting between a doctor and a patient.",
    "domain": "conv",
    "query": "Are home health monitoring tips discussed?",
    "documents": [
        {"document_id": f"d{i}", "document_index": i, "document_text": f"Text {i}."}
        for i in range(1, 5)
    ],
    "insights": [
        {"insight_id": "i1", "insight": "The doctor advises tracking blood pressure."},
        {"insight_id": "i2", "insight": "The patient is asked to log symptoms."},
    ],
    "insightid2ref_citations": {"i1": [1, 2], "i2": [3, 4]},
    "shards": [
        {"shard_id": 0, "shard": "", "doc_idxs": [2, 1]},
        {"shard_id": 1, "shard": "", "doc_idxs": [3, 4]},
    ],
}
SAMPLES = (MATH, CODE, LCB_CODE, ACTIONS, DATA2TEXT, SUMMARY)

CORRECT_DEDUPE = (
    "Here you go:\n```python\nfrom collections import Counter\n\n"
    "def dedupe(numbers):\n    counts = Counter(numbers)\n"
    "    return [n for n in numbers if counts[n] == 1]\n```"
)
WRONG_DEDUPE = "```python\ndef dedupe(numbers):\n    return sorted(set(numbers))\n```"


@pytest.fixture(autouse=True)
def samples(monkeypatch):
    """Serve the fixture samples instead of downloading the dataset."""
    monkeypatch.setattr(lic, "_load_dataset", lambda path, filename, revision: SAMPLES)


def _instance(spec: str, task_id: str):
    task = get_task(spec)
    return next(i for i in task.instances if i.metadata["id"] == task_id)


def _output(replies: list[str]) -> LMOutput:
    return LMOutput(text=replies[-1], metadata={"turns": [{"text": r} for r in replies]})


class ScriptedJudge:
    """A judge that answers each kind of prompt from its own queue, recording the prompts."""

    def __init__(self, categorizations=(), extractions=(), coverages=()) -> None:
        self.queues = {
            "categorization": [json.dumps({"response_type": r}) for r in categorizations],
            "extraction": [
                e if isinstance(e, str) and e.startswith("{") else json.dumps({"answer": e})
                for e in extractions
            ],
            "coverage": [json.dumps(c) for c in coverages],
        }
        self.calls: dict[str, list[str]] = {kind: [] for kind in self.queues}

    @staticmethod
    def kind(prompt: str) -> str:
        if "You must classify the response of the assistant" in prompt:
            return "categorization"
        if "Your goal is to extract verbatim what the answer is" in prompt:
            return "extraction"
        if "Reference Insight:" in prompt:
            return "coverage"
        raise AssertionError(f"unexpected judge prompt {prompt[:80]!r}")

    async def __call__(self, prompt: str, *, system_prompt: str | None = None) -> str:
        kind = self.kind(prompt)
        self.calls[kind].append(prompt)
        return self.queues[kind].pop(0)


class LocalGraderEnv:
    """A staging sandbox that runs the grader in a local temporary directory."""

    def __init__(self) -> None:
        self.staged: dict[str, str] = {}

    @property
    def is_running(self) -> bool:
        return True

    async def execute(self, command: str, timeout: float | None = None) -> str:
        return ""

    async def execute_command(self, command: str, timeout: float | None = None) -> ExecutionResult:
        return ExecutionResult(success=True)

    async def execute_code(
        self, code: str, language: str = "python", timeout: float | None = None
    ) -> ExecutionResult:
        return ExecutionResult(success=True)

    async def execute_with_files(
        self, command: str, files: Mapping[str, str], timeout: float | None = None
    ) -> ExecutionResult:
        self.staged.update(files)
        with tempfile.TemporaryDirectory() as work_dir:
            for path, content in files.items():
                with open(os.path.join(work_dir, os.path.basename(path)), "w") as handle:
                    handle.write(content)
            process = subprocess.run(
                [sys.executable, "grade.py"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
        return ExecutionResult(
            success=process.returncode == 0, output=process.stdout, exit_code=process.returncode
        )


class TestReferenceData:
    @pytest.mark.parametrize(("name", "sha"), PROMPT_BLOB_SHAS.items())
    def test_prompts_match_the_reference_commit(self, name, sha):
        data = getattr(prompts, name).encode("utf-8")

        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == sha

    def test_dataset_is_pinned(self):
        source = get_task("lost_in_conversation_math").config.get_data_source()

        assert source.path.removeprefix("hf://") == "microsoft/lost_in_conversation"
        assert source.revision == "b6b71f418423c61cef0f439c610a4a655cc0e2d5"


class TestRegistration:
    @pytest.mark.parametrize("name", TASK_NAMES)
    def test_specs_are_registered(self, name):
        for spec in (name, f"{name}:full", f"{name}:judge_panel"):
            assert task_exists(spec)

    @pytest.mark.parametrize("name", TASK_NAMES)
    def test_runs_under_the_multi_turn_scaffold_with_the_reference_sampling(self, name):
        task = get_task(name)
        params = task.config.sampling_params

        assert task.required_scaffold == "multi_turn_chat"
        assert (params.temperature, params.max_tokens) == (1.0, 1000)
        assert task.config.required_secrets == ("OPENAI_API_KEY",)

    def test_default_judge(self):
        config = get_task("lost_in_conversation_math").config.to_dict()
        judge_fn = config["metrics"][0]["scorer"]["judge_fn"]

        assert (judge_fn["model"], judge_fn["max_tokens"], judge_fn["temperature"]) == (
            JUDGE_MODEL,
            JUDGE_MAX_TOKENS,
            0.0,
        )

    def test_the_judge_model_can_be_overridden(self):
        task = get_task("lost_in_conversation_math", {"judge_model": "gpt-4.1-2025-04-14"})
        judge_fn = task.config.to_dict()["metrics"][0]["scorer"]["judge_fn"]

        assert judge_fn["model"] == "gpt-4.1-2025-04-14"

    def test_metrics(self):
        names = {name: [m.name for m in get_task(name).config.metrics] for name in TASK_NAMES}
        rates = ["any__any__premature_answer_rate", "any__any__no_answer_rate"]

        assert names["lost_in_conversation_math"] == ["any__any__score", *rates]
        assert names["lost_in_conversation_code"] == [
            "last_reply_pass_rate",
            "any__any__score",
            *rates,
            "source__humaneval__score",
            "source__lcb_easy__score",
            "source__lcb_medium__score",
        ]
        assert names["lost_in_conversation_summary"] == [
            "any__any__score",
            "domain__conv__score",
            "domain__news__score",
        ]
        for name in TASK_NAMES:
            assert get_task(name).config.get_primary_metric().name == "any__any__score"

    def test_full_variant_asks_one_turn_and_changes_the_task_hash(self):
        sharded, full = (
            get_task("lost_in_conversation_math"),
            get_task("lost_in_conversation_math:full"),
        )

        assert sharded.config.to_dict()["formatter"]["setting"] == "sharded"
        assert full.config.to_dict()["formatter"]["setting"] == "full"
        assert sharded.config.to_dict() != full.config.to_dict()

    def test_judge_panel_variant(self):
        task = get_task("lost_in_conversation_math:judge_panel")

        assert len(task._get_scorers()) == len(DEFAULT_JUDGE_PANEL)
        assert len(task.config.metrics) == 3 * len(DEFAULT_JUDGE_PANEL)

    def test_code_panel_keeps_one_judge_free_execution_metric(self):
        task = get_task("lost_in_conversation_code:judge_panel")

        assert len(task._get_scorers()) == len(DEFAULT_JUDGE_PANEL) + 1
        assert [m.name for m in task.config.metrics].count("last_reply_pass_rate") == 1


class TestInstances:
    def test_each_task_asks_its_own_samples(self):
        counts = {name: len(list(get_task(name).instances)) for name in TASK_NAMES}

        assert counts == {
            "lost_in_conversation_math": 1,
            "lost_in_conversation_code": 2,
            "lost_in_conversation_actions": 1,
            "lost_in_conversation_data2text": 1,
            "lost_in_conversation_summary": 1,
        }

    def test_sharded_request_sends_each_shard_after_the_system_prompt(self):
        task = get_task("lost_in_conversation_math")
        request = task.format_request(_instance("lost_in_conversation_math", MATH["task_id"]))
        shards = [shard["shard"] for shard in MATH["shards"]]

        assert request.messages == (
            {"role": "system", "content": prompts.MATH_SYSTEM_PROMPT},
            {"role": "user", "content": shards[0]},
        )
        assert request.system_prompt == prompts.MATH_SYSTEM_PROMPT
        assert [turn["content"] for turn in request.metadata["follow_up_turns"]] == shards[1:]

    def test_full_prompts(self):
        def first_turn(name: str, task_id: str) -> str:
            return _instance(f"{name}:full", task_id).metadata["turns"]

        assert first_turn("lost_in_conversation_math", MATH["task_id"]) == [
            f"Q: {MATH['question']}\nA:"
        ]
        [humaneval] = first_turn("lost_in_conversation_code", CODE["task_id"])
        assert humaneval.startswith(
            "You must generate a valid Python function that satisfies the following instruction"
        )
        assert (
            "Complete the following incomplete function signature:\n"
            f"```python\n{HUMANEVAL_PROMPT}\n```"
        ) in humaneval
        [lcb] = first_turn("lost_in_conversation_code", LCB_CODE["task_id"])
        assert lcb.startswith("### Question:\nReturn the number doubled.\n\n### Format: You will")
        assert f"```python\n{LCB_CODE['starter_code']}\n```" in lcb
        assert first_turn("lost_in_conversation_actions", ACTIONS["task_id"]) == [
            ACTIONS["fully_specified_question"][0][0]["content"]
        ]
        [table] = first_turn("lost_in_conversation_data2text", DATA2TEXT["task_id"])
        assert "table_page_title: 2012 in AFC\ntable_section_title: Events list\n" in table
        [summary] = first_turn("lost_in_conversation_summary", SUMMARY["task_id"])
        assert "You are given 4 conversations" in summary
        assert "Document 4:\nText 4.\n\n" in summary

    def test_actions_system_prompt_lists_the_functions_as_a_python_literal(self):
        instance = _instance("lost_in_conversation_actions", ACTIONS["task_id"])

        assert instance.metadata["system_prompt"].endswith(str(FUNCTIONS))

    def test_summary_turns_send_the_first_documents_twice(self):
        turns = _instance("lost_in_conversation_summary", SUMMARY["task_id"]).metadata["turns"]

        assert len(turns) == len(SUMMARY["shards"]) + 1
        assert "```\nDocument 2:\nText 2.\n\nDocument 1:\nText 1.\n\n\n```" in turns[0]
        assert turns[1] == prompts.SUMMARY_FOLLOW_UP_PROMPT + (
            "Document 2:\nText 2.\n\nDocument 1:\nText 1.\n\n"
        )
        assert turns[2] == prompts.SUMMARY_FOLLOW_UP_PROMPT + (
            "Document 3:\nText 3.\n\nDocument 4:\nText 4.\n\n"
        )

    def test_examples_are_rejected(self):
        task = get_task("lost_in_conversation_math", {"num_fewshot": 2})

        with pytest.raises(ValueError, match="without examples"):
            task.format_request(_instance("lost_in_conversation_math", MATH["task_id"]))


class TestEvaluators:
    @pytest.mark.parametrize(
        ("answer", "score"),
        [
            ("3360", 1.0),
            ("She scored 3,360 points.", 1.0),
            ("First 160 games, so 3360", 1.0),
            ("3360.", 1.0),
            ("3361", 0.0),
            # The reference strips "." and "0" from a dollar amount ending ".00"
            ("$3360.00", 0.0),
        ],
    )
    def test_math(self, answer, score):
        assert evaluate_math(answer, MATH)["score"] == score

    def test_math_without_a_number(self):
        result = evaluate_math("I don't know", MATH)

        assert result["score"] == 0.0
        assert "could not be extracted" in result["error"]

    @pytest.mark.parametrize(
        ("reply", "code"),
        [
            ("```python\ndef f(x):\n    return x\n```", "def f(x):\n    return x"),
            (
                "```python\nimport math\n\ndef g(x):\n    return math.sqrt(x)\n```\n"
                "```\nprint(g(4))\n```",
                "import math\n\ndef g(x):\n    return math.sqrt(x)",
            ),
            (
                "```python\ndef outer():\n    def inner():\n        pass\n    return 1\n```",
                "def outer():\n    def inner():\n        pass\n    return 1",
            ),
            ("Sure.\ndef h():\n    return 1", "def h():\n    return 1"),
            ("No code here.", ""),
            (
                "```python\nclass Solution:\n    def f(self):\n        return 1\n```",
                "class Solution:\n    def f(self):\n        return 1",
            ),
        ],
    )
    def test_extract_code(self, reply, code):
        assert extract_code(reply) == code

    def test_prepare_solution_adds_prompt_imports_and_renames_the_function(self):
        solution = prepare_solution(extract_code(CORRECT_DEDUPE), CODE)

        assert solution is not None
        assert solution.startswith("from typing import List\n\nfrom collections import Counter")
        assert "def remove_duplicates(numbers):" in solution
        assert "dedupe" not in solution
        assert prepare_solution("", CODE) is None

    @pytest.mark.parametrize(
        "reply",
        [
            "[create_histogram(data=[1, 2], bins=5), create_histogram(data=[3, 4], bins=5)]",
            "[create_histogram(data=[3, 4], bins=5), create_histogram(data=[1, 2], bins=5)]",
            "```python\n[create_histogram(data=[1, 2], bins=5),\n"
            " create_histogram(data=[3, 4], bins=5)]\n```",
        ],
    )
    def test_actions_correct(self, reply):
        assert evaluate_actions(reply, ACTIONS)["score"] == 1.0

    def test_actions_wrong_and_unparseable(self):
        wrong = "[create_histogram(data=[1, 2], bins=5), create_histogram(data=[3, 4], bins=10)]"

        assert evaluate_actions(wrong, ACTIONS)["score"] == 0.0
        assert evaluate_actions("Which data sets?", ACTIONS) == {
            "is_correct": False,
            "error": "Failing to parse the predicted answer as an AST",
        }
        assert parse_function_calls("f(a=1)") == [{"f": {"a": 1}}]

    def test_data2text_bleu(self):
        # sacrebleu's float arithmetic lands just above 1, so as in the reference an exact
        # description does not count as correct and the conversation continues
        assert evaluate_data2text(f" {REFERENCE_SENTENCE} ", DATA2TEXT)["score"] == pytest.approx(1)
        partial = evaluate_data2text("AFC 3 was held at Geelong Arena.", DATA2TEXT)["score"]
        assert 0.0 < partial < 1.0

    def test_summary_bullets_are_trimmed_to_the_word_limit(self):
        long_line = " ".join(["word"] * 400)

        assert summary_bullets("a\nb") == ["a", "b"]
        assert summary_bullets(long_line) == [" ".join(["word"] * 300)]

    def test_citations(self):
        assert citations("A [1][4], then [2, 3].") == [1, 4, 2, 3]

    def test_summary_scores(self):
        bullets = ["- Tracking blood pressure [1][2]", "- Logging symptoms [3]"]
        verdicts = [
            {"insight_id": "i1", "coverage": "FULL_COVERAGE", "bullet_id": 1},
            {"insight_id": "i2", "coverage": "PARTIAL_COVERAGE", "bullet_id": 2},
        ]

        scores = summary_scores(bullets, verdicts, SUMMARY["insightid2ref_citations"])

        assert scores["coverage_score"] == pytest.approx(0.75)
        assert scores["citation_score"] == pytest.approx((1 + 2 / 3) / 2)
        assert scores["score"] == pytest.approx((1 + 0.5 * 2 / 3) / 2)

    def test_an_unreadable_bullet_id_reads_the_second_to_last_bullet(self):
        bullets = ["- One [3]", "- Two [1][2]", "- Three"]
        verdict = [{"insight_id": "i1", "coverage": "FULL_COVERAGE", "bullet_id": "NA"}]

        assert summary_scores(bullets, verdict, SUMMARY["insightid2ref_citations"])["score"] == 1


@pytest.mark.usefixtures("judge_test_env")
class TestJudging:
    @pytest.mark.anyio
    async def test_conversation_ends_at_the_first_correct_answer(self):
        judge = ScriptedJudge(
            categorizations=["clarification", "answer_attempt", "answer_attempt"],
            extractions=["3000", "3360"],
        )
        instance = _instance("lost_in_conversation_math", MATH["task_id"])
        replies = ["How many games?", "It is 3000.", "The total is 3360.", "Still 3360."]

        score = await MathJudge(judge_fn=judge).ascore_with_context(
            instance, _output(replies), ScoringContext()
        )

        assert score == 1.0
        result = instance.metadata["judge_result"]["lic_judge"]
        assert [t["response_type"] for t in result["turns"]] == [
            "clarification",
            "answer_attempt",
            "answer_attempt",
        ]
        assert [t.get("score") for t in result["turns"]] == [None, 0.0, 1.0]
        assert (result["first_answer_turn"], result["num_turns"], result["is_correct"]) == (
            1,
            4,
            True,
        )
        assert len(judge.calls["categorization"]) == 3
        raw = instance.metadata["judge_raw_response"]["lic_judge"]
        assert [entry["step"] for entry in raw] == [
            "categorization",
            "categorization",
            "extraction",
            "categorization",
            "extraction",
        ]

    @pytest.mark.anyio
    async def test_categorization_prompt_follows_the_reference(self):
        judge = ScriptedJudge(categorizations=["discussion"] * 4)
        instance = _instance("lost_in_conversation_math", MATH["task_id"])

        await MathJudge(judge_fn=judge).ascore_with_context(
            instance, _output(["<think>Hmm.</think>Tell me more.", "a", "b", "c"]), ScoringContext()
        )

        prompt = judge.calls["categorization"][0]
        assert prompt.endswith("Conversation's last turn:\n[assistant] Tell me more.")
        assert f"solve:\n{MATH['shards'][0]['shard']}\n\nSpecification:\n" in prompt
        assert json.dumps(MATH["shards"][1:]) in prompt
        assert f"[Task Specific Answer] {prompts.MATH_ANSWER_DESCRIPTION}" in prompt

    @pytest.mark.anyio
    async def test_a_conversation_scores_its_last_answer_attempt(self):
        judge = ScriptedJudge(
            categorizations=["answer_attempt", "clarification", "answer_attempt", "hedge"],
            extractions=["3000", "3001"],
        )
        instance = _instance("lost_in_conversation_math", MATH["task_id"])

        score = await MathJudge(judge_fn=judge).ascore_with_context(
            instance, _output(["3000", "Which?", "3001", "Either 1 or 2"]), ScoringContext()
        )

        assert score == 0.0
        assert instance.metadata["judge_result"]["lic_judge"]["first_answer_turn"] == 0

    @pytest.mark.anyio
    async def test_no_answer_attempt_scores_zero(self):
        judge = ScriptedJudge(categorizations=["interrogation"] * 4)
        instance = _instance("lost_in_conversation_math", MATH["task_id"])

        score = await MathJudge(judge_fn=judge).ascore_with_context(
            instance, _output(["Why?"] * 4), ScoringContext()
        )

        assert score == 0.0
        assert instance.metadata["judge_result"]["lic_judge"]["first_answer_turn"] is None

    @pytest.mark.anyio
    async def test_extraction_must_be_verbatim_and_falls_back_to_empty(self):
        retried = ScriptedJudge(categorizations=["answer_attempt"], extractions=["3,360", "3360"])
        instance = _instance("lost_in_conversation_math:full", MATH["task_id"])

        assert (
            await MathJudge(judge_fn=retried).ascore_with_context(
                instance, _output(["It is 3360."]), ScoringContext()
            )
            == 1.0
        )
        assert len(retried.calls["extraction"]) == 2

        failed = ScriptedJudge(extractions=["9", "{}", "not json"])
        score = await MathJudge(judge_fn=failed).ascore_with_context(
            instance, _output(["It is 3360."]), ScoringContext()
        )
        assert score == 0.0
        assert instance.metadata["judge_result"]["lic_judge"]["turns"][0]["answer"] == ""

    @pytest.mark.anyio
    async def test_the_full_setting_is_not_categorized(self):
        judge = ScriptedJudge(extractions=["3360"])
        instance = _instance("lost_in_conversation_math:full", MATH["task_id"])

        score = await MathJudge(judge_fn=judge).ascore_with_context(
            instance, _output(["The answer is 3360."]), ScoringContext()
        )

        assert score == 1.0
        assert judge.calls["categorization"] == []
        assert instance.metadata["judge_result"]["lic_judge"]["setting"] == "full"

    @pytest.mark.anyio
    async def test_an_unreadable_categorization_leaves_the_output_unscored(self):
        judge = ScriptedJudge()
        judge.queues["categorization"] = ['{"response_type": "maybe"}', "nope", "{}"]
        instance = _instance("lost_in_conversation_math", MATH["task_id"])

        with pytest.raises(JudgeParseError):
            await MathJudge(judge_fn=judge).ascore_with_context(
                instance, _output(["a", "b", "c", "d"]), ScoringContext()
            )

        assert instance.metadata["judge_result"] == {"lic_judge": None}
        assert len(instance.metadata["judge_raw_response"]["lic_judge"]) == 1

    @pytest.mark.anyio
    async def test_an_output_without_turns_needs_the_scaffold(self):
        instance = _instance("lost_in_conversation_math", MATH["task_id"])

        with pytest.raises(ValueError, match="multi_turn_chat"):
            await MathJudge(judge_fn=ScriptedJudge()).ascore_with_context(
                instance, LMOutput(text="3360"), ScoringContext()
            )

    @pytest.mark.anyio
    async def test_actions_and_data2text_score_whole_replies(self):
        actions = _instance("lost_in_conversation_actions", ACTIONS["task_id"])
        calls = "[create_histogram(data=[1, 2], bins=5), create_histogram(data=[3, 4], bins=5)]"
        judge = ScriptedJudge(categorizations=["clarification", "answer_attempt"])

        score = await ActionsJudge(judge_fn=judge).ascore_with_context(
            actions, _output(["Which data?", calls, "unused"]), ScoringContext()
        )
        assert score == 1.0

        table = _instance("lost_in_conversation_data2text", DATA2TEXT["task_id"])
        judge = ScriptedJudge(categorizations=["answer_attempt", "answer_attempt"])
        score = await Data2TextJudge(judge_fn=judge).ascore_with_context(
            table,
            _output(["A table about AFC.", "AFC 3 was held at Geelong Arena."]),
            ScoringContext(),
        )
        assert score == evaluate_data2text("AFC 3 was held at Geelong Arena.", DATA2TEXT)["score"]
        assert len(judge.calls["categorization"]) == 2

    @pytest.mark.anyio
    async def test_summary_evaluates_only_the_last_turn(self):
        judge = ScriptedJudge(
            coverages=[
                {"coverage": "FULL_COVERAGE", "bullet_id": 1},
                {"coverage": "PARTIAL_COVERAGE", "bullet_id": 2},
            ]
        )
        instance = _instance("lost_in_conversation_summary", SUMMARY["task_id"])
        last = "- Tracking blood pressure [1][2]\n- Logging symptoms [3]"

        score = await SummaryJudge(judge_fn=judge).ascore_with_context(
            instance, _output(["- Draft [2]", "- Draft again [1]", last]), ScoringContext()
        )

        assert score == pytest.approx((1 + 0.5 * 2 / 3) / 2)
        assert judge.calls["categorization"] == []
        assert len(judge.calls["coverage"]) == len(SUMMARY["insights"])
        bullets = bullets_json(last.split("\n"))
        assert judge.calls["coverage"][0] == prompts.SUMMARY_COVERAGE_PROMPT.replace(
            "[[BULLETS]]", bullets
        ).replace("[[INSIGHT]]", SUMMARY["insights"][0]["insight"])
        turns = instance.metadata["judge_result"]["lic_judge"]["turns"]
        assert [t["score"] for t in turns[:2]] == [0.0, 0.0]

    @pytest.mark.anyio
    async def test_a_coverage_verdict_naming_no_bullet_is_asked_for_again(self):
        judge = ScriptedJudge(
            coverages=[
                {"coverage": "FULL_COVERAGE", "bullet_id": 7},
                {"coverage": "NO_COVERAGE", "bullet_id": "NA"},
                {"coverage": "NO_COVERAGE", "bullet_id": "NA"},
            ]
        )
        instance = _instance("lost_in_conversation_summary:full", SUMMARY["task_id"])

        score = await SummaryJudge(judge_fn=judge).ascore_with_context(
            instance, _output(["- One bullet [1]"]), ScoringContext()
        )

        assert score == 0.0
        assert len(judge.calls["coverage"]) == 3


@pytest.mark.usefixtures("judge_test_env")
class TestCode:
    @pytest.mark.anyio
    @pytest.mark.parametrize(("reply", "expected"), [(CORRECT_DEDUPE, 1.0), (WRONG_DEDUPE, 0.0)])
    async def test_answer_attempts_run_against_the_tests(self, reply, expected):
        env = LocalGraderEnv()
        judge = ScriptedJudge(categorizations=["clarification", "answer_attempt", "discussion"])
        instance = _instance("lost_in_conversation_code", CODE["task_id"])

        score = await CodeJudge(judge_fn=judge).ascore_with_context(
            instance, _output(["Which list?", reply, "Done."]), ScoringContext(execution_env=env)
        )

        assert score == expected
        solution = next(v for k, v in env.staged.items() if k.endswith("solution.py"))
        assert "def remove_duplicates(numbers):" in solution
        problem = json.loads(next(v for k, v in env.staged.items() if k.endswith("problem.json")))
        assert (problem["fn_name"], problem["timeout"], problem["private_test_cases"]) == (
            "remove_duplicates",
            6.0,
            "",
        )

    @pytest.mark.anyio
    async def test_a_reply_without_a_function_is_wrong_without_running(self):
        env = LocalGraderEnv()
        judge = ScriptedJudge(categorizations=["answer_attempt", "refuse", "refuse"])
        instance = _instance("lost_in_conversation_code", CODE["task_id"])

        score = await CodeJudge(judge_fn=judge).ascore_with_context(
            instance, _output(["Use a set.", "No.", "No."]), ScoringContext(execution_env=env)
        )

        assert score == 0.0
        assert env.staged == {}

    @pytest.mark.anyio
    async def test_code_needs_a_sandbox(self):
        judge = ScriptedJudge(categorizations=["answer_attempt"])
        instance = _instance("lost_in_conversation_code", CODE["task_id"])

        with pytest.raises(SandboxRequiredError):
            await CodeJudge(judge_fn=judge).ascore_with_context(
                instance, _output([CORRECT_DEDUPE, "", ""]), ScoringContext()
            )

    @pytest.mark.anyio
    async def test_last_reply_scorer_needs_no_judge(self):
        instance = _instance("lost_in_conversation_code", CODE["task_id"])
        output = _output([WRONG_DEDUPE, "Fixed:", CORRECT_DEDUPE])

        score = await LastReplyCodeScorer().ascore(instance, output, LocalGraderEnv())

        assert score == 1.0
        assert output.metadata["execution_result"]["is_correct"] is True


@pytest.mark.usefixtures("judge_test_env")
class TestRuns:
    @pytest.mark.anyio
    async def test_metrics_over_a_run(self, monkeypatch):
        async def judge(prompt: str, *, system_prompt: str | None = None) -> str:
            # Replies holding a number are answers; conversations are judged concurrently
            reply = prompt.rpartition("[assistant] ")[2]
            number = "".join(ch for ch in reply if ch.isdigit())
            if ScriptedJudge.kind(prompt) == "categorization":
                return json.dumps({"response_type": "answer_attempt" if number else "discussion"})
            return json.dumps({"answer": number})

        monkeypatch.setitem(lic._MATH_JUDGE.__dict__, "judge_fn", judge)
        task = get_task("lost_in_conversation_math")
        early, late = (task.process_doc(MATH, index) for index in range(2))
        responses = [
            Response(
                instance=instance,
                request=task.format_request(instance),
                outputs=[_output(replies)],
            )
            for instance, replies in (
                (early, ["3360", "a", "b", "c"]),
                (late, ["a", "b", "c", "3000"]),
            )
        ]

        await task.score_responses(responses, ScoringContext())
        metrics = task.compute_metrics(responses)

        assert metrics["any__any__score"]["lic_judge"] == 0.5
        assert metrics["any__any__premature_answer_rate"]["lic_judge"] == 0.5
        assert metrics["any__any__no_answer_rate"]["lic_judge"] == 0.0

    @pytest.mark.anyio
    async def test_every_judge_of_a_served_panel_judges_the_same_conversation(self):
        models = ("Qwen/Qwen3-30B-A3B-Instruct-2507", "moonshotai/Kimi-Linear-48B-A3B-Instruct")

        class ServedJudge:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name
                self.judge = ScriptedJudge(
                    categorizations=["answer_attempt"] * 4, extractions=["3360"]
                )

            async def agenerate(self, requests, sampling_params=None):
                reply = await self.judge(requests[0].messages[-1]["content"])
                return [[LMOutput(text=reply)]]

        judges = {"qwen": ServedJudge(models[0]), "kimi": ServedJudge(models[1])}
        task = get_task(
            "lost_in_conversation_math",
            {"judge_models": [f"qwen={models[0]}", f"kimi={models[1]}"]},
        )
        context = ScoringContext(
            inference_pool=SimpleNamespace(names=sorted(judges), get=judges.__getitem__)
        )
        instance = _instance("lost_in_conversation_math", MATH["task_id"])
        response = Response(
            instance=instance,
            request=task.format_request(instance),
            outputs=[_output(["3360", "a", "b", "c"])],
        )

        await task.score_responses([response], context)
        metrics = task.compute_metrics([response])

        names = {f"lic_judge[{model}]" for model in models}
        assert set(instance.metadata["judge_result"]) == names
        assert metrics["any__any__score"] == dict.fromkeys(names, 1.0)
        assert all(len(j.judge.calls["categorization"]) == 1 for j in judges.values())
