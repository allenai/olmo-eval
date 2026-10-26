"""Tests for the EQ-Bench 3 rubric task."""

import hashlib
import json

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers import ScoringIncompleteError
from olmo_eval.common.types import LMOutput, Response
from olmo_eval.evals.tasks import eq_bench_3
from olmo_eval.evals.tasks.common import get_task, task_exists
from olmo_eval.evals.tasks.common.judge import JudgeParseError
from olmo_eval.evals.tasks.common.judge_panel import DEFAULT_JUDGE_PANEL
from olmo_eval.evals.tasks.eq_bench_3 import (
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    EQBench3RubricScorer,
    RubricScoreMetric,
    conversation_score,
    format_sectioned_reply,
    output_format,
    parse_rubric_scores,
    parse_scenario_prompts,
    reference_files,
    robust_json_loads,
    rubric_prompt,
)

# Git blob SHA-1s of the reference files at affc5a000bbdb965b807e09ca7875f8530fa8f56
BLOB_SHAS = {
    "LICENSE": "2124bf8da0d50abc1eb52ab616595794426524e2",
    "data/debrief_prompt.txt": "6cceb2cedf5922eac189dbaa537b8385e41640bf",
    "data/rubric_scoring_criteria.txt": "f52c5c28bb19dfc259bacaedab0593bb39fb800c",
    "data/rubric_scoring_criteria_analysis.txt": "ed6e352692ee3e7eec15a7b7733e0c320ab840fb",
    "data/rubric_scoring_prompt.txt": "b31db2ba56acbbf0a58ea67de7a4d649bfa8f3fa",
    "data/rubric_scoring_prompt_analysis.txt": "6ca712a23e5e156385fade6375dff2bd80c98938",
    "data/scenario_master_prompt.txt": "845e4611cde3e336aae86a83f1d7bcfaca6f24f5",
    "data/scenario_master_prompt_analysis.txt": "d45ff1e7d1b2c6a4db7716ce7b3ea10460dce4e0",
    "data/scenario_master_prompt_message_drafting.txt": "9a804fba2a3ceafd54abb6054894f8e3201b270e",
    "data/scenario_prompts.txt": "030e1dcd39d6e0f89448607c930c9f6ae92a4026",
}
ROLEPLAY_REPLY = (
    "# I'm thinking & feeling\n**I feel** scapegoated.\n\n# They're thinking & feeling\n"
    "My boss wants this gone.\n\n# My response\nI'll talk to HR.\n--\nextra"
)
ANALYSIS_REPLY = "An analysis. " * 10
STANDARD_SCORES = {
    "demonstrated_empathy": 14,
    "pragmatic_ei": 12,
    "depth_of_insight": 16,
    "social_dexterity": 10,
    "emotional_reasoning": 15,
    "message_tailoring": 11,
    "warmth": 3,
    "sycophantic": 20,
}


def _judge_reply(scores: dict, reasoning: str = "Solid.") -> str:
    return "```json\n" + json.dumps({"chain_of_thought_reasoning": reasoning, **scores}) + "\n```"


def _replies_judge(reply: str | list[str]):
    """A judge that gives ``reply``, or each of a list in turn, and records its prompts."""
    calls: list[str] = []

    async def judge(prompt: str) -> str:
        calls.append(prompt)
        return reply.pop(0) if isinstance(reply, list) else reply

    return judge, calls


def _instance(scenario_id: str):
    task = get_task("eq_bench_3")
    prompts = parse_scenario_prompts(reference_files()["data/scenario_prompts.txt"])[scenario_id]
    return task.process_doc({"id": scenario_id, "prompts": prompts})


def _output(replies: list[str]) -> LMOutput:
    return LMOutput(text=replies[-1], metadata={"turns": [{"text": r} for r in replies]})


@pytest.fixture
def scenarios():
    return parse_scenario_prompts(reference_files()["data/scenario_prompts.txt"])


class TestReferenceData:
    def test_files_match_the_reference_commit(self):
        files = reference_files()

        assert set(files) == set(BLOB_SHAS)
        for path, sha in BLOB_SHAS.items():
            data = files[path].encode("utf-8")
            assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == sha, path

    def test_the_license_is_kept(self):
        assert reference_files()["LICENSE"].startswith(
            "MIT License\n\nCopyright (c) 2025 Sam Paech"
        )

    def test_scenarios_parse_as_in_the_reference(self, scenarios):
        kinds = [eq_bench_3.task_type(sid) for sid in scenarios]

        assert len(scenarios) == 45
        assert "409" not in scenarios
        assert (kinds.count("roleplay"), kinds.count("message_drafting")) == (25, 1)
        assert kinds.count("analysis") == 19
        assert len(scenarios["1"]) == 3
        assert scenarios["1"][0].startswith("[This is a role-play, with you playing")
        assert all(
            len(scenarios[sid]) == 1 for sid in eq_bench_3.ANALYSIS_SCENARIO_IDS & set(scenarios)
        )

    def test_parser_follows_the_reference_delimiters(self):
        text = (
            "Notes before any scenario.\n"
            "####### Prompt1\nNo scenario yet.\n"
            "######## 7   | Friend Issue | Title\n"
            "####### Prompt1\nFirst line\n  indented\n\n"
            "####### Prompt2  \nTurn two.\n"
            "######## 8   | Malformed\n"
            "####### 1\nSkipped.\n"
            "######## 9   | Blank\n"
            "####### Prompt1\n   \n"
        )

        assert parse_scenario_prompts(text) == {"7": ["First line\n  indented", "Turn two."]}


class TestRegistration:
    def test_specs_are_registered(self):
        assert task_exists("eq_bench_3")
        assert task_exists("eq_bench_3:judge_panel")

    def test_runs_under_the_multi_turn_scaffold_with_the_reference_sampling(self):
        task = get_task("eq_bench_3")
        params = task.config.sampling_params

        assert task.required_scaffold == "multi_turn_chat"
        assert (params.temperature, params.max_tokens) == (0.7, 12000)

    def test_default_judge(self):
        config = get_task("eq_bench_3").config.to_dict()
        judge_fn = config["metrics"][0]["scorer"]["judge_fn"]

        assert (judge_fn["model"], judge_fn["max_tokens"], judge_fn["temperature"]) == (
            JUDGE_MODEL,
            JUDGE_MAX_TOKENS,
            0.0,
        )

    def test_metrics(self):
        config = get_task("eq_bench_3").config

        assert [m.name for m in config.metrics] == [
            "any__any__rubric_score",
            "task_type__roleplay__rubric_score",
            "task_type__message_drafting__rubric_score",
            "task_type__analysis__rubric_score",
        ]
        assert config.get_primary_metric().name == "any__any__rubric_score"
        assert {m.display_format() for m in config.metrics} == {"raw"}
        assert config.required_secrets == ("OPENAI_API_KEY",)

    def test_judge_panel_variant(self):
        task = get_task("eq_bench_3:judge_panel")

        assert len(task._get_scorers()) == len(DEFAULT_JUDGE_PANEL)
        assert len(task.config.metrics) == 4 * len(DEFAULT_JUDGE_PANEL)
        assert task.config.judge_max_tokens == JUDGE_MAX_TOKENS


class TestInstances:
    def test_every_scenario_becomes_an_instance(self):
        instances = list(get_task("eq_bench_3").instances)

        assert len(instances) == 45
        assert [i.metadata["id"] for i in instances][:3] == ["1", "2", "3"]

    def test_roleplay_turns_are_wrapped_and_end_with_the_debrief(self, scenarios):
        instance = _instance("1")
        turns = instance.metadata["turns"]
        master = reference_files()["data/scenario_master_prompt.txt"]

        assert len(turns) == 4
        assert turns[0] == master.format(scenario_prompt=scenarios["1"][0])
        assert turns[-1] == reference_files()["data/debrief_prompt.txt"]
        assert instance.metadata["task_type"] == "roleplay"

    def test_drafting_and_analysis_use_their_master_prompts(self, scenarios):
        drafting = _instance("208").metadata["turns"]
        analysis = _instance("401").metadata["turns"]

        assert drafting[0].endswith(
            reference_files()["data/scenario_master_prompt_message_drafting.txt"].split(
                "{scenario_prompt}"
            )[1]
        )
        assert analysis == [
            reference_files()["data/scenario_master_prompt_analysis.txt"].format(
                scenario_prompt=scenarios["401"][0]
            )
        ]

    def test_request_scripts_every_turn(self):
        task = get_task("eq_bench_3")
        instance = _instance("1")
        request = task.format_request(instance)

        turns = instance.metadata["turns"]
        assert request.messages == ({"role": "user", "content": turns[0]},)
        assert [m["content"] for m in request.metadata["follow_up_turns"]] == turns[1:]


class TestTranscripts:
    def test_roleplay_replies_are_rebuilt_from_their_sections(self):
        rebuilt = format_sectioned_reply("1", ROLEPLAY_REPLY)

        assert rebuilt == (
            "# I'm thinking & feeling\n**I feel** scapegoated.\n\n# They're thinking & "
            "feeling\nMy boss wants this gone.\n\n# My response\nI'll talk to HR.\n\n--"
        )

    def test_curly_apostrophes_and_missing_sections(self):
        rebuilt = format_sectioned_reply("1", "# I’m thinking & feeling\nWorried.")

        assert "I'm thinking & feeling\nWorried." in rebuilt
        assert "# My response\n\n\n--" in rebuilt

    def test_drafting_sections(self):
        reply = "# Perspective-taking\nThey hurt.\n# Draft brainstorming\nBe kind.\n# Draft\nHi."

        assert format_sectioned_reply("208", reply).endswith("# Draft\nHi.\n\n--")

    def test_rubric_prompt_for_a_roleplay(self):
        instance = _instance("1")
        turns = instance.metadata["turns"]
        replies = [ROLEPLAY_REPLY, ROLEPLAY_REPLY, ROLEPLAY_REPLY, "  I could have listened. "]

        prompt = rubric_prompt("1", turns, replies)

        assert "*" not in prompt and "#" not in prompt
        assert prompt.startswith("You are an expert in assessing emotional intelligence.")
        assert "User:\n" + turns[0].replace("*", "").replace("#", "") + "\n---\nAssistant:\n" in (
            prompt
        )
        assert "Assistant:\nI could have listened.\n\n[TRANSCRIPT END]" in prompt
        assert "This was a role-play scenario" not in prompt
        assert '"demonstrated_empathy": 0-20' in prompt
        assert '"humanlike": 0-20' in prompt

    def test_rubric_prompt_for_a_drafting_conversation(self):
        turns = _instance("208").metadata["turns"]
        reply = "# Perspective-taking\nThey hurt.\n# Draft brainstorming\nBe kind.\n# Draft\nHi."

        prompt = rubric_prompt("208", turns, [reply] * 3 + ["Debrief."])

        assert (
            prompt.count(
                "Assistant:\n Perspective-taking\nThey hurt.\n\n Draft brainstorming\nBe kind.\n\n"
                " Draft\nHi.\n\n--\n"
            )
            == 3
        )

    def test_rubric_prompt_for_an_analysis(self):
        instance = _instance("401")

        prompt = rubric_prompt("401", instance.metadata["turns"], ["A deep analysis."])

        assert prompt.startswith("[RESPONDENT ANALYSIS TRANSCRIPT]")
        assert "Assistant:\nA deep analysis.\n" in prompt
        assert '"correctness": 0-20' in prompt
        assert "debrief" not in prompt.lower()

    def test_output_format_matches_the_reference(self):
        assert output_format(["warmth"]) == (
            '{\n  "chain_of_thought_reasoning": "detailed chain of thought reasoning about the '
            'coming scoring decisions",\n  "warmth": 0-20\n}'
        )


class TestJudgeReplies:
    @pytest.mark.parametrize(
        "reply,expected",
        [
            ('{"a": 1}', {"a": 1}),
            ('Here:\n```json\n{"a": 1}\n```', {"a": 1}),
            ('```json\n{"a": 1}\n```\nFixed:\n```json\n{"a": 2}\n```', {"a": 2}),
            ('{"a": "line one\nline two"}', {"a": "line one\nline two"}),
            ('{"a": "she said "hi" twice"}', {"a": 'she said "hi" twice'}),
            ('prefix {"a": {"b": 1}} suffix {"c": 2}', {"a": {"b": 1}}),
        ],
    )
    def test_robust_json_loads(self, reply, expected):
        assert robust_json_loads(reply) == expected

    def test_robust_json_loads_needs_an_object(self):
        with pytest.raises(json.JSONDecodeError):
            robust_json_loads("no json here")

    def test_rubric_scores(self):
        reply = json.dumps(
            {
                "chain_of_thought_reasoning": "It scores 3 here",
                "warmth": 12,
                "pragmatic_ei": "15/20",
                "humanlike": "n/a",
                "flag": True,
            }
        )

        assert parse_rubric_scores(reply) == {"warmth": 12.0, "pragmatic_ei": 15.0, "flag": 1.0}

    @pytest.mark.parametrize("reply", ["nothing", '{"chain_of_thought_reasoning": "x"}', "[1]"])
    def test_replies_without_scores(self, reply):
        assert parse_rubric_scores(reply) is None

    def test_conversation_scores_count_only_the_reference_criteria(self):
        standard = {k: float(v) for k, v in STANDARD_SCORES.items()}

        assert conversation_score("1", standard) == pytest.approx(78 / 6)
        assert conversation_score("401", {"theory_of_mind": 10.0, "warmth": 2.0}) == 10.0
        assert conversation_score("1", {"warmth": 3.0}) is None

    def test_parse_judge_response(self):
        scorer = EQBench3RubricScorer(judge_fn=_replies_judge("")[0])

        assert scorer.parse_judge_response(_judge_reply(STANDARD_SCORES)) == pytest.approx(
            78 / 6 / 20
        )
        with pytest.raises(ValueError, match="no rubric scores"):
            scorer.parse_judge_response('{"warmth": 3}')


@pytest.mark.usefixtures("judge_test_env")
class TestScoring:
    @pytest.mark.anyio
    async def test_scores_a_conversation(self):
        judge, calls = _replies_judge(_judge_reply(STANDARD_SCORES))
        instance = _instance("1")
        replies = [ROLEPLAY_REPLY] * 3 + ["<think>hmm</think>I could have listened."]

        score = await EQBench3RubricScorer(judge_fn=judge).ascore_with_context(
            instance, _output(replies), ScoringContext()
        )

        assert score == pytest.approx(78 / 6 / 20)
        assert calls == [
            rubric_prompt("1", instance.metadata["turns"], [*replies[:3], replies[3][18:]])
        ]
        result = instance.metadata["judge_result"]["eq_bench_3_judge"]
        assert result["conversation_score"] == pytest.approx(13.0)
        assert result["rubric_scores"]["warmth"] == 3.0

    @pytest.mark.anyio
    async def test_an_incomplete_conversation_is_left_unscored(self):
        judge, calls = _replies_judge(_judge_reply(STANDARD_SCORES))
        instance = _instance("1")

        with pytest.raises(Exception, match="ended after 2 of 4 turns"):
            await EQBench3RubricScorer(judge_fn=judge).ascore_with_context(
                instance, _output([ROLEPLAY_REPLY] * 2), ScoringContext()
            )

        assert calls == []
        assert instance.metadata["judge_result"] == {"eq_bench_3_judge": None}

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "scenario_id,replies",
        [
            ("1", ["Short."] * 3 + [ROLEPLAY_REPLY]),
            ("401", ["<think>" + "x" * 200 + "</think>Short."]),
            ("401", ["\n" * 100 + "Short." + " " * 100]),
        ],
    )
    async def test_a_conversation_without_a_long_scenario_reply_is_left_unscored(
        self, scenario_id, replies
    ):
        judge, calls = _replies_judge(_judge_reply(STANDARD_SCORES))

        with pytest.raises(ScoringIncompleteError, match="100 characters"):
            await EQBench3RubricScorer(judge_fn=judge).ascore_with_context(
                _instance(scenario_id), _output(replies), ScoringContext()
            )

        assert calls == []

    @pytest.mark.anyio
    async def test_an_analysis_reply_reaches_the_judge_as_written(self):
        judge, calls = _replies_judge(_judge_reply({"theory_of_mind": 12}))
        instance = _instance("401")
        reply = "\n" + ANALYSIS_REPLY + "\n\n"

        await EQBench3RubricScorer(judge_fn=judge).ascore_with_context(
            instance, _output([reply]), ScoringContext()
        )

        assert calls == [rubric_prompt("401", instance.metadata["turns"], [reply])]
        assert f"Assistant:\n\n{ANALYSIS_REPLY}\n\n\n" in calls[0]

    @pytest.mark.anyio
    async def test_one_long_scenario_reply_is_enough(self):
        judge, calls = _replies_judge(_judge_reply(STANDARD_SCORES))
        replies = ["Short.", ROLEPLAY_REPLY, "Short.", "Debrief."]

        await EQBench3RubricScorer(judge_fn=judge).ascore_with_context(
            _instance("1"), _output(replies), ScoringContext()
        )

        assert len(calls) == 1

    @pytest.mark.anyio
    async def test_an_output_without_turns_needs_the_scaffold(self):
        judge, _ = _replies_judge("")

        with pytest.raises(ValueError, match="scaffold=multi_turn_chat"):
            await EQBench3RubricScorer(judge_fn=judge).ascore_with_context(
                _instance("401"), LMOutput(text="analysis"), ScoringContext()
            )

    @pytest.mark.anyio
    async def test_a_reply_without_counted_scores_is_asked_for_again(self):
        judge, calls = _replies_judge(
            ['{"warmth": 3}', "oops", _judge_reply({"theory_of_mind": 12, "correctness": 8})]
        )
        instance = _instance("401")

        score = await EQBench3RubricScorer(judge_fn=judge).ascore_with_context(
            instance, _output([ANALYSIS_REPLY]), ScoringContext()
        )

        assert len(calls) == 3
        assert score == pytest.approx(10 / 20)

    @pytest.mark.anyio
    async def test_no_counted_scores_leaves_the_output_unscored(self):
        judge, calls = _replies_judge('{"warmth": 3}')
        instance = _instance("401")

        with pytest.raises(JudgeParseError):
            await EQBench3RubricScorer(judge_fn=judge).ascore_with_context(
                instance, _output([ANALYSIS_REPLY]), ScoringContext()
            )

        assert len(calls) == 3
        assert instance.metadata["judge_raw_response"] == {"eq_bench_3_judge": '{"warmth": 3}'}

    @pytest.mark.anyio
    async def test_metrics_over_a_run(self, monkeypatch):
        judge, _ = _replies_judge(
            [
                _judge_reply(STANDARD_SCORES),
                _judge_reply({"theory_of_mind": 18, "correctness": 18}),
            ]
        )
        monkeypatch.setitem(eq_bench_3._JUDGE.__dict__, "judge_fn", judge)
        task = get_task("eq_bench_3")
        roleplay, analysis = _instance("1"), _instance("401")
        responses = [
            Response(
                instance=roleplay,
                request=task.format_request(roleplay),
                outputs=[_output([ROLEPLAY_REPLY] * 3 + ["Debrief."])],
            ),
            Response(
                instance=analysis,
                request=task.format_request(analysis),
                outputs=[_output([ANALYSIS_REPLY])],
            ),
        ]

        await task.score_responses(responses, ScoringContext())
        metrics = task.compute_metrics(responses)

        assert metrics["any__any__rubric_score"]["eq_bench_3_judge"] == pytest.approx(
            5 * (13 + 18) / 2
        )
        assert metrics["task_type__roleplay__rubric_score"]["eq_bench_3_judge"] == 65.0
        assert metrics["task_type__analysis__rubric_score"]["eq_bench_3_judge"] == 90.0
        assert RubricScoreMetric().higher_is_better() is True
