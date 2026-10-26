"""Tests for the WildBench tasks."""

import hashlib
import json
import random

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response, Split
from olmo_eval.data import DataLoader
from olmo_eval.evals.tasks import wildbench
from olmo_eval.evals.tasks.common import get_task, task_exists
from olmo_eval.evals.tasks.common.judge import JudgeParseError
from olmo_eval.evals.tasks.common.judge_panel import DEFAULT_JUDGE_PANEL
from olmo_eval.evals.tasks.constants.wildbench import PAIRWISE_TEMPLATE, SCORE_TEMPLATE
from olmo_eval.evals.tasks.wildbench import (
    EMPTY_RESPONSE,
    OUTPUTS_REVISION,
    REFERENCE_MODELS,
    WILDBENCH_REVISION,
    WBRewardMetric,
    WBRewardTaskMacroMetric,
    WBScoreMetric,
    WBScoreTaskMacroMetric,
    WildBenchRewardScorer,
    WildBenchScoreScorer,
    clear_output,
    judge_value,
    pairwise_prompt,
    pairwise_reward,
    parse_choice,
    parse_score,
    score_prompt,
    shorten,
)

DOC = {
    "id": "ae006110bb364606",
    "session_id": "ae006110bb364606",
    "conversation_input": [
        {"role": "user", "content": "Write a haiku about rain."},
        {"role": "assistant", "content": "Soft rain on the roof."},
        {"role": "user", "content": "Now one about snow."},
    ],
    "references": {"gpt-4": "unused"},
    "length": 2,
    "checklist": ["Is it a haiku?", "Is it about snow?"],
    "intent": "The user wants a haiku.",
    "primary_tag": "Creative Writing",
    "secondary_tags": ["Editing", "Reasoning"],
}


def _doc(session_id: str, **fields) -> dict:
    return {**DOC, "id": session_id, "session_id": session_id, **fields}


def _output(text: str) -> LMOutput:
    """An output with its answer extracted, as the task does before scoring."""
    output = LMOutput(text=text)
    output.extracted_answer = get_task("wildbench_score").extract_answer(output)
    return output


def _replies_judge(reply):
    """A judge that gives ``reply`` (or each of a list in turn), recording its prompts."""
    calls: list[str] = []

    async def judge(prompt: str) -> str:
        calls.append(prompt)
        return reply.pop(0) if isinstance(reply, list) else reply

    return judge, calls


def _reward_instance(response_is_a: bool = True, **references) -> Instance:
    instance = get_task("wildbench_reward").process_doc(DOC)
    instance.metadata["references"] = {
        "gpt4t": "GPT answer.",
        "haiku": "Haiku answer.",
        "llama": "Llama answer.",
        **references,
    }
    instance.metadata["response_is_a"] = response_is_a
    return instance


def _judged(result: dict | None, judge: str, task: tuple[str, ...] = ("creative",)) -> Response:
    return Response(
        instance=Instance(
            question="q", metadata={"task": list(task), "judge_result": {judge: result}}
        ),
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text="a")],
    )


def _scored(score: float, task=("creative",), empty: bool = False) -> Response:
    return _judged({"score": score, "empty_response": empty}, "wildbench_score_judge", task)


def _rewarded(rewards: dict, task=("creative",)) -> Response:
    """``rewards`` maps a reference to its (reward, reward_k500)."""
    result = {
        ref: {"choice": "A+", "skipped": None, "reward": r, "reward_k500": k}
        for ref, (r, k) in rewards.items()
    }
    return _judged(result, "wildbench_reward_judge", task)


class TestRegistration:
    @pytest.mark.parametrize("spec", ["wildbench_score", "wildbench_reward"])
    def test_specs_are_registered(self, spec):
        for variant in ("", ":thinking", ":judge_panel"):
            assert task_exists(spec + variant)
        assert get_task(spec).request_type == RequestType.CHAT

    def test_data_is_pinned(self):
        source = get_task("wildbench_score").config.get_data_source()

        assert get_task("wildbench_score").config.split == Split.TEST
        assert (source.path, source.subset, source.revision, source.split) == (
            "allenai/WildBench",
            "v2",
            WILDBENCH_REVISION,
            "test",
        )

    def test_sampling_follows_the_reference(self):
        params = get_task("wildbench_reward").config.sampling_params

        assert (params.temperature, params.top_p, params.max_tokens) == (0.0, 1.0, 4096)

    @pytest.mark.parametrize(
        "spec,model",
        [("wildbench_score", "gpt-4o-2024-05-13"), ("wildbench_reward", "gpt-4-turbo-2024-04-09")],
    )
    def test_default_judges_follow_the_reference(self, spec, model):
        judge_fn = get_task(spec).config.to_dict()["metrics"][0]["scorer"]["judge_fn"]

        assert judge_fn["model"] == model
        assert (judge_fn["max_tokens"], judge_fn["temperature"]) == (1024, 0.0)
        assert judge_fn["response_format"] == {"type": "json_object"}

    def test_score_metrics(self):
        config = get_task("wildbench_score").config

        assert [m.name for m in config.metrics] == [
            "any__any__wb_score_task_macro",
            "any__any__wb_score",
            "task__creative__wb_score",
            "task__planning_reasoning__wb_score",
            "task__math_data__wb_score",
            "task__info_advice__wb_score",
            "task__coding__wb_score",
        ]
        assert config.get_primary_metric().name == "any__any__wb_score_task_macro"
        assert {m.display_format() for m in config.metrics} == {"raw"}

    def test_reward_metrics(self):
        config = get_task("wildbench_reward").config

        assert [m.name for m in config.metrics] == [
            "any__any__reward_mix",
            "any__any__reward_mix_k500",
            "any__any__reward_mix_task_macro",
            "any__any__reward_gpt4t",
            "any__any__reward_gpt4t_k500",
            "any__any__reward_haiku",
            "any__any__reward_haiku_k500",
            "any__any__reward_llama",
            "any__any__reward_llama_k500",
        ]
        assert config.get_primary_metric().name == "any__any__reward_mix"

    def test_judge_panel_variant(self):
        task = get_task("wildbench_score:judge_panel")

        assert len(task._get_scorers()) == len(DEFAULT_JUDGE_PANEL)


class TestTemplates:
    @pytest.mark.parametrize(
        "template,sha",
        [
            (SCORE_TEMPLATE, "4ee5dd146c8c24785903edad16e8c90b5cd64a7f"),
            (PAIRWISE_TEMPLATE, "f00b6e82d9e711234de85ea6da71f808413e450d"),
        ],
        ids=["score", "pairwise"],
    )
    def test_templates_match_the_reference_files(self, template, sha):
        data = template.encode("utf-8")

        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == sha


class TestInstances:
    def test_conversation_and_judge_context(self):
        instance = get_task("wildbench_score").process_doc(DOC)

        assert instance.metadata["history"] == (
            "USER: Write a haiku about rain.\n\nASSISTANT: Soft rain on the roof.\n\n"
        )
        assert instance.metadata["last_query"] == "Now one about snow."
        assert instance.metadata["task"] == ["creative", "planning_reasoning"]

    def test_request_is_the_whole_conversation(self):
        task = get_task("wildbench_score")
        request = task.format_request(task.process_doc(DOC))

        assert request.messages == tuple(
            {"role": t["role"], "content": t["content"]} for t in DOC["conversation_input"]
        )
        assert request.system_prompt is None

    def test_a_conversation_must_end_with_a_user_turn(self):
        doc = _doc("x", conversation_input=DOC["conversation_input"][:2])

        with pytest.raises(ValueError, match="does not end with a user turn"):
            get_task("wildbench_score").process_doc(doc)

    def test_reward_instances_get_references_and_a_seeded_order(self, monkeypatch):
        docs = [_doc(f"s{i}") for i in range(6)]
        sources = []

        def load(self, source, streaming=False):
            sources.append(source)
            if source.path == "allenai/WildBench":
                return iter(docs)
            return iter(
                {"session_id": d["id"], "output": [f"{source.subset} on {d['id']}"]} for d in docs
            )

        monkeypatch.setattr(DataLoader, "load", load)
        instances = list(get_task("wildbench_reward").instances)

        rng = random.Random(42)
        assert [i.metadata["response_is_a"] for i in instances] == [
            rng.random() < 0.5 for _ in docs
        ]
        assert instances[2].metadata["references"] == {
            "gpt4t": "gpt-4-turbo-2024-04-09 on s2",
            "haiku": "claude-3-haiku-20240307 on s2",
            "llama": "Llama-2-70b-chat-hf on s2",
        }
        assert {s.revision for s in sources if s.path != "allenai/WildBench"} == {OUTPUTS_REVISION}

    def test_a_missing_reference_response_is_an_error(self, monkeypatch):
        def load(self, source, streaming=False):
            if source.path == "allenai/WildBench":
                return iter([DOC])
            return iter([])

        monkeypatch.setattr(DataLoader, "load", load)

        with pytest.raises(ValueError, match="has no response from gpt-4-turbo-2024-04-09"):
            list(get_task("wildbench_reward").instances)


class TestPrompts:
    def test_score_prompt(self):
        instance = get_task("wildbench_score").process_doc(DOC)
        long_answer = " ".join(["word"] * 1500)

        prompt = score_prompt(instance, long_answer)

        assert "{$" not in prompt
        assert "<|begin_of_history|>\n\nUSER: Write a haiku about rain.\n\n" in prompt
        assert "<|begin_of_query|>\n\nNow one about snow.\n\n<|end_of_query|>" in prompt
        assert "- Is it a haiku?\n- Is it about snow?\n" in prompt
        assert long_answer in prompt

    def test_pairwise_prompt_cuts_and_marks_empty_responses(self):
        instance = get_task("wildbench_reward").process_doc(DOC)

        prompt = pairwise_prompt(instance, shorten(" ".join(["w"] * 1001), 1000), "  ")

        assert "<|begin_of_response_A|>\n\n" + "w " * 999 + "w... (truncated)" in prompt
        assert f"<|begin_of_response_B|>\n\n{EMPTY_RESPONSE}\n\n" in prompt

    def test_only_the_pairwise_prompt_cuts_the_history_and_query(self):
        long_text = " ".join(["word"] * 1001)
        turns = [
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": long_text},
        ]
        instance = get_task("wildbench_reward").process_doc(_doc("x", conversation_input=turns))

        pairwise = pairwise_prompt(instance, "a", "b")
        score = score_prompt(instance, "a")

        cut_query = " ".join(["word"] * 1000) + "... (truncated)"
        assert f"<|begin_of_query|>\n\n{cut_query}\n\n<|end_of_query|>" in pairwise
        # "USER:" is the history's first word
        cut_history = "USER: " + " ".join(["word"] * 999) + "... (truncated)"
        assert f"<|begin_of_history|>\n\n{cut_history}\n\n<|end_of_history|>" in pairwise
        assert f"<|begin_of_query|>\n\n{long_text}\n\n<|end_of_query|>" in score
        assert f"<|begin_of_history|>\n\n{instance.metadata['history']}\n\n" in score

    def test_shorten(self):
        assert shorten("a b c", 3) == "a b c"
        assert shorten("a b c d", 3) == "a b c... (truncated)"
        assert shorten("a b c d", -1) == "a b c d"

    def test_clear_output(self):
        assert clear_output(" a</s>b<|endoftext|><pad>c<end_of_turn>\n") == "a b  c"


class TestJudgeReplies:
    @pytest.mark.parametrize(
        "reply,expected",
        [
            ('{"strengths": "x", "score": "7"}', 7.0),
            ('{"score": 8}', 8.0),
            ('  {"score": "9"}  ', 9.0),
            ('The JSON: {"score": "6", "weaknesses": "bad "quote"}', 6.0),
        ],
    )
    def test_scores(self, reply, expected):
        assert parse_score(reply) == expected

    @pytest.mark.parametrize(
        "reply",
        ['{"score": "ten"}', '{"score": "NaN"}', '{"strengths": "x"}', "no json", "[]"],
    )
    def test_replies_without_a_score(self, reply):
        assert parse_score(reply) is None

    @pytest.mark.parametrize("choice", ["A++", "A+", "A=B", "B+", "B++"])
    def test_choices(self, choice):
        assert parse_choice(json.dumps({"choice": choice})) == choice

    def test_the_templates_trailing_comma_is_read_by_the_fallback(self):
        assert parse_choice('{\n  "analysis of A": "ok",\n  "choice": "B+",\n}') == "B+"

    @pytest.mark.parametrize("reply", ['{"choice": "A>B"}', '{"choice": "B+++"}', "A++"])
    def test_invalid_choices(self, reply):
        assert parse_choice(reply) is None

    def test_judge_value_needs_the_key(self):
        assert judge_value("nothing here", "score") is None


class TestPairwiseReward:
    @pytest.mark.parametrize(
        "choice,response_is_a,expected",
        [
            ("A++", True, 1.0),
            ("A+", True, 0.5),
            ("A=B", True, 0.0),
            ("B+", True, -0.5),
            ("B++", True, -1.0),
            ("A++", False, -1.0),
            ("B+", False, 0.5),
        ],
    )
    def test_rewards(self, choice, response_is_a, expected):
        assert pairwise_reward(choice, response_is_a, 100, 100, -1) == expected

    def test_length_penalty_turns_a_long_slight_win_into_a_tie(self):
        assert pairwise_reward("A+", True, 701, 200, 500) == 0.0
        assert pairwise_reward("A+", True, 700, 200, 500) == 0.5
        assert pairwise_reward("A++", True, 2000, 200, 500) == 1.0
        assert pairwise_reward("B+", True, 100, 900, 500) == 0.0
        assert pairwise_reward("B+", True, 900, 100, 500) == -0.5


@pytest.mark.usefixtures("judge_test_env")
class TestScorers:
    @pytest.mark.anyio
    async def test_score_judge(self):
        judge, calls = _replies_judge('{"strengths": "s", "weaknesses": "w", "score": "8"}')
        instance = get_task("wildbench_score").process_doc(DOC)

        score = await WildBenchScoreScorer(judge_fn=judge).ascore_with_context(
            instance, _output("<think>hmm</think>Snow falls."), ScoringContext()
        )

        assert score == 0.8
        assert calls == [score_prompt(instance, "Snow falls.")]
        assert instance.metadata["judge_result"]["wildbench_score_judge"] == {
            "score": 8.0,
            "empty_response": False,
        }

    @pytest.mark.anyio
    async def test_an_empty_response_scores_one_without_a_judge_call(self):
        judge, calls = _replies_judge("")
        instance = get_task("wildbench_score").process_doc(DOC)

        await WildBenchScoreScorer(judge_fn=judge).ascore_with_context(
            instance, _output("<think>never finished"), ScoringContext()
        )

        assert calls == []
        assert instance.metadata["judge_result"]["wildbench_score_judge"] == {
            "score": 1.0,
            "empty_response": True,
        }

    @pytest.mark.anyio
    async def test_the_judge_sees_the_response_cleaned_as_the_reference_saves_it(self):
        judge, calls = _replies_judge('{"score": "8"}')
        instance = get_task("wildbench_score").process_doc(DOC)

        await WildBenchScoreScorer(judge_fn=judge).ascore_with_context(
            instance, _output("Snow</s> falls.<|endoftext|>"), ScoringContext()
        )

        assert calls == [score_prompt(instance, "Snow  falls.")]

    @pytest.mark.anyio
    async def test_a_score_judge_with_no_score_leaves_the_output_unscored(self):
        judge, calls = _replies_judge("I refuse")
        instance = get_task("wildbench_score").process_doc(DOC)

        with pytest.raises(JudgeParseError):
            await WildBenchScoreScorer(judge_fn=judge).ascore_with_context(
                instance, LMOutput(text="Snow."), ScoringContext()
            )

        assert len(calls) == 3
        assert instance.metadata["judge_result"] == {"wildbench_score_judge": None}

    @pytest.mark.anyio
    async def test_reward_judge_compares_with_every_reference(self):
        judge, calls = _replies_judge(
            ['{"choice": "A+"}', '{"choice": "B++"}', '{"choice": "A=B"}']
        )
        instance = _reward_instance(response_is_a=True)

        score = await WildBenchRewardScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text="Snow falls."), ScoringContext()
        )

        assert len(calls) == 3
        assert all("<|begin_of_response_A|>\n\nSnow falls.\n\n" in c for c in calls)
        assert {c.split("<|begin_of_response_B|>\n\n")[1].split("\n\n")[0] for c in calls} == {
            "GPT answer.",
            "Haiku answer.",
            "Llama answer.",
        }
        verdicts = instance.metadata["judge_result"]["wildbench_reward_judge"]
        assert set(verdicts) == set(REFERENCE_MODELS)
        assert sorted(v["reward"] for v in verdicts.values()) == [-1.0, 0.0, 0.5]
        assert score == pytest.approx((-0.5 / 3 + 1) / 2)

    @pytest.mark.anyio
    async def test_the_response_is_b_when_drawn_so(self):
        judge, calls = _replies_judge('{"choice": "A++"}')
        instance = _reward_instance(response_is_a=False)

        await WildBenchRewardScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text="Snow falls."), ScoringContext()
        )

        assert all("<|begin_of_response_B|>\n\nSnow falls.\n\n" in c for c in calls)
        verdicts = instance.metadata["judge_result"]["wildbench_reward_judge"]
        assert {v["reward"] for v in verdicts.values()} == {-1.0}

    @pytest.mark.anyio
    async def test_empty_and_cut_pairs_are_ties_without_a_judge_call(self):
        judge, calls = _replies_judge('{"choice": "A++"}')
        long_reference = " ".join(["word"] * 1200)
        instance = _reward_instance(haiku="   ", llama=long_reference)

        await WildBenchRewardScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text="Snow falls."), ScoringContext()
        )

        verdicts = instance.metadata["judge_result"]["wildbench_reward_judge"]
        assert len(calls) == 1
        assert verdicts["haiku"] == {
            "choice": None,
            "skipped": "empty",
            "reward": 0.0,
            "reward_k500": 0.0,
        }
        assert verdicts["llama"]["skipped"] == "truncated"
        assert verdicts["gpt4t"]["reward"] == 1.0

    @pytest.mark.anyio
    @pytest.mark.parametrize("text", ["", "<pad></s>"])
    async def test_an_empty_response_ties_every_reference(self, text):
        judge, calls = _replies_judge('{"choice": "A++"}')
        instance = _reward_instance()

        score = await WildBenchRewardScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text=text), ScoringContext()
        )

        assert calls == []
        assert score == 0.5

    @pytest.mark.anyio
    @pytest.mark.parametrize("words,judge_calls", [(1000, 3), (1001, 0)])
    async def test_a_response_over_1000_words_ties_every_reference(self, words, judge_calls):
        judge, calls = _replies_judge('{"choice": "A=B"}')
        instance = _reward_instance()

        await WildBenchRewardScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text=" ".join(["word"] * words)), ScoringContext()
        )

        assert len(calls) == judge_calls
        verdicts = instance.metadata["judge_result"]["wildbench_reward_judge"]
        assert {v["skipped"] for v in verdicts.values()} == {None if judge_calls else "truncated"}

    @pytest.mark.anyio
    async def test_no_choice_from_one_reference_leaves_the_output_unscored(self):
        calls: list[str] = []

        async def judge(prompt: str) -> str:
            calls.append(prompt)
            return "?" if "Haiku answer." in prompt else '{"choice": "A+"}'

        instance = _reward_instance()

        with pytest.raises(JudgeParseError, match="no choice"):
            await WildBenchRewardScorer(judge_fn=judge).ascore_with_context(
                instance, LMOutput(text="Snow falls."), ScoringContext()
            )

        assert sum("Haiku answer." in c for c in calls) == 3
        assert instance.metadata["judge_result"] == {"wildbench_reward_judge": None}
        assert instance.metadata["judge_raw_response"]["wildbench_reward_judge"] == {
            "gpt4t": '{"choice": "A+"}',
            "haiku": "?",
            "llama": '{"choice": "A+"}',
        }

    @pytest.mark.anyio
    async def test_length_penalty_uses_the_cut_responses(self):
        judge, _ = _replies_judge('{"choice": "A+"}')
        instance = _reward_instance(gpt4t="short", haiku="short", llama="short")

        await WildBenchRewardScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text="x" * 600), ScoringContext()
        )

        verdict = instance.metadata["judge_result"]["wildbench_reward_judge"]["gpt4t"]
        assert (verdict["reward"], verdict["reward_k500"]) == (0.5, 0.0)


class TestMetrics:
    def test_wb_score_counts_empty_responses_but_categories_leave_them_out(self):
        responses = [_scored(9), _scored(7), _scored(1, empty=True)]

        assert WBScoreMetric().compute(responses) == pytest.approx((17 / 3 - 5) * 20)
        creative = WBScoreMetric(name="task__creative__wb_score")
        assert creative.compute(responses) == pytest.approx((8 - 5) * 20)

    def test_categories_hold_every_conversation_tagged_with_them(self):
        responses = [_scored(9, task=("creative", "coding")), _scored(5, task=("coding",))]

        coding = WBScoreMetric(name="task__coding__wb_score")
        assert coding.compute(responses) == pytest.approx((7 - 5) * 20)
        assert coding.compute_instance(_scored(9, task=("math_data",))) is None

    def test_task_macro_divides_by_every_weight(self):
        responses = [_scored(10, task=("coding",)), _scored(5, task=("creative",))]

        # Coding 100 * 1.25 + creative 0 * 0.5, over the sum of all weights
        assert WBScoreTaskMacroMetric().compute(responses) == pytest.approx(125 / 4.75)
        assert WBScoreTaskMacroMetric().compute_instance(responses[0]) is None

    def test_rewards(self):
        responses = [
            _rewarded({"gpt4t": (1.0, 1.0), "haiku": (0.5, 0.0), "llama": (-0.5, -0.5)}),
            _rewarded({"gpt4t": (0.0, 0.0), "haiku": (0.5, 0.5), "llama": (1.0, 1.0)}),
        ]

        assert WBRewardMetric().compute(responses) == pytest.approx(100 * 2.5 / 6)
        mix_k500 = WBRewardMetric(name="any__any__reward_mix_k500")
        assert mix_k500.compute(responses) == pytest.approx(100 * 2.0 / 6)
        haiku = WBRewardMetric(name="any__any__reward_haiku")
        assert haiku.compute(responses) == 50.0
        assert WBRewardMetric(name="any__any__reward_gpt4t_k500").compute(responses) == 50.0

    def test_reward_task_macro(self):
        responses = [
            _rewarded({ref: (1.0, 1.0) for ref in REFERENCE_MODELS}, task=("math_data",)),
            _rewarded({ref: (-1.0, -1.0) for ref in REFERENCE_MODELS}, task=("info_advice",)),
        ]

        assert WBRewardTaskMacroMetric().compute(responses) == pytest.approx(
            (100 * 1.0 - 100 * 0.75) / 4.75
        )

    def test_reward_task_macro_leaves_skipped_pairs_out_of_each_reference(self):
        cut = _rewarded({ref: (0.0, 0.0) for ref in REFERENCE_MODELS}, task=("coding",))
        for verdict in cut.instance.metadata["judge_result"]["wildbench_reward_judge"].values():
            verdict.update(choice=None, skipped="truncated")
        partly_cut = _rewarded(
            {"gpt4t": (1.0, 1.0), "haiku": (0.0, 0.0), "llama": (0.5, 0.5)}, task=("coding",)
        )
        partly_cut.instance.metadata["judge_result"]["wildbench_reward_judge"]["haiku"].update(
            choice=None, skipped="empty"
        )

        # The skipped pairs still tie in the overall reward
        assert WBRewardMetric().compute([cut, partly_cut]) == pytest.approx(100 * 1.5 / 6)
        # gpt4t's coding mean is 100 and llama's 50; haiku has no coding verdict
        assert WBRewardTaskMacroMetric().compute([cut, partly_cut]) == pytest.approx(
            (1.25 * 100 / 4.75 + 0 + 1.25 * 50 / 4.75) / 3
        )

    @pytest.mark.anyio
    async def test_end_to_end(self, monkeypatch, judge_test_env):
        judge, _ = _replies_judge(['{"score": "9"}', '{"score": "5"}'])
        monkeypatch.setitem(wildbench._SCORE_JUDGE.__dict__, "judge_fn", judge)
        task = get_task("wildbench_score")
        responses = [
            Response(
                instance=task.process_doc(_doc(f"s{i}")),
                request=task.format_request(task.process_doc(_doc(f"s{i}"))),
                outputs=[LMOutput(text=f"Snow {i}.")],
            )
            for i in range(2)
        ]

        await task.score_responses(responses, ScoringContext())
        metrics = task.compute_metrics(responses)

        assert metrics["any__any__wb_score"]["wildbench_score_judge"] == pytest.approx(40.0)
        assert metrics["task__creative__wb_score"]["wildbench_score_judge"] == pytest.approx(40.0)
        assert metrics["any__any__wb_score_task_macro"]["wildbench_score_judge"] == pytest.approx(
            (40 * 0.5 + 40 * 1.25) / 4.75
        )
