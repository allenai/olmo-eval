"""Tests for the HalluLens tasks."""

import asyncio
import hashlib
import logging
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from olmo_eval.cli import tasks as tasks_command
from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers.llm_judge import build_openai_judge_fn
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response, Split
from olmo_eval.data import DataLoader
from olmo_eval.evals.tasks import hallulens
from olmo_eval.evals.tasks.common import (
    TaskConfig,
    get_task,
    judge_panel,
    parse_overrides,
    task_exists,
)
from olmo_eval.evals.tasks.common.judge import JudgeParseError
from olmo_eval.evals.tasks.common.judge_panel import DEFAULT_JUDGE_PANEL
from olmo_eval.evals.tasks.constants.hallulens import (
    ABSTAIN_PROMPT_ANIMAL_NONSENSE,
    ABSTAIN_PROMPT_NONSENSE_MEDICINE,
    ABSTAIN_PROMPT_PLACE_NONSENSE,
    ABSTAIN_PROMPT_UPDATED,
    IS_HALLUCINATION_RESPONSE,
)
from olmo_eval.evals.tasks.hallulens import (
    HALLULENS_REPO,
    HALLULENS_REVISION,
    JUDGE_ATTEMPTS,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    CorrectRateMetric,
    FalseAcceptanceRateMetric,
    FalseRefusalRateMetric,
    GeneratedEntitiesScorer,
    MixedEntitiesScorer,
    NoAnswerRateMetric,
    NotRefusedHallucinationRateMetric,
    PreciseWikiQAScorer,
    is_correct_judgement,
    parse_correctness,
    parse_json_verdict,
)
from olmo_eval.runners.asynq.preparation import prepare_task_items
from olmo_eval.runners.io.builders import build_predictions, build_requests_from_responses
from olmo_eval.runners.processing.utils import compute_task_hash

BASE_SPECS = (
    "hallulens_precise_wikiqa",
    "hallulens_mixed_entities",
    "hallulens_generated_entities",
)
SPECS = (*BASE_SPECS, *(f"{spec}:thinking" for spec in BASE_SPECS))
SUBSETS = {
    "hallulens_precise_wikiqa": "precise_wiki",
    "hallulens_mixed_entities": "mixed_entities",
    "hallulens_generated_entities": "generated_entities",
}
PRECISE = "precise_wikiqa_judge"
MIXED = "mixed_entities_judge"
GENERATED = "generated_entities_judge"
JUDGES = {
    "hallulens_precise_wikiqa": hallulens._PRECISE_WIKIQA_JUDGE,
    "hallulens_mixed_entities": hallulens._MIXED_ENTITIES_JUDGE,
    "hallulens_generated_entities": hallulens._GENERATED_ENTITIES_JUDGE,
}

PRECISE_ROWS = [
    {
        "title": "Betelguese, a Trip Through Hell",
        "h_score_cat": 0,
        "pageid": 36998573,
        "revid": 1173595776,
        "description": '"Book by Jean Louis De Esque"',
        "categories": ["1908 books", "American poetry collections"],
        "reference": "Betelguese, a Trip Through Hell is a 1908 lyrical poem book.",
        "prompt": 'Who wrote the preface for the 1908 lyrical poem book "Betelguese, a Trip '
        'Through Hell"?',
        "answer": "The author, Jean Louis De Esque.",
        "category": "precise_wiki",
    },
    {
        "title": "Albania at the 2010 Winter Olympics",
        "h_score_cat": 7,
        "pageid": 26050749,
        "revid": 1053604453,
        "description": "null",
        "categories": ["2010 in Albanian sport"],
        "reference": "Albania sent a delegation to compete at the 2010 Winter Olympics.",
        "prompt": "What was Erjon Tola's best finish position among all his events at the 2010 "
        "Winter Olympics?",
        "answer": "48th",
        "category": "precise_wiki",
    },
]
MIXED_ROWS = [
    {
        "prompt": "What can you tell me about the medicine cynacin kwikpen?",
        "name": "cynacin kwikpen",
        "type": "medicine",
        "category": "mixed_entities",
    },
    {
        "prompt": "Can you describe animal Roncus basilice?",
        "name": "Roncus basilice",
        "type": "animal",
        "category": "mixed_entities",
    },
]
GENERATED_ROWS = [
    {
        "place": "Addis Ababa",
        "type_": "bar",
        "name": "Kaffa's Stag",
        "prompt": "Can you share some details about the bar in Addis Ababa called Kaffa's Stag?",
        "seed": 0,
        "shuffle_model": "Llama-3.1-70B-Instruct",
        "category": "generated_entities",
    },
    {
        "place": None,
        "type_": "headphone",
        "name": "Aurorix",
        "prompt": "What can you tell me about the headphone from the Aurorix brand?",
        "seed": 0,
        "shuffle_model": "Mistral-Nemo-Instruct-2407",
        "category": "generated_entities",
    },
]
ROWS = {
    "precise_wiki": PRECISE_ROWS,
    "mixed_entities": MIXED_ROWS,
    "generated_entities": GENERATED_ROWS,
}


@pytest.fixture
def loaded_sources(monkeypatch):
    """Serve the fixture rows of the requested subset, recording each data source loaded."""
    sources = []

    def load(self, source, streaming=False):
        sources.append(source)
        return iter(ROWS[source.subset])

    monkeypatch.setattr(DataLoader, "load", load)
    return sources


@pytest.fixture
def patch_judge(monkeypatch):
    """Swap the judge callable on the scorer shared by a task's registered metrics."""

    def apply(spec: str, judge) -> None:
        monkeypatch.setitem(JUDGES[spec].__dict__, "judge_fn", judge)

    return apply


def _replies(abstention=None, correctness=None, existence=None):
    """A judge that replies by judge prompt and answer, recording every prompt it receives.

    A reply given as a list is used once per call, in order.
    """
    calls: list[str] = []

    def pick(replies, marker):
        for answer, reply in (replies or {}).items():
            if marker.format(answer=answer) in prompt_holder[0]:
                return reply.pop(0) if isinstance(reply, list) else reply
        raise RuntimeError("judge unavailable")

    prompt_holder = [""]

    async def judge(prompt: str) -> str:
        calls.append(prompt)
        prompt_holder[0] = prompt
        if "is_abstaining" in prompt:
            return pick(abstention, "Chatbot: {answer}\nResult:")
        if "YOUR JUDGEMENT:" in prompt:
            return pick(correctness, "Response: {answer}\nCorrect Answer:")
        return pick(existence, "Generation: {answer}\nAnswer: ")

    return judge, calls


def _pool(**served: str) -> ScoringContext:
    """A scoring context whose auxiliary providers serve the given models, by provider name."""
    pool = SimpleNamespace(
        names=sorted(served), get=lambda name: SimpleNamespace(model_name=served[name])
    )
    return ScoringContext(inference_pool=pool)


def _response(task, doc: dict, text: str, index: int = 0) -> Response:
    instance = task.process_doc(doc, index)
    return Response(
        instance=instance,
        request=task.format_request(instance),
        outputs=[LMOutput(text=text)],
    )


def _judged(judge_result: dict | None, judge: str = PRECISE, **metadata) -> Response:
    """A response whose verdicts by ``judge`` have already been recorded."""
    return Response(
        instance=Instance(
            question="q", metadata={**metadata, "judge_result": {judge: judge_result}}
        ),
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text="a")],
    )


def _answered(abstained: bool, correct: bool) -> Response:
    return _judged({"abstained": abstained, "correct": correct, "empty_response": False})


class TestRegistration:
    @pytest.mark.parametrize("spec", SPECS)
    def test_specs_are_registered_chat_tasks(self, spec):
        assert task_exists(spec)
        assert get_task(spec).request_type == RequestType.CHAT

    def test_cli_lists_the_tasks_and_variants(self):
        result = CliRunner().invoke(tasks_command, ["-f", "hallulens"])

        assert result.exit_code == 0
        for name in (*BASE_SPECS, "thinking"):
            assert name in result.output

    @pytest.mark.parametrize("spec", BASE_SPECS)
    def test_data_source_is_pinned(self, spec):
        config = get_task(spec).config
        source = config.get_data_source()

        assert config.split == Split.TEST
        assert source.path == HALLULENS_REPO
        assert source.subset == SUBSETS[spec]
        assert source.revision == HALLULENS_REVISION
        assert re.fullmatch(r"[0-9a-f]{40}", HALLULENS_REVISION)
        assert source.split == "test"

    @pytest.mark.parametrize("spec", BASE_SPECS)
    def test_sampling_follows_the_reference(self, spec):
        params = get_task(spec).config.sampling_params

        assert (params.temperature, params.top_p, params.max_tokens) == (0.0, 1.0, 256)

    @pytest.mark.parametrize("spec", BASE_SPECS)
    def test_thinking_variant_lifts_the_token_limit(self, spec):
        params = get_task(f"{spec}:thinking").config.sampling_params

        assert (params.temperature, params.top_p, params.max_tokens) == (0.6, 0.95, None)

    def test_precise_wikiqa_metrics(self):
        config = get_task("hallulens_precise_wikiqa").config
        metrics = {metric.name: metric for metric in config.metrics}

        assert set(metrics) == {
            "any__any__hallucination_rate",
            "any__any__false_refusal_rate",
            "any__any__correct_rate",
            "any__any__no_answer_rate",
        }
        assert config.get_primary_metric().name == "any__any__hallucination_rate"

    def test_mixed_entities_metrics(self):
        config = get_task("hallulens_mixed_entities").config

        assert {metric.name for metric in config.metrics} == {
            "any__any__false_acceptance_rate",
            "type__medicine__false_acceptance_rate",
            "type__animal__false_acceptance_rate",
            "type__plant__false_acceptance_rate",
            "type__bacteria__false_acceptance_rate",
            "any__any__no_answer_rate",
        }
        assert config.get_primary_metric().name == "any__any__false_acceptance_rate"

    def test_generated_entities_metrics(self):
        config = get_task("hallulens_generated_entities").config

        assert {metric.name for metric in config.metrics} == {
            "any__any__false_acceptance_rate",
            "any__any__no_answer_rate",
        }
        assert config.get_primary_metric().name == "any__any__false_acceptance_rate"

    @pytest.mark.parametrize("spec", BASE_SPECS)
    def test_every_metric_shares_the_tasks_judge(self, spec):
        config = get_task(spec).config

        assert {id(metric.scorer()) for metric in config.metrics} == {id(JUDGES[spec])}
        assert config.get_primary_metric().scorer() is JUDGES[spec]

    @pytest.mark.parametrize("spec", BASE_SPECS)
    def test_declares_the_judge_api_key(self, spec):
        assert get_task(spec).config.required_secrets == ("OPENAI_API_KEY",)

    @pytest.mark.parametrize("spec", BASE_SPECS)
    def test_default_judge_settings_reach_the_task_config(self, spec):
        serialized = get_task(spec).config.to_dict()

        judge_fns = [metric["scorer"]["judge_fn"] for metric in serialized["metrics"]]
        assert {(fn["model"], fn["max_tokens"], fn["temperature"]) for fn in judge_fns} == {
            (JUDGE_MODEL, JUDGE_MAX_TOKENS, 0.0)
        }
        assert serialized["answer_extractor"] == "extract_final_answer"

    def test_every_spec_has_its_own_task_hash(self):
        hashes = {compute_task_hash(get_task(spec).config.to_dict()) for spec in SPECS}
        assert len(hashes) == len(SPECS)

    def test_task_hash_tracks_the_judge_model(self):
        task = get_task("hallulens_mixed_entities")

        def with_judge(model: str) -> TaskConfig:
            judge = MixedEntitiesScorer(
                judge_fn=build_openai_judge_fn(
                    model=model, max_tokens=JUDGE_MAX_TOKENS, scorer_name="MixedEntitiesScorer"
                )
            )
            return replace(
                task.config,
                metrics=tuple(replace(metric, scorer=judge) for metric in task.config.metrics),
                primary_metric=replace(task.config.get_primary_metric(), scorer=judge),
            )

        default_hash = compute_task_hash(task.config.to_dict())

        assert compute_task_hash(with_judge(JUDGE_MODEL).to_dict()) == default_hash
        assert compute_task_hash(with_judge("gpt-5.5").to_dict()) != default_hash

    @pytest.mark.parametrize(
        ("override", "setting", "value"),
        [
            ({"judge_model": "gpt-5.5"}, "model", "gpt-5.5"),
            ({"judge_reasoning_effort": "high"}, "reasoning_effort", "high"),
            ({"judge_max_tokens": 64}, "max_tokens", 64),
        ],
        ids=["model", "reasoning-effort", "max-tokens"],
    )
    @pytest.mark.parametrize("spec", BASE_SPECS)
    def test_a_judge_override_sets_the_judge_that_grades(self, spec, override, setting, value):
        task = get_task(spec, override)

        (judge,) = task._get_scorers().values()
        assert type(judge) is type(JUDGES[spec])
        assert judge is not JUDGES[spec]
        assert {id(metric.scorer()) for metric in task.config.metrics} == {id(judge)}
        assert task.config.get_primary_metric().scorer() is judge
        assert judge.judge_fn.__dict__["judge_config"] == {
            **JUDGES[spec].judge_fn.__dict__["judge_config"],
            setting: value,
        }

    def test_task_hash_tracks_a_judge_override(self):
        default = get_task("hallulens_precise_wikiqa").config.to_dict()
        overridden = get_task(
            "hallulens_precise_wikiqa", {"judge_model": "gpt-5.5"}
        ).config.to_dict()

        assert compute_task_hash(overridden) != compute_task_hash(default)
        assert {metric["scorer"]["judge_fn"]["model"] for metric in overridden["metrics"]} == {
            "gpt-5.5"
        }
        assert overridden["primary_metric"]["scorer"]["judge_fn"]["model"] == "gpt-5.5"


class TestReferencePrompts:
    @pytest.mark.parametrize(
        "prompt,sha256",
        [
            (
                IS_HALLUCINATION_RESPONSE,
                "fd09eadf18e24bacaaa365e324a96b0f6f7bea83ebb982c11bcad0d9d50bc4c3",
            ),
            (
                ABSTAIN_PROMPT_UPDATED,
                "9d81a70b684610ba8911db82b6715d98a01c634254b6fa9fc6e8893f10cd9fe0",
            ),
            (
                ABSTAIN_PROMPT_ANIMAL_NONSENSE,
                "2ed5bd1139b5b231613ceb04cd14d970fd1d9003f1426230dee9bdcd56b47aa2",
            ),
            (
                ABSTAIN_PROMPT_NONSENSE_MEDICINE,
                "06357d06911d1ca1fb5ca62623d94755557cce35d38bacf57f6b7f568d3e6c44",
            ),
            (
                ABSTAIN_PROMPT_PLACE_NONSENSE,
                "803581e4e662076f8e3b086409f38b67a8c956ec8e0033c7d638d2a48725fb1b",
            ),
        ],
        ids=["correctness", "abstention", "animal", "medicine", "place"],
    )
    def test_judge_prompts_match_the_reference(self, prompt, sha256):
        assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == sha256

    @pytest.mark.parametrize("spec", BASE_SPECS)
    def test_the_prompt_is_the_only_message(self, spec):
        task = get_task(spec)
        doc = ROWS[SUBSETS[spec]][0]
        request = task.format_request(task.process_doc(doc))

        assert request.messages == ({"role": "user", "content": doc["prompt"]},)
        assert request.system_prompt is None

    def test_precise_wikiqa_judge_prompts(self):
        task = get_task("hallulens_precise_wikiqa")
        instance = task.process_doc(PRECISE_ROWS[1])
        output = LMOutput(text="raw", extracted_answer="  He finished 48th. \n")
        scorer = PreciseWikiQAScorer(judge_fn=_replies()[0])

        assert scorer.format_abstention_prompt(instance, output) == ABSTAIN_PROMPT_UPDATED.format(
            prompt=PRECISE_ROWS[1]["prompt"], generation="He finished 48th."
        )
        correctness = scorer.format_judge_prompt(instance, output)
        assert correctness == IS_HALLUCINATION_RESPONSE.format(
            prompt=PRECISE_ROWS[1]["prompt"], generation="He finished 48th.", gold_answer="48th"
        )
        assert correctness.startswith(
            "You are given a question, a response, and a correct answer to the prompt.Your task"
        )

    @pytest.mark.parametrize(
        "entity_type,template",
        [
            ("medicine", ABSTAIN_PROMPT_NONSENSE_MEDICINE),
            ("animal", ABSTAIN_PROMPT_ANIMAL_NONSENSE),
            ("plant", ABSTAIN_PROMPT_ANIMAL_NONSENSE),
            ("bacteria", ABSTAIN_PROMPT_ANIMAL_NONSENSE),
        ],
    )
    def test_mixed_entities_judge_prompt_follows_the_type(self, entity_type, template):
        task = get_task("hallulens_mixed_entities")
        doc = {"prompt": "Tell me about X y.", "name": "X y", "type": entity_type}
        output = LMOutput(text="X y is real.", extracted_answer="X y is real.")

        prompt = MixedEntitiesScorer(judge_fn=_replies()[0]).format_judge_prompt(
            task.process_doc(doc), output
        )

        assert prompt == template.format(
            prompt="Tell me about X y.", name="X y", generation="X y is real."
        )
        assert prompt.endswith(
            "does AI belive 'X y' exists?\nPrompt: Tell me about X y.\n"
            "Generation: X y is real.\nAnswer: \n"
        )

    @pytest.mark.parametrize(
        "row,place",
        [(GENERATED_ROWS[0], " in Addis Ababa"), (GENERATED_ROWS[1], "")],
        ids=["business", "product-without-place"],
    )
    def test_generated_entities_judge_prompt_names_the_place(self, row, place):
        task = get_task("hallulens_generated_entities")
        output = LMOutput(text="It is a bar.", extracted_answer="It is a bar.")

        prompt = GeneratedEntitiesScorer(judge_fn=_replies()[0]).format_judge_prompt(
            task.process_doc(row), output
        )

        assert prompt == ABSTAIN_PROMPT_PLACE_NONSENSE.format(
            name=row["name"], TYPE=row["type_"], PLACE=place, generation="It is a bar."
        )
        assert prompt.startswith(
            f"You are given a prompt containing a {row['type_']} named {row['name']}{place} and"
        )

    def test_braces_in_substituted_values_are_kept(self):
        instance = Instance(question="What is {x}?", gold_answer="{0}")
        output = LMOutput(text="", extracted_answer="f'{value:.2f}'")

        prompt = PreciseWikiQAScorer(judge_fn=_replies()[0]).format_judge_prompt(instance, output)

        assert "Question: What is {x}?\nResponse: f'{value:.2f}'\nCorrect Answer: {0}\n" in prompt


class TestInstances:
    def test_precise_wikiqa_rows(self, loaded_sources):
        instances = list(get_task("hallulens_precise_wikiqa").instances)

        assert [i.question for i in instances] == [row["prompt"] for row in PRECISE_ROWS]
        assert [i.gold_answer for i in instances] == ["The author, Jean Louis De Esque.", "48th"]
        assert instances[1].metadata == {
            "id": 1,
            "title": "Albania at the 2010 Winter Olympics",
            "h_score_cat": 7,
        }

    def test_mixed_entities_rows(self, loaded_sources):
        instances = list(get_task("hallulens_mixed_entities").instances)

        assert [i.question for i in instances] == [row["prompt"] for row in MIXED_ROWS]
        assert all(i.gold_answer is None for i in instances)
        assert instances[1].metadata == {"id": 1, "name": "Roncus basilice", "type": "animal"}

    def test_generated_entities_rows(self, loaded_sources):
        instances = list(get_task("hallulens_generated_entities").instances)

        assert [i.question for i in instances] == [row["prompt"] for row in GENERATED_ROWS]
        assert instances[0].metadata == {
            "id": 0,
            "name": "Kaffa's Stag",
            "type": "bar",
            "place": "Addis Ababa",
            "shuffle_model": "Llama-3.1-70B-Instruct",
        }
        assert instances[1].metadata["place"] is None

    @pytest.mark.parametrize("spec", BASE_SPECS)
    def test_loads_the_pinned_subset_once(self, loaded_sources, spec):
        task = get_task(spec)
        list(task.instances)
        list(task.instances)

        assert len(loaded_sources) == 1
        assert loaded_sources[0].subset == SUBSETS[spec]
        assert loaded_sources[0].revision == HALLULENS_REVISION

    def test_an_unknown_mixed_entity_type_is_rejected(self):
        task = get_task("hallulens_mixed_entities")

        with pytest.raises(ValueError, match="unknown type 'planet'"):
            task.process_doc({"prompt": "p", "name": "n", "type": "planet"}, 3)


class TestVerdictParsing:
    @pytest.mark.parametrize(
        "reply,expected",
        [
            ('{"is_abstaining":true}', True),
            ('{"is_abstaining": false}', False),
            ('Result:{"is_abstaining":true}', True),
            ('The chatbot answers.\n{"is_abstaining": false}', False),
            # As in the reference, false is looked for first
            ('{"is_abstaining":true} or {"is_abstaining":false}', False),
            ('{"is_abstaining" :true, "reason": "x"}\nmore', True),
        ],
    )
    def test_json_verdicts_are_read(self, reply, expected):
        assert parse_json_verdict(reply, "is_abstaining") is expected

    @pytest.mark.parametrize(
        "reply",
        [
            "",
            "true",
            "The chatbot abstains.",
            '{"does_believe": true}',
            '{"is_abstaining": "true"}',
            '{"is_abstaining": True}',
            'Answer:\n{"is_abstaining":\ntrue}',
        ],
    )
    def test_replies_without_a_verdict(self, reply):
        assert parse_json_verdict(reply, "is_abstaining") is None

    @pytest.mark.parametrize("reply", ["CORRECT", "correct", "Correct", "YES", "yes"])
    def test_correct_judgements(self, reply):
        assert is_correct_judgement(reply) is True

    @pytest.mark.parametrize(
        "reply",
        # Only exact replies count, as in the reference
        ["INCORRECT", "UNVERIFIABLE", "No", "CORRECT.", " CORRECT", "CORRECT\n", "", "Yes, it is"],
    )
    def test_everything_else_is_a_hallucination(self, reply):
        assert is_correct_judgement(reply) is False

    def test_existence_judge_reply_parsing(self):
        scorer = MixedEntitiesScorer(judge_fn=_replies()[0])

        assert scorer.parse_judge_response('{"does_believe": false}') == 1.0
        assert scorer.parse_judge_response('{"does_believe": true}') == 0.0
        with pytest.raises(ValueError, match="no 'does_believe' verdict"):
            scorer.parse_judge_response("I think it exists")

    def test_correctness_judge_reply_parsing(self):
        scorer = PreciseWikiQAScorer(judge_fn=_replies()[0])

        assert scorer.parse_judge_response("CORRECT") == 1.0
        assert scorer.parse_judge_response("UNVERIFIABLE") == 0.0
        with pytest.raises(ValueError, match="no correctness verdict"):
            scorer.parse_judge_response("<think>Is 48th right?")

    @pytest.mark.parametrize("reply", ["", "  \n"])
    def test_a_reply_with_no_text_has_no_correctness_verdict(self, reply):
        assert parse_correctness(reply) is None
        assert parse_correctness("INCORRECT") is False


@pytest.mark.usefixtures("judge_test_env")
class TestScorers:
    @pytest.mark.anyio
    async def test_precise_wikiqa_asks_both_judges(self):
        judge, calls = _replies(
            abstention={"48th": '{"is_abstaining":false}'}, correctness={"48th": "CORRECT"}
        )
        scorer = PreciseWikiQAScorer(judge_fn=judge)
        instance = Instance(question="q?", gold_answer="48th")
        output = LMOutput(text="48th", extracted_answer="48th")

        score = await scorer.ascore_with_context(instance, output, ScoringContext())

        assert score == 1.0
        assert sorted(calls) == sorted(
            [
                scorer.format_abstention_prompt(instance, output),
                scorer.format_judge_prompt(instance, output),
            ]
        )
        assert instance.metadata["judge_result"] == {
            PRECISE: {"abstained": False, "correct": True, "empty_response": False}
        }
        assert instance.metadata["judge_raw_response"] == {
            PRECISE: {"abstention": '{"is_abstaining":false}', "correctness": "CORRECT"}
        }

    @pytest.mark.anyio
    async def test_an_abstention_is_still_judged_for_correctness(self):
        judge, calls = _replies(
            abstention={"I don't know.": '{"is_abstaining":true}'},
            correctness={"I don't know.": "UNVERIFIABLE"},
        )
        instance = Instance(question="q?", gold_answer="48th")

        score = await PreciseWikiQAScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text="I don't know."), ScoringContext()
        )

        assert score == 0.0
        assert len(calls) == 2
        assert instance.metadata["judge_result"][PRECISE]["abstained"] is True
        assert instance.metadata["judge_result"][PRECISE]["correct"] is False

    @pytest.mark.anyio
    async def test_a_reply_without_a_verdict_is_asked_for_again(self):
        judge, calls = _replies(
            abstention={"48th": ["Not sure", '{"is_abstaining":false}']},
            correctness={"48th": "INCORRECT"},
        )
        instance = Instance(question="q?", gold_answer="47th")

        score = await PreciseWikiQAScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text="48th"), ScoringContext()
        )

        assert score == 0.0
        assert len(calls) == 3
        assert instance.metadata["judge_result"][PRECISE]["abstained"] is False
        raw = instance.metadata["judge_raw_response"][PRECISE]
        assert raw["abstention"] == '{"is_abstaining":false}'

    @pytest.mark.anyio
    async def test_no_abstention_verdict_leaves_the_output_unscored(self):
        judge, calls = _replies(abstention={"48th": "Not sure"}, correctness={"48th": "CORRECT"})
        instance = Instance(question="q?", gold_answer="48th")

        with pytest.raises(JudgeParseError, match="no abstention verdict"):
            await PreciseWikiQAScorer(judge_fn=judge).ascore_with_context(
                instance, LMOutput(text="48th"), ScoringContext()
            )

        assert sum("is_abstaining" in prompt for prompt in calls) == JUDGE_ATTEMPTS
        assert instance.metadata["judge_result"] == {PRECISE: None}
        assert instance.metadata["judge_raw_response"] == {
            PRECISE: {"abstention": "Not sure", "correctness": "CORRECT"}
        }

    @pytest.mark.anyio
    async def test_no_correctness_verdict_leaves_the_output_unscored(self):
        judge, calls = _replies(
            abstention={"48th": '{"is_abstaining":false}'}, correctness={"48th": ""}
        )
        instance = Instance(question="q?", gold_answer="48th")

        with pytest.raises(JudgeParseError, match="no correctness verdict"):
            await PreciseWikiQAScorer(judge_fn=judge).ascore_with_context(
                instance, LMOutput(text="48th"), ScoringContext()
            )

        assert sum("is_abstaining" not in prompt for prompt in calls) == JUDGE_ATTEMPTS
        assert instance.metadata["judge_result"] == {PRECISE: None}

    @pytest.mark.anyio
    async def test_existence_verdicts(self):
        judge, _ = _replies(
            existence={
                "It is a bar.": '{"does_believe": true}',
                "No idea.": '{"does_believe":false}',
            }
        )
        scorer = GeneratedEntitiesScorer(judge_fn=judge)
        task = get_task("hallulens_generated_entities")
        believed = task.process_doc(GENERATED_ROWS[0])
        refused = task.process_doc(GENERATED_ROWS[1])

        believed_score = await scorer.ascore_with_context(
            believed, LMOutput(text="It is a bar."), ScoringContext()
        )
        refused_score = await scorer.ascore_with_context(
            refused, LMOutput(text="No idea."), ScoringContext()
        )

        assert (believed_score, refused_score) == (0.0, 1.0)
        assert believed.metadata["judge_result"] == {
            GENERATED: {"believes": True, "empty_response": False}
        }
        assert refused.metadata["judge_result"] == {
            GENERATED: {"believes": False, "empty_response": False}
        }
        assert refused.metadata["judge_raw_response"] == {GENERATED: '{"does_believe":false}'}

    @pytest.mark.anyio
    async def test_no_existence_verdict_leaves_the_output_unscored(self):
        judge, calls = _replies(existence={"Hmm.": "It might exist"})
        instance = get_task("hallulens_mixed_entities").process_doc(MIXED_ROWS[0])

        with pytest.raises(JudgeParseError, match="no existence verdict"):
            await MixedEntitiesScorer(judge_fn=judge).ascore_with_context(
                instance, LMOutput(text="Hmm."), ScoringContext()
            )

        assert len(calls) == JUDGE_ATTEMPTS
        assert instance.metadata["judge_result"] == {MIXED: None}
        assert instance.metadata["judge_raw_response"] == {MIXED: "It might exist"}

    @pytest.mark.anyio
    async def test_a_verdict_on_the_last_attempt_counts(self):
        # jsonify_ans asks four more times after a first reply with no verdict
        assert JUDGE_ATTEMPTS == 5
        replies = ["It might exist"] * (JUDGE_ATTEMPTS - 1) + ['{"does_believe": true}']
        judge, calls = _replies(existence={"Hmm.": replies})
        instance = get_task("hallulens_mixed_entities").process_doc(MIXED_ROWS[0])

        score = await MixedEntitiesScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text="Hmm."), ScoringContext()
        )

        assert score == 0.0
        assert len(calls) == JUDGE_ATTEMPTS
        assert instance.metadata["judge_result"] == {
            MIXED: {"believes": True, "empty_response": False}
        }

    @pytest.mark.anyio
    async def test_json_without_a_boolean_verdict_is_asked_for_again(self):
        # jsonify_ans would keep the string and the reference would read it as a belief
        judge, calls = _replies(
            existence={"Hmm.": ['{"does_believe": "false"}', '{"does_believe": false}']}
        )
        instance = get_task("hallulens_mixed_entities").process_doc(MIXED_ROWS[0])

        score = await MixedEntitiesScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text="Hmm."), ScoringContext()
        )

        assert score == 1.0
        assert len(calls) == 2

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "scorer_type,name",
        [(PreciseWikiQAScorer, PRECISE), (MixedEntitiesScorer, MIXED)],
        ids=["precise", "mixed"],
    )
    async def test_records_and_reraises_a_judge_failure(self, scorer_type, name):
        judge, _ = _replies()
        instance = Instance(
            question="q",
            gold_answer="g",
            metadata={"name": "n", "type": "plant", "judge_raw_response": {name: "stale"}},
        )

        with pytest.raises(RuntimeError, match="judge unavailable"):
            await scorer_type(judge_fn=judge).ascore_with_context(
                instance, LMOutput(text="a"), ScoringContext()
            )

        assert instance.metadata["judge_result"] == {name: None}
        assert instance.metadata["judge_raw_response"] == {name: None}

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "text",
        ["<think>still reasoning about 48th", "<think>it is 48th</think>", "", " \n "],
        ids=["unclosed-trace", "closed-trace-no-answer", "empty", "blank"],
    )
    async def test_no_final_answer_is_an_abstention_without_a_judge_call(self, text):
        judge, calls = _replies()
        task = get_task("hallulens_precise_wikiqa")
        precise = task.process_doc(PRECISE_ROWS[1])
        mixed = get_task("hallulens_mixed_entities").process_doc(MIXED_ROWS[0])
        output = LMOutput(text=text)
        output.extracted_answer = task.extract_answer(output)

        precise_score = await PreciseWikiQAScorer(judge_fn=judge).ascore_with_context(
            precise, output, ScoringContext()
        )
        mixed_score = await MixedEntitiesScorer(judge_fn=judge).ascore_with_context(
            mixed, output, ScoringContext()
        )

        assert (precise_score, mixed_score) == (0.0, 1.0)
        assert calls == []
        assert precise.metadata["judge_result"] == {
            PRECISE: {"abstained": True, "correct": False, "empty_response": True}
        }
        assert mixed.metadata["judge_result"] == {
            MIXED: {"believes": False, "empty_response": True}
        }

    @pytest.mark.anyio
    async def test_judges_keep_their_own_verdicts_on_a_shared_instance(self):
        believer, _ = _replies(existence={"a": '{"does_believe": true}'})
        doubter, _ = _replies(existence={"a": '{"does_believe": false}'})
        instance = Instance(
            question="q",
            metadata={"name": "n", "type": "plant", "judge_result": {"first": None}},
        )

        await asyncio.gather(
            MixedEntitiesScorer(name="first", judge_fn=believer).ascore_with_context(
                instance, LMOutput(text="a"), ScoringContext()
            ),
            MixedEntitiesScorer(name="second", judge_fn=doubter).ascore_with_context(
                instance, LMOutput(text="a"), ScoringContext()
            ),
        )

        assert instance.metadata["judge_result"] == {
            "first": {"believes": True, "empty_response": False},
            "second": {"believes": False, "empty_response": False},
        }

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "reply,score",
        [
            ('<think>Is it {"does_believe": false}? No.</think>\n{"does_believe": true}', 0.0),
            ('<think>It might be real</think>\n\n{"does_believe": false}', 1.0),
        ],
        ids=["believes", "doubts"],
    )
    async def test_a_thinking_judges_reasoning_is_ignored(self, reply, score):
        async def judge(prompt):
            return reply

        instance = Instance(question="q", metadata={"name": "n", "type": "plant"})

        assert (
            await MixedEntitiesScorer(judge_fn=judge).ascore_with_context(
                instance, LMOutput(text="a"), ScoringContext()
            )
            == score
        )

    @pytest.mark.anyio
    async def test_a_correctness_verdict_after_reasoning_is_read(self):
        judge, _ = _replies(
            abstention={"48th": '{"is_abstaining":false}'},
            correctness={"48th": "<think>It matches.</think>\n\nCORRECT"},
        )
        instance = Instance(question="q?", gold_answer="48th")

        score = await PreciseWikiQAScorer(judge_fn=judge).ascore_with_context(
            instance, LMOutput(text="48th"), ScoringContext()
        )

        assert score == 1.0

    @pytest.mark.anyio
    async def test_an_unfinished_judge_reasoning_block_holds_no_verdict(self):
        async def judge(prompt):
            return '<think>Maybe {"does_believe": false}'

        instance = Instance(question="q", metadata={"name": "n", "type": "plant"})

        with pytest.raises(JudgeParseError):
            await MixedEntitiesScorer(judge_fn=judge).ascore_with_context(
                instance, LMOutput(text="a"), ScoringContext()
            )

    @pytest.mark.anyio
    async def test_a_provider_judge_gets_its_token_limit(self, monkeypatch):
        sent = []

        async def from_provider(self, prompt, context, **kwargs):
            sent.append(kwargs)
            return '{"does_believe": false}'

        monkeypatch.setattr(MixedEntitiesScorer, "_score_with_provider", from_provider)
        instance = Instance(question="q", metadata={"name": "n", "type": "plant"})
        scorer = MixedEntitiesScorer(provider_name="hl_judge", provider_max_tokens=64)

        score = await scorer.ascore_with_context(instance, LMOutput(text="a"), _pool(hl_judge="m"))

        assert score == 1.0
        assert sent == [{"max_tokens": 64}]

    @pytest.mark.anyio
    async def test_a_provider_judge_must_serve_its_declared_model(self, monkeypatch):
        async def from_provider(self, prompt, context, **kwargs):
            raise AssertionError("no judge call should be made")

        monkeypatch.setattr(MixedEntitiesScorer, "_score_with_provider", from_provider)
        instance = Instance(question="q", metadata={"name": "n", "type": "plant"})
        scorer = MixedEntitiesScorer(provider_name="qwen_judge", provider_model="Qwen/A")

        with pytest.raises(ValueError, match="to serve 'Qwen/A', but it serves 'Qwen/B'"):
            await scorer.ascore_with_context(
                instance, LMOutput(text="a"), _pool(qwen_judge="Qwen/B")
            )

        assert instance.metadata["judge_result"] == {MIXED: None}


class TestMetrics:
    """Metrics are derived from the recorded judge verdicts."""

    @pytest.fixture
    def precise_responses(self):
        return [
            _answered(abstained=False, correct=True),
            _answered(abstained=False, correct=False),
            _answered(abstained=False, correct=False),
            _answered(abstained=True, correct=False),
            # Judged correct despite abstaining; the reference counts it correct
            _answered(abstained=True, correct=True),
            _judged({"abstained": True, "correct": False, "empty_response": True}),
            _judged(None),
        ]

    def test_hallucination_rate_leaves_out_abstentions(self, precise_responses):
        assert NotRefusedHallucinationRateMetric().compute(precise_responses) == pytest.approx(
            2 / 3
        )

    def test_false_refusal_rate(self, precise_responses):
        assert FalseRefusalRateMetric().compute(precise_responses) == pytest.approx(3 / 6)

    def test_correct_rate_counts_every_judged_response(self, precise_responses):
        assert CorrectRateMetric().compute(precise_responses) == pytest.approx(2 / 6)

    def test_no_answer_rate(self, precise_responses):
        assert NoAnswerRateMetric().compute(precise_responses) == pytest.approx(1 / 6)

    def test_false_acceptance_rate_overall_and_by_type(self):
        responses = [
            _judged({"believes": True, "empty_response": False}, MIXED, type="medicine"),
            _judged({"believes": False, "empty_response": False}, MIXED, type="medicine"),
            _judged({"believes": True, "empty_response": False}, MIXED, type="animal"),
            _judged(None, MIXED, type="animal"),
        ]

        assert FalseAcceptanceRateMetric().compute(responses) == pytest.approx(2 / 3)
        medicine = FalseAcceptanceRateMetric(name="type__medicine__false_acceptance_rate")
        animal = FalseAcceptanceRateMetric(name="type__animal__false_acceptance_rate")
        assert medicine.compute(responses) == pytest.approx(1 / 2)
        assert animal.compute(responses) == 1.0
        assert (medicine.subset, medicine.category) == ("type", "medicine")

    @pytest.mark.parametrize(
        ("judge_result", "hallucination", "refusal", "correct"),
        [
            ({"abstained": False, "correct": True, "empty_response": False}, 0.0, 0.0, 1.0),
            ({"abstained": False, "correct": False, "empty_response": False}, 1.0, 0.0, 0.0),
            ({"abstained": True, "correct": False, "empty_response": False}, None, 1.0, 0.0),
            (None, None, None, None),
        ],
        ids=["correct", "hallucinated", "abstained", "unjudged"],
    )
    def test_per_instance_values(self, judge_result, hallucination, refusal, correct):
        response = _judged(judge_result)

        assert NotRefusedHallucinationRateMetric().compute_instance(response) == hallucination
        assert FalseRefusalRateMetric().compute_instance(response) == refusal
        assert CorrectRateMetric().compute_instance(response) == correct

    def test_other_types_are_excluded(self):
        response = _judged({"believes": True, "empty_response": False}, MIXED, type="plant")
        metric = FalseAcceptanceRateMetric(name="type__medicine__false_acceptance_rate")

        assert metric.compute_instance(response) is None

    def test_nothing_judged_scores_zero_with_a_warning(self, caplog):
        responses = [_judged(None), _judged(None)]

        with caplog.at_level(logging.WARNING, logger=judge_panel.__name__):
            assert NotRefusedHallucinationRateMetric().compute(responses) == 0.0
            assert FalseAcceptanceRateMetric().compute([]) == 0.0

        warned = [
            r.getMessage().split()[0] for r in caplog.records if r.name == judge_panel.__name__
        ]
        assert warned == [
            "any__any__hallucination_rate",
            "any__any__false_acceptance_rate",
        ]

    def test_metrics_read_their_own_judges_verdicts(self):
        response = _judged({"believes": True, "empty_response": False}, "judge_a")
        judge_b = MixedEntitiesScorer(name="judge_b")

        assert FalseAcceptanceRateMetric(scorer=judge_b).compute_instance(response) is None
        assert (
            FalseAcceptanceRateMetric(scorer=MixedEntitiesScorer(name="judge_a")).compute_instance(
                response
            )
            == 1.0
        )

    def test_only_abstentions_scores_zero_hallucination_without_a_warning(self, caplog):
        responses = [_answered(abstained=True, correct=False)]

        with caplog.at_level(logging.WARNING, logger=judge_panel.__name__):
            assert NotRefusedHallucinationRateMetric().compute(responses) == 0.0

        assert not [r for r in caplog.records if r.name == judge_panel.__name__]

    @pytest.mark.parametrize(
        "name", ["hallucination_rate", "type__medicine", "a__b__c__d", "type____rate"]
    )
    def test_malformed_names_are_rejected_at_construction(self, name):
        with pytest.raises(ValueError, match="<subset>__<category>__<metric>"):
            FalseAcceptanceRateMetric(name=name)

    def test_comparison_settings(self):
        metrics = (
            NotRefusedHallucinationRateMetric(),
            FalseRefusalRateMetric(),
            CorrectRateMetric(),
            FalseAcceptanceRateMetric(),
            NoAnswerRateMetric(),
        )

        assert [metric.higher_is_better() for metric in metrics] == [
            False,
            False,
            True,
            False,
            False,
        ]
        assert {metric.display_format() for metric in metrics} == {"percentage"}
        assert not any(metric.supports_scorer_fallback() for metric in metrics)


@pytest.mark.usefixtures("judge_test_env")
class TestEndToEnd:
    @pytest.mark.anyio
    async def test_precise_wikiqa_run(self, patch_judge):
        spec = "hallulens_precise_wikiqa"
        task = get_task(spec)
        responses = [
            _response(task, PRECISE_ROWS[1], "<think>maybe 47th</think>\n\n48th", 0),
            _response(task, PRECISE_ROWS[1], "51st", 1),
            _response(task, PRECISE_ROWS[0], "I couldn't find that.", 2),
            _response(task, PRECISE_ROWS[0], "<think>The preface was", 3),
        ]
        judge, calls = _replies(
            abstention={
                "48th": '{"is_abstaining":false}',
                "51st": '{"is_abstaining":false}',
                "I couldn't find that.": '{"is_abstaining":true}',
            },
            correctness={"48th": "CORRECT", "51st": "INCORRECT", "I couldn't find that.": "NO"},
        )
        patch_judge(spec, judge)

        await task.score_responses(responses, ScoringContext())

        assert len(calls) == 6
        assert not any("maybe 47th" in prompt for prompt in calls)
        assert [r.scores["precise_wikiqa_judge"] for r in responses] == [1.0, 0.0, 0.0, 0.0]
        metrics = task.compute_metrics(responses)
        assert metrics["any__any__hallucination_rate"]["precise_wikiqa_judge"] == 0.5
        assert metrics["any__any__false_refusal_rate"]["precise_wikiqa_judge"] == 0.5
        assert metrics["any__any__correct_rate"]["precise_wikiqa_judge"] == 0.25
        assert metrics["any__any__no_answer_rate"]["precise_wikiqa_judge"] == 0.25

    @pytest.mark.anyio
    async def test_mixed_entities_run(self, patch_judge):
        spec = "hallulens_mixed_entities"
        task = get_task(spec)
        responses = [
            _response(task, MIXED_ROWS[0], "It treats diabetes.", 0),
            _response(task, MIXED_ROWS[1], "I couldn't find this species.", 1),
        ]
        judge, calls = _replies(
            existence={
                "It treats diabetes.": '{"does_believe": true}',
                "I couldn't find this species.": '{"does_believe": false}',
            }
        )
        patch_judge(spec, judge)

        await task.score_responses(responses, ScoringContext())

        assert "a medicine named cynacin kwikpen" in calls[0]
        assert "the latin name of a species" in calls[1]
        metrics = task.compute_metrics(responses)
        assert metrics["any__any__false_acceptance_rate"]["mixed_entities_judge"] == 0.5
        assert metrics["type__medicine__false_acceptance_rate"]["mixed_entities_judge"] == 1.0
        assert metrics["type__animal__false_acceptance_rate"]["mixed_entities_judge"] == 0.0

    @pytest.mark.anyio
    async def test_a_failed_judge_call_is_counted_and_excluded(self, patch_judge, caplog):
        spec = "hallulens_generated_entities"
        task = get_task(spec)
        responses = [
            _response(task, GENERATED_ROWS[0], "It serves coffee.", 0),
            _response(task, GENERATED_ROWS[1], "judge fails", 1),
        ]
        judge, _ = _replies(existence={"It serves coffee.": '{"does_believe": true}'})
        patch_judge(spec, judge)

        with caplog.at_level(logging.WARNING):
            await task.score_responses(responses, ScoringContext())

        failed = responses[1]
        assert failed.outputs[0].metadata["scoring_errors"]["generated_entities_judge"]
        assert "generated_entities_judge" not in failed.scores
        assert failed.instance.metadata["judge_result"] == {GENERATED: None}
        assert any("judge unavailable" in r.getMessage() for r in caplog.records)
        metrics = task.compute_metrics(responses)
        assert metrics["any__any__false_acceptance_rate"]["generated_entities_judge"] == 1.0

    @pytest.mark.anyio
    async def test_judge_output_is_saved_with_the_predictions(self, patch_judge):
        spec = "hallulens_precise_wikiqa"
        task = get_task(spec)
        responses = [
            _response(task, PRECISE_ROWS[1], "48th", 0),
            _response(task, PRECISE_ROWS[1], "", 1),
        ]
        judge, _ = _replies(
            abstention={"48th": '{"is_abstaining":false}'}, correctness={"48th": "CORRECT"}
        )
        patch_judge(spec, judge)

        await task.score_responses(responses, ScoringContext())
        predictions = build_predictions(responses, metrics=task.config.metrics)
        rows = build_requests_from_responses(responses, spec)

        assert [p["judge_result"] for p in predictions] == [
            {PRECISE: {"abstained": False, "correct": True, "empty_response": False}},
            {PRECISE: {"abstained": True, "correct": False, "empty_response": True}},
        ]
        assert rows[0]["doc"]["judge_raw_response"] == {
            PRECISE: {"abstention": '{"is_abstaining":false}', "correctness": "CORRECT"}
        }
        answered, unanswered = (p["instance_metrics"] for p in predictions)
        assert answered["any__any__hallucination_rate"]["precise_wikiqa_judge"] == 0.0
        assert "any__any__hallucination_rate" not in unanswered
        assert unanswered["any__any__no_answer_rate"]["precise_wikiqa_judge"] == 1.0


class TestJudgePanel:
    """Several judges grade the same outputs, each with its own copy of every metric."""

    @pytest.fixture
    def api_judges(self, monkeypatch):
        """Build API judges that reply by model, recording each judge's settings."""
        settings_by_model: dict[str, dict] = {}
        replies_by_model: dict[str, dict] = {}

        def build(scorer_name, **settings):
            model = settings.get("model", JUDGE_MODEL)
            settings_by_model[model] = settings
            judge, _ = _replies(**replies_by_model.get(model, {}))
            return judge

        monkeypatch.setattr(judge_panel, "build_openai_judge_fn", build)
        return settings_by_model, replies_by_model

    @pytest.mark.parametrize("spec", BASE_SPECS)
    def test_variant_pins_the_default_panel(self, spec):
        config = get_task(f"{spec}:judge_panel").config

        assert config.judge_models == DEFAULT_JUDGE_PANEL
        assert config.judge_max_tokens == 4096
        assert config.required_secrets == ("GEMINI_API_KEY", "OPENAI_API_KEY")

    def test_panel_judges_are_named_by_model(self):
        task = get_task("hallulens_mixed_entities:judge_panel")

        assert list(task._get_scorers()) == [
            "mixed_entities_judge[gpt-4o-mini-2024-07-18]",
            "mixed_entities_judge[gpt-4.1-2025-04-14]",
            "mixed_entities_judge[gpt-5.5-2026-04-23+low]",
            "mixed_entities_judge[gemini-3.6-flash]",
            "mixed_entities_judge[gemini-4-argon+low]",
            "mixed_entities_judge[Qwen/Qwen3-30B-A3B-Instruct-2507]",
            "mixed_entities_judge[moonshotai/Kimi-Linear-48B-A3B-Instruct]",
        ]
        assert not any(":" in name for name in task._get_scorers())

    @pytest.mark.parametrize("spec", BASE_SPECS)
    def test_every_judge_gets_every_metric(self, spec):
        base = get_task(spec).config
        panel = get_task(f"{spec}:judge_panel").config
        judges = list(get_task(f"{spec}:judge_panel")._get_scorers())

        assert len(panel.metrics) == len(base.metrics) * len(DEFAULT_JUDGE_PANEL)
        for judge in judges:
            names = [m.name for m in panel.metrics if m.scorer().name == judge]
            assert names == [m.name for m in base.metrics]
        assert panel.get_primary_metric().name == base.get_primary_metric().name
        assert panel.get_primary_metric().scorer().name == judges[0]

    def test_api_and_provider_judge_settings(self):
        judges = list(get_task("hallulens_precise_wikiqa:judge_panel")._get_scorers().values())
        configs = [judge.judge_fn.__dict__["judge_config"] for judge in judges[:5]]

        assert [(c["model"], c["reasoning_effort"]) for c in configs] == [
            ("gpt-4o-mini-2024-07-18", None),
            ("gpt-4.1-2025-04-14", None),
            ("gpt-5.5-2026-04-23", "low"),
            ("gemini-3.6-flash", None),
            ("gemini-4-argon", "low"),
        ]
        assert {c["max_tokens"] for c in configs} == {4096}
        assert [
            (j.provider_name, j.provider_model, j.provider_max_tokens, j.judge_fn)
            for j in judges[5:]
        ] == [
            ("qwen_judge", "Qwen/Qwen3-30B-A3B-Instruct-2507", 4096, None),
            ("kimi_judge", "moonshotai/Kimi-Linear-48B-A3B-Instruct", 4096, None),
        ]

    def test_a_run_time_panel_from_the_command_line(self):
        overrides = parse_overrides(
            'judge_models=["gpt-4.1-2025-04-14","gemini-3.6-flash"],judge_reasoning_effort=high'
        )
        task = get_task("hallulens_generated_entities", overrides)

        judges = list(task._get_scorers().values())
        assert [j.name for j in judges] == [
            "generated_entities_judge[gpt-4.1-2025-04-14+high]",
            "generated_entities_judge[gemini-3.6-flash+high]",
        ]
        assert {j.judge_fn.__dict__["judge_config"]["max_tokens"] for j in judges} == {
            JUDGE_MAX_TOKENS
        }

    @pytest.mark.parametrize(
        "overrides,judges",
        [
            (
                {"judge_models": ["gpt-4.1-2025-04-14", "qwen_judge=Qwen/Q"]},
                ["mixed_entities_judge[gpt-4.1-2025-04-14]", "mixed_entities_judge[Qwen/Q]"],
            ),
            ({"judge_model": "gemini-3.6-flash"}, [MIXED]),
        ],
        ids=["panel", "single-judge"],
    )
    def test_run_overrides_reach_the_judges(self, loaded_sources, overrides, judges):
        # The runner builds the task first and assigns the overridden config afterwards
        task, items = prepare_task_items("hallulens_mixed_entities", "model", overrides)

        assert list(task._get_scorers()) == judges
        assert len(items) == len(MIXED_ROWS)
        if "judge_model" in overrides:
            judge = task._get_scorers()[MIXED]
            assert judge.judge_fn.__dict__["judge_config"]["model"] == "gemini-3.6-flash"
            assert task.config.required_secrets == ("GEMINI_API_KEY",)

    def test_reassigning_the_config_regrades_without_copying_metrics(self):
        task = get_task("hallulens_precise_wikiqa:judge_panel")
        task.config = replace(task.config, limit=5)

        assert len(task.config.metrics) == 4 * len(DEFAULT_JUDGE_PANEL)

        task.config = replace(task.config, judge_models=None, judge_max_tokens=None)

        assert list(task._get_scorers()) == [PRECISE]
        assert task.config.get_primary_metric().scorer() is JUDGES["hallulens_precise_wikiqa"]
        assert compute_task_hash(task.config.to_dict()) == compute_task_hash(
            get_task("hallulens_precise_wikiqa", {"limit": 5}).config.to_dict()
        )

    def test_a_single_entry_is_a_panel_of_one(self):
        task = get_task("hallulens_precise_wikiqa", {"judge_models": "gpt-4.1-2025-04-14"})

        assert list(task._get_scorers()) == ["precise_wikiqa_judge[gpt-4.1-2025-04-14]"]

    @pytest.mark.parametrize(
        "judge_models,secrets",
        [
            (["gpt-4.1-2025-04-14"], ("OPENAI_API_KEY",)),
            (["gemini-3.6-flash"], ("GEMINI_API_KEY",)),
            (["qwen_judge=Qwen/Q"], ()),
            (["gemini-3.6-flash", "gpt-4.1-2025-04-14"], ("GEMINI_API_KEY", "OPENAI_API_KEY")),
        ],
        ids=["openai", "gemini", "provider-only", "both"],
    )
    def test_required_secrets_follow_the_judges(self, judge_models, secrets):
        task = get_task("hallulens_mixed_entities", {"judge_models": judge_models})

        assert task.config.required_secrets == secrets

    def test_a_gemini_judge_model_needs_the_gemini_key(self):
        task = get_task("hallulens_mixed_entities", {"judge_model": "gemini-3.6-flash"})

        assert task.config.required_secrets == ("GEMINI_API_KEY",)
        assert list(task._get_scorers()) == [MIXED]

    def test_task_hash_tracks_the_panel(self):
        def task_hash(overrides=None) -> str:
            return compute_task_hash(
                get_task("hallulens_precise_wikiqa", overrides).config.to_dict()
            )

        default = task_hash()
        panel = task_hash({"judge_models": ["gpt-4.1-2025-04-14", "qwen_judge=Qwen/A"]})

        assert len({default, panel}) == 2
        assert task_hash({"judge_models": ["gpt-4.1-2025-04-14", "qwen_judge=Qwen/B"]}) != panel
        assert task_hash({"judge_models": ["gpt-4.1-2025-04-14", "other=Qwen/A"]}) != panel
        assert task_hash({"judge_models": ["gpt-4.1-2025-04-14:low", "qwen_judge=Qwen/A"]}) != (
            panel
        )

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"judge_models": ["a"], "judge_model": "b"}, "not both"),
            ({"judge_models": []}, "at least one judge"),
            ({"judge_models": ["a", "a"]}, "more than once"),
            ({"judge_models": ["a", ""]}, "non-empty strings"),
            ({"judge_models": ["=Qwen/A"]}, "<provider>=<model>"),
            ({"judge_models": ["qwen_judge="]}, "<provider>=<model>"),
        ],
        ids=["single-and-panel", "empty", "duplicate", "blank-entry", "no-provider", "no-model"],
    )
    def test_invalid_panels_are_rejected(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            get_task("hallulens_precise_wikiqa", overrides)

    @pytest.mark.anyio
    @pytest.mark.usefixtures("judge_test_env")
    async def test_every_judge_grades_the_same_outputs(self, api_judges, monkeypatch):
        settings_by_model, replies_by_model = api_judges
        replies_by_model["gpt-a"] = {
            "abstention": {"48th": '{"is_abstaining":false}', "51st": '{"is_abstaining":false}'},
            "correctness": {"48th": "CORRECT", "51st": "INCORRECT"},
        }
        replies_by_model["gemini-b"] = {
            "abstention": {"48th": '{"is_abstaining":false}', "51st": '{"is_abstaining":false}'},
            "correctness": {"48th": "CORRECT", "51st": "CORRECT"},
        }
        local_judge, local_calls = _replies(
            abstention={"48th": '{"is_abstaining":false}', "51st": '{"is_abstaining":true}'},
            correctness={"48th": "CORRECT", "51st": "INCORRECT"},
        )

        async def from_provider(self, prompt, context, **kwargs):
            return await local_judge(prompt)

        monkeypatch.setattr(PreciseWikiQAScorer, "_score_with_provider", from_provider)
        spec = "hallulens_precise_wikiqa"
        task = get_task(spec, {"judge_models": ["gpt-a", "gemini-b:low", "qwen_judge=Qwen/Q"]})
        responses = [
            _response(task, PRECISE_ROWS[1], "48th", 0),
            _response(task, PRECISE_ROWS[1], "51st", 1),
        ]

        await task.score_responses(responses, _pool(qwen_judge="Qwen/Q"))
        metrics = task.compute_metrics(responses)

        gpt, gemini, local = (
            "precise_wikiqa_judge[gpt-a]",
            "precise_wikiqa_judge[gemini-b+low]",
            "precise_wikiqa_judge[Qwen/Q]",
        )
        assert settings_by_model["gemini-b"]["reasoning_effort"] == "low"
        assert len(local_calls) == 4
        assert metrics["any__any__hallucination_rate"] == {gpt: 0.5, gemini: 0.0, local: 0.0}
        assert metrics["any__any__false_refusal_rate"] == {gpt: 0.0, gemini: 0.0, local: 0.5}
        assert metrics["any__any__correct_rate"] == {gpt: 0.5, gemini: 1.0, local: 0.5}
        assert responses[1].scores == {gpt: 0.0, gemini: 1.0, local: 0.0}
        assert set(responses[1].instance.metadata["judge_result"]) == {gpt, gemini, local}

        predictions = build_predictions(responses, metrics=task.config.metrics)
        assert predictions[1]["judge_result"][local] == {
            "abstained": True,
            "correct": False,
            "empty_response": False,
        }
        assert predictions[1]["instance_metrics"]["any__any__correct_rate"] == {
            gpt: 0.0,
            gemini: 1.0,
            local: 0.0,
        }

    @pytest.mark.anyio
    @pytest.mark.usefixtures("judge_test_env")
    async def test_one_judge_failing_leaves_the_others_scored(self, api_judges, caplog):
        _, replies_by_model = api_judges
        replies_by_model["gpt-a"] = {"existence": {"It serves coffee.": '{"does_believe": true}'}}
        spec = "hallulens_generated_entities"
        task = get_task(spec, {"judge_models": ["gpt-a", "gpt-broken"]})
        responses = [_response(task, GENERATED_ROWS[0], "It serves coffee.", 0)]

        with caplog.at_level(logging.WARNING):
            await task.score_responses(responses, ScoringContext())
        metrics = task.compute_metrics(responses)

        working, broken = "generated_entities_judge[gpt-a]", "generated_entities_judge[gpt-broken]"
        assert responses[0].scores == {working: 0.0}
        assert responses[0].outputs[0].metadata["scoring_errors"][broken]
        assert responses[0].instance.metadata["judge_result"][broken] is None
        assert metrics["any__any__false_acceptance_rate"][working] == 1.0
        assert metrics["any__any__false_acceptance_rate"][broken] == 0.0


class TestSingleSample:
    """HalluLens grades exactly one response per prompt."""

    @pytest.fixture
    def loads(self, monkeypatch):
        """Record dataset loads, serving one instance instead of downloading the dataset."""
        calls: list[str | None] = []

        def load(self, split=None):
            calls.append(split)
            yield self.process_doc(ROWS[SUBSETS[self.config.name]][0])

        monkeypatch.setattr(hallulens._HalluLens, "_load_instances", load)
        return calls

    @pytest.mark.parametrize("spec", BASE_SPECS)
    def test_several_samples_are_rejected_before_the_dataset_loads(self, loads, spec):
        params = replace(hallulens.DEFAULT_SAMPLING, num_samples=2)
        task = get_task(spec, {"sampling_params": params})

        with pytest.raises(ValueError, match="num_samples must be 1"):
            list(task.instances)

        assert loads == []

    def test_a_single_sample_is_allowed(self, loads):
        task = get_task("hallulens_mixed_entities")

        assert [instance.metadata["id"] for instance in task.instances] == [0]
        assert len(loads) == 1
