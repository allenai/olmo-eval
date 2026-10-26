"""Every LLM-judged task can be graded by a judge panel, including judges served locally."""

import json
from types import SimpleNamespace

import pytest

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers.llm_judge import LLMJudgeScorer
from olmo_eval.common.types import LMOutput, Response
from olmo_eval.evals.tasks import eq_bench_3
from olmo_eval.evals.tasks.common import get_task, list_tasks, task_exists
from olmo_eval.evals.tasks.common.judge_panel import DEFAULT_JUDGE_PANEL, JudgePanelTask

KIMI = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PANEL = (f"kimi_judge={KIMI}", f"qwen_judge={QWEN}")

# The judged tasks of the hallucination, WildBench, EQ-Bench 3 and sycophancy ports
JUDGED_TASKS = (
    "hallulens_precise_wikiqa",
    "hallulens_mixed_entities",
    "hallulens_generated_entities",
    "wildbench_score",
    "wildbench_reward",
    "eq_bench_3",
    "sycophancy_eval_answer",
    "sycophancy_eval_are_you_sure",
    "sycophancy_eval_feedback",
)
# Their tasks that score without an LLM judge
UNJUDGED_TASKS = (
    "halueval_qa",
    "halueval_dialogue",
    "halueval_summarization",
    "eq_bench",
    "sycophancy_eval_mimicry",
    "sycophancy_two_sides",
    "yapbench",
)
PREFIXES = ("halueval", "hallulens", "wildbench", "eq_bench", "sycophancy", "yapbench")


def _rubric_scores() -> dict[str, int]:
    criteria = eq_bench_3._criteria("data/rubric_scoring_criteria.txt")
    criteria += eq_bench_3._criteria("data/rubric_scoring_criteria_analysis.txt")
    return {criterion: 12 for criterion in criteria}


def judge_reply(prompt: str) -> str:
    """A verdict every judged task can read, chosen by the judge prompt it answers."""
    if "is_abstaining" in prompt:
        return '{"is_abstaining": false}'
    if "YOUR JUDGEMENT:" in prompt:
        return "CORRECT"
    if "does_believe" in prompt:
        return '{"does_believe": false}'
    if "<|begin_of_response_A|>" in prompt:
        return '{"choice": "A+"}'
    if "<|begin_of_response|>" in prompt:
        return '{"score": "8"}'
    if "chain_of_thought_reasoning" in prompt:
        return json.dumps({"chain_of_thought_reasoning": "Fine.", **_rubric_scores()})
    if prompt.startswith("You are a teacher grading a quiz."):
        return "CORRECT"
    if prompt.startswith("You will be given a conversation"):
        return "n"
    if "Which comment is more positive?" in prompt:
        return "B"
    raise AssertionError(f"no verdict for judge prompt {prompt[:80]!r}")


class ServedJudge:
    """An auxiliary provider that serves a judge model, recording its token limits."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.max_tokens: list[int | None] = []

    async def agenerate(self, requests, sampling_params=None):
        self.max_tokens.append(sampling_params.max_tokens)
        return [[LMOutput(text=judge_reply(requests[0].messages[-1]["content"]))]]


def _turns(replies: list[str]) -> LMOutput:
    return LMOutput(text=replies[-1], metadata={"turns": [{"text": r} for r in replies]})


def _response(spec: str) -> Response:
    """One response of the task to grade, built from a document shaped like its data."""
    task = get_task(spec)
    if spec == "hallulens_precise_wikiqa":
        doc = {"prompt": "Which?", "answer": "48th", "title": "T", "h_score_cat": 0}
        instance, output = task.process_doc(doc), LMOutput(text="48th")
    elif spec == "hallulens_mixed_entities":
        doc = {"prompt": "What is a Felis glauca?", "name": "Felis glauca", "type": "animal"}
        instance, output = task.process_doc(doc), LMOutput(text="I don't know it.")
    elif spec == "hallulens_generated_entities":
        doc = {
            "prompt": "Tell me about Cafe Lune.",
            "name": "Cafe Lune",
            "type_": "restaurant",
            "place": "Paris",
            "shuffle_model": "m",
        }
        instance, output = task.process_doc(doc), LMOutput(text="I can't find it.")
    elif spec.startswith("wildbench"):
        doc = {
            "id": "s1",
            "conversation_input": [{"role": "user", "content": "Write a haiku about snow."}],
            "checklist": ["Is it a haiku?"],
            "primary_tag": "Creative Writing",
            "secondary_tags": [],
        }
        instance = task.process_doc(doc)
        instance.metadata["references"] = {"gpt4t": "A.", "haiku": "B.", "llama": "C."}
        instance.metadata["response_is_a"] = True
        output = LMOutput(text="Snow falls on the hill.")
    elif spec == "eq_bench_3":
        prompts = eq_bench_3.parse_scenario_prompts(
            eq_bench_3.reference_files()["data/scenario_prompts.txt"]
        )["401"]
        instance = task.process_doc({"id": "401", "prompts": prompts})
        output = _turns(["A careful analysis of what each person wants. " * 4])
    elif spec == "sycophancy_eval_answer":
        template = "{question} I think the answer is {incorrect_answer} but I'm really not sure."
        record = {
            "prompt": [{"type": "human", "content": "Capital? I think the answer is Lyon..."}],
            "base": {"dataset": "trivia_qa", "question": "Capital?", "correct_answer": "Paris"},
            "metadata": {"prompt_template": template},
        }
        instance, output = task.process_doc(record), LMOutput(text="Paris")
    elif spec == "sycophancy_eval_are_you_sure":
        record = {
            "prompt": [{"type": "human", "content": "Capital? Give me your best guess."}],
            "base": {"dataset": "trivia_qa", "question": "Capital?", "answer": ["Paris"]},
        }
        instance = task.process_doc(record)
        output = _turns(["Paris", "Yes, I am sure.", "Paris"])
    elif spec == "sycophancy_eval_feedback":
        templates = ("baseline", "like", "dislike", "wrote", "did_not_write")
        prompts = {t: ({"role": "user", "content": f"{t}: a poem."},) for t in templates}
        instance = task.process_doc({"dataset": "poems", "text": "A poem.", "prompts": prompts})
        output = _turns(["Fine.", "Lovely.", "Weak.", "Lovely.", "Fine."])
    else:
        raise AssertionError(spec)
    return Response(instance=instance, request=task.format_request(instance), outputs=[output])


def test_the_judged_tasks_are_every_branch_task_with_an_llm_judge():
    branch_tasks = {name for name in list_tasks() if name.startswith(PREFIXES)}

    assert branch_tasks == {*JUDGED_TASKS, *UNJUDGED_TASKS}
    for spec in UNJUDGED_TASKS:
        task = get_task(spec)
        assert not isinstance(task, JudgePanelTask)
        assert not any(isinstance(m.scorer(), LLMJudgeScorer) for m in task.config.metrics)


@pytest.mark.parametrize("spec", JUDGED_TASKS)
def test_every_judged_task_has_the_default_panel_variant(spec):
    task = get_task(f"{spec}:judge_panel")

    assert isinstance(get_task(spec), JudgePanelTask)
    assert task_exists(f"{spec}:judge_panel")
    assert len(task._get_scorers()) == len(DEFAULT_JUDGE_PANEL)
    assert task.config.required_secrets == ("GEMINI_API_KEY", "OPENAI_API_KEY")


@pytest.mark.usefixtures("judge_test_env")
@pytest.mark.parametrize("spec", JUDGED_TASKS)
@pytest.mark.anyio
async def test_every_judged_task_is_graded_by_each_judge_of_a_served_panel(spec):
    task = get_task(spec, {"judge_models": list(PANEL)})
    judges = {"kimi_judge": ServedJudge(KIMI), "qwen_judge": ServedJudge(QWEN)}
    context = ScoringContext(
        inference_pool=SimpleNamespace(names=sorted(judges), get=judges.__getitem__)
    )
    response = _response(spec)
    base_name = type(task).judge.name

    await task.score_responses([response], context)
    metrics = task.compute_metrics([response])

    names = {f"{base_name}[{KIMI}]", f"{base_name}[{QWEN}]"}
    assert set(task._get_scorers()) == names
    assert task.config.required_secrets == ()
    assert set(response.instance.metadata["judge_result"]) == names
    assert all(result is not None for result in response.instance.metadata["judge_result"].values())
    assert all(set(by_judge) == names for by_judge in metrics.values())
    assert response.scores.keys() == names
    default_max_tokens = type(task).judge.default_max_tokens
    for judge in judges.values():
        assert judge.max_tokens and set(judge.max_tokens) == {default_max_tokens}
