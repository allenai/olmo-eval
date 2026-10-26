"""
WildBench Evaluation Tasks

WildBench evaluates chat models on 1,024 real user conversations from WildChat, each with a
checklist of what a good response should do. A judge grades the model's response to the
conversation's last turn in two ways:

- ``wildbench_score`` (WB-Score): the judge scores the response from 1 to 10.
- ``wildbench_reward`` (WB-Reward): the judge compares the response with each of three
  reference models' responses, from much better to much worse.

Paper: https://arxiv.org/abs/2406.04770
Reference: https://github.com/allenai/WildBench/tree/d6b8dcaf377d173d031980f97c16e1a82618c03d
Data: ``allenai/WildBench`` (config ``v2``) and the reference models' responses in
``allenai/WildBench-V2-Model-Outputs``, both pinned.

Prompting, judge prompts, judge settings and scoring follow the reference's ``src/eval.py``,
``src/unified_utils.py``, ``src/openai_batch_eval/batch_results_format.py`` and
``leaderboard/data_dir/_create_tables.py`` and ``_merge_results.py``. The model sees the
conversation with no system prompt, as the reference's vLLM inference sends it, and answers
greedily with up to 4096 tokens. Its response is cleaned as the reference's ``clear_output``
saves it: ``<|endoftext|>``, ``</s>``, ``<pad>`` and ``<end_of_turn>`` become spaces and the
text is stripped. The judge gets one user message at temperature 0 with up to 1024 tokens and
a JSON response format, as in the reference's batch evaluation.

WB-Score is ``(mean score - 5) * 20``, from -80 to 100, as the leaderboard shows it. Each task
category's score leaves empty responses out, and the task-macro score weights the categories
(Creative Tasks 0.5, Planning & Reasoning 1.25, Math & Data Analysis 1, Information/Advice
seeking 0.75, Coding & Debugging 1.25) and divides by the sum of all the weights. A
conversation counts in every category its tags map to.

WB-Reward counts a much better response as 100, slightly better as 50, a tie as 0, slightly
worse as -50 and much worse as -100. As in the reference, the history, the query, the
response and the reference's response are cut to 1,000 words for this judge (the WB-Score
judge sees them whole), the order of the two responses is drawn once per conversation with
seed 42, and a pair in which either response is empty or cut counts as a tie in the overall
reward. ``reward_mix`` is the mean over the three references. The ``_k500`` metrics apply the
length penalty: a slight win by a response more than 500 characters longer than the other,
both as cut, counts as a tie. The reward task macro leaves the empty and cut pairs out of
each category, weights the categories as WB-Score does for each reference, and averages the
three references' task macros.

Deviations from the reference:

- An empty response scores 1 without a judge call, as the reference's ``eval.py`` intends;
  its evaluation still sends the response to the judge and keeps the judge's score.
- A pair in which either response is empty or cut is not sent to the judge. The reference
  sends it and then counts it as a tie, or drops it when the judge's reply does not parse.
- A judge reply that does not parse is asked for again, then the output is left unscored,
  and a reply that is not valid JSON is read with the key-matching fallback of the
  reference's ``parse_result``. The reference's batch path drops a reply that is not JSON,
  and drops only the comparison with that reference; here a comparison with no verdict
  leaves the whole output unscored.
- Reasoning traces are removed before judging, and an output whose ``<think>`` never closed
  leaves an empty response. The judge sees the answer stripped of surrounding whitespace
  and ``<answer>`` tags.
- The reference's inference through the OpenAI, Anthropic, Mistral, Cohere and Yi APIs adds
  the system prompt "You are a helpful AI assistant.", which is not sent here.
- The leaderboard also reports per-category rewards, rewards and reward task macros for
  other length penalties, and the mean of the WB-Score and reward task macros; here the
  rewards are reported without a penalty and with K = 500, and the reward task macro
  without a penalty.

The paper names GPT-4-Turbo as the WB-Score judge and rescales scores by ``(score - 5) * 2``;
the reference's scripts and leaderboard, followed here, judge with ``gpt-4o-2024-05-13`` and
multiply that by a further 10.

``-o judge_model=...`` replaces the default judge (``gpt-4o-2024-05-13`` for WB-Score,
``gpt-4-turbo-2024-04-09`` for WB-Reward), and ``judge_panel`` grades with every judge of the
shared panel (see ``olmo_eval.evals.tasks.common.judge_panel``). The ``thinking`` variant
samples with temperature 0.6, top_p 0.95 and no token limit, for reasoning models.

Example command to run:

olmo-eval run -m allenai/Olmo-3-7B-Instruct -t wildbench_score -t wildbench_reward
"""

import asyncio
import json
import math
import random
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.scorers import Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.tasks.common.judge import JudgeParseError
from olmo_eval.evals.tasks.common.judge_panel import (
    JudgePanelTask,
    JudgeVerdictMetric,
    PanelJudgeScorer,
    judge_result,
    register_judge_panel_variant,
)
from olmo_eval.evals.tasks.constants.wildbench import PAIRWISE_TEMPLATE, SCORE_TEMPLATE
from olmo_eval.evals.tasks.omniscience import extract_final_answer

WILDBENCH_REPO = "allenai/WildBench"
WILDBENCH_CONFIG = "v2"
WILDBENCH_REVISION = "26c49eb39d7d5ce2099b0bbafed5a88dcce954ec"
OUTPUTS_REPO = "allenai/WildBench-V2-Model-Outputs"
OUTPUTS_REVISION = "d6755bc68220df853c0825a733430f73f5af2501"

# The reference's generation settings (scripts/_common_vllm.sh)
DEFAULT_SAMPLING = SamplingParams(max_tokens=4096, temperature=0.0, top_p=1.0)
# Room for reasoning models to finish thinking before they answer
THINKING_SAMPLING = SamplingParams(max_tokens=None, temperature=0.6, top_p=0.95)

SCORE_JUDGE_MODEL = "gpt-4o-2024-05-13"
REWARD_JUDGE_MODEL = "gpt-4-turbo-2024-04-09"
JUDGE_MAX_TOKENS = 1024
JUDGE_ATTEMPTS = 3

# The reference models of WB-Reward, under the short names their metrics use
REFERENCE_MODELS = {
    "gpt4t": "gpt-4-turbo-2024-04-09",
    "haiku": "claude-3-haiku-20240307",
    "llama": "Llama-2-70b-chat-hf",
}
PAIRWISE_MAX_WORDS = 1000
PAIRWISE_SEED = 42
LENGTH_PENALTY_K = 500
EMPTY_RESPONSE = "[This model response is empty.]"
TRUNCATION_SUFFIX = "... (truncated)"
# Strings the reference's clear_output (src/unified_utils.py) replaces with a space
CLEARED_STRINGS = ("<|endoftext|>", "</s>", "<pad>", "<end_of_turn>")

# Task tags -> task categories, and the categories' task-macro weights
TASK_GROUPS = {
    "Information seeking": "info_advice",
    "Creative Writing": "creative",
    "Coding & Debugging": "coding",
    "Reasoning": "planning_reasoning",
    "Editing": "creative",
    "Math": "math_data",
    "Planning": "planning_reasoning",
    "Brainstorming": "creative",
    "Role playing": "creative",
    "Advice seeking": "info_advice",
    "Data Analysis": "math_data",
    "Others": "creative",
}
TASK_WEIGHTS = {
    "creative": 0.5,
    "planning_reasoning": 1.25,
    "math_data": 1.0,
    "info_advice": 0.75,
    "coding": 1.25,
}

# Pairwise choices -> (the better response, how much better)
CHOICES = {"A++": ("A", 2), "A+": ("A", 1), "A=B": (None, 0), "B+": ("B", 1), "B++": ("B", 2)}

# =============================================================================
# Judge Prompts
# =============================================================================


def clear_output(text: str) -> str:
    """Clean a response as the reference saves it: special-token strings become spaces."""
    for token in CLEARED_STRINGS:
        text = text.replace(token, " ")
    return text.strip()


def shorten(text: str, max_words: int = -1) -> str:
    """Cut text to ``max_words`` space-separated words, as the reference does; -1 keeps all."""
    words = text.split(" ")
    if max_words > 0 and len(words) > max_words:
        return " ".join(words[:max_words]) + TRUNCATION_SUFFIX
    return text


def _fill(template: str, instance: Instance, max_words: int, **responses: str) -> str:
    """Fill a judge template's placeholders in the reference's order."""
    metadata = instance.metadata
    prompt = template.replace("{$history}", shorten(metadata["history"], max_words))
    prompt = prompt.replace("{$user_query}", shorten(metadata["last_query"], max_words))
    for placeholder, response in responses.items():
        prompt = prompt.replace(f"{{${placeholder}}}", response)
    checklist = "".join(f"- {item}\n" for item in metadata["checklist"])
    return prompt.replace("{$checklist}", checklist)


def score_prompt(instance: Instance, answer: str) -> str:
    """The WB-Score judge prompt, with nothing cut."""
    return _fill(SCORE_TEMPLATE, instance, -1, model_output=answer)


def pairwise_prompt(instance: Instance, response_a: str, response_b: str) -> str:
    """The WB-Reward judge prompt for two already cut responses."""
    return _fill(
        PAIRWISE_TEMPLATE,
        instance,
        PAIRWISE_MAX_WORDS,
        candidate_A=response_a if response_a.strip() else EMPTY_RESPONSE,
        candidate_B=response_b if response_b.strip() else EMPTY_RESPONSE,
    )


def judge_value(reply: str, key: str) -> str | None:
    """Read a key's value from a judge reply, as the reference's ``parse_result`` does.

    The reply is parsed as JSON; failing that, ``"key": "value"`` is searched for.
    """
    reply = reply.strip()
    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        value = parsed.get(key)
        return None if value is None else str(value)
    if parsed is not None or key not in reply:
        return None
    match = re.search(f'"{key}"\\s*:\\s*"([^"]*?)"', reply) or re.search(
        f'"{key}"\\s*:\\s*"(.*?)"', reply, re.DOTALL
    )
    return match.group(1) if match else None


def parse_score(reply: str) -> float | None:
    """The 1-10 score in a WB-Score judge reply, or None if it gives no finite number."""
    value = judge_value(reply, "score")
    try:
        score = None if value is None else float(value)
    except ValueError:
        return None
    return score if score is not None and math.isfinite(score) else None


def parse_choice(reply: str) -> str | None:
    """The choice in a WB-Reward judge reply, or None if it is not one of the five."""
    value = judge_value(reply, "choice")
    return value if value in CHOICES else None


def pairwise_reward(
    choice: str, response_is_a: bool, response_length: int, reference_length: int, k: int
) -> float:
    """The reward of a judged pair, from -1 to 1, with length penalty ``k`` (-1 for none).

    A slight win by a response more than ``k`` characters longer than the other is a tie.
    """
    better, extent = CHOICES[choice]
    if better is None:
        return 0.0
    response_won = (better == "A") == response_is_a
    winner_length, loser_length = (
        (response_length, reference_length) if response_won else (reference_length, response_length)
    )
    if extent == 1 and k >= 0 and winner_length > loser_length + k:
        return 0.0
    magnitude = 1.0 if extent == 2 else 0.5
    return magnitude if response_won else -magnitude


# =============================================================================
# Judges
# =============================================================================


@dataclass(frozen=True)
class _WildBenchJudge(PanelJudgeScorer):
    """A WildBench panel judge, which grades responses as the reference saves them."""

    def final_answer(self, output: LMOutput) -> str | None:
        """Return the final answer cleaned by ``clear_output``, or None when it is empty."""
        return clear_output(super().final_answer(output) or "") or None


@dataclass(frozen=True)
class WildBenchScoreScorer(_WildBenchJudge):
    """Panel judge that scores a response from 1 to 10 (WB-Score); scores the score over 10."""

    name: str = "wildbench_score_judge"
    default_model: ClassVar[str] = SCORE_JUDGE_MODEL
    default_max_tokens: ClassVar[int] = JUDGE_MAX_TOKENS
    attempts: ClassVar[int] = JUDGE_ATTEMPTS
    judge_response_format: ClassVar[dict[str, Any] | None] = {"type": "json_object"}

    def format_judge_prompt(self, instance: Instance, output: LMOutput) -> str:
        return score_prompt(instance, self.final_answer(output) or "")

    def parse_judge_response(self, response: str) -> float:
        """Return the score over 10.

        Raises:
            ValueError: If the reply gives no score.
        """
        score = parse_score(response)
        if score is None:
            raise ValueError(f"no score in judge reply {response!r}")
        return score / 10

    async def ascore_with_context(
        self, instance: Instance, output: LMOutput, context: ScoringContext
    ) -> float:
        """Score the response, recording the score on the instance; empty responses score 1."""
        answer = self.final_answer(output)
        if answer is None:
            self.record(instance, {"score": 1.0, "empty_response": True}, raw_response=None)
            return 0.1
        try:
            self._validate_provider(context)
            score, raw_response = await self._judge(
                score_prompt(instance, answer), parse_score, context
            )
        except Exception:
            self.record(instance, None, raw_response=None)
            raise
        if score is None:
            self.record(instance, None, raw_response=raw_response)
            raise JudgeParseError(f"no score after {self.attempts} judge attempts")
        self.record(instance, {"score": score, "empty_response": False}, raw_response=raw_response)
        return score / 10


@dataclass(frozen=True)
class WildBenchRewardScorer(_WildBenchJudge):
    """Panel judge that compares a response with each reference model's (WB-Reward).

    Scores the mean reward, mapped from -1..1 to 0..1.
    """

    name: str = "wildbench_reward_judge"
    default_model: ClassVar[str] = REWARD_JUDGE_MODEL
    default_max_tokens: ClassVar[int] = JUDGE_MAX_TOKENS
    attempts: ClassVar[int] = JUDGE_ATTEMPTS
    judge_response_format: ClassVar[dict[str, Any] | None] = {"type": "json_object"}

    def format_judge_prompt(self, instance: Instance, output: LMOutput) -> str:
        """Format the prompt comparing the response with the first reference model's."""
        reference = next(iter(REFERENCE_MODELS))
        return self._pair(instance, self.final_answer(output) or "", reference)[0]

    def _pair(self, instance: Instance, answer: str, reference: str) -> tuple[str, str, str]:
        """The judge prompt for a reference, with the cut response and reference response."""
        response = shorten(answer, PAIRWISE_MAX_WORDS)
        reference_response = shorten(instance.metadata["references"][reference], PAIRWISE_MAX_WORDS)
        if instance.metadata["response_is_a"]:
            prompt = pairwise_prompt(instance, response, reference_response)
        else:
            prompt = pairwise_prompt(instance, reference_response, response)
        return prompt, response, reference_response

    def parse_judge_response(self, response: str) -> float:
        """Return the reward of the response as Response A, from 0 to 1, with no penalty.

        Raises:
            ValueError: If the reply gives no valid choice.
        """
        choice = parse_choice(response)
        if choice is None:
            raise ValueError(f"no choice in judge reply {response!r}")
        return (pairwise_reward(choice, True, 0, 0, -1) + 1) / 2

    async def _compare(
        self, instance: Instance, answer: str, reference: str, context: ScoringContext
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Compare the response with one reference model's; return the verdict and the reply.

        The verdict is None when no reply gave a valid choice.
        """
        prompt, response, reference_response = self._pair(instance, answer, reference)
        verdict: dict[str, Any] = {"choice": None, "skipped": None}
        if not response.strip() or not reference_response.strip():
            verdict["skipped"] = "empty"
        elif response.endswith(TRUNCATION_SUFFIX) or reference_response.endswith(TRUNCATION_SUFFIX):
            verdict["skipped"] = "truncated"
        if verdict["skipped"]:
            return {**verdict, "reward": 0.0, "reward_k500": 0.0}, None

        choice, raw_response = await self._judge(prompt, parse_choice, context)
        if choice is None:
            return None, raw_response
        lengths = (len(response.strip()), len(reference_response.strip()))
        response_is_a = instance.metadata["response_is_a"]
        return {
            **verdict,
            "choice": choice,
            "reward": pairwise_reward(choice, response_is_a, *lengths, -1),
            "reward_k500": pairwise_reward(choice, response_is_a, *lengths, LENGTH_PENALTY_K),
        }, raw_response

    async def ascore_with_context(
        self, instance: Instance, output: LMOutput, context: ScoringContext
    ) -> float:
        """Compare the response with every reference model's, recording the rewards."""
        answer = self.final_answer(output) or ""
        try:
            self._validate_provider(context)
            # Every comparison settles before a failure is raised
            results = await asyncio.gather(
                *(self._compare(instance, answer, ref, context) for ref in REFERENCE_MODELS),
                return_exceptions=True,
            )
        except Exception:
            self.record(instance, None, raw_response=None)
            raise
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            self.record(instance, None, raw_response=None)
            raise failures[0]

        verdicts: dict[str, dict[str, Any]] = {}
        raw_responses: dict[str, str | None] = {}
        for ref, result in zip(REFERENCE_MODELS, results, strict=True):
            assert not isinstance(result, BaseException)
            verdict, raw_responses[ref] = result
            if verdict is not None:
                verdicts[ref] = verdict
        if len(verdicts) < len(REFERENCE_MODELS):
            self.record(instance, None, raw_response=raw_responses)
            raise JudgeParseError(f"no choice after {self.attempts} judge attempts")
        self.record(instance, verdicts, raw_response=raw_responses)
        mean_reward = sum(v["reward"] for v in verdicts.values()) / len(verdicts)
        return (mean_reward + 1) / 2


# =============================================================================
# Metrics
# =============================================================================


@dataclass(frozen=True, slots=True)
class _WildBenchMetric(JudgeVerdictMetric):
    """A WildBench metric; a ``task__<category>`` subset holds every conversation in it."""

    def _in_subset(self, response: Response) -> bool:
        if self.subset == "any":
            return True
        return self.category in response.instance.metadata.get(self.subset, ())

    def display_format(self) -> str:
        return "raw"


@dataclass(frozen=True, slots=True)
class WBScoreMetric(_WildBenchMetric):
    """``(mean score - 5) * 20``; a task category leaves empty responses out."""

    name: str = "any__any__wb_score"
    scorer: type[Scorer] | Scorer = WildBenchScoreScorer

    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        if self.subset != "any" and judge_result["empty_response"]:
            return None
        return (judge_result["score"] - 5) * 20


@dataclass(frozen=True, slots=True)
class WBRewardMetric(_WildBenchMetric):
    """Reward from -100 to 100, against one reference model or the mean of all three.

    The metric name is ``<subset>__<category>__reward_<reference or mix>[_k500]``.
    """

    name: str = "any__any__reward_mix"
    scorer: type[Scorer] | Scorer = WildBenchRewardScorer

    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        metric = self.name.split("__")[2].removeprefix("reward_")
        reference, _, penalty = metric.partition("_")
        key = "reward_k500" if penalty == "k500" else "reward"
        references = list(REFERENCE_MODELS) if reference == "mix" else [reference]
        return 100 * sum(judge_result[ref][key] for ref in references) / len(references)


@dataclass(frozen=True, slots=True)
class _TaskMacroMetric(_WildBenchMetric):
    """Task-macro aggregate: category means weighted by ``TASK_WEIGHTS`` over their sum."""

    def task_macro(
        self,
        responses: Sequence[Response],
        group_value: Callable[[dict[str, Any]], float | None],
    ) -> float:
        """Weight each category's mean ``group_value``, leaving out None, over every weight."""
        judge_name = self.scorer().name
        total = 0.0
        for group, weight in TASK_WEIGHTS.items():
            values = [
                value
                for response in responses
                if group in response.instance.metadata.get("task", ())
                and (result := judge_result(response, judge_name)) is not None
                and (value := group_value(result)) is not None
            ]
            if values:
                total += weight * sum(values) / len(values)
        return total / sum(TASK_WEIGHTS.values())

    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        return None

    def compute_instance(self, response: Response) -> float | None:
        """A weighted mean of category means has no per-instance equivalent."""
        return None


@dataclass(frozen=True, slots=True)
class WBScoreTaskMacroMetric(_TaskMacroMetric):
    """Task-macro WB-Score, the reference's ranking score; empty responses are left out."""

    name: str = "any__any__wb_score_task_macro"
    scorer: type[Scorer] | Scorer = WildBenchScoreScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return self.task_macro(responses, _non_empty_wb_score)


@dataclass(frozen=True, slots=True)
class WBRewardTaskMacroMetric(_TaskMacroMetric):
    """Mean over the reference models of the task-macro reward against each, with no penalty.

    As in the reference, each reference's category means leave out the pairs it skipped as
    empty or cut.
    """

    name: str = "any__any__reward_mix_task_macro"
    scorer: type[Scorer] | Scorer = WildBenchRewardScorer

    def compute(self, responses: Sequence[Response]) -> float:
        macros = [
            self.task_macro(responses, partial(_judged_reward, reference))
            for reference in REFERENCE_MODELS
        ]
        return sum(macros) / len(macros)


def _non_empty_wb_score(judge_result: dict[str, Any]) -> float | None:
    """A judged response's WB-Score, or None for an empty response."""
    return None if judge_result["empty_response"] else (judge_result["score"] - 5) * 20


def _judged_reward(reference: str, judge_result: dict[str, Any]) -> float | None:
    """The reward against one reference model, or None for a pair skipped without a judge."""
    verdict = judge_result[reference]
    return None if verdict["skipped"] else 100 * verdict["reward"]


# =============================================================================
# Task Definitions
# =============================================================================


class _WildBench(JudgePanelTask):
    """Base for the WildBench tasks: the conversation as the request, judged after."""

    data_source = DataSource(
        path=WILDBENCH_REPO, subset=WILDBENCH_CONFIG, revision=WILDBENCH_REVISION, split="test"
    )
    split = Split.TEST
    sampling_params = DEFAULT_SAMPLING
    answer_extractor = extract_final_answer
    required_secrets = ("OPENAI_API_KEY",)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a conversation to an Instance, with its judge context and task categories."""
        conversation = doc["conversation_input"]
        if not conversation or conversation[-1]["role"] != "user":
            raise ValueError(f"WildBench conversation {doc['id']} does not end with a user turn")
        history = "".join(
            f"{'USER' if turn['role'] == 'user' else 'ASSISTANT'}: {turn['content']}\n\n"
            for turn in conversation[:-1]
            if turn["role"] in ("user", "assistant")
        )
        tags = [doc["primary_tag"], *doc["secondary_tags"]]
        return Instance(
            question=conversation[-1]["content"],
            metadata={
                "id": doc["id"],
                "messages": [{"role": t["role"], "content": t["content"]} for t in conversation],
                "history": history,
                "last_query": conversation[-1]["content"],
                "checklist": list(doc["checklist"]),
                "primary_tag": doc["primary_tag"],
                "task": sorted({TASK_GROUPS[tag] for tag in tags}),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Send the conversation, with no system prompt."""
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=tuple(dict(message) for message in instance.metadata["messages"]),
        )

    @property
    def request_type(self) -> RequestType:
        return RequestType.CHAT


_SCORE_JUDGE = WildBenchScoreScorer()
_REWARD_JUDGE = WildBenchRewardScorer()


@register("wildbench_score")
class WildBenchScore(_WildBench):
    """WB-Score: the judge scores each response from 1 to 10 against its checklist."""

    judge = _SCORE_JUDGE
    metrics = (
        WBScoreTaskMacroMetric(scorer=_SCORE_JUDGE),
        WBScoreMetric(scorer=_SCORE_JUDGE),
        *(WBScoreMetric(name=f"task__{g}__wb_score", scorer=_SCORE_JUDGE) for g in TASK_WEIGHTS),
    )
    primary_metric = WBScoreTaskMacroMetric(scorer=_SCORE_JUDGE)


@register("wildbench_reward")
class WildBenchReward(_WildBench):
    """WB-Reward: the judge compares each response with three reference models' responses."""

    judge = _REWARD_JUDGE
    metrics = (
        WBRewardMetric(scorer=_REWARD_JUDGE),
        WBRewardMetric(name="any__any__reward_mix_k500", scorer=_REWARD_JUDGE),
        WBRewardTaskMacroMetric(scorer=_REWARD_JUDGE),
        *(
            WBRewardMetric(name=f"any__any__reward_{ref}{penalty}", scorer=_REWARD_JUDGE)
            for ref in REFERENCE_MODELS
            for penalty in ("", "_k500")
        ),
    )
    primary_metric = WBRewardMetric(scorer=_REWARD_JUDGE)

    def _load_instances(self, split: str | None = None) -> Iterator[Instance]:
        """Yield the conversations with the reference responses and the pair order.

        The order is drawn over the whole set in dataset order with seed 42, as the
        reference does, so it does not depend on which conversations a run evaluates.
        """
        from olmo_eval.data import DataLoader

        loader = DataLoader()
        references = {
            ref: {
                str(record["session_id"]): _first_output(record["output"])
                for record in loader.load(
                    DataSource(
                        path=OUTPUTS_REPO, subset=model, revision=OUTPUTS_REVISION, split="train"
                    )
                )
            }
            for ref, model in REFERENCE_MODELS.items()
        }
        rng = random.Random(PAIRWISE_SEED)
        source = self.config.get_data_source(split=split)
        for index, doc in enumerate(loader.load(source)):
            response_is_a = rng.random() < 0.5
            instance = self.process_doc(doc, index)
            missing = [ref for ref in references if instance.metadata["id"] not in references[ref]]
            if missing:
                raise ValueError(
                    f"WildBench conversation {instance.metadata['id']} has no response from "
                    f"{', '.join(REFERENCE_MODELS[ref] for ref in missing)}"
                )
            instance.metadata["references"] = {
                ref: outputs[instance.metadata["id"]] for ref, outputs in references.items()
            }
            instance.metadata["response_is_a"] = response_is_a
            yield instance


def _first_output(output: Any) -> str:
    """A reference record's response: the first of a list, as the reference reads it."""
    return output[0] if isinstance(output, list) else output


for _task_name in ("wildbench_score", "wildbench_reward"):
    register_variant(_task_name, "thinking", sampling_params=THINKING_SAMPLING)
    register_judge_panel_variant(_task_name)
