"""
EQ-Bench 3 Rubric Evaluation Task

EQ-Bench 3 measures emotional intelligence in conversation. In role-play scenarios the model
answers the scripted turns (one or more) of an emotionally charged situation in character,
writing its own thoughts and feelings, its read of the other person, and its response, then
reflects on the role-play out of character. In message-drafting scenarios it drafts a
message, and in analysis scenarios it analyzes a role-play transcript in one turn. A judge
scores each conversation on a 0-20 rubric.

Reference: https://github.com/EQ-bench/eqbench3/tree/affc5a000bbdb965b807e09ca7875f8530fa8f56
Leaderboard: https://eqbench.com/eqbench3.html

``eq_bench_3.json`` holds the reference's scenario, master, debrief and rubric prompt files
and its rubric criteria, verbatim from that commit (MIT License, copyright 2025 Sam Paech;
the license text is included in the file). Scenarios are parsed, prompted, transcribed and
scored as in the reference's ``core/benchmark.py`` and ``core/conversation.py``:

- Each scenario turn is wrapped in its master prompt, and role-play and message-drafting
  scenarios end with the debrief prompt. As in the reference, the parser drops scenario 409,
  whose prompt delimiter is malformed.
- A conversation is scored only if one of its scenario replies, the debrief aside, has at
  least ``MIN_REPLY_CHARS`` characters; others are left unscored, as the reference leaves
  them out of its average.
- The judge sees the conversation with role-play and drafting replies rebuilt from their
  parsed sections, analysis replies as written, and the debrief, in the rubric prompt with
  ``*`` and ``#`` removed. Its JSON reply is read with the reference's lenient parser,
  which counts a boolean as 0 or 1.
- A role-play or drafting conversation scores the mean of the qualitative criteria the
  reference counts (demonstrated_empathy, pragmatic_ei, depth_of_insight, social_dexterity,
  emotional_reasoning, message_tailoring); an analysis scores the mean of its six criteria.
  The rubric score is 5 times the mean over conversations, from 0 to 100.

Deviations from the reference:

- Only the rubric score is computed. The leaderboard ranks by an Elo from pairwise
  comparisons with other models' published transcripts, which this task does not run, so
  its score is not comparable to the leaderboard's.
- The reference's README names claude-opus-4-6, called with thinking disabled, as its
  default judge; this task judges with ``JUDGE_MODEL`` by default, and rubric scores depend
  on the judge.
- Every turn samples at temperature 0.7 with up to 12000 tokens; the reference samples the
  debrief at temperature 0.5 and sets ``min_p`` 0.1 on the other turns.
- Earlier replies go back to the model as it wrote them, so the chat template decides
  whether a reasoning trace is kept. The judge sees each reply after its last ``</think>``,
  and a reply whose ``<think>`` never closed as empty. The reference removes reasoning
  before both, cutting a reply at the first ``</think>``, ``</thinking>`` or
  ``</reasoning>`` only when it also opens that block, so it keeps whole a reply whose
  block never closed or was opened by the chat template.
- A judge reply with no scores for the counted criteria is asked for again up to
  ``JUDGE_ATTEMPTS`` times, then left unscored.
- Each scenario runs once, the reference's default of ``--iterations``, and the score is
  not rounded; the reference rounds the 0-20 mean to two decimals before scaling it.

The task runs under the ``multi_turn_chat`` scaffold, and ``judge_panel`` grades with every
judge of the shared panel (see ``olmo_eval.evals.tasks.common.judge_panel``), keeping the
reference's 8000-token judge limit.

Example command to run:

olmo-eval run -m allenai/Olmo-3-7B-Instruct --harness multi_turn_chat -t eq_bench_3
"""

import io
import json
import re
import statistics
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any, ClassVar

from olmo_eval.common.execution import ScoringContext
from olmo_eval.common.multi_turn import missing_replies_error, scripted_chat_request, turn_replies
from olmo_eval.common.scorers import Scorer, ScoringIncompleteError
from olmo_eval.common.types import Instance, LMOutput, LMRequest, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.common.judge import JudgeParseError, visible_response
from olmo_eval.evals.tasks.common.judge_panel import (
    JudgePanelTask,
    JudgeVerdictMetric,
    PanelJudgeScorer,
    register_judge_panel_variant,
)

DATA_FILE = "eq_bench_3.json"

# The reference's turn settings (core/conversation.py)
DEFAULT_SAMPLING = SamplingParams(max_tokens=12000, temperature=0.7)

JUDGE_MODEL = "gpt-5.5-2026-04-23"
# The reference's rubric judge token limit (core/benchmark.py)
JUDGE_MAX_TOKENS = 8000
JUDGE_ATTEMPTS = 3
# A conversation needs a scenario reply this long to be scored (core/benchmark.py's
# _MIN_RAW_LEN)
MIN_REPLY_CHARS = 100

# Scenario types, from the reference's utils/constants.py
NO_RP_SCENARIO_IDS = frozenset({"8", "10", "12", "14"})
MESSAGE_DRAFTING_SCENARIO_IDS = frozenset({str(i) for i in range(201, 210)})
ANALYSIS_SCENARIO_IDS = frozenset({str(i) for i in range(401, 421)})

# The criteria a role-play or drafting conversation's score averages (core/benchmark.py);
# the last four are analysis criteria, which the role-play rubric does not ask for
STANDARD_ALLOWED_CRITERIA = frozenset(
    {
        "demonstrated_empathy",
        "pragmatic_ei",
        "depth_of_insight",
        "social_dexterity",
        "emotional_reasoning",
        "message_tailoring",
        "theory_of_mind",
        "subtext_identification",
        "intellectual_grounding",
        "correctness",
    }
)

TASK_TYPES = ("roleplay", "message_drafting", "analysis")

# =============================================================================
# Reference Data
# =============================================================================


@cache
def reference_files() -> dict[str, str]:
    """The vendored reference files, keyed by their path in the reference repository."""
    data = files(__package__).joinpath(DATA_FILE).read_text(encoding="utf-8")
    return json.loads(data)["files"]


def parse_scenario_prompts(text: str) -> dict[str, list[str]]:
    """Parse the scenario prompts file into each scenario's turns, as the reference does.

    A ``########`` line starts a scenario and a ``####### PromptN`` line starts a turn; a
    scenario with no well-formed turn is dropped.
    """
    scenarios: dict[str, list[str]] = {}
    scenario_id: str | None = None
    prompts: list[str] = []
    lines: list[str] = []
    in_prompt = False

    def finish_prompt() -> None:
        prompt = "\n".join(lines).strip()
        if prompt:
            prompts.append(prompt)

    for raw_line in io.StringIO(text):
        line = raw_line.strip()
        scenario_match = re.match(r"^########\s*(\S+)", line)
        prompt_match = re.match(r"^#######\s*Prompt(\d+)", line)
        if scenario_match:
            if lines:
                finish_prompt()
                lines = []
            if scenario_id and prompts:
                scenarios[scenario_id] = prompts
            scenario_id = scenario_match.group(1)
            prompts = []
            in_prompt = False
        elif prompt_match:
            if scenario_id is None:
                continue
            if lines:
                finish_prompt()
            lines = []
            in_prompt = True
        elif scenario_id and in_prompt:
            lines.append(raw_line.rstrip("\n\r"))
    if lines:
        finish_prompt()
    if scenario_id and prompts:
        scenarios[scenario_id] = prompts
    return scenarios


def _criteria(path: str) -> list[str]:
    lines = reference_files()[path].splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def output_format(criteria: list[str]) -> str:
    """The JSON format the rubric prompt asks the judge for, as the reference builds it."""
    fields: dict[str, Any] = {
        "chain_of_thought_reasoning": (
            "detailed chain of thought reasoning about the coming scoring decisions"
        )
    }
    fields.update({criterion: 0 for criterion in criteria})
    return json.dumps(fields, indent=2).replace(": 0", ": 0-20")


def task_type(scenario_id: str) -> str:
    """Whether a scenario is a role-play, a message-drafting or an analysis task."""
    if scenario_id in ANALYSIS_SCENARIO_IDS:
        return "analysis"
    if scenario_id in MESSAGE_DRAFTING_SCENARIO_IDS:
        return "message_drafting"
    return "roleplay"


# =============================================================================
# Transcripts
# =============================================================================

_DRAFT_PATTERNS = {
    "perspective_taking": r"#\s*Perspective[- ]taking\s*\n([\s\S]*?)(?=#|\Z)",
    "draft_brainstorming": r"#\s*Draft brainstorming\s*\n([\s\S]*?)(?=#|\Z)",
    "draft": r"#\s*Draft\s*\n([\s\S]*?)(?=--|\Z)",
}
_ROLEPLAY_PATTERNS = {
    "thinking_feeling": r"#\s*I[’']m thinking & feeling\s*\n([\s\S]*?)(?=#|\Z)",
    "their_thinking_feeling": r"#\s*They[’']re thinking & feeling\s*\n([\s\S]*?)(?=#|\Z)",
    "response": r"#\s*My response\s*\n([\s\S]*?)(?=--|\Z)",
}


def judged_reply(reply: str) -> str:
    """A reply as the judge sees it: its visible response, or the reply as written.

    A reply with no reasoning tag keeps its surrounding whitespace, as the reference's
    transcript does.
    """
    if "<think>" in reply or "</think>" in reply:
        return visible_response(reply)
    return reply


def format_sectioned_reply(scenario_id: str, reply: str) -> str:
    """Rebuild a role-play or drafting reply from its parsed sections, as the judge sees it.

    Follows the reference's ``_parse_response`` and ``_format_parsed_response``; a missing
    section is left empty.
    """
    drafting = scenario_id in MESSAGE_DRAFTING_SCENARIO_IDS
    patterns = _DRAFT_PATTERNS if drafting else _ROLEPLAY_PATTERNS
    sections = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, reply, flags=re.IGNORECASE | re.DOTALL)
        sections[key] = match.group(1).strip() if match else ""
    if drafting:
        parts = [
            "# Perspective-taking",
            sections["perspective_taking"],
            "\n# Draft brainstorming",
            sections["draft_brainstorming"],
            "\n# Draft",
            sections["draft"],
        ]
    else:
        parts = [
            "# I'm thinking & feeling",
            sections["thinking_feeling"],
            "\n# They're thinking & feeling",
            sections["their_thinking_feeling"],
            "\n# My response",
            sections["response"],
        ]
    return "\n".join([*parts, "\n--"])


def rubric_prompt(scenario_id: str, turns: list[str], replies: list[str]) -> str:
    """The rubric prompt for a conversation, as the reference's judge receives it.

    ``turns`` and ``replies`` include the debrief for role-play and drafting scenarios.
    """
    analysis = scenario_id in ANALYSIS_SCENARIO_IDS
    scenario_turns = turns if analysis else turns[:-1]
    parts = []
    for turn, reply in zip(scenario_turns, replies, strict=False):
        parts.append(f"User:\n{turn}\n")
        if analysis or scenario_id in NO_RP_SCENARIO_IDS:
            content = reply
        else:
            content = format_sectioned_reply(scenario_id, reply)
        parts.append(f"Assistant:\n{content}\n")
    transcript = "---\n".join(parts)

    if analysis:
        template = reference_files()["data/rubric_scoring_prompt_analysis.txt"]
        criteria = _criteria("data/rubric_scoring_criteria_analysis.txt")
        prompt = template.format(transcript=transcript, output_format=output_format(criteria))
    else:
        template = reference_files()["data/rubric_scoring_prompt.txt"]
        criteria = _criteria("data/rubric_scoring_criteria.txt")
        prompt = template.format(
            transcript=transcript,
            output_format=output_format(criteria),
            debrief=replies[-1].strip(),
        )
    return prompt.replace("*", "").replace("#", "")


# =============================================================================
# Judge Verdicts
# =============================================================================


def robust_json_loads(raw_text: str) -> Any:
    """Parse the JSON object in a judge reply, as the reference's ``robust_json_loads`` does.

    Keeps only the last of several ```` ```json ```` blocks, drops fence lines, takes the
    first balanced object, and repairs unescaped newlines and stray quotes inside strings.

    Raises:
        json.JSONDecodeError: If no object can be read.
    """
    if raw_text.count("```json") > 1:
        raw_text = raw_text[raw_text.rfind("```json") + len("```json") :]
    cleaned = "\n".join(
        line for line in raw_text.splitlines() if not line.lstrip().startswith("```")
    ).strip()

    json_str = _first_json_object(cleaned)
    if json_str is None:
        raise json.JSONDecodeError("No balanced JSON object found", cleaned, 0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return json.loads(_repair_inner_quotes(_repair_newlines(json_str)))


def _first_json_object(text: str) -> str | None:
    in_string = escape_next = False
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if in_string:
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def _repair_newlines(text: str) -> str:
    out, in_string, escape = [], False, False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            elif ch in "\r\n":
                ch = "\\n"
        elif ch == '"':
            in_string = True
        out.append(ch)
    return "".join(out)


def _repair_inner_quotes(text: str) -> str:
    out, in_string, escape = [], False, False
    n = len(text)
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                j = i + 1
                while j < n and text[j].isspace():
                    j += 1
                if j < n and text[j] not in {",", "}", "]", ":"}:
                    out.append("\\")
                else:
                    in_string = False
        elif ch == '"':
            in_string = True
        out.append(ch)
    return "".join(out)


def parse_rubric_scores(reply: str) -> dict[str, float] | None:
    """Read the numeric criterion scores from a judge reply, or None if it holds none.

    Follows the reference's ``_parse_rubric_scores``: numbers are kept, booleans as 0 or 1,
    and other values except the reasoning are searched for a number such as the 15 in "15/20".
    """
    try:
        parsed = robust_json_loads(reply)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    scores = {}
    for key, value in parsed.items():
        if isinstance(value, (int, float)):
            scores[key] = float(value)
        elif key != "chain_of_thought_reasoning":
            match = re.search(r"\b(\d+(\.\d+)?)\b", str(value))
            if match:
                scores[key] = float(match.group(1))
    return scores or None


def conversation_score(scenario_id: str, scores: dict[str, float]) -> float | None:
    """Mean of a conversation's counted criteria, from 0 to 20, or None if it has none."""
    if scenario_id in ANALYSIS_SCENARIO_IDS:
        allowed = set(_criteria("data/rubric_scoring_criteria_analysis.txt"))
    else:
        allowed = STANDARD_ALLOWED_CRITERIA
    values = [score for metric, score in scores.items() if metric in allowed]
    return statistics.mean(values) if values else None


# =============================================================================
# Scoring and Metrics
# =============================================================================


@dataclass(frozen=True)
class EQBench3RubricScorer(PanelJudgeScorer):
    """Panel judge that scores a conversation on the EQ-Bench 3 rubric.

    Scores the conversation's rubric score over 20, from 0 to 1.
    """

    name: str = "eq_bench_3_judge"
    default_model: ClassVar[str] = JUDGE_MODEL
    default_max_tokens: ClassVar[int] = JUDGE_MAX_TOKENS
    attempts: ClassVar[int] = JUDGE_ATTEMPTS

    def _replies(self, instance: Instance, output: LMOutput) -> list[str]:
        """Each turn's ``judged_reply``, raising if the conversation is incomplete.

        As in the reference, a conversation is incomplete when it ended early or when no
        scenario reply, the debrief aside, has ``MIN_REPLY_CHARS`` characters once stripped.
        """
        replies = turn_replies(output)
        if replies is None:
            raise missing_replies_error("EQ-Bench 3")
        turns = instance.metadata["turns"]
        if len(replies) < len(turns):
            raise ScoringIncompleteError(
                f"the conversation ended after {len(replies)} of {len(turns)} turns"
            )
        visible = [judged_reply(reply) for reply in replies]
        analysis = instance.metadata["id"] in ANALYSIS_SCENARIO_IDS
        scenario_replies = visible[: len(turns) if analysis else len(turns) - 1]
        if not any(len(reply.strip()) >= MIN_REPLY_CHARS for reply in scenario_replies):
            raise ScoringIncompleteError(
                f"no scenario reply has the {MIN_REPLY_CHARS} characters the reference requires"
            )
        return visible

    def format_judge_prompt(self, instance: Instance, output: LMOutput) -> str:
        """Format the rubric prompt for a complete conversation."""
        return rubric_prompt(
            instance.metadata["id"], instance.metadata["turns"], self._replies(instance, output)
        )

    def parse_judge_response(self, response: str) -> float:
        """Return a role-play or drafting conversation's score over 20, from a judge reply.

        Analysis conversations count other criteria; ``ascore_with_context`` reads replies
        for the conversation's own scenario type.

        Raises:
            ValueError: If the reply has no scores for the counted criteria.
        """
        scores = parse_rubric_scores(response)
        score = None if scores is None else conversation_score("", scores)
        if score is None:
            raise ValueError(f"no rubric scores for the counted criteria in {response!r}")
        return score / 20

    async def ascore_with_context(
        self, instance: Instance, output: LMOutput, context: ScoringContext
    ) -> float:
        """Judge the conversation, recording its criterion scores on the instance."""
        scenario_id = instance.metadata["id"]

        def parse(reply: str) -> dict[str, float] | None:
            scores = parse_rubric_scores(reply)
            if scores is None or conversation_score(scenario_id, scores) is None:
                return None
            return scores

        try:
            prompt = self.format_judge_prompt(instance, output)
            self._validate_provider(context)
            scores, raw_response = await self._judge(prompt, parse, context)
        except Exception:
            self.record(instance, None, raw_response=None)
            raise
        if scores is None:
            self.record(instance, None, raw_response=raw_response)
            raise JudgeParseError(
                f"no scores for the counted criteria after {self.attempts} judge attempts"
            )
        score = conversation_score(scenario_id, scores)
        assert score is not None
        self.record(
            instance,
            {"rubric_scores": scores, "conversation_score": score},
            raw_response=raw_response,
        )
        return score / 20


@dataclass(frozen=True, slots=True)
class RubricScoreMetric(JudgeVerdictMetric):
    """5 times the mean conversation score: the rubric score, from 0 to 100."""

    name: str = "any__any__rubric_score"
    scorer: type[Scorer] | Scorer = EQBench3RubricScorer

    def judged_value(self, judge_result: dict[str, Any]) -> float | None:
        return 5 * judge_result["conversation_score"]

    def display_format(self) -> str:
        return "raw"


# =============================================================================
# Task Definition
# =============================================================================

_JUDGE = EQBench3RubricScorer()


@register("eq_bench_3")
class EQBench3(JudgePanelTask):
    """EQ-Bench 3 scenarios, scored on the judge's rubric."""

    split = Split.TEST
    sampling_params = DEFAULT_SAMPLING
    required_scaffold = "multi_turn_chat"
    required_secrets = ("OPENAI_API_KEY",)
    judge = _JUDGE
    metrics = (
        RubricScoreMetric(scorer=_JUDGE),
        *(
            RubricScoreMetric(name=f"task_type__{kind}__rubric_score", scorer=_JUDGE)
            for kind in TASK_TYPES
        ),
    )
    primary_metric = RubricScoreMetric(scorer=_JUDGE)

    def _load_instances(self, split: str | None = None) -> Iterator[Instance]:
        """Yield the parsed scenarios from the vendored reference files."""
        reference = reference_files()
        scenarios = parse_scenario_prompts(reference["data/scenario_prompts.txt"])
        for index, (scenario_id, prompts) in enumerate(scenarios.items()):
            yield self.process_doc({"id": scenario_id, "prompts": prompts}, index)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Wrap a scenario's prompts in its master prompt and add the debrief turn."""
        scenario_id = doc["id"]
        kind = task_type(scenario_id)
        reference = reference_files()
        master = {
            "roleplay": reference["data/scenario_master_prompt.txt"],
            "message_drafting": reference["data/scenario_master_prompt_message_drafting.txt"],
            "analysis": reference["data/scenario_master_prompt_analysis.txt"],
        }[kind]
        turns = [
            prompt if scenario_id in NO_RP_SCENARIO_IDS else master.format(scenario_prompt=prompt)
            for prompt in doc["prompts"]
        ]
        if kind != "analysis":
            turns.append(reference["data/debrief_prompt.txt"])
        return Instance(
            question=turns[0],
            metadata={"id": scenario_id, "index": index, "task_type": kind, "turns": turns},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Ask every turn in order, each after the model's earlier replies."""
        return scripted_chat_request(instance.metadata["turns"])


register_judge_panel_variant("eq_bench_3", judge_max_tokens=JUDGE_MAX_TOKENS)
