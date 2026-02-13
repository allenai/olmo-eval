"""LAB-Bench evaluation tasks.

LAB-Bench (futurehouse/lab-bench) is a biology research benchmark with 8 subtasks.
This module implements the text-only subtasks, starting with LitQA2.

Usage:
    # Chat-based generation (default)
    olmo-eval run -m llama3.1-8b -t lab_bench_litqa2

    # Logprob-based multiple choice
    olmo-eval run -m llama3.1-8b -t lab_bench_litqa2:mc

    # Bits-per-byte perplexity
    olmo-eval run -m llama3.1-8b -t lab_bench_litqa2:bpb
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import MCQAChatFormatter, MultipleChoiceFormatter, PPLFormatter
from olmo_eval.common.metrics import AccuracyMetric, BPBMetric, Metric
from olmo_eval.common.scorers import MultipleChoiceScorer, Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

# Regex for "ANSWER: X" pattern (case-insensitive)
_ANSWER_PATTERN = re.compile(r"ANSWER\s*:\s*([A-Z])", re.IGNORECASE)

# Matches REFUSE_CHOICE from the official LAB-Bench evaluation code
_REFUSE_CHOICE = "Insufficient information to answer the question"


@dataclass(frozen=True, slots=True)
class PrecisionMetric(Metric):
    """Accuracy excluding abstentions (correct / non-abstained).

    Matches the "precision" metric from the LAB-Bench evaluation protocol.
    A response is an abstention if the model chose the refuse option
    (identified by `unsure_idx` in instance metadata).
    """

    name: str = "precision"
    scorer: type[Scorer] = MultipleChoiceScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        committed = 0
        correct = 0.0
        for r in responses:
            unsure_idx = r.instance.metadata.get("unsure_idx")
            if unsure_idx is not None:
                unsure_letter = chr(ord("A") + unsure_idx)
                extracted = r.outputs[0].extracted_answer if r.outputs else None
                if extracted is not None and str(extracted).strip().upper() == unsure_letter:
                    continue
            committed += 1
            correct += r.scores.get(scorer_name, 0.0)
        return correct / committed if committed > 0 else 0.0


class LabBenchTask(Task):
    """Base class for text-only LAB-Bench subtasks.

    Handles the shared pattern: combine `ideal` + `distractors` into shuffled
    choices, extract letter answers from model output.
    """

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self.config.get_data_source()
            for idx, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, idx)
                if instance is not None:
                    self._instances_cache.append(instance)
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a LAB-Bench document to an Instance with shuffled choices."""
        question = doc.get("question", "")
        ideal = doc.get("ideal", "")
        distractors = doc.get("distractors", [])

        if not question or not ideal:
            return None

        # Build choices: ideal + distractors (deduplicate in case ideal appears in distractors)
        choices = [ideal] + [d for d in distractors if d != ideal]

        # Inject abstention option if not already present (the HF dataset omits it,
        # but the LAB-Bench evaluation protocol includes it)
        if not any("insufficient" in c.lower() for c in choices):
            choices.append(_REFUSE_CHOICE)

        # Deterministic per-question shuffle
        rng = random.Random(f"{self.config.seed}:{index}")
        rng.shuffle(choices)

        # Find where the ideal and refuse options landed after shuffle
        gold_idx = choices.index(ideal)
        gold_letter = chr(ord("A") + gold_idx)

        # Find the refuse option index (may not exist if one was already in distractors)
        try:
            unsure_idx = choices.index(_REFUSE_CHOICE)
        except ValueError:
            unsure_idx = None

        metadata: dict[str, Any] = {
            "id": doc.get("id", f"lab_bench_{index}"),
            "index": index,
            "gold_idx": gold_idx,
            "gold_text": ideal,
        }
        if unsure_idx is not None:
            metadata["unsure_idx"] = unsure_idx

        return Instance(
            question=question,
            gold_answer=gold_letter,
            choices=tuple(choices),
            metadata=metadata,
        )

    @property
    def request_type(self) -> RequestType:
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.CHAT

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract a letter answer from model output.

        Looks for the "ANSWER: X" pattern that the system prompt requests.
        Returns None if the model doesn't follow the expected format.
        """
        text = output.text.strip()
        if not text:
            return None

        matches = list(_ANSWER_PATTERN.finditer(text))
        if matches:
            return matches[-1].group(1).upper()

        return None


LITQA2_SYSTEM_PROMPT = """\
You are a scientific research assistant. Answer the following multiple choice \
question by reasoning through the options and selecting the best answer.

End your response with "ANSWER: X" where X is the letter of your chosen answer."""


@register("lab_bench_litqa2")
class LabBenchLitQA2(LabBenchTask):
    """LitQA2: Literature-based question answering from LAB-Bench."""

    data_source = DataSource(path="futurehouse/lab-bench", subset="LitQA2", split="train")
    formatter = MCQAChatFormatter(system_prompt=LITQA2_SYSTEM_PROMPT)
    metrics = (AccuracyMetric(scorer=MultipleChoiceScorer), PrecisionMetric())
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)


# =============================================================================
# Variant Registrations
# =============================================================================

# Logprob-based multiple choice
register_variant(
    "lab_bench_litqa2",
    "mc",
    formatter=MultipleChoiceFormatter(),
    metrics=(AccuracyMetric(scorer=MultipleChoiceScorer), PrecisionMetric()),
)

# Bits-per-byte perplexity
register_variant(
    "lab_bench_litqa2",
    "bpb",
    formatter=PPLFormatter(),
    metrics=(BPBMetric(),),
)
