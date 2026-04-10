"""GPQA (Graduate-Level Google-Proof Q&A) evaluation tasks.

GPQA (Idavidrein/gpqa) is a multiple-choice benchmark with 4-way questions
across biology, physics, and chemistry. Three quality-tier subsets are available
(diamond/main/extended) and each question has a fine-grained ``Subdomain`` field.

Tasks (12 total):
    gpqa_diamond             Full diamond subset (198 questions)
    gpqa_main                Full main subset (448 questions)
    gpqa_extended            Full extended subset (546 questions)
    gpqa_{subset}_{subject}  Filtered by broad subject (biology/chemistry/physics)

Each task supports :mc (logprob-based) and :bpb (bits-per-byte) variants.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.formatters import MCQAChatFormatter, MultipleChoiceFormatter, PPLFormatter
from olmo_eval.common.metrics import AccuracyMetric, BPBMetric
from olmo_eval.common.scorers import MultipleChoiceScorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

# Matches "ANSWER: X" or "(X)" where X is a capital letter
_ANSWER_PATTERN = re.compile(r"ANSWER\s*:\s*\(?([A-Z])\)?", re.IGNORECASE)
_PAREN_LETTER = re.compile(r"\(([A-Z])\)")
_LAST_CAPITAL = re.compile(r"\b([A-Z])\b")


# ---------------------------------------------------------------------------
# Subject grouping
# ---------------------------------------------------------------------------

_SUBJECT_GROUPS: dict[str, set[str]] = {
    "biology": {
        "Molecular Biology",
        "Genetics",
    },
    "chemistry": {
        "Organic Chemistry",
        "Chemistry",
        "Chemistry (general)",
        "Analytical Chemistry",
        "Inorganic Chemistry",
        "Physical Chemistry",
    },
    "physics": {
        "Quantum Mechanics",
        "Physics",
        "Physics (general)",
        "Astrophysics",
        "High-energy particle physics",
        "Condensed Matter Physics",
        "Electromagnetism and Photonics",
        "Relativistic Mechanics",
        "Statistical Mechanics",
        "Optics and Acoustics",
    },
}

# Reverse lookup: subdomain string -> group name
_SUBDOMAIN_TO_GROUP: dict[str, str] = {
    subdomain: group for group, subdomains in _SUBJECT_GROUPS.items() for subdomain in subdomains
}

_WARNED_SUBDOMAINS: set[str] = set()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert scientist. Answer the following multiple choice question by \
reasoning through the problem step by step, then selecting the best answer.

End your response with "ANSWER: X" where X is the letter of your chosen answer."""

_DEFAULT_ACCURACY = AccuracyMetric(scorer=MultipleChoiceScorer)
_DEFAULT_METRICS = (_DEFAULT_ACCURACY,)
_DEFAULT_SAMPLING = SamplingParams(temperature=0.0, max_tokens=1024)


# ---------------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------------


class GPQATask(Task):
    """Base class for GPQA tasks."""

    split = Split.TRAIN
    subject_group: str | None = None

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
        question = doc.get("Question", "")
        correct = doc.get("Correct Answer", "")
        if not question or not correct:
            return None

        # Filter by subject group if set
        subdomain = doc.get("Subdomain", "")
        if self.subject_group is not None:
            group = _SUBDOMAIN_TO_GROUP.get(subdomain)
            if group is None and subdomain not in _WARNED_SUBDOMAINS:
                _WARNED_SUBDOMAINS.add(subdomain)
                log.warning("GPQA: unmapped subdomain %r", subdomain)
            if group != self.subject_group:
                return None

        # Build choices
        incorrect = [
            doc.get("Incorrect Answer 1", ""),
            doc.get("Incorrect Answer 2", ""),
            doc.get("Incorrect Answer 3", ""),
        ]
        choices = [_clean_text(correct)] + [_clean_text(a) for a in incorrect if a]

        # Deterministic per-question shuffle (seed=111 matches old oe-eval convention)
        rng = random.Random(f"{self.config.seed}:{index}")
        rng.shuffle(choices)

        gold_idx = choices.index(_clean_text(correct))
        gold_letter = chr(ord("A") + gold_idx)

        metadata: dict[str, Any] = {
            "index": index,
            "gold_idx": gold_idx,
            "gold_text": _clean_text(correct),
            "subdomain": subdomain,
        }
        explanation = doc.get("Explanation")
        if explanation:
            metadata["explanation"] = explanation

        return Instance(
            question=_clean_text(question),
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

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        for response in responses:
            if (
                response.request.request_type == RequestType.LOGLIKELIHOOD
                and len(response.outputs) > 1
            ):
                # MC logprob mode: pick the continuation with highest logprob
                best_idx = 0
                best_logprob = float("-inf")
                for i, output in enumerate(response.outputs):
                    logprob = (
                        output.metadata.get("total_logprob", float("-inf"))
                        if output.metadata
                        else float("-inf")
                    )
                    if logprob > best_logprob:
                        best_logprob = logprob
                        best_idx = i
                letter = chr(ord("A") + best_idx)
                for output in response.outputs:
                    output.extracted_answer = letter
            else:
                for output in response.outputs:
                    output.extracted_answer = self.extract_answer(output)

    def extract_answer(self, output: LMOutput) -> str | None:
        text = output.text

        # 1. Try "ANSWER: X"
        matches = list(_ANSWER_PATTERN.finditer(text))
        if matches:
            return matches[-1].group(1).upper()

        # 2. Try parenthesized letter "(X)"
        paren = list(_PAREN_LETTER.finditer(text))
        if paren:
            return paren[-1].group(1).upper()

        # 3. Fallback: last standalone capital letter
        caps = list(_LAST_CAPITAL.finditer(text))
        if caps:
            last = caps[-1].group(1).upper()
            if last in {"A", "B", "C", "D"}:
                return last

        return None


_TITLE_MARKER = re.compile(r"\s*\[title\]\s*", re.IGNORECASE)
_MULTI_SPACE = re.compile(r" {2,}")


def _clean_text(text: str) -> str:
    """Normalize GPQA text while preserving bracketed scientific content."""
    text = _TITLE_MARKER.sub(". ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_SUBSETS = ("gpqa_diamond", "gpqa_main", "gpqa_extended")
_SUBJECTS = ("biology", "chemistry", "physics")

for _subset in _SUBSETS:
    # Full-subset task (no filtering)
    _cls = type(
        _subset.title().replace("_", ""),
        (GPQATask,),
        {
            "__module__": __name__,
            "__qualname__": _subset.title().replace("_", ""),
            "data_source": DataSource(path="Idavidrein/gpqa", subset=_subset),
            "formatter": MCQAChatFormatter(system_prompt=_SYSTEM_PROMPT),
            "metrics": _DEFAULT_METRICS,
            "primary_metric": _DEFAULT_ACCURACY,
            "sampling_params": _DEFAULT_SAMPLING,
        },
    )
    globals()[_cls.__name__] = _cls
    register(_subset)(_cls)
    register_variant(_subset, "mc", formatter=MultipleChoiceFormatter(), metrics=_DEFAULT_METRICS)
    register_variant(_subset, "bpb", formatter=PPLFormatter(), metrics=(BPBMetric(),))

    # Subject-filtered tasks
    for _subject in _SUBJECTS:
        _task_name = f"{_subset}_{_subject}"
        _class_name = _task_name.title().replace("_", "")
        _cls_subj = type(
            _class_name,
            (GPQATask,),
            {
                "__module__": __name__,
                "__qualname__": _class_name,
                "data_source": DataSource(path="Idavidrein/gpqa", subset=_subset),
                "subject_group": _subject,
                "formatter": MCQAChatFormatter(system_prompt=_SYSTEM_PROMPT),
                "metrics": _DEFAULT_METRICS,
                "primary_metric": _DEFAULT_ACCURACY,
                "sampling_params": _DEFAULT_SAMPLING,
            },
        )
        globals()[_class_name] = _cls_subj
        register(_task_name)(_cls_subj)
        register_variant(
            _task_name, "mc", formatter=MultipleChoiceFormatter(), metrics=_DEFAULT_METRICS
        )
        register_variant(_task_name, "bpb", formatter=PPLFormatter(), metrics=(BPBMetric(),))
