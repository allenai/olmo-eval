"""Multilingual MBPP task implementations.

Multilingual MBPP contains MBPP problems translated to 17 programming languages
using o4-mini. The v2fix version includes fixes for Windows line endings.

Languages:
- bash, c, cpp, csharp, go, haskell, java, javascript, matlab,
  php, python, r, ruby, rust, scala, swift, typescript

Dataset: allenai/multilingual_mbpp
"""

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import PPLFormatter
from olmo_eval.common.metrics import BPBMetric
from olmo_eval.common.types import Instance, LMOutput, LMRequest, SamplingParams
from olmo_eval.data import DataLoader
from olmo_eval.evals.tasks.common import Task, register_subtasks

# Supported languages in multilingual MBPP
MULTILINGUAL_MBPP_LANGUAGES: tuple[str, ...] = (
    "bash",
    "c",
    "cpp",
    "csharp",
    "go",
    "haskell",
    "java",
    "javascript",
    "matlab",
    "php",
    "python",
    "r",
    "ruby",
    "rust",
    "scala",
    "swift",
    "typescript",
)


class MultilingualMBPPTask(Task):
    """Base class for Multilingual MBPP tasks.

    Each language variant loads from a different subset of the dataset.
    The v2fix version normalizes Windows line endings (\\r\\n -> \\n).
    """

    normalize_line_endings: bool = False  # Set True for v2fix
    language: str = "python"  # Override in subclasses

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _normalize(self, text: str) -> str:
        """Normalize line endings if v2fix mode."""
        if self.normalize_line_endings:
            return text.replace("\r\n", "\n")
        return text

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        text = self._normalize(doc["text"]).strip()
        code = self._normalize(doc["code"]).strip()

        # Build prompt: task description + code fence start (matches oe-eval fewshot_context)
        # In oe-eval, the prompt ends with the code fence opening
        question = text + f"\n```{self.language}\n"

        # Gold answer is just the code with closing fence (matches oe-eval choices[0])
        gold_answer = code + "\n```"

        return Instance(
            question=question,
            gold_answer=gold_answer,
            metadata={
                "id": doc["task_id"],
                "language": self.language,
                "text": text,
                "code": code,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract code from model output."""
        text = output.text
        # Remove closing fence if present
        if "```" in text:
            text = text.split("```")[0]
        return text.strip() if text else None

    def _build_fewshot(self) -> list[Instance]:
        """Build few-shot examples from the prompt split.

        Multilingual MBPP uses the same structure as MBPP with a 'prompt' split
        containing examples for few-shot prompting.
        Falls back to 'train' split if 'prompt' is not available.
        """
        return self._build_fewshot_from_source(
            split="prompt",
            sample=True,
            fallback_splits=["train"],
        )


class MultilingualMBPPV2FixTask(MultilingualMBPPTask):
    """Multilingual MBPP with Windows line ending fixes."""

    normalize_line_endings: bool = True


# =============================================================================
# Task Registration
# =============================================================================

_SHARED_ATTRS: dict = {
    "metrics": (),
    "sampling_params": SamplingParams(max_tokens=1024, temperature=0.0, stop_sequences=("\n\n",)),
}

_VARIANTS: dict = {
    "bpb": {
        "formatter": PPLFormatter(leading_space=False, always_prepend_separator=True),
        "metrics": (BPBMetric(),),
    },
    "3shot": {"num_fewshot": 3},
}

for _base, _prefix in [
    (MultilingualMBPPTask, "mt_mbpp"),
    (MultilingualMBPPV2FixTask, "mt_mbpp_v2fix"),
]:
    register_subtasks(
        _base,
        list(MULTILINGUAL_MBPP_LANGUAGES),
        task_prefix=_prefix,
        data_source="allenai/multilingual_mbpp",
        subtask_attr="language",
        class_attrs=_SHARED_ATTRS,
        variants=_VARIANTS,
    )


# Export constants
__all__ = [
    "MULTILINGUAL_MBPP_LANGUAGES",
    "MultilingualMBPPTask",
    "MultilingualMBPPV2FixTask",
]
