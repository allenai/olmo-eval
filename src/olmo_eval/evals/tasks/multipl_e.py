"""MULTIPL_E task implementations.

MULTIPL_E contains HumanEval and MBPP problems translated to multiple programming
languages. This implementation supports 6 languages with code execution evaluation:
cpp, java, js, php, rs, sh.

Dataset: nuprl/MultiPL-E
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import PPLFormatter
from olmo_eval.common.metrics import BPBMetric, PassAtKMetric
from olmo_eval.common.scorers import MultiplEScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, SamplingParams
from olmo_eval.data import DataLoader
from olmo_eval.evals.constants.code import MULTIPL_E_LANGUAGES, MULTIPL_E_STOP_SEQUENCES
from olmo_eval.evals.tasks.common import Task, register_subtasks


class MultiplETask(Task):
    """Base class for MULTIPL_E tasks.

    Each language variant loads from a different subset of the dataset.
    Supports both HumanEval and MBPP problem sets.
    """

    language: str = "cpp"

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

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        return Instance(
            question=doc["prompt"],
            gold_answer="",
            metadata={
                "id": doc["name"],
                "language": self.language,
                "test": doc["tests"],
                "answer_prefix": doc["prompt"],
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
        """Extract the answer from model output.

        Note: The actual answer with prefix is computed in _extract_answers
        which has access to the instance metadata.
        """
        return output.text

    def _extract_answers(self, responses: Any) -> None:
        """Extract code and prepend answer prefix.

        MULTIPL_E follows HumanEval's setup by adding the prompt to the
        generated code completion, as the prompt may provide necessary
        imports and function signatures.
        """
        for response in responses:
            for output in response.outputs:
                code = self.extract_answer(output)
                if code:
                    output.extracted_answer = response.instance.metadata["answer_prefix"] + code
                else:
                    output.extracted_answer = None


# =============================================================================
# Scorer Factory
# =============================================================================


def _make_scorer_for_language(lang: str) -> type[MultiplEScorer]:
    """Create a scorer class for a specific language.

    We need a factory because PassAtKMetric takes a scorer type, not an instance.
    """
    # Capture the language value in a local variable for the closure
    default_lang = lang

    @dataclass(frozen=True, slots=True)
    class LanguageScorer(MultiplEScorer):
        language: str = default_lang

    # Set a unique name for each language scorer
    LanguageScorer.__name__ = f"MultiplEScorer_{lang}"
    LanguageScorer.__qualname__ = f"MultiplEScorer_{lang}"

    return LanguageScorer


# =============================================================================
# Task Registration
# =============================================================================


def _get_variants(language: str) -> dict[str, dict[str, Any]]:
    """Get variant configurations for a language."""
    scorer_cls = _make_scorer_for_language(language)

    return {
        "bpb": {
            "formatter": PPLFormatter(leading_space=False, always_prepend_separator=True),
            "metrics": (BPBMetric(),),
        },
        "pass_at_1": {
            "metrics": (PassAtKMetric(k=1, scorer=scorer_cls),),
        },
        "pass_at_10": {
            "metrics": (PassAtKMetric(k=10, scorer=scorer_cls),),
            "sampling_params": SamplingParams(
                max_tokens=512,
                temperature=0.8,
                num_samples=20,
            ),
        },
    }


# Register HumanEval tasks for each language
for _lang in MULTIPL_E_LANGUAGES:
    register_subtasks(
        MultiplETask,
        [_lang],
        task_prefix="multipl_e_humaneval",
        data_source=f"nuprl/MultiPL-E:humaneval-{_lang}",
        subtask_attr="language",
        class_attrs={
            "metrics": (),
            "sampling_params": SamplingParams(
                max_tokens=512,
                temperature=0.0,
                stop_sequences=MULTIPL_E_STOP_SEQUENCES[_lang],
            ),
        },
        variants=_get_variants(_lang),
    )

# Register MBPP tasks for each language
for _lang in MULTIPL_E_LANGUAGES:
    register_subtasks(
        MultiplETask,
        [_lang],
        task_prefix="multipl_e_mbpp",
        data_source=f"nuprl/MultiPL-E:mbpp-{_lang}",
        subtask_attr="language",
        class_attrs={
            "metrics": (),
            "sampling_params": SamplingParams(
                max_tokens=512,
                temperature=0.0,
                stop_sequences=MULTIPL_E_STOP_SEQUENCES[_lang],
            ),
        },
        variants=_get_variants(_lang),
    )


# Export
__all__ = [
    "MultiplETask",
]
