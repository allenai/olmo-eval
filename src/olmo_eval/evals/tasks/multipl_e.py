"""MULTIPL_E task implementations.

MULTIPL_E contains HumanEval and MBPP problems translated to multiple programming
languages. This implementation supports 6 languages with code execution evaluation:
cpp, java, js, php, rs, sh.

Dataset: nuprl/MultiPL-E
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics import PassAtKMetric
from olmo_eval.common.scorers import MultiplEScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import MULTIPL_E_LANGUAGES, MULTIPL_E_STOP_TOKENS
from olmo_eval.evals.tasks.common import Task, register, register_variant


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
        """Convert a dataset document to an Instance.

        Dataset schema:
        - name: Task identifier (e.g., "HumanEval_0_has_close_elements")
        - language: Programming language (e.g., "cpp")
        - prompt: Code prefix with includes/imports and function signature
        - tests: Test code (closing brace + main with assertions)
        - stop_tokens: Stop sequences for this item
        """
        return Instance(
            question=doc["prompt"],
            gold_answer="",
            metadata={
                "id": doc["name"],
                "language": doc["language"],
                "prompt": doc["prompt"],
                "test": doc["tests"],
                "stop_tokens": doc.get("stop_tokens", []),
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

    def get_sampling_params(self, instance: Instance) -> SamplingParams | None:
        """Get sampling params with instance-specific stop tokens."""
        from dataclasses import replace

        base_params = self.config.sampling_params or SamplingParams()

        # If config already has explicit stop_sequences, use those only
        # (matches oe-eval-internal behavior where hardcoded stop tokens override per-doc)
        if base_params.stop_sequences:
            return base_params

        # Otherwise, use per-document stop_tokens from the dataset
        stop_tokens = instance.metadata.get("stop_tokens", [])
        if stop_tokens:
            return replace(base_params, stop_sequences=tuple(stop_tokens))

        return base_params

    def _extract_answers(self, responses: Any) -> None:
        """Extract code and prepend the prompt.

        MULTIPL_E follows HumanEval's setup by adding the prompt to the
        generated code completion, as the prompt contains necessary
        imports and function signatures.
        """
        for response in responses:
            for output in response.outputs:
                code = self.extract_answer(output)
                if code:
                    output.extracted_answer = response.instance.metadata["prompt"] + code
                else:
                    output.extracted_answer = None


# =============================================================================
# Scorer Factory
# =============================================================================

# Cache for language-specific scorer classes (needed for pickling)
_SCORER_CACHE: dict[str, type[MultiplEScorer]] = {}


def _make_scorer_for_language(lang: str) -> type[MultiplEScorer]:
    """Create a scorer class for a specific language.

    We need a factory because PassAtKMetric takes a scorer type, not an instance.
    Scorer classes are cached and added to module namespace for pickling support.
    """
    if lang in _SCORER_CACHE:
        return _SCORER_CACHE[lang]

    # Capture the language value in a local variable for the closure
    default_lang = lang
    class_name = f"MultiplEScorer_{lang}"

    @dataclass(frozen=True, slots=True)
    class LanguageScorer(MultiplEScorer):
        language: str = default_lang

    # Set a unique name for each language scorer
    LanguageScorer.__name__ = class_name
    LanguageScorer.__qualname__ = class_name

    # Make class picklable by adding to module namespace
    globals()[class_name] = LanguageScorer
    _SCORER_CACHE[lang] = LanguageScorer

    return LanguageScorer


# =============================================================================
# Task Registration
# =============================================================================


def _register_humaneval_task(lang: str) -> None:
    """Register a MULTIPL_E HumanEval task for a specific language."""
    task_name = f"multipl_e_humaneval_{lang}"
    subset_name = f"humaneval-{lang}"
    scorer_cls = _make_scorer_for_language(lang)

    class_name = f"MultiplETask_humaneval_{lang}"
    task_cls = type(
        class_name,
        (MultiplETask,),
        {
            "language": lang,
            "data_source": DataSource(path="nuprl/MultiPL-E", subset=subset_name),
            "metrics": (),
            "sampling_params": SamplingParams(
                max_tokens=512,
                temperature=0.2,
                do_sample=True,
                top_p=0.95,
                num_samples=20,
            ),
            "__module__": __name__,
            "__qualname__": class_name,
        },
    )

    register(task_name)(task_cls)

    register_variant(
        task_name,
        "pass_at_1",
        metrics=(PassAtKMetric(k=1, scorer=scorer_cls),),
    )
    register_variant(
        task_name,
        "pass_at_10",
        metrics=(PassAtKMetric(k=10, scorer=scorer_cls),),
        sampling_params=SamplingParams(
            max_tokens=512,
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            num_samples=20,
        ),
    )
    register_variant(
        task_name,
        "olmo3base",
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.6,
            top_p=0.6,
            do_sample=True,
            num_samples=32,
            stop_sequences=MULTIPL_E_STOP_TOKENS[lang],
        ),
        metrics=(
            PassAtKMetric(k=1, scorer=scorer_cls),
            PassAtKMetric(k=2, scorer=scorer_cls),
            PassAtKMetric(k=4, scorer=scorer_cls),
            PassAtKMetric(k=8, scorer=scorer_cls),
            PassAtKMetric(k=16, scorer=scorer_cls),
        ),
        primary_metric=PassAtKMetric(k=1, scorer=scorer_cls),
    )


def _register_mbpp_task(lang: str) -> None:
    """Register a MULTIPL_E MBPP task for a specific language."""
    task_name = f"multipl_e_mbpp_{lang}"
    subset_name = f"mbpp-{lang}"
    scorer_cls = _make_scorer_for_language(lang)

    class_name = f"MultiplETask_mbpp_{lang}"
    task_cls = type(
        class_name,
        (MultiplETask,),
        {
            "language": lang,
            "data_source": DataSource(path="nuprl/MultiPL-E", subset=subset_name),
            "metrics": (),
            "sampling_params": SamplingParams(
                max_tokens=512,
                temperature=0.2,
                do_sample=True,
                top_p=0.95,
                num_samples=20,
            ),
            "__module__": __name__,
            "__qualname__": class_name,
        },
    )

    register(task_name)(task_cls)

    register_variant(
        task_name,
        "pass_at_1",
        metrics=(PassAtKMetric(k=1, scorer=scorer_cls),),
    )
    register_variant(
        task_name,
        "pass_at_10",
        metrics=(PassAtKMetric(k=10, scorer=scorer_cls),),
        sampling_params=SamplingParams(
            max_tokens=512,
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            num_samples=20,
        ),
    )
    register_variant(
        task_name,
        "olmo3base",
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.6,
            top_p=0.6,
            do_sample=True,
            num_samples=32,
            stop_sequences=MULTIPL_E_STOP_TOKENS[lang],
        ),
        metrics=(
            PassAtKMetric(k=1, scorer=scorer_cls),
            PassAtKMetric(k=2, scorer=scorer_cls),
            PassAtKMetric(k=4, scorer=scorer_cls),
            PassAtKMetric(k=8, scorer=scorer_cls),
            PassAtKMetric(k=16, scorer=scorer_cls),
        ),
        primary_metric=PassAtKMetric(k=1, scorer=scorer_cls),
    )


# Register HumanEval tasks for each language (20 samples for pass@k)
for _lang in MULTIPL_E_LANGUAGES:
    _register_humaneval_task(_lang)

# Register MBPP tasks for each language (1 sample, pass@1 only)
for _lang in MULTIPL_E_LANGUAGES:
    _register_mbpp_task(_lang)


# Export
__all__ = [
    "MultiplETask",
]
