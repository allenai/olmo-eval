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

from olmo_eval.core import (
    BitsPerByteScorer,
    BPBMetric,
    Instance,
    LMOutput,
    LMRequest,
    PPLFormatter,
    RequestType,
    SamplingParams,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.core import Task, TaskConfig, register, register_variant

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

    default_source: str = "allenai/multilingual_mbpp"
    normalize_line_endings: bool = False  # Set True for v2fix

    def __init__(self, config: TaskConfig, language: str) -> None:
        super().__init__(config)
        self.language = language

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

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split).with_subset(self.language)
        except ValueError:
            return DataSource(
                path=self.default_source,
                subset=self.language,
                split=split,
            )

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
            request_type=RequestType.COMPLETION,
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
# Task Configurations and Registration
# =============================================================================


def _make_mt_mbpp_config(language: str) -> TaskConfig:
    """Create config for mt_mbpp_{language} task."""
    return TaskConfig(
        name=f"mt_mbpp_{language}",
        data_source=DataSource(path="allenai/multilingual_mbpp", subset=language),
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=("\n\n",),
        ),
    )


def _make_mt_mbpp_v2fix_config(language: str) -> TaskConfig:
    """Create config for mt_mbpp_v2fix_{language} task."""
    return TaskConfig(
        name=f"mt_mbpp_v2fix_{language}",
        data_source=DataSource(path="allenai/multilingual_mbpp", subset=language),
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=("\n\n",),
        ),
    )


# Register all mt_mbpp_{language} tasks
for _lang in MULTILINGUAL_MBPP_LANGUAGES:

    def _make_config_factory(lang: str):
        return lambda: _make_mt_mbpp_config(lang)

    def _make_class_factory(lang: str):
        class _MultilingualMBPP(MultilingualMBPPTask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, language=lang)

        _MultilingualMBPP.__name__ = f"MultilingualMBPP_{lang.title()}"
        return _MultilingualMBPP

    register(f"mt_mbpp_{_lang}", _make_config_factory(_lang))(_make_class_factory(_lang))


# Register all mt_mbpp_v2fix_{language} tasks
for _lang in MULTILINGUAL_MBPP_LANGUAGES:

    def _make_v2fix_config_factory(lang: str):
        return lambda: _make_mt_mbpp_v2fix_config(lang)

    def _make_v2fix_class_factory(lang: str):
        class _MultilingualMBPPV2Fix(MultilingualMBPPV2FixTask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, language=lang)

        _MultilingualMBPPV2Fix.__name__ = f"MultilingualMBPPV2Fix_{lang.title()}"
        return _MultilingualMBPPV2Fix

    register(f"mt_mbpp_v2fix_{_lang}", _make_v2fix_config_factory(_lang))(
        _make_v2fix_class_factory(_lang)
    )


# =============================================================================
# Variant Registrations
# =============================================================================

# Register bpb and 3shot variants for all mt_mbpp_{language} tasks
for _lang in MULTILINGUAL_MBPP_LANGUAGES:
    # BPB variant - use mt_mbpp_{language}:bpb
    register_variant(
        f"mt_mbpp_{_lang}",
        "bpb",
        # Matches oe-eval: no leading space, always prepend \n\n separator
        formatter=PPLFormatter(leading_space=False, always_prepend_separator=True),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )

    # 3shot variant - composable with bpb (e.g., mt_mbpp_{language}:3shot:bpb)
    register_variant(
        f"mt_mbpp_{_lang}",
        "3shot",
        num_fewshot=3,
    )


# Register bpb and 3shot variants for all mt_mbpp_v2fix_{language} tasks
for _lang in MULTILINGUAL_MBPP_LANGUAGES:
    # BPB variant - use mt_mbpp_v2fix_{language}:bpb
    register_variant(
        f"mt_mbpp_v2fix_{_lang}",
        "bpb",
        # Matches oe-eval: no leading space, always prepend \n\n separator
        formatter=PPLFormatter(leading_space=False, always_prepend_separator=True),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )

    # 3shot variant - composable with bpb (e.g., mt_mbpp_v2fix_{language}:3shot:bpb)
    register_variant(
        f"mt_mbpp_v2fix_{_lang}",
        "3shot",
        num_fewshot=3,
    )


# Export constants
__all__ = [
    "MULTILINGUAL_MBPP_LANGUAGES",
    "MultilingualMBPPTask",
    "MultilingualMBPPV2FixTask",
]
