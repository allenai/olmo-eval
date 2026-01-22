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

from datasets import load_dataset

from olmo_eval.core import (
    BPBMetric,
    BitsPerByteScorer,
    Instance,
    LMOutput,
    LMRequest,
    PPLFormatter,
    RequestType,
    SamplingParams,
)
from olmo_eval.evals.tasks.core import Task, TaskConfig, register

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

    hf_path: str = "allenai/multilingual_mbpp"
    normalize_line_endings: bool = False  # Set True for v2fix

    def __init__(self, config: TaskConfig, language: str) -> None:
        super().__init__(config)
        self.language = language
        self._instances_cache: list[Instance] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            dataset = load_dataset(
                self.hf_path,
                name=self.language,
                split="test",
            )
            for doc in dataset:
                self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _normalize(self, text: str) -> str:
        """Normalize line endings if v2fix mode."""
        if self.normalize_line_endings:
            return text.replace("\r\n", "\n")
        return text

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        text = self._normalize(doc["text"]).strip()
        code = self._normalize(doc["code"]).strip()

        # Build prompt: task description + code fence start
        question = text + f"\n```{self.language}\n"

        # Gold answer is the code with closing fence
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
        hf_dataset="allenai/multilingual_mbpp",
        hf_subsets=(language,),
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
        hf_dataset="allenai/multilingual_mbpp",
        hf_subsets=(language,),
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=("\n\n",),
        ),
    )


def _make_mt_mbpp_bpb_config(language: str) -> TaskConfig:
    """Create config for mt_mbpp_{language}:bpb task."""
    return TaskConfig(
        name=f"mt_mbpp_{language}:bpb",
        hf_dataset="allenai/multilingual_mbpp",
        hf_subsets=(language,),
        formatter=PPLFormatter(),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )


def _make_mt_mbpp_v2fix_bpb_config(language: str) -> TaskConfig:
    """Create config for mt_mbpp_v2fix_{language}:bpb task."""
    return TaskConfig(
        name=f"mt_mbpp_v2fix_{language}:bpb",
        hf_dataset="allenai/multilingual_mbpp",
        hf_subsets=(language,),
        formatter=PPLFormatter(),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
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


# Register all mt_mbpp_{language}:bpb tasks
for _lang in MULTILINGUAL_MBPP_LANGUAGES:

    def _make_bpb_config_factory(lang: str):
        return lambda: _make_mt_mbpp_bpb_config(lang)

    def _make_bpb_class_factory(lang: str):
        class _MultilingualMBPPBPB(MultilingualMBPPTask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, language=lang)

        _MultilingualMBPPBPB.__name__ = f"MultilingualMBPPBPB_{lang.title()}"
        return _MultilingualMBPPBPB

    register(f"mt_mbpp_{_lang}:bpb", _make_bpb_config_factory(_lang))(
        _make_bpb_class_factory(_lang)
    )


# Register all mt_mbpp_v2fix_{language}:bpb tasks
for _lang in MULTILINGUAL_MBPP_LANGUAGES:

    def _make_v2fix_bpb_config_factory(lang: str):
        return lambda: _make_mt_mbpp_v2fix_bpb_config(lang)

    def _make_v2fix_bpb_class_factory(lang: str):
        class _MultilingualMBPPV2FixBPB(MultilingualMBPPV2FixTask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, language=lang)

        _MultilingualMBPPV2FixBPB.__name__ = f"MultilingualMBPPV2FixBPB_{lang.title()}"
        return _MultilingualMBPPV2FixBPB

    register(f"mt_mbpp_v2fix_{_lang}:bpb", _make_v2fix_bpb_config_factory(_lang))(
        _make_v2fix_bpb_class_factory(_lang)
    )


# Export constants
__all__ = [
    "MULTILINGUAL_MBPP_LANGUAGES",
    "MultilingualMBPPTask",
    "MultilingualMBPPV2FixTask",
]
