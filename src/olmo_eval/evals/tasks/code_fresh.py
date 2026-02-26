"""C4 perplexity task implementations."""

from collections.abc import Iterator
from typing import Any
from unicode_segmentation_rs import split_word_bound_indices

from olmo_eval.common.metrics import CorpusPerplexityMetric
from olmo_eval.common.types import (
    Instance,
    LMRequest,
    RequestType,
    Split,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.common.types import Instance, LMOutput, LMRequest, SamplingParams
from olmo_eval.evals.tasks.common import Task, register, register_variant, register_subtasks
from olmo_eval.common.formatters import PPLFormatter
from olmo_eval.common.metrics import BPBMetric

MAX_LENGTH = 4096


class CodeFresh(Task):
    """Base class for CodeFresh perplexity tasks."""

    split = Split.TRAIN

    @property
    def request_type(self) -> RequestType:
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.LOGLIKELIHOOD

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self.config.get_data_source()
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        text = doc["file_contents"].strip()[:MAX_LENGTH * 4] # assuming a generous 4 tokens per character

        return Instance(
            question="",  # Context
            gold_answer=text,  # The text we score as the "continuation"
            metadata={
                "id": index,
                "num_chars": len(text),
                "num_words": len(text.split()),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        gold = instance.gold_answer
        continuations = (gold,) if gold is not None else None
        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
            continuations=continuations,
        )


# =============================================================================
# Variant Registrations
# =============================================================================

LANGUAGES = [
    "C"
    "C#"
    "C++"
    "CSS"
    "Clojure"
    "Common Lisp"
    "Dart"
    "Erlang"
    "Fortran"
    "Go"
    "HTML"
    "Haskell"
    "Java"
    "Java Server Page"
    "JavaScript"
    "Julia"
    "Kotlin"
    "Lua"
    "Markdown"
    "Mathematica"
    "Matlab"
    "OCaml"
    "Objective-C"
    "Objective-C++"
    "PHP"
    "Perl"
    "PowerShell"
    "Python"
    "Ruby"
    "Rust"
    "Scala"
    "Scheme"
    "Swift"
    "Tcl"
    "TeX"
    "TypeScript"
    "Vue"
    "reStructuredText"
    "systemverilog"
    "verilog"
    "vhdl"
]

SHARED_ATTRS: dict = {
    "metrics": (),
    "sampling_params": SamplingParams(max_tokens=MAX_LENGTH, temperature=0.0, stop_sequences=("\n\n",)),
}

VARIANTS: dict = {
    "bpb": {
        "formatter": PPLFormatter(leading_space=False, always_prepend_separator=True),
        "metrics": (BPBMetric(),),
    },
    "3shot": {"num_fewshot": 0},
}

register_subtasks(
    CodeFresh,
    subtasks=LANGUAGES,
    task_prefix="code_fresh_file",
    data_source="allenai/dolma_eval_code_perplexity_T3_2025_1M_file",
    subtask_attr="subset",
    class_attrs=SHARED_ATTRS,
    variants=VARIANTS,
)

__all__ = ["CodeFresh"]
