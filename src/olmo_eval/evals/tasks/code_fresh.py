"""Code fresh perplexity task implementations."""

from collections.abc import Iterator
from typing import Any

from olmo_eval.common.formatters import PPLFormatter
from olmo_eval.common.metrics import BPBMetric, CorpusPerplexityMetric
from olmo_eval.common.types import (
    Instance,
    LMRequest,
    RequestType,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.common import Task, register_subtasks

MAX_LENGTH = 4096


class CodeFreshBase(Task):
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
                if doc["file_tokens"] * 1.05 > MAX_LENGTH:
                    # the 1.05 is a bit of buffer to avoid edge cases
                    continue

                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        text = doc["file_contents"].strip()

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


# @register("code_fresh")
class CodeFreshFile(CodeFreshBase):
    """MBPP code generation task."""

    data_source = DataSource(path="allenai/dolma_eval_code_perplexity_T3_2025_1M_file")


# =============================================================================
# Variant Registrations
# =============================================================================

LANGUAGES = [
    "blade",
    "c",
    "clojure",
    "common_lisp",
    "cpp",
    "csharp",
    "css",
    "dart",
    "erlang",
    "fortran",
    "go",
    "haskell",
    "html",
    "java",
    "java_server_page",
    "javascript",
    "julia",
    "kotlin",
    "lua",
    "markdown",
    "mathematica",
    "matlab",
    "objective_c",
    "objective_cpp",
    "ocaml",
    "perl",
    "php",
    "powershell",
    "python",
    "restructuredtext",
    "ruby",
    "rust",
    "scala",
    "scheme",
    "swift",
    "systemverilog",
    "tcl",
    "tex",
    "typescript",
    "verilog",
    "vhdl",
    "vue",
]

SHARED_ATTRS: dict = {
    "metrics": (),
    "sampling_params": SamplingParams(max_tokens=MAX_LENGTH, temperature=0.0, stop_sequences=()),
}

VARIANTS: dict = {
    "bpb": {
        "formatter": PPLFormatter(leading_space=False, always_prepend_separator=True),
        "metrics": (BPBMetric(),),
    },
    "ppl": {
        "formatter": PPLFormatter(leading_space=False, always_prepend_separator=True),
        "metrics": (CorpusPerplexityMetric(),),
    },
}

register_subtasks(
    CodeFreshFile,
    subtasks=LANGUAGES,
    task_prefix="code_fresh",
    data_source="allenai/dolma_eval_code_perplexity_T3_2025_1M_file",
    subtask_attr="subset",
    class_attrs=SHARED_ATTRS,
    variants=VARIANTS,
)

make_suite(
    "code_fresh:bpb",
    tuple(f"code_fresh_{language}:bpb" for language in LANGUAGES),
    aggregation=AggregationStrategy.AVERAGE,
)

__all__ = ["CodeFreshFile"]
