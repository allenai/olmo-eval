"""HumanEval code generation task implementations."""

from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.core import (
    BitsPerByteScorer,
    BPBMetric,
    Instance,
    LMOutput,
    LMRequest,
    PPLFormatter,
    RequestType,
    Response,
    SamplingParams,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import HUMANEVAL_STOP_SEQUENCES
from olmo_eval.evals.extract import extract_code
from olmo_eval.evals.tasks.core import Task, TaskConfig, register, register_variant


class HumanEvalTask(Task):
    """HumanEval code generation task."""

    default_hf_path: str = "openai_humaneval"
    fewshot_split: str = "test"  # HumanEval only has a test split

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

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
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        prompt = "```python\n" + doc["prompt"]
        unit_tests = doc["test"] + f"\ncheck({doc['entry_point']})"

        return Instance(
            question=prompt,
            gold_answer=doc["canonical_solution"],
            metadata={
                "id": doc["task_id"],
                "entry_point": doc["entry_point"],
                "answer_prefix": doc["prompt"],
                "test": unit_tests,
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
        """Extract code from model output.

        Note: This base implementation just extracts code. The actual answer
        with prefix is computed in score_responses which has access to the instance.
        """
        return extract_code(output.text)

    def score_responses(self, responses: Sequence[Response]) -> Sequence[Response]:
        """Apply all scorers to extract answers and compute scores."""
        for response in responses:
            for output in response.outputs:
                code = self.extract_answer(output)
                if code:
                    # For Humaneval, we follow the original paper setup by adding the prompt
                    # to the generated code completion as the prompt may provide additional
                    # library imports needed for the code execution.
                    output.extracted_answer = response.instance.metadata["answer_prefix"] + code
                else:
                    output.extracted_answer = None
            # Apply each scorer, taking best score across outputs (for multi-sample)
            for scorer in self.config.scorers:
                scores = [scorer.score(response.instance, o) for o in response.outputs]
                response.scores[scorer.name] = max(scores) if scores else 0.0
        return responses

class HumanEvalPlusTask(HumanEvalTask):
    """HumanEval+ task with additional test cases."""

    default_hf_path: str = "evalplus/humanevalplus"


class HumanEvalInloopBPBTask(HumanEvalTask):
    """HumanEval BPB task matching oe-eval's inloop_bpb variant.

    Key differences from base HumanEvalTask:
    - No prompt_suffix (just doc["prompt"] as context)
    - Few-shot examples use space before answer (matching oe-eval's doc_to_target)
    """

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance for BPB evaluation.

        Uses raw prompt as context (no prompt_suffix), matching oe-eval's
        inloop_bpb variant where answer_prefix="".
        """
        # Just the function signature/docstring as context (no prompt_suffix)
        prompt = doc["prompt"]
        unit_tests = doc["test"] + f"\ncheck({doc['entry_point']})"

        return Instance(
            question=prompt,
            gold_answer=doc["canonical_solution"],
            metadata={
                "id": doc["task_id"],
                "entry_point": doc["entry_point"],
                "answer_prefix": doc["prompt"],
                "test": unit_tests,
            },
        )

class HumanEvalPlusInloopBPBTask(HumanEvalInloopBPBTask):
    """HumanEval+ BPB task matching oe-eval's inloop_bpb variant."""

    default_hf_path: str = "evalplus/humanevalplus"


# =============================================================================
# Task Configs
# =============================================================================


def _humaneval_config() -> TaskConfig:
    return TaskConfig(
        name="humaneval",
        data_source=DataSource(path="openai_humaneval"),
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=HUMANEVAL_STOP_SEQUENCES,
        ),
    )


def _humaneval_plus_config() -> TaskConfig:
    return TaskConfig(
        name="humaneval_plus",
        data_source=DataSource(path="evalplus/humanevalplus"),
        scorers=(),
        metrics=(),
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.0,
            stop_sequences=HUMANEVAL_STOP_SEQUENCES,
        ),
    )


def _humaneval_inloop_bpb_config() -> TaskConfig:
    """Config matching oe-eval's codex_humaneval with inloop_bpb variant.

    Key differences from standard BPB config:
    - No prompt_suffix in prompt (handled by HumanEvalInloopBPBTask)
    - PPLFormatter.answer_prefix=" " to add space before answer in few-shot examples
      (matching oe-eval's doc_to_target which returns " " + canonical_solution)
    - fewshot_seed=1234 to match oe-eval's default
    """
    return TaskConfig(
        name="humaneval_inloop_bpb",
        data_source=DataSource(path="openai_humaneval"),
        # answer_prefix=" " matches oe-eval's doc_to_target behavior
        formatter=PPLFormatter(answer_prefix=" "),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
        num_fewshot=3,
        fewshot_seed=1234,
    )


def _humaneval_plus_inloop_bpb_config() -> TaskConfig:
    """Config matching oe-eval's codex_humaneval with inloop_bpb variant for HumanEval+."""
    return TaskConfig(
        name="humaneval_plus_inloop_bpb",
        data_source=DataSource(path="evalplus/humanevalplus"),
        formatter=PPLFormatter(answer_prefix=" "),
        scorers=(BitsPerByteScorer(),),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
        num_fewshot=3,
        fewshot_seed=1234,
    )


# =============================================================================
# Task Registrations
# =============================================================================


@register("humaneval", _humaneval_config)
class HumanEval(HumanEvalTask):
    """HumanEval code generation task."""

    pass


@register("humaneval_plus", _humaneval_plus_config)
class HumanEvalPlus(HumanEvalPlusTask):
    """HumanEval+ code generation task."""

    pass


# Register with underscore naming to avoid variant parsing issues
@register("humaneval_inloop_bpb", _humaneval_inloop_bpb_config)
class HumanEvalInloopBPB(HumanEvalInloopBPBTask):
    """HumanEval BPB evaluation task matching oe-eval's inloop_bpb variant.

    Use as: humaneval_inloop_bpb or humaneval_inloop_bpb:3shot
    """

    pass


@register("humaneval_plus_inloop_bpb", _humaneval_plus_inloop_bpb_config)
class HumanEvalPlusInloopBPB(HumanEvalPlusInloopBPBTask):
    """HumanEval+ BPB evaluation task matching oe-eval's inloop_bpb variant.

    Use as: humaneval_plus_inloop_bpb or humaneval_plus_inloop_bpb:3shot
    """

    pass


# =============================================================================
# Variant Registrations
# =============================================================================

# BPB variant - use humaneval:bpb or humaneval_plus:bpb
# Uses leading_space=True and answer_prefix=" " to match oe-eval's doc_to_target
# which returns " " + canonical_solution (space before answer)
register_variant(
    "humaneval",
    "bpb",
    formatter=PPLFormatter(leading_space=True, answer_prefix=" "),
    scorers=(BitsPerByteScorer(),),
    metrics=(BPBMetric(),),
    primary_metric=BPBMetric(),
)

register_variant(
    "humaneval_plus",
    "bpb",
    formatter=PPLFormatter(leading_space=True, answer_prefix=" "),
    scorers=(BitsPerByteScorer(),),
    metrics=(BPBMetric(),),
    primary_metric=BPBMetric(),
)

# 3shot variant for inloop_bpb task
register_variant(
    "humaneval_inloop_bpb",
    "3shot",
    num_fewshot=3,
    fewshot_seed=1234,
)

# 3shot variants - composable with bpb (e.g., humaneval:3shot:bpb)
# Uses fewshot_seed=1234 to match oe-eval's default
register_variant(
    "humaneval",
    "3shot",
    num_fewshot=3,
    fewshot_seed=1234,
)

register_variant(
    "humaneval_plus",
    "3shot",
    num_fewshot=3,
    fewshot_seed=1234,
)
