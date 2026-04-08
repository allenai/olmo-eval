"""HumanEval code generation task implementations."""

from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.formatters import ChatFormatter, CompletionFormatter, PPLFormatter
from olmo_eval.common.metrics import BPBMetricByteAvg, PassAtKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    Response,
    SamplingParams,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.code import HUMANEVAL_STOP_SEQUENCES
from olmo_eval.evals.extract import extract_code, indent_code
from olmo_eval.evals.tasks.common import Task, register, register_variant


@register("humaneval")
class HumanEval(Task):
    """HumanEval code generation task."""

    data_source = DataSource(path="openai_humaneval")
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
        stop_sequences=HUMANEVAL_STOP_SEQUENCES,
    )
    fewshot_split: str = "test"  # HumanEval only has a test split

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

    def _build_fewshot(self) -> list[Instance]:
        """Sample one extra fewshot example for deduplication.

        HumanEval only has a test split, so fewshot examples come from the same
        pool as eval instances.  The old oe-eval framework samples k+1 and then
        removes the eval doc if it appears in the fewshot set.  We replicate that
        behaviour here.
        """
        import random

        if self.config.num_fewshot == 0:
            return []

        loader = DataLoader()
        source = self._get_source_for_split(self.fewshot_split)
        all_instances = [
            inst for doc in loader.load(source) if (inst := self.process_doc(doc)) is not None
        ]

        if not all_instances:
            return []

        rng = random.Random(self.config.fewshot_seed)
        # Sample one extra to allow per-instance deduplication
        k = min(self.config.num_fewshot + 1, len(all_instances))
        return rng.sample(all_instances, k)

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request.

        Excludes the current instance from the fewshot set if it appears there
        (possible because HumanEval draws fewshot from the test split).
        """
        if self.config.formatter is not None:
            fewshot = self.get_fewshot()
            instance_id = instance.metadata.get("id")
            if instance_id is not None:
                filtered = [ex for ex in fewshot if ex.metadata.get("id") != instance_id]
            else:
                filtered = list(fewshot)
            # Keep only num_fewshot examples
            filtered = filtered[: self.config.num_fewshot]
            return self.config.formatter.format(instance, filtered)

        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract code from model output.

        Note: This base implementation just extracts code. The actual answer
        with prefix is computed in score_responses which has access to the instance.
        """
        return extract_code(output.text)

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Extract code and prepend answer prefix.

        HumanEval follows the original paper setup by adding the prompt
        to the generated code completion as the prompt may provide additional
        library imports needed for the code execution.

        Chat/instruction models often output function body code without the
        leading indentation expected inside a function. We normalize the
        indentation to ensure the code is valid when concatenated with the
        function signature.
        """
        for response in responses:
            for output in response.outputs:
                code = self.extract_answer(output)
                if code:
                    # Ensure code has proper indentation for function body
                    code = indent_code(code)
                    output.extracted_answer = response.instance.metadata["answer_prefix"] + code
                else:
                    output.extracted_answer = None


@register("humaneval_plus")
class HumanEvalPlus(HumanEval):
    """HumanEval+ task with additional test cases."""

    data_source = DataSource(path="evalplus/humanevalplus")


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
    metrics=(BPBMetricByteAvg(),),
)

register_variant(
    "humaneval_plus",
    "bpb",
    formatter=PPLFormatter(leading_space=True, answer_prefix=" "),
    metrics=(BPBMetricByteAvg(),),
)

# 3shot variants - composable with bpb (e.g., humaneval:3shot:bpb)
# Uses fewshot_seed=1234 to match oe-eval's default
register_variant(
    "humaneval",
    "3shot",
    num_fewshot=3,
    fewshot_seed=1234,
    formatter=CompletionFormatter(),
)

register_variant(
    "humaneval_plus",
    "3shot",
    num_fewshot=3,
    fewshot_seed=1234,
    formatter=CompletionFormatter(),
)

# Chat variants for instruction-tuned models
# Use with agent backends: humaneval:chat:pass_at_1
# Note: System prompt is owned by the harness (e.g., codex_agent preset)
_CHAT_USER_TEMPLATE = """\
Complete this Python function. Write only the function body (the implementation \
code that goes inside the function). Do not repeat the function signature or docstring.

```python
{question}
```"""

register_variant(
    "humaneval",
    "chat",
    formatter=ChatFormatter(
        user_template=_CHAT_USER_TEMPLATE,
        assistant_template="{answer}",
    ),
)

register_variant(
    "humaneval_plus",
    "chat",
    formatter=ChatFormatter(
        user_template=_CHAT_USER_TEMPLATE,
        assistant_template="{answer}",
    ),
)

# =============================================================================
# Pass@K Execution Variants (require sandbox)
# =============================================================================
# These variants execute generated code against test cases.
# Requires HarnessConfig with sandboxes configured:
#   sandboxes=(SandboxConfig(image="..."),)

register_variant(
    "humaneval",
    "pass_at_1",
    metrics=(PassAtKMetric(k=1, scorer=CodeExecutionScorer),),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.2,
        stop_sequences=HUMANEVAL_STOP_SEQUENCES,
    ),
)

register_variant(
    "humaneval",
    "pass_at_10",
    metrics=(PassAtKMetric(k=10, scorer=CodeExecutionScorer),),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.8,
        num_samples=10,
        stop_sequences=HUMANEVAL_STOP_SEQUENCES,
    ),
)

register_variant(
    "humaneval_plus",
    "pass_at_1",
    metrics=(PassAtKMetric(k=1, scorer=CodeExecutionScorer),),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.2,
        stop_sequences=HUMANEVAL_STOP_SEQUENCES,
    ),
)

register_variant(
    "humaneval_plus",
    "pass_at_10",
    metrics=(PassAtKMetric(k=10, scorer=CodeExecutionScorer),),
    sampling_params=SamplingParams(
        max_tokens=1024,
        temperature=0.8,
        num_samples=10,
        stop_sequences=HUMANEVAL_STOP_SEQUENCES,
    ),
)
