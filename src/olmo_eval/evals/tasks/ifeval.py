"""IFEval (Instruction-Following Evaluation) task implementation."""

import re
from collections.abc import Iterator, Sequence
from typing import Any

from datasets import load_dataset

from olmo_eval.core import (
    Instance,
    LMOutput,
    LMRequest,
    Metric,
    RequestType,
    Response,
    Scorer,
)
from olmo_eval.evals.tasks.core import Task, TaskConfig, register

# Import instruction checkers from the ifeval library if available
try:
    from instruction_following_eval import instructions_registry  # type: ignore[import-not-found]

    IFEVAL_AVAILABLE = True
except ImportError:
    IFEVAL_AVAILABLE = False
    instructions_registry = None


class IFEvalScorer(Scorer):
    """Scorer for IFEval that checks instruction following."""

    name: str = "ifeval"

    def __init__(self, strictness: str = "loose"):
        """Initialize the scorer.

        Args:
            strictness: "strict" or "loose" evaluation mode
        """
        self.strictness = strictness

    def _check_instruction(
        self,
        instruction_id: str,
        instruction_kwargs: dict,
        response: str,
    ) -> bool:
        """Check if a single instruction is followed.

        Args:
            instruction_id: The type of instruction (e.g., "length_constraints:number_words")
            instruction_kwargs: Arguments for the instruction checker
            response: The model's response text

        Returns:
            True if the instruction is followed, False otherwise
        """
        if not IFEVAL_AVAILABLE:
            # Fallback: basic checks for common instruction types
            return self._basic_instruction_check(instruction_id, instruction_kwargs, response)

        # Use the official ifeval library
        try:
            assert instructions_registry is not None
            checker_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            checker = checker_cls(**instruction_kwargs)

            if self.strictness == "strict":
                return checker.check_following(response)
            else:
                # Loose mode allows minor relaxations
                return checker.check_following(response) or self._check_loose(checker, response)
        except Exception:
            return False

    def _basic_instruction_check(
        self,
        instruction_id: str,
        instruction_kwargs: dict,
        response: str,
    ) -> bool:
        """Basic fallback checks when ifeval library is not available."""
        response_lower = response.lower()

        # Word count checks
        if "number_words" in instruction_id:
            word_count = len(response.split())
            relation = instruction_kwargs.get("relation", "at least")
            num_words = instruction_kwargs.get("num_words", 0)

            relation_checks = {
                "at least": word_count >= num_words,
                "at most": word_count <= num_words,
                "less than": word_count < num_words,
            }
            return relation_checks.get(relation, word_count >= num_words)

        # Keyword checks
        if "keywords" in instruction_id:
            keywords = instruction_kwargs.get("keywords", [])
            if "inclusion" in instruction_id:
                return all(kw.lower() in response_lower for kw in keywords)
            elif "exclusion" in instruction_id or "forbidden" in instruction_id:
                return not any(kw.lower() in response_lower for kw in keywords)

        # Format checks
        if "json_format" in instruction_id:
            try:
                import json

                json.loads(response)
                return True
            except (json.JSONDecodeError, ValueError):
                return False

        if "number_bullet_points" in instruction_id:
            num_bullets = instruction_kwargs.get("num_bullets", 0)
            bullet_count = len(re.findall(r"^\s*[\*\-•]\s", response, re.MULTILINE))
            return bullet_count >= num_bullets

        # Default: assume instruction is followed
        return True

    def _check_loose(self, checker: Any, response: str) -> bool:
        """Apply loose checking with relaxations."""
        # Remove common prefixes/suffixes that models add
        cleaned = response.strip()
        prefixes = [
            "Sure, ",
            "Here is ",
            "Here's ",
            "Of course, ",
            "Certainly, ",
            "I'd be happy to ",
            "Let me ",
        ]
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]

        suffixes = [
            "Hope this helps!",
            "Let me know if you need anything else.",
            "Is there anything else I can help with?",
        ]
        for suffix in suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()

        try:
            return checker.check_following(cleaned)
        except Exception:
            return False

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score the output against the instructions."""
        if output.extracted_answer is None:
            return 0.0

        response = str(output.extracted_answer)
        instruction_ids = instance.metadata.get("instruction_id_list", [])
        instruction_kwargs = instance.metadata.get("kwargs", [])

        if not instruction_ids:
            return 1.0

        # Check each instruction
        results = []
        for inst_id, kwargs in zip(instruction_ids, instruction_kwargs, strict=True):
            results.append(self._check_instruction(inst_id, kwargs, response))

        # Store detailed results in metadata
        output.metadata = output.metadata or {}
        output.metadata["instruction_results"] = results
        output.metadata["instructions_followed"] = sum(results)
        output.metadata["total_instructions"] = len(results)

        # Prompt-level: all instructions must be followed
        prompt_acc = 1.0 if all(results) else 0.0
        # Instruction-level: proportion of instructions followed
        inst_acc = sum(results) / len(results) if results else 1.0

        output.metadata["prompt_level_acc"] = prompt_acc
        output.metadata["inst_level_acc"] = inst_acc

        return inst_acc


class IFEvalMetric(Metric):
    """Metric for IFEval that computes prompt-level and instruction-level accuracy."""

    def __init__(self, level: str = "inst", strictness: str = "loose"):
        """Initialize the metric.

        Args:
            level: "prompt" for prompt-level or "inst" for instruction-level
            strictness: "strict" or "loose"
        """
        self.level = level
        self.strictness = strictness
        self._name = f"{level}_level_{strictness}_acc"

    @property
    def name(self) -> str:
        """Return the metric name."""
        return self._name

    def compute(self, responses: Sequence[Response]) -> float:
        """Compute the metric across all responses."""
        values = []
        metric_key = f"{self.level}_level_acc"

        for response in responses:
            for output in response.outputs:
                if output.metadata and metric_key in output.metadata:
                    values.append(output.metadata[metric_key])

        if not values:
            return 0.0
        return sum(values) / len(values)


class IFEvalTask(Task):
    """IFEval (Instruction-Following Evaluation) task.

    Evaluates language models on their ability to follow verifiable instructions
    such as "write in more than 400 words" or "mention the keyword AI at least 3 times".

    See: "Instruction-Following Evaluation for Large Language Models"
    https://arxiv.org/abs/2311.07911

    Homepage: https://github.com/google-research/google-research/tree/master/instruction_following_eval
    """

    hf_path: str = "HuggingFaceH4/ifeval"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)
        self._instances_cache: list[Instance] | None = None

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the train split (the only split with labels)."""
        if self._instances_cache is None:
            self._instances_cache = []
            dataset = load_dataset(
                self.hf_path,
                split="train",
                trust_remote_code=True,
            )
            for doc in dataset:
                self._instances_cache.append(self._process_doc(doc))
        yield from self._instances_cache

    def _process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        return Instance(
            question=doc["prompt"],
            gold_answer=None,  # No gold answer, we check instruction following
            metadata={
                "key": doc.get("key"),
                "instruction_id_list": doc.get("instruction_id_list", []),
                "kwargs": doc.get("kwargs", []),
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
        """Extract the answer from model output."""
        return output.text.strip() if output.text else None


def _ifeval_config() -> TaskConfig:
    return TaskConfig(
        name="ifeval",
        hf_dataset="HuggingFaceH4/ifeval",
        scorers=(IFEvalScorer(strictness="loose"),),
        metrics=(
            IFEvalMetric(level="prompt", strictness="loose"),
            IFEvalMetric(level="inst", strictness="loose"),
            IFEvalMetric(level="prompt", strictness="strict"),
            IFEvalMetric(level="inst", strictness="strict"),
        ),
    )


@register("ifeval", _ifeval_config)
class IFEval(IFEvalTask):
    """IFEval task."""

    pass
