"""DeepMind Mathematics dataset task implementations.

Based on: https://arxiv.org/abs/1904.01557
Homepage: https://github.com/google-deepmind/mathematics_dataset
"""

import logging
import re
from collections.abc import Iterator
from tokenize import TokenError
from typing import Any

import numpy as np
import sympy

from olmo_eval.core import (
    AccuracyMetric,
    ExactMatchScorer,
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.constants.benchmarks import DEEPMIND_MATH_CATEGORIES
from olmo_eval.evals.tasks.core import Task, TaskConfig, register

logger = logging.getLogger(__name__)


def _clean_deepmind_string(s: str) -> str:
    """Clean DeepMind dataset strings from binary encoding artifacts."""
    # Remove b'...\n' wrapper if present
    s = re.sub(r"\\n'$", "", re.sub(r"^b'", "", s))
    return s.strip()


def _clean_prediction(prediction: str) -> str:
    """Clean model prediction for comparison."""
    res = re.sub(r"\s*\.?\s*I hope it is correct.*", "", prediction)
    # Strip special tokens
    special_tokens = ["<end_of_turn>", "<|eot_id|>", "<s>", "</s>"]
    for token in special_tokens:
        res = res.replace(token, "")
    res = res.strip()
    # Strip trailing period
    res = re.sub(r"\.\s*$", "", res).strip()
    # Strip math delimiters
    delimiters = [
        (r"\$", r"\$"),
        (r"\\\(", r"\\\)"),
        (r"\*\*", r"\*\*"),
        (r"\*\*\*", r"\*\*\*"),
        (r"\\\[", r"\\\]"),
    ]
    for left, right in delimiters:
        res = re.sub(f"^{left}(.*){right}$", r"\1", res).strip()
    return res


def _check_sympy_equal(expr1: Any, expr2: Any) -> bool:
    """Check if two sympy expressions are equal."""
    if not isinstance(expr1, sympy.Expr) or not isinstance(expr2, sympy.Expr):
        return expr1 == expr2
    if expr1 is None or expr2 is None:
        return False
    if expr1.free_symbols != expr2.free_symbols:
        return False
    variables = expr1.free_symbols
    test_values = np.random.default_rng(42).random(len(variables))
    expr1_num = expr1
    expr2_num = expr2
    for symbol, number in zip(variables, test_values, strict=True):
        expr1_num = expr1_num.subs(symbol, sympy.Float(number))
        expr2_num = expr2_num.subs(symbol, sympy.Float(number))
    expr1_float = float(expr1_num)
    expr2_float = float(expr2_num)
    if not np.allclose(expr1_float, expr2_float):
        return False
    return bool(expr1.equals(expr2))


def _compare_math_answers(gold: str, prediction: str) -> bool:
    """Compare mathematical answers with sympy for expression equivalence."""
    prediction = _clean_prediction(prediction)

    # Direct string match (case insensitive)
    if gold.lower() == prediction.lower():
        return True

    # Boolean answers
    if gold.lower() in ["true", "false"]:
        pred_lower = prediction.lower()
        gold_lower = gold.lower()
        if gold_lower == pred_lower:
            return True
        return (pred_lower == "yes" and gold_lower == "true") or (
            pred_lower == "no" and gold_lower == "false"
        )

    # Try sympy comparison
    try:
        gold_expr = sympy.parse_expr(gold)
        pred_expr = sympy.parse_expr(prediction)
        return _check_sympy_equal(gold_expr, pred_expr)
    except (TypeError, SyntaxError, TokenError):
        return False
    except Exception as e:
        logger.debug(f"Sympy comparison error: {e} (gold={gold}, pred={prediction})")
        return False


class DeepMindMathTask(Task):
    """Base class for DeepMind Mathematics dataset tasks."""

    default_hf_path: str = "deepmind/math_dataset"

    def __init__(self, config: TaskConfig, category: str) -> None:
        super().__init__(config)
        self.category = category

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the test split."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for idx, doc in enumerate(loader.load(source)):
                self._instances_cache.append(self.process_doc(doc, idx))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split).with_subset(self.category)
        except ValueError:
            return DataSource(
                path=self.default_hf_path,
                subset=self.category,
                split=split,
            )

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert a dataset document to an Instance."""
        question = _clean_deepmind_string(doc["question"])
        answer = _clean_deepmind_string(doc["answer"])

        return Instance(
            question=question,
            gold_answer=answer,
            metadata={
                "index": index,
                "category": self.category,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())

        prompt = f"Problem:\n{instance.question}\n\nAnswer:"
        return LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=prompt,
        )

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer from model output."""
        text = output.text.strip()
        # Look for "answer is X" pattern - stop at period followed by space/end, not decimals
        match = re.search(
            r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.(?:\s|$)|$)", text, re.I
        )
        if match:
            answer = match.group(1).strip()
            if answer.endswith("."):
                answer = answer[:-1]
            return answer
        # Otherwise take the whole output
        return _clean_prediction(text)

    def score_answer(self, extracted: str | None, gold: str) -> bool:
        """Score an extracted answer against the gold answer."""
        if extracted is None:
            return False
        return _compare_math_answers(gold, extracted)


def _make_deepmind_math_config(category: str) -> TaskConfig:
    """Create a DeepMind Math task config for a specific category."""
    return TaskConfig(
        name=f"deepmind_math_{category}",
        data_source=DataSource(path="deepmind/math_dataset", subset=category),
        scorers=(ExactMatchScorer(case_sensitive=False),),
        metrics=(AccuracyMetric(scorer=ExactMatchScorer),),
    )


# Register all DeepMind Math tasks dynamically
for category in DEEPMIND_MATH_CATEGORIES:

    def make_config_factory(c: str):
        return lambda: _make_deepmind_math_config(c)

    def make_class_factory(c: str):
        class _DeepMindMathCategory(DeepMindMathTask):
            def __init__(self, config: TaskConfig) -> None:
                super().__init__(config, category=c)

        _DeepMindMathCategory.__name__ = f"DeepMindMath_{c.replace('__', '_').title()}"
        return _DeepMindMathCategory

    register(f"deepmind_math_{category}", make_config_factory(category))(
        make_class_factory(category)
    )
