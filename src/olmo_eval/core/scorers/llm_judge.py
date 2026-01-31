"""LLM-as-judge scorers for open-ended evaluation.

This module provides scorers that use language models to evaluate
responses, following patterns from benchmarks like SimpleQA.
"""

import re
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from ..types import Instance, LMOutput
from .base import Scorer

# Type for judge function: takes prompt, returns judge response
JudgeFn = Callable[[str], str]

# Rubric-based judge prompt template
RUBRIC_JUDGE_PROMPT_TEMPLATE = """\
You are evaluating an AI assistant's response using the following rubric.

Question: {question}

Reference Answer: {gold_answer}

AI Assistant's Answer: {model_answer}

Rubric:
{rubric_text}

Provide your evaluation and end with a score in the format "Score: X" \
where X is a number from 0 to {max_score}."""

# SimpleQA-style judge prompt template
SIMPLEQA_JUDGE_PROMPT_TEMPLATE = """\
You are a judge evaluating the correctness of an AI assistant's response.

Question: {question}

Correct Answer: {gold_answer}

AI Assistant's Answer: {model_answer}

Grade the AI's answer as one of the following:
A) CORRECT - The AI's answer is factually correct and answers the question accurately.
B) INCORRECT - The AI's answer is factually wrong or does not answer the question.
C) NOT_ATTEMPTED - The AI explicitly declined to answer or said it doesn't know.

Respond with only the letter (A, B, or C) corresponding to your grade."""


@dataclass(frozen=True)
class LLMJudgeScorer(Scorer):
    """Abstract base class for LLM-as-judge scorers.

    Subclasses must implement format_judge_prompt() and parse_judge_response().
    """

    name: ClassVar[str] = "llm_judge"
    judge_fn: JudgeFn | None = field(default=None)

    @abstractmethod
    def format_judge_prompt(self, instance: Instance, output: LMOutput) -> str:
        """Format the prompt to send to the judge model.

        Args:
            instance: The evaluation instance.
            output: The model output to evaluate.

        Returns:
            Formatted prompt string for the judge.
        """
        ...

    @abstractmethod
    def parse_judge_response(self, response: str) -> float:
        """Parse the judge's response into a score.

        Args:
            response: The judge model's response.

        Returns:
            Score between 0.0 and 1.0.
        """
        ...

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score using the judge function.

        Args:
            instance: The evaluation instance.
            output: The model output to evaluate.

        Returns:
            Score from the judge (0.0 to 1.0).
        """
        if self.judge_fn is None:
            return 0.0

        prompt = self.format_judge_prompt(instance, output)
        response = self.judge_fn(prompt)
        return self.parse_judge_response(response)


# Grade type for SimpleQA-style evaluation
SimpleQAGrade = Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]


def _build_openai_judge_fn(model: str = "gpt-4o-mini") -> JudgeFn | None:
    """Build a judge function using OpenAI API if OPENAI_API_KEY is set."""
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)

    def judge(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return response.choices[0].message.content or ""

    return judge


@dataclass(frozen=True)
class SimpleQAJudgeScorer(LLMJudgeScorer):
    """LLM judge following SimpleQA's CORRECT/INCORRECT/NOT_ATTEMPTED grading.

    Uses A/B/C response format where:
    - A = CORRECT
    - B = INCORRECT
    - C = NOT_ATTEMPTED

    If judge_fn is not provided, automatically uses OpenAI API with gpt-4o-mini
    when OPENAI_API_KEY environment variable is set.
    """

    name: ClassVar[str] = "simpleqa_judge"
    judge_fn: JudgeFn | None = field(default=None)
    not_attempted_score: float = 0.0

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score using the judge function, auto-building from OpenAI if needed."""
        judge = self.judge_fn
        if judge is None:
            judge = _build_openai_judge_fn()
        if judge is None:
            return 0.0

        prompt = self.format_judge_prompt(instance, output)
        response = judge(prompt)
        return self.parse_judge_response(response)

    def format_judge_prompt(self, instance: Instance, output: LMOutput) -> str:
        """Format SimpleQA-style judge prompt."""
        return SIMPLEQA_JUDGE_PROMPT_TEMPLATE.format(
            question=instance.question,
            gold_answer=instance.gold_answer or "",
            model_answer=output.extracted_answer or output.text,
        )

    def parse_judge_response(self, response: str) -> float:
        """Parse A/B/C grade from judge response.

        Args:
            response: The judge's response.

        Returns:
            1.0 for CORRECT (A), 0.0 for INCORRECT (B),
            not_attempted_score for NOT_ATTEMPTED (C).
        """
        response = response.strip().upper()

        # Look for letter grade
        if response.startswith("A") or "CORRECT" in response and "INCORRECT" not in response:
            return 1.0
        elif response.startswith("B") or "INCORRECT" in response:
            return 0.0
        elif response.startswith("C") or "NOT_ATTEMPTED" in response or "NOT ATTEMPTED" in response:
            return self.not_attempted_score
        else:
            # Default to incorrect if can't parse
            return 0.0

    def get_grade(self, response: str) -> SimpleQAGrade:
        """Get the grade category from judge response.

        Args:
            response: The judge's response.

        Returns:
            Grade category.
        """
        response = response.strip().upper()

        if response.startswith("A") or "CORRECT" in response and "INCORRECT" not in response:
            return "CORRECT"
        elif response.startswith("B") or "INCORRECT" in response:
            return "INCORRECT"
        elif response.startswith("C") or "NOT_ATTEMPTED" in response or "NOT ATTEMPTED" in response:
            return "NOT_ATTEMPTED"
        else:
            return "INCORRECT"


@dataclass(frozen=True)
class RubricJudgeScorer(LLMJudgeScorer):
    """LLM judge with custom rubric and configurable score extraction.

    Allows defining custom evaluation rubrics and score patterns.
    """

    name: ClassVar[str] = "rubric_judge"
    judge_fn: JudgeFn | None = field(default=None)
    rubric: str = ""
    score_pattern: str = r"Score:\s*(\d+(?:\.\d+)?)"
    max_score: float = 10.0
    default_score: float = 0.0

    def format_judge_prompt(self, instance: Instance, output: LMOutput) -> str:
        """Format rubric-based judge prompt."""
        return RUBRIC_JUDGE_PROMPT_TEMPLATE.format(
            question=instance.question,
            gold_answer=instance.gold_answer or "N/A",
            model_answer=output.extracted_answer or output.text,
            rubric_text=self.rubric or self._default_rubric(),
            max_score=self.max_score,
        )

    def _default_rubric(self) -> str:
        """Default rubric when none provided."""
        return f"""Evaluate the response on a scale of 0 to {self.max_score}:
- {self.max_score}: Perfect, completely correct and comprehensive
- {self.max_score * 0.8}: Mostly correct with minor issues
- {self.max_score * 0.5}: Partially correct
- {self.max_score * 0.2}: Mostly incorrect with some relevant content
- 0: Completely incorrect or irrelevant"""

    def parse_judge_response(self, response: str) -> float:
        """Extract score from judge response using pattern.

        Args:
            response: The judge's response.

        Returns:
            Normalized score (0.0 to 1.0).
        """
        match = re.search(self.score_pattern, response, re.IGNORECASE)
        if match:
            try:
                raw_score = float(match.group(1))
                # Normalize to 0-1 range
                return min(1.0, max(0.0, raw_score / self.max_score))
            except ValueError:
                return self.default_score / self.max_score
        return self.default_score / self.max_score

    def get_raw_score(self, response: str) -> float:
        """Get the raw (unnormalized) score from response.

        Args:
            response: The judge's response.

        Returns:
            Raw score value.
        """
        match = re.search(self.score_pattern, response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return self.default_score
        return self.default_score
