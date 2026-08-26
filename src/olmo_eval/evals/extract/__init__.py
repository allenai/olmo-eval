"""Answer extraction utilities for tasks."""

from .answer_format import ExtractedAnswer, extract_answer_with_format
from .code import extract_code, extract_code_before_fence, indent_code
from .math import MathExtractor, extract_math_answer, is_equiv, normalize_final_answer
from .mcq import extract_mcq_answer
from .reasoning import extract_think_answer, extract_think_answer_only
from .sanitize import sanitize_code

__all__ = [
    "ExtractedAnswer",
    "extract_answer_with_format",
    "extract_code",
    "extract_code_before_fence",
    "extract_math_answer",
    "extract_mcq_answer",
    "indent_code",
    "is_equiv",
    "MathExtractor",
    "normalize_final_answer",
    "sanitize_code",
    "extract_think_answer",
    "extract_think_answer_only",
]
