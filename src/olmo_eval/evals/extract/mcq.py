"""Multiple-choice answer extraction from free-form model output.

Patterns are tried in priority order; the last match from the first
pattern that hits wins.  Priority reflects specificity — explicit
instruction-following and LaTeX formats are unambiguous, while
parenthesised or bold letters appear throughout running text and are
used only as fallbacks.

This is olmo-eval's native extractor (used by BBQ, WMDP, MedQA). Tasks
ported for parity with oe-eval use ``answer_format.extract_answer_with_format``
instead, whose cascade and priority order mirror oe-eval's and yield
different answers on the same text.
"""

import re

# "ANSWER: X" — the explicit instruction-following format. Tolerates markdown
# emphasis and parentheses around the label and letter ("**Answer:** B",
# "Answer: **(C)**"). The letter must stand alone, so "ANSWER: the ..." does
# not capture the "t" of "the". Same-line only ([^\S\n] = whitespace excluding
# newline) to avoid "Answer:\n\nThe ..." matching the next line.
ANSWER_LINE_PATTERN: re.Pattern[str] = re.compile(
    r"ANSWER[*_]*[^\S\n]*:[*_]*[^\S\n]*[*_(]*([A-Z])(?!\w)", re.IGNORECASE
)

# Ordered from most specific → least specific.  Each regex must have
# exactly one capture group containing the answer letter.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    ANSWER_LINE_PATTERN,
    # \boxed{X} or \boxed{\text{X}} — LaTeX (common with thinking-mode models)
    re.compile(r"\\boxed\{(?:\\text\{)?([A-Z])"),
    # (X) — parenthesized letter
    re.compile(r"\(([A-Z])\)"),
    # **X) or **X. — bold-markdown letter
    re.compile(r"\*\*([A-Z])[.)]\s"),
)


def extract_mcq_answer(text: str) -> str | None:
    """Return the MCQ letter from *text*, or ``None``.

    Tries each pattern in priority order and returns the last match from
    the first pattern that fires.
    """
    for pattern in _PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(1).upper()
    return None
