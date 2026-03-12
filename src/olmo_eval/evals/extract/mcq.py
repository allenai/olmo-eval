"""Multiple-choice answer extraction from free-form model output.

Patterns are tried in priority order; the last match from the first
pattern that hits wins.  Priority reflects specificity — explicit
instruction-following and LaTeX formats are unambiguous, while
parenthesised or bold letters appear throughout running text and are
used only as fallbacks.
"""

import re

# Ordered from most specific → least specific.  Each regex must have
# exactly one capture group containing the answer letter.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "ANSWER: X" or "ANSWER: (X)" — explicit instruction-following format.
    # Same-line only ([^\S\n] = whitespace excluding newline) to avoid
    # "Answer:\n\nThe ..." matching the T in "The".
    re.compile(r"ANSWER[^\S\n]*:[^\S\n]*\(?([A-Z])\)?", re.IGNORECASE),
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
