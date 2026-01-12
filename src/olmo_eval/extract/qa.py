"""Multiple choice question answer extraction utilities."""

import re

# Common patterns for extracting MCQA answers, in order of specificity
MCQA_PATTERNS = [
    r"(?i)therefore,?\s*the\s*answer\s*is:?\s*\(?($ANS$)\b",
    r"(?i)so\s+the\s+answer\s+is\s+($ANS$)\.?",
    r"(?i)the\s+correct\s+answer\s+is:?\s*($ANS$)",
    r"(?i)the\s+answer\s+is\s+($ANS$)\.?",
    r"(?i)answer:\s*($ANS$)",
    r"(?i)\b($ANS$)\)?\s+is\s+correct",
    r"(?i)\(($ANS$)\)",
    r"(?i)\b($ANS$)\b",
]


def extract_mcqa_answer(
    text: str,
    answer_regexes: list[str] | None = None,
) -> str | None:
    """Extract a multiple choice answer from text.

    Args:
        text: The model output text to search.
        answer_regexes: List of regex patterns for valid answers.
            Each pattern should capture the answer as group 1.
            Default is [A-D] for 4-choice questions.

    Returns:
        The extracted answer (capitalized) or None if not found.
    """
    if answer_regexes is None:
        answer_regexes = [r"[A-D]"]

    for pattern in MCQA_PATTERNS:
        for ans_pattern in answer_regexes:
            # Replace $ANS$ placeholder with actual answer pattern
            full_pattern = pattern.replace("$ANS$", ans_pattern)
            match = re.search(full_pattern, text)
            if match:
                groups = [g for g in match.groups() if g is not None]
                if groups:
                    return groups[-1].upper()

    return None


def extract_mcqa_answer_index(
    text: str,
    num_choices: int = 4,
) -> int:
    """Extract a multiple choice answer and return its index.

    Args:
        text: The model output text to search.
        num_choices: Number of choices (default 4 for A-D).

    Returns:
        The index of the answer (0-based) or -1 if not found.
    """
    letters = "ABCDEFGHIJ"[:num_choices]
    answer_regexes = [f"[{letters}]"]
    answer = extract_mcqa_answer(text, answer_regexes)
    if answer and answer in letters:
        return letters.index(answer)
    return -1
