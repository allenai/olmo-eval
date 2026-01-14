"""Math and LaTeX answer extraction utilities."""

import re

# Substitutions to normalize math answers
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

# Expressions to remove from answers
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def extract_math_answer(text: str) -> list[str]:
    """Extract mathematical answers from text.

    Tries multiple extraction strategies:
    1. Look for \\boxed{...} answers
    2. Look for "Final Answer: ..." pattern
    3. Extract from dollar sign delimiters
    4. Fall back to full text

    Args:
        text: The model output text.

    Returns:
        List of extracted answers, normalized.
    """
    all_answers = []

    # Try to extract boxed answer first
    boxed_answer = _last_boxed_only_string(text)
    if boxed_answer is not None:
        try:
            boxed_answer = _remove_boxed(boxed_answer)
        except (AssertionError, IndexError):
            boxed_answer = None

    # Try Minerva-style extraction
    minerva_answer = normalize_final_answer(_get_unnormalized_answer(text))
    if minerva_answer and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)

    # Add boxed answer if found
    if boxed_answer is not None:
        all_answers.append(normalize_final_answer(boxed_answer))

    # Try extracting from dollar signs if no answers yet
    if len(all_answers) == 0:
        dollars = [m.start() for m in re.finditer(r"\$", text)]
        if len(dollars) > 1:
            answer = normalize_final_answer(text[dollars[-2] + 1 : dollars[-1]])
            all_answers.append(answer)

    # Fall back to full result if no other extraction worked
    if len(all_answers) == 0:
        all_answers.append(normalize_final_answer(text))

    return all_answers


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Based on appendix D of Lewkowycz et al. (2022).
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold, is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def is_math_equiv(str1: str | None, str2: str | None) -> bool:
    """Check if two math answers are equivalent.

    Uses string normalization for comparison.
    """
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def _last_boxed_only_string(string: str) -> str | None:
    """Extract the last \\boxed{...} expression from a string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def _remove_boxed(s: str) -> str:
    """Remove \\boxed{} wrapper from a string."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def _get_unnormalized_answer(text: str) -> str:
    """Extract answer from Minerva-style "Final Answer: ..." format."""
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        text,
    )
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


def _strip_string(string: str) -> str:
    """Normalize a string for comparison."""
    # linebreaks
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{."
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # get rid of "k = " or "q = " at beginning
    parts = string.split("=")
    if len(parts) == 2 and len(parts[0]) <= 2:
        string = parts[1]

    # remove spaces
    string = string.replace(" ", "")

    # Normalize 0.5 -> 1/2
    if string == "0.5":
        string = "\\frac{1}{2}"

    return string
