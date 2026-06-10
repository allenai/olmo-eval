"""Prompt construction for the Molmo2 image-QA benchmarks.

Vendored from ``mm_olmo/olmo/data/data_formatter.py``:

* :data:`POINT_COUNT_TEMPLATES` — the ``GENERAL_PROMPTS_V1["point_count"]``
  list, verbatim (60 entries; duplicates and typos are intentional — the
  template *index* is what the seeded RNG selects).
* :func:`pixmo_count_question` — replicates the per-example template choice of
  mm_olmo's eval data pipeline (``DeterministicDataset`` seed arithmetic with
  the eval loader seed 691203, then ``rng.randint`` over the template list).
* :func:`format_mc_question` — the eval branch of ``template_options``
  (``"Only return the correct answer option."``).

Style-prefix rules (``demo_or_style_v2`` system prompts): short-answer styles
(``vqa2``, ``chart_qa``, ``doc_qa``, ``info_qa``, ``text_vqa``) are rendered
as ``"{style}: {question}"``; multiple-choice and counting styles get no
prefix.  Tasks bake the prefix into the instance question directly.
"""

from __future__ import annotations

import string

import numpy as np

EVAL_LOADER_SEED = 691203
"""DataLoaderConfig seed used by mm_olmo's eval pipeline."""

POINT_COUNT_TEMPLATES: list[str] = [
    "How many {label} are there?",
    "How many {label}?",
    "How many {label}.",
    "how many {label}.",
    "how many {label}?",
    'How many "{label}" are there in the image?',
    "How many {label} are there in the image?",
    "Tell me how many {label} there are",
    "Tell me how many {label} there are and point to them.",
    "how many {label}",
    "Tell me where each {label} is.",
    "Tell me how many {label} are in the image",
    "count {label}",
    "count every {label}",
    "count each {label}",
    "count {label}.",
    "Count the {label}.",
    "How many {label} do you see?",
    "How many {label} are visible?",
    "Count all the {label}",
    "how mmny {label}?",
    "Count every {label} in the picture.",
    "Count all the {label}",
    "Count each {label}",
    "Point to and count the {label} in the picture.",
    "Point and count {label}",
    "Point to every {label}",
    "Locate the {label} and count them",
    "Locate every {label} and count them",
    "Find all the {label}. How many are there?",
    "Find each {label}. How many are there?",
    "Point at {label} and then tell me the count.",
    "What is the total number of {label} in the image?",
    "What is the number of {label}?",
    "In this image, how many {label} are there?",
    "In all the picture, how many {label} are there?",
    "Point at the {label} and then count them.",
    "Point to all the visible {label} output the total count.",
    "Point to all the {label} visible and output the total count. \nPlease say 'There are none.' if it is not in the image.",
    'Point to all occurrences of "{label}" and output the total count.',
    "Show me where the {label} are and output the total count.",
    "Where are the {label}? How many are there?",
    "Generate list of points showing where the {label} are and output the total count.",
    "Object: {label}\nInstruction: Point to the object and output the total count.",
    "find any {label} in the picture and output the total count.",
    "Can you see any {label} in the image? Point to them and output the total count.",
    "Can you point out all {label} in this image? How many are there?",
    "If there are any {label} present, indicate their positions and output the total count.",
    "How many {label} are there in the image? Point to them and output the total count.",
    "How many {label} are there in the image?",
    "Give me the count of {label} in the image.",
    "How many {label} are visible in the image?",
    "How many {label} are there?",
    "In the image, how many {label} are there?",
    "Can you count the number of {label} in the image?",
    "Can you count every {label} in the picture?",
    "Can you see any {label} in the image? How many are there?",
    "Are there any {label} in the image? How many are there?",
    "If you see any {label} in the image, give me the count. Otherwise, say 'There are none.'",
    "Object: {label}\nInstruction: How many are there?",
]


def _apply_label(template: str, label: str) -> str:
    """``apply_keywords`` from mm_olmo: replaces only the first occurrence."""
    res = template.split("{label}", 2)
    return res[0] + label + res[1]


def pixmo_count_question(label: str, arrow_idx: int, seed: int = EVAL_LOADER_SEED) -> str:
    """Reproduce mm_olmo's per-example PixMo-Count question template.

    ``arrow_idx`` is the example's position in the on-disk arrow dataset; the
    RNG seed arithmetic matches ``DeterministicDataset.get`` (epoch 0) and the
    template pick matches ``apply_keyword_prompt``'s ``rng.randint``.
    """
    rng = np.random.RandomState((seed * 195172 + arrow_idx) % (2**32 - 1))
    template = POINT_COUNT_TEMPLATES[rng.randint(0, len(POINT_COUNT_TEMPLATES))]
    return _apply_label(template, label.lower())


def format_mc_question(
    question: str,
    options: list[str],
    *,
    labelled: bool = True,
) -> tuple[str, str | list[str]]:
    """Eval-time multiple-choice templating (``template_options`` eval branch).

    Returns ``(formatted_question, option_names)``.  With ``labelled=True``
    options are rendered as ``A. …`` lines and ``option_names`` is the letter
    string (e.g. ``"ABCD"``, matching mm_olmo where it is a slice of
    ``string.ascii_uppercase``).  With ``labelled=False`` (AI2D
    ``ai2_diagram_no_letter``) options are listed verbatim and
    ``option_names`` is the option list itself.
    """
    if labelled:
        prefixes = string.ascii_uppercase
        # zip-shortest on purpose: prefixes covers up to 26 options
        option_text = "\n".join(
            f"{prefix}. {opt}" for prefix, opt in zip(prefixes, options, strict=False)
        )
        option_names: str | list[str] = prefixes[: len(options)]
    else:
        option_text = "\n".join(options)
        option_names = options
    formatted = question + "\nOnly return the correct answer option.\n" + option_text
    return formatted, option_names
