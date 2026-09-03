"""Extract source-flagged FrontierScience questions for external review."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

SUBJECT_ORDER = {"physics": 0, "chemistry": 1, "biology": 2}
MATH_SPAN = re.compile(r"(\\+\(.*?\\+\)|\\+\[.*?\\+\])", re.DOTALL)
UNICODE_LATEX = {
    "α": r"\ensuremath{\alpha}",
    "β": r"\ensuremath{\beta}",
    "γ": r"\ensuremath{\gamma}",
    "Γ": r"\ensuremath{\Gamma}",
    "Δ": r"\ensuremath{\Delta}",
    "ε": r"\ensuremath{\varepsilon}",
    "η": r"\ensuremath{\eta}",
    "θ": r"\ensuremath{\theta}",
    "μ": r"\ensuremath{\mu}",
    "π": r"\ensuremath{\pi}",
    "ρ": r"\ensuremath{\rho}",
    "σ": r"\ensuremath{\sigma}",
    "τ": r"\ensuremath{\tau}",
    "ω": r"\ensuremath{\omega}",
    "−": r"\ensuremath{-}",
    "×": r"\ensuremath{\times}",
    "√": r"\ensuremath{\surd}",
    "₀": r"\ensuremath{_0}",
    "₁": r"\ensuremath{_1}",
    "₂": r"\ensuremath{_2}",
    "₃": r"\ensuremath{_3}",
    "₄": r"\ensuremath{_4}",
    "₅": r"\ensuremath{_5}",
    "₆": r"\ensuremath{_6}",
    "₇": r"\ensuremath{_7}",
    "₈": r"\ensuremath{_8}",
    "₉": r"\ensuremath{_9}",
}
PROSE_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}
BARE_GREEK_COMMANDS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "eta": "η",
    "theta": "θ",
    "mu": "μ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "omega": "ω",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latex_math(span: str) -> str:
    """Normalize possibly doubled math delimiters from the source JSON."""
    display = "[" in span[: span.find(span.lstrip("\\")) + 1]
    opener = "\\[" if display else "\\("
    closer = "\\]" if display else "\\)"
    content = re.sub(r"^\\+[\[(]", "", span)
    content = re.sub(r"\\+[\])]$", "", content)
    content = re.sub(r"\\{2,}", r"\\", content)
    # Keep the outer delimiters and discard any redundant equation wrapper
    # carried inside a source display-math span.
    content = re.sub(r"\\begin\{equation\*?\}", "", content)
    content = re.sub(r"\\end\{equation\*?\}", "", content)
    # Markdown producers sometimes escape literal brackets and subscripts
    # inside an already-delimited math span.
    content = content.replace(r"\[", "[").replace(r"\]", "]").replace(r"\_", "_")
    for source, replacement in UNICODE_LATEX.items():
        content = content.replace(source, replacement.replace(r"\ensuremath{", "").rstrip("}"))
    return f"{opener}{content}{closer}"


def latex_prose(text: str) -> str:
    """Escape prose while preserving embedded TeX math spans."""
    text = html.unescape(text)
    text = re.sub(
        r"([A-Za-z])¨",
        lambda match: rf"\(\ddot{{{match.group(1)}}}\)",
        text,
    )
    text = re.sub(
        r"([A-Za-z])˙",
        lambda match: rf"\(\dot{{{match.group(1)}}}\)",
        text,
    )
    text = re.sub(r"√\[([^\]]+)\]", lambda match: rf"\(\sqrt{{{match.group(1)}}}\)", text)
    text = text.replace(
        r"a) \[`\( S]_0 >> [E]_0 \)`",
        r"a) \([S]_0 \gg [E]_0\)",
    )
    text = re.sub(r"(?m)\\+\s*$", "", text).replace("`", "")
    parts = MATH_SPAN.split(text)
    rendered: list[str] = []
    for part in parts:
        if not part:
            continue
        if MATH_SPAN.fullmatch(part):
            rendered.append(latex_math(part))
            continue
        for command, symbol in BARE_GREEK_COMMANDS.items():
            part = re.sub(rf"\\+{command}\b", symbol, part)
        escaped = re.sub(r"[\\&%$#_{}~^<>]", lambda match: PROSE_ESCAPES[match.group()], part)
        for source, replacement in UNICODE_LATEX.items():
            escaped = escaped.replace(source, replacement)
        rendered.append(escaped)
    return "".join(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--jsonl-output", type=Path, required=True)
    parser.add_argument("--latex-output", type=Path)
    args = parser.parse_args()

    flagged = [record for record in load_jsonl(args.input) if record["source_answer_flag"]]
    flagged.sort(key=lambda record: (SUBJECT_ORDER[record["subject"]], record["task_group_id"]))

    review_records: list[dict[str, Any]] = []
    for record in flagged:
        flag = record["source_answer_flag"]
        review_records.append(
            {
                "task_group_id": record["task_group_id"],
                "subject": record["subject"],
                "question": record["problem"],
                "supplied_reference_answer": record["reference_answer"],
                "gold_assessment": flag["gold_verdict"],
                "gold_concern": flag["gold_analysis"],
                "stem_assessment": flag["stem_verdict"],
                "stem_concern": flag["stem_analysis"],
                "confirmation_requested": [
                    "Is the supplied reference answer correct under the literal question stem?",
                    "If not, what is the correct answer or the smallest stem correction needed?",
                ],
            }
        )

    args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl_output.open("w") as handle:
        for record in review_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    subject_counts: dict[str, int] = {}
    for record in review_records:
        subject = record["subject"]
        subject_counts[subject] = subject_counts.get(subject, 0) + 1

    lines = [
        "# FrontierScience Olympiad: external review packet",
        "",
        (
            "These questions were flagged during distractor curation. The supplied reference "
            "answers remain unchanged and authoritative in the 100-question evaluation set."
        ),
        "",
        "For each question, please determine:",
        "",
        "1. Is the supplied reference answer correct under the literal stem?",
        "2. If not, what is the correct answer or the smallest stem correction needed?",
        "",
        f"Questions: **{len(review_records)}**",
        "",
    ]
    for subject in ("physics", "chemistry", "biology"):
        subject_records = [record for record in review_records if record["subject"] == subject]
        if not subject_records:
            continue
        lines.extend([f"## {subject.title()} ({subject_counts[subject]})", ""])
        for record in subject_records:
            lines.extend(
                [
                    f"### `{record['task_group_id']}`",
                    "",
                    f"Internal assessment: **{record['gold_assessment']}**",
                    "",
                    "#### Question",
                    "",
                    record["question"],
                    "",
                    "#### Supplied reference answer",
                    "",
                    record["supplied_reference_answer"],
                    "",
                    "#### Reason for review",
                    "",
                    record["gold_concern"],
                    "",
                    f"Stem assessment: **{record['stem_assessment']}**",
                    "",
                    record["stem_concern"],
                    "",
                    "---",
                    "",
                ]
            )

    args.markdown_output.write_text("\n".join(lines))

    if args.latex_output:
        latex_lines = [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{textcomp}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage[hidelinks]{hyperref}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{0.7em}",
            r"\setlength{\emergencystretch}{3em}",
            r"\sloppy",
            r"\title{FrontierScience Olympiad: External Review Packet}",
            r"\author{}",
            r"\date{}",
            r"\begin{document}",
            r"\maketitle",
            (
                "These questions were flagged during distractor curation. The supplied reference "
                "answers remain unchanged and authoritative in the 100-question evaluation set."
            ),
            "",
            r"For each question, please determine:",
            r"\begin{enumerate}",
            r"\item Is the supplied reference answer correct under the literal question stem?",
            (
                r"\item If not, what is the correct answer or the smallest stem correction "
                r"needed?"
            ),
            r"\end{enumerate}",
        ]
        for subject in ("physics", "chemistry", "biology"):
            subject_records = [record for record in review_records if record["subject"] == subject]
            if not subject_records:
                continue
            latex_lines.extend([f"\\section{{{subject.title()} ({len(subject_records)})}}", ""])
            for record in subject_records:
                latex_lines.extend(
                    [
                        f"\\subsection{{\\texttt{{{record['task_group_id']}}}}}",
                        "",
                        (
                            r"\textbf{Internal assessment:} "
                            f"{latex_prose(record['gold_assessment'])}"
                        ),
                        "",
                        r"\subsubsection*{Question}",
                        "",
                        latex_prose(record["question"]),
                        "",
                        r"\subsubsection*{Supplied reference answer}",
                        "",
                        latex_prose(record["supplied_reference_answer"]),
                        "",
                        r"\subsubsection*{Reason for review}",
                        "",
                        latex_prose(record["gold_concern"]),
                        "",
                        (
                            r"\textbf{Stem assessment:} "
                            f"{latex_prose(record['stem_assessment'])}"
                        ),
                        "",
                        latex_prose(record["stem_concern"]),
                        "",
                    ]
                )
        latex_lines.append(r"\end{document}")
        args.latex_output.write_text("\n".join(latex_lines) + "\n")


if __name__ == "__main__":
    main()
