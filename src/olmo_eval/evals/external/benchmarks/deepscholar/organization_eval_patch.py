"""Replacement for DeepScholar's organization evaluator.

This file is copied into the pinned DeepScholar checkout at runtime. It targets
the pinned LOTUS revision, whose ``pairwise_judge`` accessor returns normalized
``"A"``/``"B"`` strings rather than the Pydantic response objects expected by
the upstream evaluator.
"""

from __future__ import annotations

import lotus  # noqa: F401  # registers the pandas pairwise_judge accessor
import pandas as pd

try:
    from evaluator import EvaluationFunction, Evaluator
    from parsers import Parser
except ImportError:
    from ..parsers import Parser
    from .enum import EvaluationFunction
    from .evaluator import Evaluator


ORGANIZATION_JUDGE_INSTRUCTION = """
You will receive the title and abstract of a research paper together with two
candidate related-work sections, A and B, written for that paper.

Decide which section exhibits better organization and coherence. Ignore the
text's Markdown or LaTeX formatting and assess organization only:

- Logical structure: a clear introduction, sensible thematic grouping, and a
  smooth progression of ideas.
- Paragraph cohesion: each paragraph develops one topic and flows naturally to
  the next.
- Clarity and readability: minimal redundancy or contradiction, with useful
  transitions.
- Signposting: helpful headings, topic sentences, or discourse markers.

Do not judge breadth, citation accuracy, or analytic depth. There are no ties.
Return exactly one token: A if section A is better organized, or B if section B
is better organized. Do not return JSON or an explanation.

### Paper under assessment
{paper_title}
{paper_abstract}

### Candidate related-work section A
{related_work_a}

### Candidate related-work section B
{related_work_b}
"""


def _generated_section_wins(decision: object) -> int:
    """Convert a normalized LOTUS decision to a generated-section win."""
    return int(isinstance(decision, str) and decision.strip().upper() == "A")


class OrganizationEvaluator(Evaluator):
    evaluation_function = EvaluationFunction.ORGANIZATION

    def calculate(self, parsers: list[Parser]) -> pd.DataFrame:
        system_prompt = (
            "You are an intelligent, rigorous, and fair evaluator of scholarly "
            "writing quality and relevance."
        )
        infos = [parser.get_folder_info(include_related_works_section=True) for parser in parsers]
        df = pd.DataFrame(infos)
        df.rename(
            columns={
                "generated_related_works_section": "related_work_a",
                "related_works_section": "related_work_b",
            },
            inplace=True,
        )

        # At the pinned LOTUS revision, pairwise_judge returns plain A/B strings
        # and normalizes the reversed trial back to the original column order.
        # Therefore A means the generated section won in both output columns.
        results: pd.DataFrame = df.pairwise_judge(
            col1="related_work_a",
            col2="related_work_b",
            judge_instruction=ORGANIZATION_JUDGE_INSTRUCTION,
            system_prompt=system_prompt,
            n_trials=2,
            permute_cols=True,
        )
        results["organization_v1"] = results["_judge_0"].map(_generated_section_wins)
        results["organization_v2"] = results["_judge_1"].map(_generated_section_wins)
        results[self.evaluation_function.value] = (
            results["organization_v1"] + results["organization_v2"]
        ) / 2
        results.drop(columns=["_judge_0", "_judge_1"], inplace=True)
        return results
