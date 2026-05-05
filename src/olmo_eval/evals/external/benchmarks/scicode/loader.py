"""Loader for SciCode problems from the HuggingFace ``SciCode1/SciCode`` dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from olmo_eval.data import DataLoader, DataSource


@dataclass
class SciCodeProblem:
    """A single SciCode problem with its sub-steps and metadata."""

    problem_id: str
    problem_name: str
    required_dependencies: str
    sub_steps: list[dict[str, Any]]
    general_solution: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_problems(
    split: str = "test",
    problem_ids: list[str] | None = None,
) -> list[SciCodeProblem]:
    """Load SciCode problems.

    Args:
        split: Dataset split (``test`` or ``validation``).
        problem_ids: Optional filter on ``problem_id``.

    Returns:
        List of ``SciCodeProblem`` instances (order matches the dataset).
    """
    loader = DataLoader()
    source = DataSource(path="SciCode1/SciCode", split=split)
    problems: list[SciCodeProblem] = []
    for doc in loader.load(source):
        problem_id = str(doc["problem_id"])
        if problem_ids and problem_id not in problem_ids:
            continue
        sub_steps = [dict(s) for s in doc["sub_steps"]]
        problems.append(
            SciCodeProblem(
                problem_id=problem_id,
                problem_name=doc.get("problem_name", ""),
                required_dependencies=doc["required_dependencies"],
                sub_steps=sub_steps,
                general_solution=doc.get("general_solution"),
                metadata={
                    k: v
                    for k, v in doc.items()
                    if k
                    not in (
                        "problem_id",
                        "problem_name",
                        "required_dependencies",
                        "sub_steps",
                        "general_solution",
                    )
                },
            )
        )
    return problems
