"""Terminal-Bench task dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalBenchTask:
    """A single Terminal-Bench task.

    Attributes:
        task_id: Unique identifier for the task.
        image: Docker image for the task environment.
        working_dir: Working directory inside the container (from Dockerfile WORKDIR).
        instruction: Task instruction content.
        agent_timeout: Timeout for agent execution in seconds.
        verifier_timeout: Timeout for verification in seconds.
        test_files: Mapping of relative paths to file content (as bytes).
        solution_script: Content of the solution script for oracle mode.
        difficulty: Task difficulty level.
        category: Task category.
    """

    task_id: str
    image: str
    working_dir: str
    instruction: str
    agent_timeout: float
    verifier_timeout: float
    test_files: dict[str, bytes]
    solution_script: str
    difficulty: str
    category: str
