"""SWE-Bench Pro task dataclass."""

from __future__ import annotations

import re
from dataclasses import dataclass

_INSTANCE_REPO_RE = re.compile(r"^instance_(?P<owner>.+?)__(?P<name>.+?)-[0-9a-f]{40}")


def repo_from_instance_id(instance_id: str) -> str:
    """Return the ``owner/name`` repository encoded in a SWE-Bench Pro instance ID.

    Returns ``"unknown"`` when the ID does not follow the ``instance_<owner>__<name>-<sha>``
    convention.
    """
    match = _INSTANCE_REPO_RE.match(instance_id)
    if match is None:
        return "unknown"
    return f"{match.group('owner')}/{match.group('name')}"


@dataclass(frozen=True)
class SWEBenchProTask:
    """A single SWE-Bench Pro task.

    Attributes:
        instance_id: Unique identifier for the task.
        repo: Source repository as ``owner/name``.
        image: Container image with the repository checked out at the base commit.
        working_dir: Repository location inside the container.
        instruction: Problem description shown to the agent.
        test_files: Verifier files, keyed by path relative to ``/tests``. The
            directory must contain a ``test.sh`` that writes
            ``/logs/verifier/reward.txt``.
        gold_patch: Reference solution as a unified diff.
        agent_timeout: Wall-clock budget for the agent in seconds.
        verifier_timeout: Timeout for the verifier in seconds.
        hard: Whether the task belongs to the HARD subset.
        dockerfile_extra: Extra Dockerfile lines applied when deriving the sandbox image.
    """

    instance_id: str
    repo: str
    image: str
    working_dir: str
    instruction: str
    test_files: dict[str, bytes]
    gold_patch: str
    agent_timeout: float
    verifier_timeout: float
    hard: bool = False
    dockerfile_extra: tuple[str, ...] = ()
