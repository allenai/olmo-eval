"""Terminal-Bench task loader."""

from __future__ import annotations

import logging
import subprocess
import tomllib
from pathlib import Path

from .task import TerminalBenchTask

logger = logging.getLogger(__name__)


class TerminalBenchLoader:
    """Loads Terminal-Bench tasks from a git repository."""

    REPO_URL = "https://github.com/laude-institute/terminal-bench-2.git"
    DEFAULT_REF = "f5b891cb4f7c20e306f9d05887628b43af740f43"

    def ensure_repo(self, target_dir: Path, ref: str | None = None) -> Path:
        """Clone or update the Terminal-Bench repository.

        Args:
            target_dir: Directory to clone/update the repo in.
            ref: Git ref to checkout (commit, branch, or tag).

        Returns:
            Path to the repository.
        """
        ref = ref or self.DEFAULT_REF

        if (target_dir / ".git").exists():
            logger.info(f"Updating Terminal-Bench repo at {target_dir}")
            subprocess.run(
                ["git", "-C", str(target_dir), "fetch", "origin"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(target_dir), "checkout", ref],
                check=True,
                capture_output=True,
            )
        else:
            logger.info(f"Cloning Terminal-Bench repo to {target_dir}")
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            # Clone without --depth to support checking out specific commits
            subprocess.run(
                ["git", "clone", self.REPO_URL, str(target_dir)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(target_dir), "checkout", ref],
                check=True,
                capture_output=True,
            )
        return target_dir

    def load_tasks(
        self,
        repo_dir: Path,
        task_ids: list[str] | None = None,
    ) -> list[TerminalBenchTask]:
        """Load tasks from the Terminal-Bench repository.

        Args:
            repo_dir: Path to the repository.
            task_ids: Optional list of task IDs to load (default: all).

        Returns:
            List of TerminalBenchTask instances.
        """
        tasks = []

        # Tasks are at the repo root (each directory with task.toml is a task)
        for task_dir in sorted(repo_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            if not (task_dir / "task.toml").exists():
                continue
            if task_ids and task_dir.name not in task_ids:
                continue

            try:
                task = self._load_task(task_dir)
                tasks.append(task)
                logger.debug(f"Loaded task: {task.task_id}")
            except Exception as e:
                logger.warning(f"Failed to load task {task_dir.name}: {e}")

        task_ids = [t.task_id for t in tasks]
        logger.info(f"Loaded {len(tasks)} tasks: {task_ids}")
        return tasks

    def _load_task(self, task_dir: Path) -> TerminalBenchTask:
        """Load a single task from its directory.

        Args:
            task_dir: Path to the task directory.

        Returns:
            A TerminalBenchTask instance.
        """
        # Parse task.toml
        config = tomllib.loads((task_dir / "task.toml").read_text())

        # Parse WORKDIR from Dockerfile
        dockerfile_path = task_dir / "environment" / "Dockerfile"
        workdir = "/app"  # Default
        if dockerfile_path.exists():
            for line in dockerfile_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("WORKDIR"):
                    parts = line.split(None, 1)
                    if len(parts) > 1:
                        workdir = parts[1].strip().strip('"').strip("'")

        # Read instruction
        instruction_path = task_dir / "instruction.md"
        instruction = instruction_path.read_text() if instruction_path.exists() else ""

        # Load test files recursively
        test_files: dict[str, bytes] = {}
        tests_dir = task_dir / "tests"
        if tests_dir.exists():
            for test_file in tests_dir.rglob("*"):
                if test_file.is_file():
                    rel_path = test_file.relative_to(tests_dir)
                    test_files[str(rel_path)] = test_file.read_bytes()

        # Load solution script
        solution_path = task_dir / "solution" / "solve.sh"
        solution_script = solution_path.read_text() if solution_path.exists() else ""

        # Extract metadata
        env_config = config.get("environment", {})
        agent_config = config.get("agent", {})
        verifier_config = config.get("verifier", {})
        metadata = config.get("metadata", {})

        return TerminalBenchTask(
            task_id=task_dir.name,
            image=env_config.get("docker_image", ""),
            working_dir=workdir,
            instruction=instruction,
            agent_timeout=float(agent_config.get("timeout_sec", 900.0)),
            verifier_timeout=float(verifier_config.get("timeout_sec", 900.0)),
            test_files=test_files,
            solution_script=solution_script,
            difficulty=metadata.get("difficulty", "unknown"),
            category=metadata.get("category", "unknown"),
        )
