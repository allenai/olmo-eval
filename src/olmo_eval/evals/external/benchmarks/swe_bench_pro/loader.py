"""SWE-Bench Pro task loader."""

from __future__ import annotations

import logging
import subprocess
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from . import v1
from .task import SWEBenchProTask, repo_from_instance_id

logger = logging.getLogger(__name__)

DEFAULT_WORKING_DIR = "/app"
DEFAULT_TIMEOUT_SECONDS = 3000.0


class SWEBenchProLoader:
    """Loads SWE-Bench Pro tasks from the official harness repository."""

    REPO_URL = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
    # Tag v2.0.0
    DEFAULT_REF = "66f92766bba642462d4bbe5479e83f91f9211862"

    def ensure_repo(self, target_dir: Path, ref: str | None = None) -> Path:
        """Fetch the harness repository at ``ref`` into ``target_dir``.

        Only the requested commit is fetched, so repeated calls with a
        different ref reuse the same directory.

        Args:
            target_dir: Directory to hold the checkout.
            ref: Commit SHA or tag to check out.

        Returns:
            Path to the checkout.
        """
        ref = ref or self.DEFAULT_REF
        git = ["git", "-C", str(target_dir)]

        if not (target_dir / ".git").exists():
            logger.info(f"Initializing SWE-Bench Pro repo at {target_dir}")
            target_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([*git, "init", "-q"], check=True, capture_output=True)
            subprocess.run(
                [*git, "remote", "add", "origin", self.REPO_URL], check=True, capture_output=True
            )

        head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True)
        if head.returncode == 0 and head.stdout.strip() == ref:
            return target_dir

        logger.info(f"Fetching SWE-Bench Pro repo at {ref}")
        subprocess.run(
            [*git, "fetch", "-q", "--depth", "1", "origin", ref], check=True, capture_output=True
        )
        subprocess.run(
            [*git, "checkout", "-q", "--force", "FETCH_HEAD"], check=True, capture_output=True
        )
        return target_dir

    def load_v2_tasks(
        self,
        repo_dir: Path,
        instance_ids: list[str] | None = None,
        hard_only: bool = False,
    ) -> list[SWEBenchProTask]:
        """Load the v2 Harbor task directories.

        Args:
            repo_dir: Path to the harness repository.
            instance_ids: Optional instance IDs to keep (default: all).
            hard_only: Keep only tasks in the HARD subset.

        Returns:
            Tasks sorted by instance ID.
        """
        v2_dir = repo_dir / "v2"
        tasks_dir = v2_dir / "tasks"
        if not tasks_dir.is_dir():
            raise FileNotFoundError(f"No v2 task directory at {tasks_dir}")

        hard_ids = self._read_id_list(v2_dir / "hard51_ids.txt")
        wanted = set(instance_ids) if instance_ids else None

        tasks = []
        for task_dir in sorted(tasks_dir.iterdir()):
            if not (task_dir / "task.toml").is_file():
                continue
            if wanted is not None and task_dir.name not in wanted:
                continue
            if hard_only and task_dir.name not in hard_ids:
                continue
            tasks.append(self._load_v2_task(task_dir, hard=task_dir.name in hard_ids))

        if wanted is not None:
            missing = wanted - {t.instance_id for t in tasks}
            if missing:
                logger.warning(f"Unknown SWE-Bench Pro instance IDs: {sorted(missing)}")

        logger.info(f"Loaded {len(tasks)} SWE-Bench Pro v2 tasks")
        return tasks

    def _load_v2_task(self, task_dir: Path, hard: bool) -> SWEBenchProTask:
        config = tomllib.loads((task_dir / "task.toml").read_text())
        env_config = config.get("environment", {})
        agent_config = config.get("agent", {})
        verifier_config = config.get("verifier", {})

        tests_dir = task_dir / "tests"
        test_files = {
            str(path.relative_to(tests_dir)): path.read_bytes()
            for path in sorted(tests_dir.rglob("*"))
            if path.is_file()
        }
        if "test.sh" not in test_files:
            raise FileNotFoundError(f"Task {task_dir.name} has no tests/test.sh")

        gold_patch_path = task_dir / "solution" / "gold_patch.diff"
        instance_id = task_dir.name
        return SWEBenchProTask(
            instance_id=instance_id,
            repo=repo_from_instance_id(instance_id),
            image=env_config["docker_image"],
            working_dir=DEFAULT_WORKING_DIR,
            instruction=(task_dir / "instruction.md").read_text(),
            test_files=test_files,
            gold_patch=gold_patch_path.read_text() if gold_patch_path.is_file() else "",
            agent_timeout=float(agent_config.get("timeout_sec", DEFAULT_TIMEOUT_SECONDS)),
            verifier_timeout=float(verifier_config.get("timeout_sec", DEFAULT_TIMEOUT_SECONDS)),
            hard=hard,
        )

    def load_v1_tasks(
        self,
        repo_dir: Path,
        rows: Iterable[Mapping[str, Any]] | None = None,
        instance_ids: list[str] | None = None,
    ) -> list[SWEBenchProTask]:
        """Load tasks from the original v1 release.

        Args:
            repo_dir: Path to the harness repository, which holds the v1 run scripts.
            rows: v1 dataset rows. Fetched from Hugging Face when omitted.
            instance_ids: Optional instance IDs to keep (default: all).

        Returns:
            Tasks sorted by instance ID.
        """
        run_scripts_dir = repo_dir / "run_scripts"
        if not run_scripts_dir.is_dir():
            raise FileNotFoundError(f"No v1 run script directory at {run_scripts_dir}")

        if rows is None:
            rows = self.fetch_v1_rows()
        wanted = set(instance_ids) if instance_ids else None
        selected = [r for r in rows if wanted is None or r["instance_id"] in wanted]

        if wanted is not None:
            missing = wanted - {r["instance_id"] for r in selected}
            if missing:
                logger.warning(f"Unknown SWE-Bench Pro v1 instance IDs: {sorted(missing)}")

        tasks = sorted(
            (v1.task_from_row(row, run_scripts_dir) for row in selected),
            key=lambda t: t.instance_id,
        )
        logger.info(f"Loaded {len(tasks)} SWE-Bench Pro v1 tasks")
        return tasks

    @staticmethod
    def fetch_v1_rows(revision: str = v1.DATASET_REVISION) -> list[dict[str, Any]]:
        """Download the v1 dataset rows from Hugging Face."""
        from datasets import load_dataset

        dataset = load_dataset(v1.DATASET_NAME, revision=revision, split="test")
        return [dict(row) for row in dataset]

    @staticmethod
    def _read_id_list(path: Path) -> set[str]:
        if not path.is_file():
            return set()
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}
