"""Tests for the SWE-Bench Pro external evaluation."""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import shutil
import subprocess
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from parameterized import parameterized

from olmo_eval.common.execution import ExecutionResult
from olmo_eval.common.types.trajectory import AgentTrajectory
from olmo_eval.evals.external.benchmarks.swe_bench_pro import eval as sbp_eval
from olmo_eval.evals.external.benchmarks.swe_bench_pro import verifier as sbp_verifier
from olmo_eval.evals.external.benchmarks.swe_bench_pro.loader import SWEBenchProLoader
from olmo_eval.evals.external.benchmarks.swe_bench_pro.task import (
    SWEBenchProTask,
    repo_from_instance_id,
)
from olmo_eval.evals.external.registry import get_external_eval

SHA = "0123456789abcdef0123456789abcdef01234567"
INSTANCE_ID = f"instance_element-hq__element-web-{SHA}-vnan"


def _make_task(**overrides: Any) -> SWEBenchProTask:
    fields: dict[str, Any] = {
        "instance_id": INSTANCE_ID,
        "repo": "element-hq/element-web",
        "image": "ghcr.io/example/image:tag",
        "working_dir": "/app",
        "instruction": "Fix the bug.",
        "test_files": {"test.sh": b"exit 0\n", "config.json": b"{}"},
        "gold_patch": "diff --git a/x b/x\n",
        "agent_timeout": 100.0,
        "verifier_timeout": 50.0,
        "hard": False,
    }
    fields.update(overrides)
    return SWEBenchProTask(**fields)


def _write_v2_task(tasks_dir: Path, instance_id: str, with_test_sh: bool = True) -> None:
    task_dir = tasks_dir / instance_id
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "task.toml").write_text(
        "[agent]\ntimeout_sec = 1200.0\n[verifier]\ntimeout_sec = 600.0\n"
        f'[environment]\ndocker_image = "ghcr.io/example/sbp:{instance_id}"\n'
    )
    (task_dir / "instruction.md").write_text(f"Instruction for {instance_id}")
    (task_dir / "solution" / "gold_patch.diff").write_text("gold patch")
    if with_test_sh:
        (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
    (task_dir / "tests" / "config.json").write_text("{}")


class FakeExecutor:
    """Executor stub that answers commands from a scripted handler."""

    def __init__(self, handler: Callable[[str], ExecutionResult] | None = None) -> None:
        self.handler = handler or (lambda _: ExecutionResult(success=True, output="", exit_code=0))
        self.commands: list[str] = []
        self.files: dict[str, str] = {}

    async def write_files(self, files: dict[str, str], timeout: float | None = None) -> None:
        self.files.update(files)

    async def execute_command(self, command: str, timeout: float | None = None, **_: Any):
        self.commands.append(command)
        return self.handler(command)


def _result(output: str = "", exit_code: int = 0) -> ExecutionResult:
    return ExecutionResult(success=exit_code == 0, output=output, exit_code=exit_code)


class TestRepoFromInstanceId(unittest.TestCase):
    @parameterized.expand(
        [
            (INSTANCE_ID, "element-hq/element-web"),
            (f"instance_future-architect__vuls-{SHA}", "future-architect/vuls"),
            (f"instance_NodeBB__NodeBB-{SHA}-vnan", "NodeBB/NodeBB"),
            ("not-an-instance", "unknown"),
        ]
    )
    def test_parses_owner_and_name(self, instance_id: str, expected: str) -> None:
        self.assertEqual(repo_from_instance_id(instance_id), expected)


class TestArgs(unittest.TestCase):
    def test_defaults(self) -> None:
        args = sbp_eval.SWEBenchProArgs.from_dict({})
        self.assertIsNone(args.instance_ids)
        self.assertEqual(args.subset, "all")
        self.assertIsNone(args.limit)
        self.assertIsNone(args.agent_timeout)
        self.assertFalse(args.oracle)
        self.assertTrue(args.remove_images)

    def test_parses_string_values(self) -> None:
        args = sbp_eval.SWEBenchProArgs.from_dict(
            {
                "instance_ids": "a, b,,c",
                "subset": "hard",
                "limit": "3",
                "max_turns": "7",
                "agent_timeout": "60",
                "oracle": "true",
                "remove_images": "false",
            }
        )
        self.assertEqual(args.instance_ids, ["a", "b", "c"])
        self.assertEqual(args.subset, "hard")
        self.assertEqual(args.limit, 3)
        self.assertEqual(args.max_turns, 7)
        self.assertEqual(args.agent_timeout, 60.0)
        self.assertTrue(args.oracle)
        self.assertFalse(args.remove_images)

    def test_rejects_malformed_boolean(self) -> None:
        with self.assertRaises(ValueError):
            sbp_eval.SWEBenchProArgs.from_dict({"oracle": "true -A max_concurrency=4"})

    def test_rejects_unknown_subset(self) -> None:
        with self.assertRaises(ValueError):
            sbp_eval.SWEBenchProArgs.from_dict({"subset": "easy"})


class TestLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.repo_dir)
        self.tasks_dir = self.repo_dir / "v2" / "tasks"
        self.ids = [f"instance_owner__repo-{str(i) * 40}" for i in range(3)]
        for instance_id in self.ids:
            _write_v2_task(self.tasks_dir, instance_id)
        (self.repo_dir / "v2" / "hard51_ids.txt").write_text(f"{self.ids[1]}\n")

    def test_loads_all_tasks(self) -> None:
        tasks = SWEBenchProLoader().load_v2_tasks(self.repo_dir)
        self.assertEqual([t.instance_id for t in tasks], self.ids)
        task = tasks[0]
        self.assertEqual(task.repo, "owner/repo")
        self.assertEqual(task.image, f"ghcr.io/example/sbp:{self.ids[0]}")
        self.assertEqual(task.working_dir, "/app")
        self.assertEqual(task.instruction, f"Instruction for {self.ids[0]}")
        self.assertEqual(task.gold_patch, "gold patch")
        self.assertEqual(task.agent_timeout, 1200.0)
        self.assertEqual(task.verifier_timeout, 600.0)
        self.assertEqual(set(task.test_files), {"test.sh", "config.json"})
        self.assertEqual([t.hard for t in tasks], [False, True, False])

    def test_filters_by_instance_id_and_hard(self) -> None:
        loader = SWEBenchProLoader()
        tasks = loader.load_v2_tasks(self.repo_dir, instance_ids=[self.ids[2], "missing"])
        self.assertEqual([t.instance_id for t in tasks], [self.ids[2]])
        tasks = loader.load_v2_tasks(self.repo_dir, hard_only=True)
        self.assertEqual([t.instance_id for t in tasks], [self.ids[1]])

    def test_task_without_verifier_is_rejected(self) -> None:
        _write_v2_task(self.tasks_dir, f"instance_owner__repo-{'9' * 40}", with_test_sh=False)
        with self.assertRaises(FileNotFoundError):
            SWEBenchProLoader().load_v2_tasks(self.repo_dir)

    def test_missing_task_directory_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            SWEBenchProLoader().load_v2_tasks(self.repo_dir / "nowhere")


@unittest.skipUnless(shutil.which("git"), "git is required")
class TestPatchCommands(unittest.TestCase):
    """Run the capture and apply commands against a real repository."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self._git("init", "-q")
        (self.repo / "a.txt").write_text("one\n")
        (self.repo / "logo.bin").write_bytes(b"\x00\x01\x02")
        self._git("add", "-A")
        self._git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")

    def _git(self, *args: str, cwd: Path | None = None) -> None:
        subprocess.run(["git", *args], cwd=cwd or self.repo, check=True, capture_output=True)

    def _bash(self, command: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", command], capture_output=True, text=True)

    def test_capture_includes_new_and_binary_files_and_applies_cleanly(self) -> None:
        (self.repo / "a.txt").write_text("two\n")
        (self.repo / "new.py").write_text("print('hi')\n")
        (self.repo / "logo.bin").write_bytes(b"\x00\xff\x02\x03")

        captured = self._bash(sbp_verifier.capture_patch_command(str(self.repo)))
        self.assertEqual(captured.returncode, 0, captured.stderr)
        patch = sbp_verifier.parse_captured_patch(captured.stdout)
        assert patch is not None
        self.assertIn("new.py", patch)
        self.assertIn("GIT binary patch", patch)

        # The capture leaves the index untouched.
        status = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=self.repo,
            capture_output=True,
            text=True,
        )
        self.assertEqual(status.stdout, "")

        pristine = self.root / "pristine"
        self._git("clone", "-q", str(self.repo), str(pristine), cwd=self.root)
        patch_file = self.root / "model.patch"
        patch_file.write_text(patch, errors="surrogateescape")
        applied = self._bash(sbp_verifier.apply_patch_command(str(pristine), str(patch_file)))
        self.assertEqual(applied.returncode, 0, applied.stderr)
        self.assertEqual((pristine / "a.txt").read_text(), "two\n")
        self.assertEqual((pristine / "new.py").read_text(), "print('hi')\n")
        self.assertEqual((pristine / "logo.bin").read_bytes(), b"\x00\xff\x02\x03")

    def test_capture_of_clean_repo_is_empty(self) -> None:
        captured = self._bash(sbp_verifier.capture_patch_command(str(self.repo)))
        self.assertEqual(sbp_verifier.parse_captured_patch(captured.stdout), "")

    def test_parse_rejects_incomplete_output(self) -> None:
        self.assertIsNone(sbp_verifier.parse_captured_patch("no markers here"))

    def _snapshot(self, repo: Path) -> str:
        output = self._bash(sbp_verifier.snapshot_command(str(repo))).stdout
        tree = sbp_verifier.parse_snapshot(output)
        assert tree is not None, output
        return tree

    def test_snapshot_excludes_files_the_image_left_untracked(self) -> None:
        # Images can ship untracked files, such as a database directory.
        (self.repo / "state").mkdir()
        (self.repo / "state" / "db.aof").write_text("v1\n")
        (self.repo / "untouched.log").write_text("keep\n")
        pristine = self.root / "pristine"
        shutil.copytree(self.repo, pristine)

        baseline = self._snapshot(self.repo)
        self.assertEqual(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=self.repo, capture_output=True, text=True
            ).stdout.count("??"),
            2,
        )

        (self.repo / "state" / "db.aof").write_text("v2\n")
        (self.repo / "a.txt").write_text("two\n")
        captured = self._bash(sbp_verifier.capture_patch_command(str(self.repo), baseline))
        patch = sbp_verifier.parse_captured_patch(captured.stdout)
        assert patch is not None
        self.assertNotIn("untouched.log", patch)
        self.assertNotIn("new file mode", patch)

        patch_file = self.root / "model.patch"
        patch_file.write_text(patch)
        applied = self._bash(sbp_verifier.apply_patch_command(str(pristine), str(patch_file)))
        self.assertEqual(applied.returncode, 0, applied.stdout + applied.stderr)
        self.assertEqual((pristine / "state" / "db.aof").read_text(), "v2\n")
        self.assertEqual((pristine / "a.txt").read_text(), "two\n")

    @parameterized.expand([("",), ("warning\nnot-a-sha",)])
    def test_parse_snapshot_rejects_bad_output(self, output: str) -> None:
        self.assertIsNone(sbp_verifier.parse_snapshot(output))


class TestPinLocalhost(unittest.TestCase):
    def test_removes_localhost_from_ipv6_loopback_only(self) -> None:
        hosts = Path(tempfile.mkdtemp()) / "hosts"
        self.addCleanup(shutil.rmtree, hosts.parent)
        hosts.write_text(
            "127.0.0.1\tlocalhost\n"
            "::1\tlocalhost ip6-localhost ip6-loopback\n"
            "10.0.0.5\tlocalhost-box\n"
        )
        command = sbp_eval._PIN_LOCALHOST_COMMAND.replace("/etc/hosts", str(hosts))
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = hosts.read_text().splitlines()
        self.assertEqual(lines[0], "127.0.0.1\tlocalhost")
        self.assertEqual(lines[1].split(), ["::1", "ip6-localhost", "ip6-loopback"])
        self.assertEqual(lines[2], "10.0.0.5\tlocalhost-box")


class TestWriteFiles(unittest.TestCase):
    def test_stages_base64_and_decodes_in_sandbox(self) -> None:
        executor = FakeExecutor()
        content = b"\x00binary\xff"
        asyncio.run(sbp_verifier.write_files(executor, {"/tests/sub/file.bin": content}))  # type: ignore[arg-type]

        ((staged_path, staged),) = executor.files.items()
        self.assertEqual(base64.b64decode(staged), content)
        (command,) = executor.commands
        self.assertIn(f"base64 -d < {staged_path} > /tests/sub/file.bin", command)
        self.assertIn('mkdir -p "$(dirname /tests/sub/file.bin)"', command)

    def test_raises_when_decoding_fails(self) -> None:
        executor = FakeExecutor(lambda _: _result("boom", exit_code=1))
        with self.assertRaises(RuntimeError):
            asyncio.run(sbp_verifier.write_files(executor, {"/x": b"y"}))  # type: ignore[arg-type]


class TestVerifier(unittest.TestCase):
    def _verify(self, handler: Callable[[str], ExecutionResult], patch: str):
        executor = FakeExecutor(handler)
        result = asyncio.run(
            sbp_verifier.SWEBenchProVerifier().verify(
                executor,  # type: ignore[arg-type]
                patch,
                {"test.sh": b"exit 0"},
                "/app",
                timeout=10.0,
            )
        )
        return result, executor

    def test_resolved_when_reward_is_one(self) -> None:
        def handler(command: str) -> ExecutionResult:
            if command.startswith("cat /logs/verifier/reward.txt"):
                return _result("1\n")
            return _result()

        result, executor = self._verify(handler, "diff --git a/x b/x\n")
        self.assertTrue(result.resolved)
        self.assertTrue(result.patch_applied)
        self.assertTrue(any("git apply" in c for c in executor.commands))
        self.assertTrue(any("bash /tests/test.sh" in c for c in executor.commands))

    def test_unresolved_when_reward_is_zero(self) -> None:
        def handler(command: str) -> ExecutionResult:
            if "bash /tests/test.sh" in command:
                return _result("RESULT: FAILED", exit_code=1)
            if command.startswith("cat /logs/verifier/reward.txt"):
                return _result("0\n")
            if "run-script-stdout.txt" in command:
                encoded = base64.b64encode(gzip.compress(b"AssertionError: boom\n")).decode()
                return _result(encoded)
            if "run-script-stderr.txt" in command:
                return _result("", exit_code=3)
            return _result()

        result, _ = self._verify(handler, "")
        self.assertFalse(result.resolved)
        self.assertEqual(result.test_exit_code, 1)
        self.assertEqual(result.runner_logs, {"run-script-stdout.txt": "AssertionError: boom\n"})
        self.assertIn("AssertionError: boom", result.test_output)

    def test_empty_patch_skips_apply(self) -> None:
        result, executor = self._verify(lambda _: _result("1"), "  \n")
        self.assertTrue(result.patch_applied)
        self.assertFalse(any("git apply" in c for c in executor.commands))

    def test_patch_that_does_not_apply_is_unresolved(self) -> None:
        def handler(command: str) -> ExecutionResult:
            if "git apply" in command:
                return _result("error: patch failed", exit_code=1)
            return _result("1")

        result, executor = self._verify(handler, "diff --git a/x b/x\n")
        self.assertFalse(result.resolved)
        self.assertFalse(result.patch_applied)
        self.assertFalse(any("bash /tests/test.sh" in c for c in executor.commands))


def _task_result(**overrides: Any) -> sbp_eval.TaskResult:
    fields: dict[str, Any] = {
        "instance_id": INSTANCE_ID,
        "repo": "element-hq/element-web",
        "hard": False,
        "resolved": False,
        "patch": "diff",
        "trajectory": AgentTrajectory(turns=()),
        "completion_reason": "complete",
        "agent_duration": 1.0,
    }
    fields.update(overrides)
    return sbp_eval.TaskResult(**fields)


class TestMetrics(unittest.TestCase):
    def test_compute_metrics(self) -> None:
        results = [
            _task_result(resolved=True, hard=True),
            _task_result(resolved=False, hard=True, patch=""),
            _task_result(repo="NodeBB/NodeBB", resolved=True),
            _task_result(repo="NodeBB/NodeBB", patch="", error="boom"),
        ]
        metrics = sbp_eval.SWEBenchProExternalEval.compute_metrics(results)
        self.assertEqual(metrics["pass_rate"], 0.5)
        self.assertEqual(metrics["num_tasks"], 4)
        self.assertEqual(metrics["num_resolved"], 2)
        self.assertEqual(metrics["num_errors"], 1)
        self.assertEqual(metrics["num_empty_patches"], 1)
        self.assertEqual(metrics["pass_rate_hard"], 0.5)
        self.assertEqual(metrics["pass_rate_element_hq_element_web"], 0.5)
        self.assertEqual(metrics["pass_rate_nodebb_nodebb"], 0.5)

    def test_compute_metrics_without_hard_tasks(self) -> None:
        metrics = sbp_eval.SWEBenchProExternalEval.compute_metrics([_task_result()])
        self.assertNotIn("pass_rate_hard", metrics)

    def test_save_task_results_writes_patches_and_runner_logs(self) -> None:
        output_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, output_dir)
        results = [
            _task_result(runner_logs={"run-script-stdout.txt": "log text"}),
            _task_result(instance_id="other", resolved=True, patch="p2"),
        ]
        sbp_eval.SWEBenchProExternalEval()._save_task_results(results, str(output_dir))

        tasks = json.loads((output_dir / "swe_bench_pro_tasks.json").read_text())
        self.assertEqual([t["resolved"] for t in tasks], [False, True])
        patches = (output_dir / "swe_bench_pro_patches.jsonl").read_text().splitlines()
        self.assertEqual(json.loads(patches[1]), {"instance_id": "other", "model_patch": "p2"})
        log_file = output_dir / "verifier_logs" / INSTANCE_ID / "run-script-stdout.txt"
        self.assertEqual(log_file.read_text(), "log text")
        self.assertFalse((output_dir / "verifier_logs" / "other").exists())

    def test_compute_metrics_empty(self) -> None:
        metrics = sbp_eval.SWEBenchProExternalEval.compute_metrics([])
        self.assertEqual(metrics["pass_rate"], 0.0)


class FakeSandboxManager:
    """SandboxManager stand-in that hands out one shared FakeExecutor."""

    instances: list[FakeSandboxManager] = []

    def __init__(self, configs: list[Any], owner: str | None = None) -> None:
        self.configs = configs
        self.owner = owner
        self.started = False
        self.stopped = False
        self.executor = FakeExecutor(self._handle)
        FakeSandboxManager.instances.append(self)

    @staticmethod
    def _handle(command: str) -> ExecutionResult:
        if command.startswith("cat /logs/verifier/reward.txt"):
            return _result("1")
        if "git write-tree" in command:
            return _result(f"{SHA}\n")
        return _result()

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def get_executor(self, _: frozenset[str]) -> FakeExecutor:
        return self.executor


class TestExecuteTask(unittest.TestCase):
    def setUp(self) -> None:
        FakeSandboxManager.instances = []
        patches = [
            mock.patch("olmo_eval.harness.sandbox.SandboxManager", FakeSandboxManager),
            mock.patch(
                "olmo_eval.harness.sandbox.image.get_swerex_image",
                return_value="swerex-derived:latest",
            ),
            mock.patch.object(sbp_eval, "get_docker_network_args", return_value=()),
            mock.patch.object(sbp_eval, "_remove_images"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.remove_images = sbp_eval._remove_images

    def _execute(self, args: dict[str, Any], **eval_overrides: Any) -> sbp_eval.TaskResult:
        swe_eval = sbp_eval.SWEBenchProExternalEval()
        for name, value in eval_overrides.items():
            setattr(swe_eval, name, value)
        return asyncio.run(
            swe_eval._execute_task(
                _make_task(),
                SimpleNamespace(model_name="m"),  # type: ignore[arg-type]
                "podman",
                sbp_eval.SWEBenchProArgs.from_dict(args),
            )
        )

    def test_oracle_grades_gold_patch_in_fresh_sandbox(self) -> None:
        result = self._execute({"oracle": True})
        self.assertTrue(result.resolved)
        self.assertEqual(result.patch, "diff --git a/x b/x\n")
        self.assertEqual(result.completion_reason, "oracle")
        (grader,) = FakeSandboxManager.instances
        self.assertTrue(grader.stopped)
        self.assertEqual(grader.configs[0].image, "swerex-derived:latest")
        self.remove_images.assert_called_once_with(  # type: ignore[attr-defined]
            "podman", ["swerex-derived:latest", "ghcr.io/example/image:tag"]
        )

    def test_agent_patch_is_captured_then_graded_separately(self) -> None:
        captured = base64.b64encode(b"agent diff").decode()

        async def fake_run_agent(*_: Any, **__: Any) -> tuple[AgentTrajectory, str]:
            agent_manager = FakeSandboxManager.instances[0]
            agent_manager.executor.handler = lambda _: _result(
                f"<<<SWE_BENCH_PRO_PATCH_START>>>\n{captured}\n<<<SWE_BENCH_PRO_PATCH_END>>>"
            )
            return AgentTrajectory(turns=()), "complete"

        result = self._execute({"remove_images": False}, _run_agent=fake_run_agent)
        self.assertEqual(result.patch, "agent diff")
        self.assertTrue(result.resolved)
        agent_manager, grader = FakeSandboxManager.instances
        capture = next(c for c in agent_manager.executor.commands if "git diff --cached" in c)
        self.assertIn(f"git diff --cached --binary {SHA}", capture)
        self.assertTrue(agent_manager.stopped and grader.stopped)
        self.assertIn(sbp_verifier.PATCH_PATH, " ".join(grader.executor.commands))
        self.remove_images.assert_not_called()  # type: ignore[attr-defined]

    def test_failure_is_reported_as_error(self) -> None:
        async def failing_run_agent(*_: Any, **__: Any) -> tuple[AgentTrajectory, str]:
            raise RuntimeError("agent crashed")

        result = self._execute({}, _run_agent=failing_run_agent)
        self.assertFalse(result.resolved)
        self.assertEqual(result.completion_reason, "error")
        self.assertIn("agent crashed", result.error or "")
        self.assertTrue(FakeSandboxManager.instances[0].stopped)


class TestGradeRetries(unittest.TestCase):
    def setUp(self) -> None:
        FakeSandboxManager.instances = []
        patcher = mock.patch("olmo_eval.harness.sandbox.SandboxManager", FakeSandboxManager)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _grade(self, *results: sbp_verifier.VerificationResult):
        with mock.patch.object(
            sbp_verifier.SWEBenchProVerifier, "verify", side_effect=list(results)
        ) as verify:
            outcome = asyncio.run(
                sbp_eval.SWEBenchProExternalEval()._grade(
                    SimpleNamespace(),  # type: ignore[arg-type]
                    _make_task(),
                    "diff",
                    "owner",
                )
            )
        return outcome, verify.call_count

    def test_retries_once_after_unresponsive_sandbox(self) -> None:
        unresponsive = sbp_verifier.VerificationResult(
            False, True, "Sandbox unresponsive after 3 polls", -1
        )
        passed = sbp_verifier.VerificationResult(True, True, "RESULT: PASSED", 0)
        outcome, calls = self._grade(unresponsive, passed)
        self.assertTrue(outcome.resolved)
        self.assertEqual(calls, 2)
        self.assertTrue(all(m.stopped for m in FakeSandboxManager.instances))

    def test_does_not_retry_a_test_failure(self) -> None:
        failed = sbp_verifier.VerificationResult(False, True, "RESULT: FAILED", 1)
        outcome, calls = self._grade(failed)
        self.assertFalse(outcome.resolved)
        self.assertEqual(calls, 1)

    def test_gives_up_after_retry_budget(self) -> None:
        unresponsive = sbp_verifier.VerificationResult(
            False, True, "Sandbox unresponsive after 3 polls", -1
        )
        outcome, calls = self._grade(unresponsive, unresponsive)
        self.assertFalse(outcome.resolved)
        self.assertEqual(calls, 1 + sbp_eval.GRADE_RETRIES)


class TestRegistration(unittest.TestCase):
    def test_v2_is_registered(self) -> None:
        swe_eval = get_external_eval("swe_bench_pro")
        self.assertIsInstance(swe_eval, sbp_eval.SWEBenchProExternalEval)
        self.assertEqual(swe_eval.version, "v2")  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
