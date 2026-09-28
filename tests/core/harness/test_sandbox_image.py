"""Tests for derived swe-rex sandbox images."""

from __future__ import annotations

import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

from olmo_eval.harness.sandbox import image as image_module


class TestBuildSwerexDockerfile(unittest.TestCase):
    def test_default_layout_prepends_venv_to_path(self) -> None:
        dockerfile = image_module.build_swerex_dockerfile("base:latest")
        self.assertIn("FROM base:latest", dockerfile)
        self.assertIn('ENV PATH="/root/venv/bin:$PATH"', dockerfile)
        self.assertIn("apt-get install", dockerfile)

    def test_isolated_layout_leaves_environment_untouched(self) -> None:
        dockerfile = image_module.build_swerex_dockerfile(
            "base:latest", dockerfile_extra=("RUN echo extra",), isolated=True
        )
        self.assertIn("FROM base:latest", dockerfile)
        self.assertIn("/opt/swerex/venv", dockerfile)
        self.assertIn(
            "ln -sf /opt/swerex/venv/bin/swerex-remote /usr/local/bin/swerex-remote", dockerfile
        )
        self.assertIn("RUN echo extra", dockerfile)
        self.assertNotIn("ENV PATH", dockerfile)
        self.assertNotIn("VIRTUAL_ENV", dockerfile)
        self.assertNotIn("apt-get", dockerfile)


class TestGetSwerexImage(unittest.TestCase):
    def setUp(self) -> None:
        image_module._resolved_images.clear()
        self.addCleanup(image_module._resolved_images.clear)

    def _run(self, **kwargs: object) -> tuple[str, list[list[str]]]:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess:
            calls.append(cmd)
            # Local image missing, registry pulls fail, builds succeed.
            returncode = 1 if cmd[1:3] == ["image", "inspect"] or cmd[1] == "pull" else 0
            return subprocess.CompletedProcess(cmd, returncode, stdout=b"", stderr=b"")

        config = SimpleNamespace(swerex_registry="registry.example/repo")
        with (
            mock.patch.object(image_module, "get_infra_config", return_value=config),
            mock.patch.object(image_module.subprocess, "run", side_effect=fake_run),
        ):
            result = image_module.get_swerex_image("base:latest", "podman", **kwargs)  # type: ignore[arg-type]
        return result, calls

    def test_isolated_changes_the_image_tag(self) -> None:
        default, _ = self._run(use_registry=False)
        image_module._resolved_images.clear()
        isolated, _ = self._run(isolated=True, use_registry=False)
        self.assertNotEqual(default, isolated)

    def test_use_registry_false_skips_pull_and_push(self) -> None:
        result, calls = self._run(isolated=True, use_registry=False)
        self.assertTrue(result.startswith("swerex-"))
        verbs = [cmd[1] for cmd in calls]
        self.assertIn("build", verbs)
        self.assertNotIn("pull", verbs)
        self.assertNotIn("push", verbs)

    def test_registry_is_used_by_default(self) -> None:
        _, calls = self._run(isolated=True)
        verbs = [cmd[1] for cmd in calls]
        self.assertIn("pull", verbs)
        self.assertIn("push", verbs)


if __name__ == "__main__":
    unittest.main()
