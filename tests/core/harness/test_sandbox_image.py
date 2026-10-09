"""Tests for the derived swe-rex image build."""

from __future__ import annotations

from unittest import mock

from olmo_eval.harness.sandbox import SandboxConfig, SandboxMode
from olmo_eval.harness.sandbox import image as image_mod


class TestDockerfile:
    def test_default_build_puts_the_venv_on_path(self) -> None:
        dockerfile = image_mod.build_swerex_dockerfile("base:1", ("RUN echo extra",))

        assert dockerfile.startswith("FROM base:1\n")
        assert "apt-get install" in dockerfile
        assert 'ENV PATH="/root/venv/bin:$PATH"' in dockerfile
        assert "RUN echo extra" in dockerfile

    def test_pristine_build_leaves_the_base_environment_alone(self) -> None:
        dockerfile = image_mod.build_swerex_dockerfile("base:1", ("RUN echo extra",), pristine=True)

        assert dockerfile.startswith("FROM base:1\n")
        assert "apt-get" not in dockerfile
        assert "ENV PATH" not in dockerfile
        assert "ENV VIRTUAL_ENV" not in dockerfile
        assert "pip install --python /root/venv/bin/python --no-cache-dir swe-rex" in dockerfile
        assert "ln -sf /root/venv/bin/swerex-remote /usr/local/bin/swerex-remote" in dockerfile
        assert "rm /usr/local/bin/swerex-uv" in dockerfile
        assert "/usr/local/bin/uv" not in dockerfile
        assert "RUN echo extra" in dockerfile


class TestImageResolution:
    def setup_method(self) -> None:
        image_mod._resolved_images.clear()

    def teardown_method(self) -> None:
        image_mod._resolved_images.clear()

    def test_pristine_and_default_images_have_different_tags(self) -> None:
        with mock.patch.object(image_mod, "_resolve_swerex_image", return_value="x") as resolve:
            image_mod.get_swerex_image("base:1", "docker")
            image_mod.get_swerex_image("base:1", "docker", pristine=True)

        tags = [call.args[4] for call in resolve.call_args_list]
        assert len(set(tags)) == 2
        assert resolve.call_args_list[1].args[5] is True

    def test_resolution_is_cached_per_tag(self) -> None:
        with mock.patch.object(image_mod, "_resolve_swerex_image", return_value="x") as resolve:
            image_mod.get_swerex_image("base:1", "docker", pristine=True)
            image_mod.get_swerex_image("base:1", "docker", pristine=True)

        assert resolve.call_count == 1

    def test_a_build_uses_the_pristine_dockerfile_on_the_base_platform(self) -> None:
        completed = mock.Mock(returncode=0, stderr=b"")
        missing = mock.Mock(returncode=1, stderr=b"")
        inspected = mock.Mock(returncode=0, stdout=b"linux/amd64\n")
        with (
            mock.patch.object(image_mod, "get_infra_config") as infra,
            mock.patch.object(
                image_mod.subprocess, "run", side_effect=[missing, inspected, completed]
            ) as run,
        ):
            infra.return_value.swerex_registry = None
            image = image_mod._resolve_swerex_image("base:1", "docker", (), False, "abc", True)

        assert image == "swerex-abc:latest"
        build = run.call_args_list[2]
        assert build.args[0] == [
            "docker",
            "build",
            "-t",
            "swerex-abc:latest",
            "--platform=linux/amd64",
            "-",
        ]
        assert b"ENV PATH" not in build.kwargs["input"]
        assert b"swerex-remote" in build.kwargs["input"]

    def test_a_missing_base_image_is_pulled_to_read_its_platform(self) -> None:
        not_local = mock.Mock(returncode=1)
        pulled = mock.Mock(returncode=0)
        inspected = mock.Mock(returncode=0, stdout=b"linux/arm64")
        with mock.patch.object(
            image_mod.subprocess, "run", side_effect=[not_local, pulled, inspected]
        ) as run:
            assert image_mod._image_platform("podman", "base:1") == "linux/arm64"

        assert run.call_args_list[1].args[0] == ["podman", "pull", "base:1"]

    def test_an_unreadable_platform_leaves_the_build_default(self) -> None:
        failed = mock.Mock(returncode=1, stderr=b"nope")
        with mock.patch.object(image_mod.subprocess, "run", return_value=failed):
            assert image_mod._image_platform("docker", "base:1") is None


def test_pristine_image_round_trips_through_dict() -> None:
    config = SandboxConfig(image="base:1", mode=SandboxMode.DOCKER, pristine_image=True)

    restored = SandboxConfig.from_dict(config.to_dict())

    assert restored.pristine_image is True
    assert SandboxConfig.from_dict({"image": "b", "mode": "docker"}).pristine_image is False
