"""SandboxConfig survives serialization, as a harness config does when CLI overrides apply."""

import dataclasses

from olmo_eval.cli.run.config import _apply_dotlist_overrides
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.presets import get_harness_preset
from olmo_eval.harness.sandbox.config import Capability, RegistryAuth, SandboxConfig, SandboxMode


def _non_default_config() -> SandboxConfig:
    """A config with every field away from its default, so a dropped field shows up."""
    return SandboxConfig(
        image="example/image:1",
        mode=SandboxMode.MODAL,
        capabilities=Capability.BASH | Capability.PYTHON,
        instances=7,
        max_concurrency=3,
        min_instances=5,
        container_runtime="docker",
        startup_timeout=12.0,
        command_timeout=34.0,
        remove_container=False,
        working_dir="/work",
        environment=(("A", "1"),),
        volumes=(("/host", "/container"),),
        modal_sandbox_kwargs={"cpu": 2},
        runtime_timeout=56.0,
        deployment_timeout=78.0,
        required_secrets=("SECRET",),
        docker_args=("--init",),
        log_dir="/logs",
        exec_shell=("bash", "-c"),
        enable_diagnostics=False,
        inject_swerex=True,
        pristine_image=True,
        dockerfile_extra=("RUN true",),
        image_pull="never",
        registry_auth=RegistryAuth(provider="gcp"),
        retry_delivered_commands=True,
    )


def test_every_field_is_set_away_from_its_default() -> None:
    config = _non_default_config()
    default = SandboxConfig(image="example/image:1", mode=SandboxMode.MODAL)
    same = [
        f.name
        for f in dataclasses.fields(SandboxConfig)
        if f.name not in ("image", "mode") and getattr(config, f.name) == getattr(default, f.name)
    ]
    assert same == [], f"extend _non_default_config to cover {same}"


def test_round_trip_keeps_every_field() -> None:
    config = _non_default_config()
    assert SandboxConfig.from_dict(config.to_dict()) == config


def test_preset_sandbox_settings_survive_a_sandboxes_override() -> None:
    preset = get_harness_preset("codex_universal")
    assert all(s.retry_delivered_commands for s in preset.sandboxes)

    data = _apply_dotlist_overrides(preset.to_dict(), ['sandboxes={"max_concurrency":1}'])
    config = HarnessConfig.from_dict(data)

    assert [s.max_concurrency for s in config.sandboxes] == [1] * len(preset.sandboxes)
    assert all(s.retry_delivered_commands for s in config.sandboxes)
    for got, want in zip(config.sandboxes, preset.sandboxes, strict=True):
        assert dataclasses.replace(got, max_concurrency=want.max_concurrency) == want
