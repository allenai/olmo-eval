"""SGLang uses the same automatic extra installation path as other providers."""

from unittest.mock import patch

import pytest

from olmo_eval.cli.beaker.config_loader import LaunchConfig
from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler
from olmo_eval.common.constants.infrastructure import BEAKER_DEFAULT_IMAGE, BEAKER_SGLANG_IMAGE
from olmo_eval.launch.beaker.launcher import BeakerLauncher


@pytest.mark.parametrize(
    ("provider_kind", "custom_package"),
    [("sglang", None), ("sglang", "sglang==0.5.19"), ("vllm", None)],
)
@pytest.mark.parametrize(
    "image", [None, "explicit-image", "author/olmo-eval-cu1302-trc2130-amd64-devel"]
)
@pytest.mark.parametrize("library_path", [None, "/custom/driver", "secret"])
def test_job_installs_selected_provider(provider_kind, custom_package, image, library_path):
    overrides = [f"provider.kind={provider_kind}"]
    if custom_package:
        overrides.append(f"provider.package={custom_package}")
    config = LaunchConfig(
        name="sglang-install-test",
        model_specs=["llama3.1-8b"],  # Overrides must beat the model's vLLM preset.
        task_specs=["gsm8k"],
        cluster="h100",
        workspace="ai2/test",
        harness="default",
        env_vars={"LD_LIBRARY_PATH": library_path}
        if library_path and library_path != "secret"
        else {},
        harness_overrides=overrides,
    )
    plan = ExperimentPlan(
        name=config.name,
        model_spec=config.model_specs[0],
        priority="normal",
        tasks=config.task_specs,
        original_task_specs=config.task_specs,
        total_expanded_tasks=1,
        num_gpus=1,
    )
    assembler = JobConfigAssembler(
        config=config,
        effective_image=image,
        effective_groups=[],
        beaker_username="test-user",
        common_secrets=[("LD_LIBRARY_PATH", "custom-library-path")]
        if library_path == "secret"
        else [],
        task_secrets=[],
        inject_aws_credentials=False,
        inject_gcs_credentials=False,
    )
    with patch("olmo_eval.cli.beaker.job_assembler.cluster_has_weka", return_value=False):
        job = assembler.assemble(plan)
    default_image = BEAKER_SGLANG_IMAGE if provider_kind == "sglang" else BEAKER_DEFAULT_IMAGE
    assert job.beaker_image == (image or default_image)
    if provider_kind == "sglang" and library_path is None:
        assert job.command[:2] == [
            "bash",
            "/gantry-runtime/src/olmo_eval/launch/beaker/scripts/cuda13_driver",
        ]
    else:
        assert job.command[0] == "olmo-eval"
    assert job.env_vars.get("LD_LIBRARY_PATH") == (
        None if library_path == "secret" else library_path
    )
    assert ("vllm" in job.extras) == (provider_kind == "vllm")
    assert "clients" not in job.extras
    assert not job.vllm_isolated_venv
    assert ("sglang" in job.extras) == (provider_kind == "sglang" and custom_package is None)
    if custom_package:
        assert custom_package in job.provider_packages
    # Render the real installation command without constructing a network client.
    launcher = object.__new__(BeakerLauncher)
    command = launcher._build_install_cmd(job.extras, provider_packages=job.provider_packages)
    assert provider_kind in command
    if provider_kind == "sglang":
        assert "vllm" not in command
    else:
        assert "sglang" not in command
    assert "-c /tmp/cuda-constraints.txt" in command
