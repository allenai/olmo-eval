"""SGLang uses the same automatic extra installation path as other providers."""

from unittest.mock import patch

import pytest

from olmo_eval.cli.beaker.config_loader import LaunchConfig
from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler
from olmo_eval.launch.beaker.launcher import BeakerLauncher


@pytest.mark.parametrize("custom_package", [None, "sglang==0.5.19"])
def test_sglang_job_installs_selected_provider(custom_package):
    overrides = ["provider.kind=sglang"]
    if custom_package:
        overrides.append(f"provider.package={custom_package}")
    config = LaunchConfig(
        name="sglang-install-test",
        model_specs=["llama3.1-8b"],  # Overrides must beat the model's vLLM preset.
        task_specs=["gsm8k"],
        cluster="h100",
        workspace="ai2/test",
        harness="default",
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
        effective_image="test-cuda13-image",
        effective_groups=[],
        beaker_username="test-user",
        common_secrets=[],
        task_secrets=[],
        inject_aws_credentials=False,
        inject_gcs_credentials=False,
    )
    with patch("olmo_eval.cli.beaker.job_assembler.cluster_has_weka", return_value=False):
        job = assembler.assemble(plan)
    assert "vllm" not in job.extras
    assert "clients" not in job.extras
    assert not job.vllm_isolated_venv
    assert ("sglang" in job.extras) == (custom_package is None)
    if custom_package:
        assert custom_package in job.provider_packages
    # Render the real installation command without constructing a network client.
    launcher = object.__new__(BeakerLauncher)
    command = launcher._build_install_cmd(job.extras, provider_packages=job.provider_packages)
    assert "sglang" in command
    assert "vllm" not in command
    assert "-c /tmp/cuda-constraints.txt" in command
