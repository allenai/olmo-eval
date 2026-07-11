import stat
from pathlib import Path
from unittest import mock

from olmo_eval.evals.external.base import SandboxedExternalEval
from olmo_eval.evals.external.benchmarks.deepscholar.args import DeepScholarArgs
from olmo_eval.evals.external.benchmarks.deepscholar.eval import DeepScholarExternalEval
from olmo_eval.harness.sandbox.config import SandboxConfig, SandboxMode


def test_output_directories_are_bind_mounted(tmp_path: Path) -> None:
    evaluator = DeepScholarExternalEval()
    base_config = SandboxConfig(image="test", mode=SandboxMode.DOCKER)

    with mock.patch.object(
        SandboxedExternalEval, "_create_sandbox_config", return_value=base_config
    ):
        config = evaluator._create_sandbox_config("podman", str(tmp_path))

    destination = tmp_path / "deepscholar_results"
    assert config.inject_swerex
    assert config.volumes == (
        (str(destination / "generation"), evaluator._gen_dir),
        (str(destination / "evaluation"), evaluator._eval_dir),
    )
    assert (destination / "generation").is_dir()
    assert (destination / "evaluation").is_dir()
    assert stat.S_IMODE((destination / "generation").stat().st_mode) == 0o777
    assert stat.S_IMODE((destination / "evaluation").stat().st_mode) == 0o777


def test_max_batch_size_reaches_lotus_lm_config() -> None:
    evaluator = DeepScholarExternalEval()
    args = DeepScholarArgs.from_dict({"max_batch_size": "8"})

    lm_config = evaluator._build_lm_config("model", "http://vllm/v1", True, args)

    assert args.max_batch_size == 8
    assert lm_config["max_batch_size"] == 8
