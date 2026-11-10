"""Public launch errors and an explicit escape hatch for unclaimed workspaces."""

import importlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from olmo_eval.cli.beaker.launch import launch
from olmo_eval.launch.beaker.volume import launch_with_volume


@pytest.fixture
def submission(monkeypatch):
    module = importlib.import_module("olmo_eval.cli.beaker.launch")
    launcher = Mock()
    launcher.beaker.user_name = "tester"
    launcher.experiment_url.return_value = "https://beaker.org/ex/test"
    monkeypatch.setattr("olmo_eval.launch.BeakerLauncher", Mock(return_value=launcher))
    monkeypatch.setattr(module, "_handle_group_creation", Mock())
    monkeypatch.setattr(module, "_prepare_secrets", Mock(return_value=([], [])))
    credentials = Mock()
    credentials.detect_and_setup.return_value = (False, False)
    monkeypatch.setattr(
        importlib.import_module("olmo_eval.cli.beaker.credentials"),
        "CredentialManager",
        Mock(return_value=credentials),
    )
    return launcher


def args(*extra):
    return [
        "-m",
        "mock",
        "-t",
        "gsm8k",
        "-c",
        "h100",
        "-w",
        "ai2/unclaimed",
        "-B",
        "ai2/test",
        "--no-upload",
        "--no-follow",
        "--yes",
        *extra,
    ]


@pytest.mark.parametrize(
    "error",
    [
        'Error: workspace "ai2/unclaimed" has no claim on volume "ai2/olmo-eval"',
        'Error: volume "ai2/olmo-eval" claim is read-only; cannot mount with read-write mode',
    ],
)
def test_claim_rejection_is_actionable_cli_error(submission, monkeypatch, error):
    def gantry(**kwargs):
        Path(kwargs["save_spec"]).write_text('{"version":"v2","tasks":[{}]}')

    submit = Mock(side_effect=subprocess.CalledProcessError(1, "beaker", stderr=error))
    monkeypatch.setattr("olmo_eval.launch.beaker.volume.subprocess.run", submit)

    def rejected(job):
        return launch_with_volume(
            gantry, {"name": job.name, "workspace": job.workspace}, job.resume_volume, dry_run=False
        )

    submission.launch.side_effect = rejected
    result = CliRunner().invoke(launch, args())
    assert result.exit_code == 1
    assert result.output.count("Error: Evaluation was not submitted") == 1
    assert "Traceback" not in result.output
    assert "RuntimeError" not in result.output
    assert "--no-resume" in result.output
    assert "Interrupted or retried attempts will start over" in result.output
    assert (
        "beaker volume claim create --workspace ai2/unclaimed "
        "--volume ai2/olmo-eval --mode readwrite"
    ) in result.output
    assert error in result.output
    assert submit.call_count == 1


def test_no_resume_cli_submits_without_volume_or_progress(submission):
    submission.launch.return_value = SimpleNamespace(id="test")
    result = CliRunner().invoke(launch, args("--no-resume"))
    assert result.exit_code == 0, result.output
    job = submission.launch.call_args.args[0]
    assert job.resume_volume is None
    assert job.command[:2] == ["olmo-eval", "run"]
    assert "--resume" not in job.command
    assert "olmo_eval.launch.beaker.resume" not in job.command
    assert job.command[job.command.index("-O") + 1] == "/results"


@pytest.mark.parametrize(
    ("yaml_resume", "cli_resume", "expected"),
    [(False, None, False), (False, True, True), (True, False, False), (True, None, True)],
)
def test_yaml_resume_and_cli_precedence(tmp_path, yaml_resume, cli_resume, expected):
    from olmo_eval.cli.beaker.config_loader import LaunchConfigLoader

    path = tmp_path / "eval.yaml"
    path.write_text(
        "name: test\nmodels: [mock]\ntasks: [gsm8k]\ncluster: h100\n"
        "workspace: ai2/unclaimed\nbudget: ai2/test\nresume_volume: ai2/custom\n"
        f"resume: {str(yaml_resume).lower()}\n"
    )
    config = LaunchConfigLoader(str(path), {"resume": cli_resume, "upload": False}).load()
    assert config.resume is expected
    assert config.resume_volume == ("ai2/custom" if expected else None)


def test_other_submission_errors_are_not_mislabeled(monkeypatch):
    def gantry(**kwargs):
        Path(kwargs["save_spec"]).write_text('{"version":"v2","tasks":[{}]}')

    monkeypatch.setattr(
        "olmo_eval.launch.beaker.volume.subprocess.run",
        Mock(side_effect=subprocess.CalledProcessError(1, "beaker", stderr="budget exhausted")),
    )
    with pytest.raises(RuntimeError, match="budget exhausted") as error:
        launch_with_volume(
            gantry, {"name": "test", "workspace": "ai2/unclaimed"}, "ai2/cache", dry_run=False
        )
    assert "--no-resume" not in str(error.value)
