"""Tests for SWE-bench external result parsing."""

import json
import sys
import time

import pytest

from olmo_eval.evals.external.benchmarks.swebench.eval import SWEBenchExternalEval


def test_parse_results_includes_instances_from_exit_statuses(tmp_path):
    eval_obj = SWEBenchExternalEval()
    run_id = "swe_bench_test1234"

    summary = {
        "resolved_ids": ["django__django-10914"],
        "unresolved_ids": ["astropy__astropy-14182"],
        "resolved": 1,
        "unresolved": 1,
    }
    (tmp_path / f"model.{run_id}.json").write_text(json.dumps(summary))

    report = {
        "django__django-10914": {
            "patch_is_None": False,
            "patch_exists": True,
            "patch_successfully_applied": True,
            "resolved": True,
            "tests_status": {},
        },
        "astropy__astropy-14182": {
            "patch_is_None": False,
            "patch_exists": True,
            "patch_successfully_applied": True,
            "resolved": False,
            "tests_status": {},
        },
    }
    (tmp_path / "report.json").write_text(json.dumps(report))

    exit_statuses = """
instances_by_exit_status:
    ContextWindowExceededError:
    - astropy__astropy-12907
    LimitsExceeded:
    - astropy__astropy-14995
    - astropy__astropy-7746
    - astropy__astropy-14365
    - django__django-10924
    Submitted:
    - django__django-10914
    - astropy__astropy-14182
    - astropy__astropy-693
"""
    (tmp_path / "exit_statuses.yaml").write_text(exit_statuses.strip())

    result = eval_obj._parse_results(
        work_dir=tmp_path,
        run_id=run_id,
        score_ok=True,
        raw_output="",
        start_time=time.time(),
    )

    assert result.success is True
    assert result.metrics == {
        "resolve_rate": 1 / 8,
        "resolved": 1.0,
        "total": 8.0,
    }
    assert result.predictions is not None
    assert [pred["native_id"] for pred in result.predictions] == [
        "django__django-10914",
        "astropy__astropy-14182",
        "astropy__astropy-12907",
        "astropy__astropy-14995",
        "astropy__astropy-7746",
        "astropy__astropy-14365",
        "django__django-10924",
        "astropy__astropy-693",
    ]

    resolved_by_instance = {
        pred["native_id"]: pred["instance_metrics"]["resolved"]["external"]
        for pred in result.predictions
    }
    assert resolved_by_instance["django__django-10914"] == 1.0
    assert resolved_by_instance["astropy__astropy-14182"] == 0.0
    assert resolved_by_instance["astropy__astropy-693"] == 0.0
    assert resolved_by_instance["astropy__astropy-12907"] == 0.0

    assert result.metadata["instance_exit_statuses"] == {
        "astropy__astropy-12907": "ContextWindowExceededError",
        "astropy__astropy-14995": "LimitsExceeded",
        "astropy__astropy-7746": "LimitsExceeded",
        "astropy__astropy-14365": "LimitsExceeded",
        "django__django-10924": "LimitsExceeded",
        "django__django-10914": "Submitted",
        "astropy__astropy-14182": "Submitted",
        "astropy__astropy-693": "Submitted",
    }


def test_parse_results_handles_schema_v2_submitted_ids(tmp_path):
    eval_obj = SWEBenchExternalEval()
    run_id = "swe_bench_test5678"

    summary = {
        "total_instances": 300,
        "submitted_instances": 8,
        "completed_instances": 2,
        "resolved_instances": 1,
        "unresolved_instances": 1,
        "empty_patch_instances": 6,
        "error_instances": 0,
        "completed_ids": [
            "astropy__astropy-14182",
            "django__django-10914",
        ],
        "empty_patch_ids": [
            "astropy__astropy-12907",
            "astropy__astropy-14365",
            "astropy__astropy-14995",
            "astropy__astropy-6938",
            "astropy__astropy-7746",
            "django__django-10924",
        ],
        "submitted_ids": [
            "astropy__astropy-12907",
            "astropy__astropy-14182",
            "astropy__astropy-14365",
            "astropy__astropy-14995",
            "astropy__astropy-6938",
            "astropy__astropy-7746",
            "django__django-10914",
            "django__django-10924",
        ],
        "resolved_ids": ["django__django-10914"],
        "unresolved_ids": ["astropy__astropy-14182"],
        "error_ids": [],
        "schema_version": 2,
    }
    (tmp_path / f"model.{run_id}.json").write_text(json.dumps(summary))

    result = eval_obj._parse_results(
        work_dir=tmp_path,
        run_id=run_id,
        score_ok=True,
        raw_output="",
        start_time=time.time(),
    )

    assert result.success is True
    assert result.metrics == {
        "resolve_rate": 1 / 8,
        "resolved": 1.0,
        "total": 8.0,
    }
    assert result.predictions is not None
    assert [pred["native_id"] for pred in result.predictions] == [
        "django__django-10914",
        "astropy__astropy-14182",
        "astropy__astropy-12907",
        "astropy__astropy-14365",
        "astropy__astropy-14995",
        "astropy__astropy-6938",
        "astropy__astropy-7746",
        "django__django-10924",
    ]


@pytest.mark.anyio
async def test_run_scoring_uses_max_workers_without_modal(tmp_path, monkeypatch):
    eval_obj = SWEBenchExternalEval()
    captured: dict[str, object] = {}

    async def fake_run_subprocess(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return True, ""

    monkeypatch.setattr(eval_obj, "_run_subprocess", fake_run_subprocess)
    from olmo_eval.evals.external.benchmarks.swebench.eval import SWEBenchArgs

    await eval_obj._run_scoring(
        swe_args=SWEBenchArgs(max_workers_eval=7, use_modal=False),
        preds_path=tmp_path / "preds.json",
        run_id="run123",
        work_dir=tmp_path,
        container_runtime="podman",
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:3] == [sys.executable, "-m", "swebench.harness.run_evaluation"]
    assert "--max_workers" in cmd
    assert cmd[cmd.index("--max_workers") + 1] == "7"
    assert "--parallelism" not in cmd
    assert "--modal" not in cmd


@pytest.mark.anyio
async def test_run_scoring_uses_parallelism_with_modal(tmp_path, monkeypatch):
    eval_obj = SWEBenchExternalEval()
    captured: dict[str, object] = {}

    async def fake_run_subprocess(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return True, ""

    monkeypatch.setattr(eval_obj, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(
        "olmo_eval.evals.external.benchmarks.swebench.eval._ensure_modal_config_exists",
        lambda: None,
    )

    from olmo_eval.evals.external.benchmarks.swebench.eval import SWEBenchArgs

    await eval_obj._run_scoring(
        swe_args=SWEBenchArgs(max_workers_eval=7, use_modal=True),
        preds_path=tmp_path / "preds.json",
        run_id="run123",
        work_dir=tmp_path,
        container_runtime="podman",
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--parallelism" in cmd
    assert cmd[cmd.index("--parallelism") + 1] == "7"
    assert "--modal" in cmd
    assert cmd[cmd.index("--modal") + 1] == "true"
    assert "--max_workers" not in cmd
