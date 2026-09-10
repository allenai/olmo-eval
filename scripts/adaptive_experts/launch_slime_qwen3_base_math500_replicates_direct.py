#!/usr/bin/env python3
"""Submit MATH-500 checkpoint replicates from a validated Beaker eval spec."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
RUN_ROOT = PROJECT / "slime-runs/qwen3-30b-a3b-base-dapo"
HF_ROOT = RUN_ROOT / "hf-checkpoints"
ORIGIN_HF_DIR = RUN_ROOT / "hf/Qwen3-30B-A3B-Base"
LEDGER = ROOT / "notes/beaker_jobs.jsonl"
RECORD_SCRIPT = ROOT / "scripts/adaptive_experts/record_beaker_job.py"
BASE_EXPERIMENT = "01KYJER2PV95MEETA2HWX1F3S3"
WORKSPACE = "ai2/holmes-testing"
CLUSTER = "ai2/jupiter"
PRIORITY = "urgent"
GROUP = "adaptive-compute-slime-qwen3-base-math500-replicates-20260728"
PHASE = "slime-checkpoint-math500-replicate"
SOURCE_COMMIT = "e8b88f196767630acaf8a383d12a33d314788d27"
SOURCE_HASH = "52060bdc2f8988ab7b74635ddadc00a71ebab5f88fa227c37d4797f265975f35"
SNAPSHOT = PROJECT / f".run_snapshots/olmo-eval/{SOURCE_HASH}"


@dataclass(frozen=True)
class Family:
    key: str
    run_name: str
    active_k: int
    reference_scaled: bool
    steps: tuple[int, ...]


FAMILIES = (
    Family(
        "k8-normalized",
        "slime-qwen3-base-dapo-k8-pilot-100step-8gpu-v5-20260725",
        8,
        False,
        (0, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450),
    ),
    Family(
        "k8-trained-k6-eval",
        "slime-qwen3-base-dapo-k8-pilot-100step-8gpu-v5-20260725",
        6,
        False,
        (20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450),
    ),
    Family(
        "k6-normalized",
        "slime-qwen3-base-dapo-k6-pilot-100step-8gpu-v1-20260725",
        6,
        False,
        (0, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500),
    ),
    Family(
        "k6-trained-k8-eval",
        "slime-qwen3-base-dapo-k6-pilot-100step-8gpu-v1-20260725",
        8,
        False,
        (20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500),
    ),
    Family(
        "k4-reference-k8",
        "slime-qwen3-base-dapo-k4-reference-k8-pilot-100step-8gpu-v1-20260725",
        4,
        True,
        (0, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500),
    ),
    Family(
        "k6-reference-k8",
        "slime-qwen3-base-dapo-k6-reference-k8-500step-8gpu-v1-20260726",
        6,
        True,
        (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500),
    ),
)


def run_json(command: list[str]) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {shlex.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Expected JSON from {shlex.join(command)}; got:\n{completed.stdout}\n{completed.stderr}"
        ) from error


def recorded_tags() -> set[str]:
    if not LEDGER.exists():
        return set()
    tags = set()
    with LEDGER.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                if record.get("phase") == PHASE:
                    tags.add(record.get("run_tag", ""))
    return tags


def model_path(family: Family, step: int) -> Path:
    if step == 0:
        return ORIGIN_HF_DIR
    return HF_ROOT / f"{family.run_name}-step{step}-hf"


def build_spec(base: dict, family: Family, step: int, replicate: int) -> tuple[dict, str, str, Path]:
    spec = copy.deepcopy(base)
    model = model_path(family, step)
    run_tag = (
        f"qwen3-base-dapo-{family.key}-step{step}-math500-replicate-"
        f"rep{replicate}-v1-20260728"
    )
    name = f"adaptive-{run_tag}"
    spec["description"] = (
        f"MATH-500 replicate {replicate}/3 for {family.key}, checkpoint step {step}; "
        "replicate 1 is the MATH task in the original checkpoint eval"
    )
    task = spec["tasks"][0]
    # `beaker experiment spec --format json` emits this duration as nanoseconds,
    # while `experiment create` expects a duration string in an input spec.
    task["timeout"] = "24h"
    task["arguments"] = [
        "olmo-eval",
        "run",
        "-O",
        "/results",
        "-m",
        str(model),
        "-t",
        "math500:chat",
        "-o",
        "sampling_params.max_tokens=20480",
        "--experiment-group",
        GROUP,
        "--experiment-name",
        name,
        "--harness",
        "codex_python",
        "-o",
        "provider.num_instances=4",
        "-o",
        f'provider.kwargs.hf_overrides={{"num_experts_per_tok":{8 if family.reference_scaled else family.active_k}}}',
        "-o",
        "metrics.collect_gpu=true",
        "-o",
        "provider.max_model_len=32768",
        "-o",
        "provider.kwargs.gpu_memory_utilization=0.9",
        "-o",
        "provider.kwargs.max_num_seqs=16",
        "-o",
        "provider.kwargs.startup_timeout=900",
    ]
    env = {item["name"]: item for item in task["envVars"]}
    for key in (
        "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_MODE",
        "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_SEED",
        "OLMO_EVAL_VLLM_QWEN_EXPERT_ROUTER_K",
        "VLLM_USE_FLASHINFER_MOE_FP16",
        "OLMO_EVAL_VLLM_QWEN_EXPERT_KEEP_K",
        "OLMO_EVAL_VLLM_QWEN_EXPERT_REFERENCE_K",
        "OLMO_EVAL_VLLM_QWEN_EXPERT_RENORMALIZE",
    ):
        env.pop(key, None)
    if family.reference_scaled:
        env.update(
            {
                "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_MODE": {
                    "name": "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_MODE",
                    "value": "truncate",
                },
                "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_SEED": {
                    "name": "OLMO_EVAL_VLLM_QWEN_EXPERT_WEIGHT_SEED",
                    "value": "0",
                },
                "OLMO_EVAL_VLLM_QWEN_EXPERT_ROUTER_K": {
                    "name": "OLMO_EVAL_VLLM_QWEN_EXPERT_ROUTER_K",
                    "value": "8",
                },
                "VLLM_USE_FLASHINFER_MOE_FP16": {
                    "name": "VLLM_USE_FLASHINFER_MOE_FP16",
                    "value": "0",
                },
                "OLMO_EVAL_VLLM_QWEN_EXPERT_KEEP_K": {
                    "name": "OLMO_EVAL_VLLM_QWEN_EXPERT_KEEP_K",
                    "value": str(family.active_k),
                },
                "OLMO_EVAL_VLLM_QWEN_EXPERT_REFERENCE_K": {
                    "name": "OLMO_EVAL_VLLM_QWEN_EXPERT_REFERENCE_K",
                    "value": "8",
                },
                "OLMO_EVAL_VLLM_QWEN_EXPERT_RENORMALIZE": {
                    "name": "OLMO_EVAL_VLLM_QWEN_EXPERT_RENORMALIZE",
                    "value": "false",
                },
            }
        )
    task["envVars"] = list(env.values())
    task["context"]["priority"] = PRIORITY
    task["constraints"]["cluster"] = [CLUSTER]
    return spec, run_tag, name, model


def record_job(
    experiment_id: str,
    run_tag: str,
    name: str,
    model: Path,
    family: Family,
    step: int,
    replicate: int,
) -> None:
    logical_command = shlex.join(
        [
            "olmo-eval",
            "beaker",
            "launch",
            "-m",
            str(model),
            "-t",
            "math500:chat",
            "-o",
            "sampling_params.max_tokens=20480",
            "--name",
            name,
            "--cluster",
            CLUSTER,
            "--workspace",
            WORKSPACE,
            "--priority",
            PRIORITY,
            "--group",
            GROUP,
            "#",
            f"direct-spec-clone={BASE_EXPERIMENT}",
            f"family={family.key}",
            f"step={step}",
            f"replicate={replicate}",
        ]
    )
    subprocess.run(
        [
            "python3",
            str(RECORD_SCRIPT),
            "--ledger",
            str(LEDGER),
            "--phase",
            PHASE,
            "--run-tag",
            run_tag,
            "--expert-count",
            str(family.active_k),
            "--model",
            str(model),
            "--task",
            "math500:chat",
            "--cluster",
            CLUSTER,
            "--workspace",
            WORKSPACE,
            "--priority",
            PRIORITY,
            "--group",
            GROUP,
            "--source-commit",
            SOURCE_COMMIT,
            "--source-hash",
            SOURCE_HASH,
            "--source-snapshot",
            str(SNAPSHOT),
            "--result-storage",
            "beaker_dataset",
            "--experiment-id",
            experiment_id,
            "--command",
            logical_command,
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--replicate", type=int, choices=(2, 3), action="append")
    parser.add_argument("--family", choices=tuple(family.key for family in FAMILIES), action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = run_json(["beaker", "experiment", "spec", BASE_EXPERIMENT, "--format", "json"])
    existing = recorded_tags()
    replicates = args.replicate or [2, 3]
    selected_families = [family for family in FAMILIES if not args.family or family.key in args.family]
    submitted = 0
    skipped = 0
    for replicate in replicates:
        for family in selected_families:
            for step in family.steps:
                spec, run_tag, name, model = build_spec(base, family, step, replicate)
                if run_tag in existing:
                    print(f"SKIP {run_tag}", flush=True)
                    skipped += 1
                    continue
                if args.limit is not None and submitted >= args.limit:
                    print(f"Reached limit: submitted={submitted}, skipped={skipped}", flush=True)
                    return
                if args.dry_run:
                    print(json.dumps({"run_tag": run_tag, "spec": spec}, sort_keys=True))
                    submitted += 1
                    continue
                with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as stream:
                    json.dump(spec, stream)
                    stream.flush()
                    output = run_json(
                        [
                            "beaker",
                            "experiment",
                            "create",
                            stream.name,
                            "--workspace",
                            WORKSPACE,
                            "--name",
                            name,
                            "--format",
                            "json",
                        ]
                    )
                if isinstance(output, list):
                    experiment_id = output[0].get("id") if output else None
                else:
                    experiment_id = output.get("id") or output.get("experiment", {}).get("id")
                if not experiment_id:
                    match = re.search(r"01[A-Z0-9]{24}", json.dumps(output))
                    if not match:
                        raise RuntimeError(f"Could not find experiment ID in: {output}")
                    experiment_id = match.group(0)
                record_job(experiment_id, run_tag, name, model, family, step, replicate)
                existing.add(run_tag)
                submitted += 1
                print(f"SUBMITTED {submitted}: https://beaker.org/ex/{experiment_id} {run_tag}", flush=True)
    print(f"Complete: submitted={submitted}, skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
