#!/usr/bin/env python3
"""Append scalar Beaker metrics to the canonical long-form results CSV.

The CSV intentionally uses the same columns as the Results Google Sheet.  A
metric gets its own row.  For DeepScholar-Bench, only metrics whose base name
ends in ``_fixed`` are retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "data" / "results.csv"
DEFAULT_SEED_CSV = HERE / "data" / "eval_matrix.csv"
CSV_FIELDS = (
    "Model Name",
    "Eval Name",
    "Metric",
    "Score",
    "Beaker Run ID",
    "Notes",
    "Valid for analysis",
)

MODEL_NAMES = {
    "allenai/Olmo-3-7B-Instruct": "OLMo-3 7B Instruct",
    "allenai/Olmo-3-7B-Think": "Olmo-3 7B Think",
    "Qwen/Qwen3.5-9B": "Qwen3.5 9B Instruct",
    "zai-org/GLM-4.1V-9B-Thinking": "GLM-4.1V-9B-Thinking",
    "google/gemma-4-12B-it": "Gemma4 12B-Unified",
    "google/gemma-4-26B-A4B-it": "Gemma4 26B-A4B",
    "openai/gpt-oss-20b": "GPT-OSS-20b",
    "Qwen/Qwen3.5-35B-A3B": "Qwen3.5-35B-A3B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "Nemotron 3 Nano 30B-A3B",
}

TASK_NAMES = {
    "expertqa": "ExpertQA",
    "litsearch": "LitSearch-open",
    "litsearch_rerank": "LitSearch-rerank",
    "sage_open_ended": "SAGE-open",
    "sage_short_form": "SAGE-short",
    "deepscholar_bench": "DeepScholar-Bench",
    "ifeval_ood": "IFEval",
    "mmlu": "MMLU",
    "math500": "MATH-500",
}


def beaker_json(*args: str) -> Any:
    command = ["beaker", *args, "--format", "json"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def scalar_metrics(value: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[str, float]]:
    """Flatten nested metric dictionaries into colon-delimited scalar names."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from scalar_metrics(child, (*prefix, str(key)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
        if math.isfinite(score):
            yield ":".join(prefix), score


def base_task_name(task_name: str) -> str:
    if task_name.startswith("mmlu_"):
        return "mmlu"
    return task_name


def display_eval(task_name: str, config: dict[str, Any]) -> str:
    base = TASK_NAMES.get(base_task_name(task_name), task_name)
    limit = config.get("limit")
    return f"{base}-dev{limit}" if isinstance(limit, int) and limit > 0 else base


def scope_note(task_name: str, config: dict[str, Any]) -> str:
    limit = config.get("limit")
    if not isinstance(limit, int) or limit <= 0:
        return ""
    if base_task_name(task_name) == "deepscholar_bench":
        start = int(config.get("start_idx", 0))
        return f"Fixed dev{limit} slice; indices {start}-{start + limit - 1}; _fixed metrics only"
    seed = config.get("seed", 42)
    return f"Fixed dev{limit} sample; seed {seed}"


def keep_metric(task_name: str, metric: str) -> bool:
    if base_task_name(task_name) != "deepscholar_bench":
        return True
    return metric.split(":", 1)[0].endswith("_fixed")


def model_display_name(model_ref: str, overrides: dict[str, str]) -> str:
    return overrides.get(model_ref) or MODEL_NAMES.get(model_ref) or model_ref.rsplit("/", 1)[-1]


def result_rows(
    payload: dict[str, Any], experiment_id: str, model_overrides: dict[str, str]
) -> tuple[list[dict[str, str]], int]:
    metrics = payload.get("metrics") or {}
    errors = metrics.get("errors") or []
    if errors:
        raise ValueError(f"{experiment_id} exposes evaluation errors: {errors}")

    tasks = metrics.get("tasks") or []
    if not tasks:
        raise ValueError(f"{experiment_id} has no task metrics")

    run_config = metrics.get("config") or {}
    provider_config = run_config.get("provider") or {}
    model_ref = str(
        provider_config.get("model") or run_config.get("model") or tasks[0].get("model") or ""
    )
    model = model_display_name(model_ref, model_overrides)
    eval_counts = Counter(
        display_eval(str(task.get("task", "")), task.get("config") or {}) for task in tasks
    )

    rows: list[dict[str, str]] = []
    skipped = 0
    for task in tasks:
        task_name = str(task.get("task", ""))
        config = task.get("config") or {}
        eval_name = display_eval(task_name, config)
        note = scope_note(task_name, config)
        subject_prefix = f"{task_name}/" if eval_counts[eval_name] > 1 else ""

        limit = config.get("limit")
        num_instances = task.get("num_instances")
        if (
            base_task_name(task_name) == "deepscholar_bench"
            and isinstance(limit, int)
            and num_instances != limit
        ):
            raise ValueError(
                f"{experiment_id} expected {limit} DeepScholar instances, got {num_instances}"
            )

        for metric, score in scalar_metrics(task.get("metrics") or {}):
            if not keep_metric(task_name, metric):
                skipped += 1
                continue
            rows.append(
                {
                    "Model Name": model,
                    "Eval Name": eval_name,
                    "Metric": f"{subject_prefix}{metric}",
                    "Score": repr(score),
                    "Beaker Run ID": experiment_id,
                    "Notes": note,
                    "Valid for analysis": "True",
                }
            )

    # Summary metrics are aggregate values that may not exist in task metrics
    # (notably the overall MMLU average). Add them unless already present.
    existing = {(row["Eval Name"], row["Metric"]) for row in rows}
    for summary_task, summary in (metrics.get("summary") or {}).items():
        if not isinstance(summary, dict):
            continue
        metric = summary.get("metric")
        score = summary.get("score")
        if not isinstance(metric, str) or not isinstance(score, (int, float)):
            continue
        summary_task = str(summary_task)
        exact_matches = [task for task in tasks if str(task.get("task", "")) == summary_task]
        matching = [
            task
            for task in tasks
            if base_task_name(str(task.get("task", ""))) == base_task_name(summary_task)
        ]
        if summary_task == "mmlu:chat":
            # The chat protocol is a top-level MMLU suite, not a category.
            # Store its overall score under the canonical metric so a vetted
            # full rerun can supersede the invalid raw-loglikelihood result.
            mmlu_tasks = [
                task for task in tasks if base_task_name(str(task.get("task", ""))) == "mmlu"
            ]
            config = (mmlu_tasks[0].get("config") or {}) if mmlu_tasks else {}
            eval_name = display_eval("mmlu", config)
            output_metric = metric
        elif summary_task.startswith("mmlu:"):
            # Category aggregates such as ``mmlu:humanities`` belong to the
            # MMLU eval; they are metrics, not additional matrix evaluations.
            mmlu_tasks = [
                task for task in tasks if base_task_name(str(task.get("task", ""))) == "mmlu"
            ]
            config = (mmlu_tasks[0].get("config") or {}) if mmlu_tasks else {}
            eval_name = display_eval("mmlu", config)
            output_metric = f"{summary_task}/{metric}"
        elif exact_matches:
            config = exact_matches[0].get("config") or {}
            eval_name = display_eval(summary_task, config)
            output_metric = f"{summary_task}/{metric}" if eval_counts[eval_name] > 1 else metric
        else:
            config = (matching[0].get("config") or {}) if matching else {}
            eval_name = display_eval(summary_task, config)
            output_metric = metric
        if not keep_metric(summary_task, metric):
            skipped += 1
            continue
        if (eval_name, output_metric) in existing:
            continue
        rows.append(
            {
                "Model Name": model,
                "Eval Name": eval_name,
                "Metric": output_metric,
                "Score": repr(float(score)),
                "Beaker Run ID": experiment_id,
                "Notes": scope_note(summary_task, config),
                "Valid for analysis": "True",
            }
        )
        existing.add((eval_name, output_metric))

    rows.sort(key=lambda row: (row["Eval Name"], row["Metric"]))
    return rows, skipped


def successful_job(experiment: dict[str, Any]) -> dict[str, Any]:
    successful = [
        job
        for job in experiment.get("jobs", [])
        if (job.get("status") or {}).get("exitCode") == 0
        and (job.get("status") or {}).get("finalized")
        and (job.get("result") or {}).get("beaker")
    ]
    if not successful:
        raise ValueError(f"{experiment.get('id')} has no finalized successful job")
    return max(successful, key=lambda job: (job.get("status") or {}).get("finalized", ""))


def load_experiments(groups: list[str], experiment_ids: list[str]) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for group in groups:
        result = beaker_json("group", "experiments", group)
        experiments.extend(result if isinstance(result, list) else [result])
    if experiment_ids:
        result = beaker_json("experiment", "get", *experiment_ids)
        experiments.extend(result if isinstance(result, list) else [result])

    unique: dict[str, dict[str, Any]] = {}
    for experiment in experiments:
        unique[str(experiment["id"])] = experiment
    return sorted(unique.values(), key=lambda experiment: str(experiment.get("created", "")))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"{path} columns are {reader.fieldnames}, expected {CSV_FIELDS}")
        return [
            {
                field: (row.get(field) or ("True" if field == "Valid for analysis" else ""))
                for field in CSV_FIELDS
            }
            for row in reader
        ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def upsert_rows(existing: list[dict[str, str]], incoming: list[dict[str, str]]) -> tuple[int, int]:
    positions = {
        (
            row["Model Name"],
            row["Eval Name"],
            row["Metric"],
            row["Beaker Run ID"],
        ): index
        for index, row in enumerate(existing)
    }
    added = updated = 0
    for row in incoming:
        key = (row["Model Name"], row["Eval Name"], row["Metric"], row["Beaker Run ID"])
        # Repair rows imported by the older model lookup, which missed model
        # refs stored at metrics.config.provider.model.
        blank_model_key = ("", row["Eval Name"], row["Metric"], row["Beaker Run ID"])
        if key not in positions and row["Model Name"] and blank_model_key in positions:
            index = positions.pop(blank_model_key)
            existing[index] = row
            positions[key] = index
            updated += 1
            continue
        if key in positions:
            index = positions[key]
            incoming_validity = existing[index]["Valid for analysis"]
            if incoming_validity:
                row["Valid for analysis"] = incoming_validity
            if existing[index] != row:
                existing[index] = row
                updated += 1
        else:
            positions[key] = len(existing)
            existing.append(row)
            added += 1
    return added, updated


def parse_model_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model-name must be REF=DISPLAY, got {value!r}")
        ref, display = value.split("=", 1)
        overrides[ref] = display
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", default=[], help="Beaker group to import")
    parser.add_argument(
        "--experiment", action="append", default=[], help="Beaker experiment ID to import"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--seed-from",
        type=Path,
        default=DEFAULT_SEED_CSV,
        help="Initial seven-column CSV copied when --csv does not yet exist",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        metavar="REF=DISPLAY",
        help="Override the display name for a model ref",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Pull and print rows without writing"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.group and not args.experiment:
        print("At least one --group or --experiment is required", file=sys.stderr)
        return 2

    try:
        model_overrides = parse_model_overrides(args.model_name)
        experiments = load_experiments(args.group, args.experiment)
        incoming: list[dict[str, str]] = []
        skipped = 0
        for experiment in experiments:
            job = successful_job(experiment)
            payload = beaker_json("job", "results", str(job["id"]))
            rows, run_skipped = result_rows(payload, str(experiment["id"]), model_overrides)
            incoming.extend(rows)
            skipped += run_skipped

        if args.dry_run:
            writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(incoming)
            print(
                f"Would import {len(incoming)} rows; skipped {skipped} filtered metrics",
                file=sys.stderr,
            )
            return 0

        if args.csv.exists():
            existing = read_csv(args.csv)
        elif args.seed_from.exists():
            existing = read_csv(args.seed_from)
        else:
            existing = []
        added, updated = upsert_rows(existing, incoming)
        write_csv(args.csv, existing)
        print(
            f"Imported {len(incoming)} metric rows into {args.csv} "
            f"({added} added, {updated} updated, {skipped} filtered)."
        )
        return 0
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
