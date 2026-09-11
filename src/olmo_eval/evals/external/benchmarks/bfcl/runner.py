"""Run and validate a pinned BFCL v4 checkout inside the evaluation sandbox."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

BFCL_REGISTRY_NAME = "olmo-eval-provider-FC"
BFCL_DISPLAY_NAME = "OLMo Eval Provider (FC)"
NON_WEB_COLLECTIONS = ("non_live", "live", "multi_turn", "memory")
EXPECTED_RESULT_FILES = 23
EXPECTED_SCORE_FILES = 20
EXPECTED_RESULT_ROWS = 5_017
EXPECTED_COVERED_WEIGHT = 0.8


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open() as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            entries.append(value)
    return entries


def _validate_result_inventory(
    result_root: Path,
    expected: dict[Path, set[str]],
) -> int:
    actual_files = {
        path.relative_to(result_root) for path in result_root.rglob("BFCL_v4_*_result.json")
    }
    if actual_files != set(expected):
        missing = sorted(str(path) for path in set(expected) - actual_files)
        unexpected = sorted(str(path) for path in actual_files - set(expected))
        raise ValueError(
            f"BFCL result-file inventory mismatch: missing={missing}, unexpected={unexpected}"
        )

    row_count = 0
    for relative_path, expected_ids in expected.items():
        entries = _read_jsonl(result_root / relative_path)
        ids = [entry.get("id") for entry in entries]
        if any(not isinstance(entry_id, str) for entry_id in ids):
            raise ValueError(f"Malformed BFCL result ID in {relative_path}")
        duplicate_ids = sorted(entry_id for entry_id, count in Counter(ids).items() if count > 1)
        if duplicate_ids:
            raise ValueError(f"Duplicate BFCL result IDs in {relative_path}: {duplicate_ids}")
        if set(ids) != expected_ids:
            missing = sorted(expected_ids - set(ids))
            unexpected = sorted(set(ids) - expected_ids)
            raise ValueError(
                f"BFCL result-ID mismatch in {relative_path}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for entry in entries:
            if "result" not in entry:
                raise ValueError(f"BFCL result {entry.get('id')!r} has no result field")
            if "traceback" in entry:
                raise ValueError(f"BFCL inference traceback recorded for {entry.get('id')!r}")
            if isinstance(entry["result"], str) and entry["result"].startswith(
                "Error during inference:"
            ):
                raise ValueError(f"BFCL inference error recorded for {entry.get('id')!r}")
        row_count += len(entries)
    return row_count


def _validate_score_inventory(
    score_root: Path,
    expected: dict[Path, set[str]],
) -> dict[str, dict[str, float | int]]:
    actual_files = {
        path.relative_to(score_root) for path in score_root.rglob("BFCL_v4_*_score.json")
    }
    if actual_files != set(expected):
        missing = sorted(str(path) for path in set(expected) - actual_files)
        unexpected = sorted(str(path) for path in actual_files - set(expected))
        raise ValueError(
            f"BFCL score-file inventory mismatch: missing={missing}, unexpected={unexpected}"
        )

    scores: dict[str, dict[str, float | int]] = {}
    for relative_path, expected_ids in expected.items():
        entries = _read_jsonl(score_root / relative_path)
        if not entries:
            raise ValueError(f"Empty BFCL score file: {relative_path}")
        header, failures = entries[0], entries[1:]
        category = relative_path.name.removeprefix("BFCL_v4_").removesuffix("_score.json")
        total_count = header.get("total_count")
        correct_count = header.get("correct_count")
        accuracy = header.get("accuracy")
        if total_count != len(expected_ids):
            raise ValueError(
                f"BFCL score count mismatch for {category}: {total_count} != {len(expected_ids)}"
            )
        if (
            not isinstance(total_count, int)
            or not isinstance(correct_count, int)
            or not isinstance(accuracy, (int, float))
        ):
            raise ValueError(f"Malformed BFCL score header for {category}: {header}")
        failure_ids = [entry.get("id") for entry in failures]
        if any(not isinstance(entry_id, str) for entry_id in failure_ids):
            raise ValueError(f"Malformed BFCL failure ID in {relative_path}")
        if any(entry.get("valid") is not False for entry in failures):
            raise ValueError(f"Malformed BFCL failure record in {relative_path}")
        if len(failure_ids) != len(set(failure_ids)):
            raise ValueError(f"Duplicate BFCL failure IDs in {relative_path}")
        if not set(failure_ids).issubset(expected_ids):
            raise ValueError(f"Unexpected BFCL failure IDs in {relative_path}")
        if correct_count != total_count - len(failures):
            raise ValueError(f"BFCL correct-count mismatch for {category}")
        expected_accuracy = correct_count / total_count
        if not math.isclose(float(accuracy), expected_accuracy, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"BFCL accuracy mismatch for {category}")
        scores[category] = {
            "accuracy": float(accuracy),
            "correct_count": correct_count,
            "total_count": total_count,
        }
    return scores


def _unweighted(scores: dict[str, dict[str, float | int]], categories: tuple[str, ...]) -> float:
    return sum(float(scores[category]["accuracy"]) for category in categories) / len(categories)


def _weighted(scores: dict[str, dict[str, float | int]], categories: tuple[str, ...]) -> float:
    total = sum(int(scores[category]["total_count"]) for category in categories)
    return (
        sum(
            float(scores[category]["accuracy"]) * int(scores[category]["total_count"])
            for category in categories
        )
        / total
    )


def _compute_metrics(scores: dict[str, dict[str, float | int]]) -> dict[str, float]:
    simple = _unweighted(scores, ("simple_python", "simple_java", "simple_javascript"))
    non_live = (
        simple
        + float(scores["multiple"]["accuracy"])
        + float(scores["parallel"]["accuracy"])
        + float(scores["parallel_multiple"]["accuracy"])
    ) / 4
    live = _weighted(
        scores,
        ("live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"),
    )
    irrelevance = _unweighted(scores, ("irrelevance", "live_irrelevance"))
    multi_turn = _unweighted(
        scores,
        (
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
        ),
    )
    memory = _unweighted(scores, ("memory_kv", "memory_vector", "memory_rec_sum"))
    contribution = 0.1 * non_live + 0.1 * live + 0.1 * irrelevance + 0.3 * multi_turn + 0.2 * memory
    return {
        "non_web_weighted_contribution": contribution,
        "non_live_accuracy": non_live,
        "live_accuracy": live,
        "irrelevance_accuracy": irrelevance,
        "multi_turn_accuracy": multi_turn,
        "memory_accuracy": memory,
        "live_relevance_accuracy": float(scores["live_relevance"]["accuracy"]),
    }


def _register_model(provider_model: str) -> None:
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )

    MODEL_CONFIG_MAPPING[BFCL_REGISTRY_NAME] = ModelConfig(
        model_name=provider_model,
        display_name=BFCL_DISPLAY_NAME,
        url="https://github.com/allenai/olmo-eval",
        org="Allen Institute for AI",
        license="N/A",
        model_handler=OpenAICompletionsHandler,
        is_fc_model=True,
        underscore_to_dot=True,
    )


def _build_expected_inventories() -> tuple[dict[Path, set[str]], dict[Path, set[str]]]:
    from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
    from bfcl_eval.utils import (
        extract_test_category_from_id,
        get_directory_structure_by_category,
        get_directory_structure_by_id,
        get_file_name_by_category,
        load_dataset_entry,
    )

    result_inventory: dict[Path, set[str]] = defaultdict(set)
    score_inventory: dict[Path, set[str]] = {}
    for collection in NON_WEB_COLLECTIONS:
        for category in TEST_COLLECTION_MAPPING[collection]:
            entries = load_dataset_entry(category)
            for entry in entries:
                result_category = extract_test_category_from_id(entry["id"])
                path = (
                    Path(BFCL_REGISTRY_NAME)
                    / get_directory_structure_by_id(entry["id"])
                    / get_file_name_by_category(result_category, is_result_file=True)
                )
                result_inventory[path].add(entry["id"])

            scoring_entries = load_dataset_entry(category, include_prereq=False)
            score_path = (
                Path(BFCL_REGISTRY_NAME)
                / get_directory_structure_by_category(category)
                / get_file_name_by_category(category, is_score_file=True)
            )
            score_inventory[score_path] = {entry["id"] for entry in scoring_entries}

    if len(result_inventory) != EXPECTED_RESULT_FILES:
        raise ValueError(f"Pinned BFCL checkout produced {len(result_inventory)} result files")
    if len(score_inventory) != EXPECTED_SCORE_FILES:
        raise ValueError(f"Pinned BFCL checkout produced {len(score_inventory)} score files")
    expected_rows = sum(len(ids) for ids in result_inventory.values())
    if expected_rows != EXPECTED_RESULT_ROWS:
        raise ValueError(f"Pinned BFCL checkout produced {expected_rows} expected rows")
    return dict(result_inventory), score_inventory


def _prepare_encoder(cache_dir: Path, repository: str, revision: str) -> None:
    from huggingface_hub import snapshot_download

    hub_cache = cache_dir / "hub"
    snapshot = Path(
        snapshot_download(repo_id=repository, revision=revision, cache_dir=hub_cache)
    ).resolve()
    if snapshot.name != revision:
        raise ValueError(f"Encoder resolved to {snapshot.name}, expected {revision}")

    cache_repo = hub_cache / f"models--{repository.replace('/', '--')}"
    ref = cache_repo / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    # huggingface_hub reads this file verbatim; a trailing newline would become
    # part of the snapshot directory name and break offline resolution.
    ref.write_text(revision)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    SentenceTransformer("all-MiniLM-L6-v2", device="cpu", local_files_only=True)


def _run(args: argparse.Namespace) -> None:
    os.environ["OPENAI_BASE_URL"] = args.provider_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    _register_model(args.provider_model)

    from bfcl_eval._llm_response_generation import main as generation_main
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

    output_dir = Path(args.output_dir).resolve()
    result_dir = output_dir / "result"
    score_dir = output_dir / "score"
    result_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    generation_main(
        SimpleNamespace(
            model=[BFCL_REGISTRY_NAME],
            test_category=list(NON_WEB_COLLECTIONS),
            temperature=args.temperature,
            include_input_log=False,
            exclude_state_log=False,
            num_gpus=1,
            num_threads=args.num_threads,
            gpu_memory_utilization=0.9,
            backend="vllm",
            skip_server_setup=False,
            local_model_path=None,
            result_dir=result_dir,
            allow_overwrite=False,
            run_ids=False,
            enable_lora=False,
            max_lora_rank=None,
            lora_modules=None,
        )
    )
    evaluation_main(
        [BFCL_REGISTRY_NAME],
        list(NON_WEB_COLLECTIONS),
        str(result_dir),
        str(score_dir),
    )

    expected_results, expected_scores = _build_expected_inventories()
    row_count = _validate_result_inventory(result_dir, expected_results)
    scores = _validate_score_inventory(score_dir, expected_scores)
    metrics = _compute_metrics(scores)

    summary = {
        "metrics": metrics,
        "metadata": {
            "benchmark": "BFCL v4",
            "bfcl_commit": args.bfcl_commit,
            "scorer_patch_sha256": args.scorer_patch_sha256,
            "encoder_revision": args.encoder_revision,
            "covered_weight": EXPECTED_COVERED_WEIGHT,
            "excluded_categories": ["web_search_base", "web_search_no_snippet"],
            "result_file_count": len(expected_results),
            "score_file_count": len(expected_scores),
            "num_tasks": row_count,
            "provider_model": args.provider_model,
            # BFCL renormalizes missing agentic subsections in its partial-run
            # top-line CSV. The fixed-weight contribution above is therefore
            # computed from the validated native leaf scores, not that partial
            # aggregate.
            "official_partial_overall_used": False,
        },
        "predictions": [
            {
                "native_id": category,
                "instance_metrics": {"accuracy": {"external": score["accuracy"]}},
                "correct_count": score["correct_count"],
                "total_count": score["total_count"],
            }
            for category, score in sorted(scores.items())
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--cache-dir", type=Path, required=True)
    prepare.add_argument("--encoder-repository", required=True)
    prepare.add_argument("--encoder-revision", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--bfcl-root", type=Path, required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--provider-model", required=True)
    run.add_argument("--provider-url", required=True)
    run.add_argument("--temperature", type=float, required=True)
    run.add_argument("--num-threads", type=int, required=True)
    run.add_argument("--encoder-revision", required=True)
    run.add_argument("--bfcl-commit", default="6ea57973c7a6097fd7c5915698c54c17c5b1b6c8")
    run.add_argument("--scorer-patch-sha256", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "prepare":
        _prepare_encoder(args.cache_dir, args.encoder_repository, args.encoder_revision)
    else:
        if args.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if args.num_threads < 1:
            raise ValueError("num_threads must be positive")
        os.chdir(args.bfcl_root)
        _run(args)


if __name__ == "__main__":
    main()
