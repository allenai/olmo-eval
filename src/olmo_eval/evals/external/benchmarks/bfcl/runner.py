"""Run and validate a pinned BFCL v4 checkout inside the evaluation sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import threading
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

BFCL_REGISTRY_NAME = "olmo-eval-provider-FC"
BFCL_DISPLAY_NAME = "OLMo Eval Provider (FC)"
NON_WEB_COLLECTIONS = ("non_live", "live", "multi_turn", "memory")
WEB_COLLECTIONS = ("web_search",)
NON_WEB_EXPECTED_RESULT_FILES = 23
NON_WEB_EXPECTED_SCORE_FILES = 20
NON_WEB_EXPECTED_RESULT_ROWS = 5_017
NON_WEB_EXPECTED_COVERED_WEIGHT = 0.8
WEB_EXPECTED_RESULT_FILES = 2
WEB_EXPECTED_SCORE_FILES = 2
WEB_EXPECTED_RESULT_ROWS = 200
WEB_EXPECTED_COVERED_WEIGHT = 0.2


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


def _compute_non_web_metrics(scores: dict[str, dict[str, float | int]]) -> dict[str, float]:
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


def _compute_web_metrics(scores: dict[str, dict[str, float | int]]) -> dict[str, float]:
    web_search = _unweighted(scores, ("web_search_base", "web_search_no_snippet"))
    return {
        "web_weighted_contribution": 0.2 * web_search,
        "web_search_accuracy": web_search,
        "web_search_base_accuracy": float(scores["web_search_base"]["accuracy"]),
        "web_search_no_snippet_accuracy": float(scores["web_search_no_snippet"]["accuracy"]),
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


def _build_expected_inventories(
    collections: tuple[str, ...],
    *,
    expected_result_files: int,
    expected_score_files: int,
    expected_result_rows: int,
) -> tuple[dict[Path, set[str]], dict[Path, set[str]]]:
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
    for collection in collections:
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

    if len(result_inventory) != expected_result_files:
        raise ValueError(f"Pinned BFCL checkout produced {len(result_inventory)} result files")
    if len(score_inventory) != expected_score_files:
        raise ValueError(f"Pinned BFCL checkout produced {len(score_inventory)} score files")
    expected_rows = sum(len(ids) for ids in result_inventory.values())
    if expected_rows != expected_result_rows:
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


def _read_serpapi_secret(path: Path) -> str:
    secret = path.read_text().strip()
    if not secret:
        raise ValueError("The mounted SerpAPI secret file is empty")
    return secret


def _get_serpapi_remaining(api_key: str) -> int:
    import requests

    try:
        response = requests.get(
            "https://serpapi.com/account.json",
            params={"api_key": api_key},
            timeout=30,
        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"SerpAPI account preflight request failed ({type(error).__name__})"
        ) from None
    if response.status_code != 200:
        raise RuntimeError(f"SerpAPI account preflight returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError("SerpAPI account preflight returned invalid JSON") from None
    remaining = payload.get("total_searches_left")
    if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 0:
        raise ValueError("SerpAPI account response has no valid total_searches_left value")
    return remaining


def _classify_serpapi_response(response: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(response, dict):
        return "malformed_response", {"response_type": type(response).__name__}

    # Treat an explicit provider error as authoritative even if the response
    # also contains stale or partial result data.
    error = response.get("error")
    if error is not None:
        error_text = str(error)
        status = "rate_limited" if "429" in error_text else "error_payload"
        return status, {"error_sha256": hashlib.sha256(error_text.encode()).hexdigest()}

    organic_results = response.get("organic_results")
    if isinstance(organic_results, list):
        status = "organic_results" if organic_results else "empty_organic_results"
        return status, {"organic_result_count": len(organic_results)}
    if "organic_results" in response:
        return "malformed_organic_results", {"organic_results_type": type(organic_results).__name__}

    search_metadata = response.get("search_metadata")
    search_information = response.get("search_information")
    return "missing_organic_results", {
        "response_keys": sorted(str(key) for key in response),
        "search_status": (
            search_metadata.get("status") if isinstance(search_metadata, dict) else None
        ),
        "organic_results_state": (
            search_information.get("organic_results_state")
            if isinstance(search_information, dict)
            else None
        ),
    }


def _install_serpapi_audit(audit_path: Path, web_search: Any | None = None) -> None:
    if web_search is None:
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import web_search

    original_get_dict = web_search.GoogleSearch.get_dict
    write_lock = threading.Lock()
    sequence = 0
    audit_path.touch(exist_ok=False)

    def audited_get_dict(search: Any) -> Any:
        nonlocal sequence
        params = getattr(search, "params_dict", {})
        query = str(params.get("q", "")) if isinstance(params, dict) else ""
        event: dict[str, Any] = {
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "started_at": datetime.now(UTC).isoformat(),
        }
        started = time.monotonic()
        try:
            response = original_get_dict(search)
        except Exception as error:
            error_text = str(error)
            rate_limited = "429" in error_text
            event.update(
                {
                    "status": ("rate_limited_exception" if rate_limited else "exception"),
                    "error_type": type(error).__name__,
                    "error_sha256": hashlib.sha256(error_text.encode()).hexdigest(),
                }
            )
            # The SerpAPI client may include its full request URL (and therefore
            # the API key and raw query) in exception text. Preserve BFCL's 429
            # retry signal without allowing either secret into captured output.
            message = "429 from SerpAPI" if rate_limited else "SerpAPI request failed"
            raise RuntimeError(message) from None
        else:
            status, fields = _classify_serpapi_response(response)
            event.update({"status": status, **fields})
            return response
        finally:
            event["duration_ms"] = round((time.monotonic() - started) * 1_000, 3)
            with write_lock:
                sequence += 1
                event["sequence"] = sequence
                with audit_path.open("a") as file_handle:
                    file_handle.write(json.dumps(event, sort_keys=True) + "\n")

    web_search.GoogleSearch.get_dict = audited_get_dict


def _validate_serpapi_audit(audit_path: Path) -> dict[str, int]:
    events = _read_jsonl(audit_path)
    expected_sequences = list(range(1, len(events) + 1))
    if [event.get("sequence") for event in events] != expected_sequences:
        raise ValueError("SerpAPI audit sequence is incomplete or out of order")

    forbidden_fields = {"api_key", "serpapi_api_key", "query", "keywords"}
    for event in events:
        if forbidden_fields.intersection(key.lower() for key in event):
            raise ValueError("SerpAPI audit contains a sensitive field")
        query_sha256 = event.get("query_sha256")
        if (
            not isinstance(query_sha256, str)
            or len(query_sha256) != 64
            or any(character not in "0123456789abcdef" for character in query_sha256)
        ):
            raise ValueError("SerpAPI audit contains an invalid query hash")

    statuses = Counter(str(event.get("status")) for event in events)
    terminal_statuses = {
        "empty_organic_results",
        "error_payload",
        "exception",
        "malformed_organic_results",
        "malformed_response",
        "missing_organic_results",
    }
    failures = {status: statuses[status] for status in terminal_statuses if statuses[status]}
    if failures:
        raise ValueError(f"SerpAPI audit contains terminal failures: {failures}")
    allowed_statuses = {"organic_results", "rate_limited", "rate_limited_exception"}
    unexpected = sorted(set(statuses) - allowed_statuses)
    if unexpected:
        raise ValueError(f"SerpAPI audit contains unexpected statuses: {unexpected}")
    return dict(statuses)


def _run(args: argparse.Namespace) -> None:
    os.environ["OPENAI_BASE_URL"] = args.provider_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    collections = NON_WEB_COLLECTIONS if args.suite == "non_web" else WEB_COLLECTIONS
    if args.suite == "web":
        serpapi_key = _read_serpapi_secret(args.serpapi_secret_file)
        os.environ["SERPAPI_API_KEY"] = serpapi_key
        quota_before = _get_serpapi_remaining(serpapi_key)
        if quota_before < args.minimum_serpapi_credits:
            raise ValueError(
                f"SerpAPI has {quota_before} searches left; "
                f"{args.minimum_serpapi_credits} are required"
            )
    else:
        serpapi_key = None
        quota_before = None

    _register_model(args.provider_model)

    from bfcl_eval._llm_response_generation import main as generation_main
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

    output_dir = Path(args.output_dir).resolve()
    result_dir = output_dir / "result"
    score_dir = output_dir / "score"
    result_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "serpapi_audit.jsonl"
    if args.suite == "web":
        _install_serpapi_audit(audit_path)

    generation_main(
        SimpleNamespace(
            model=[BFCL_REGISTRY_NAME],
            test_category=list(collections),
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
        list(collections),
        str(result_dir),
        str(score_dir),
    )

    if args.suite == "non_web":
        expected_results, expected_scores = _build_expected_inventories(
            NON_WEB_COLLECTIONS,
            expected_result_files=NON_WEB_EXPECTED_RESULT_FILES,
            expected_score_files=NON_WEB_EXPECTED_SCORE_FILES,
            expected_result_rows=NON_WEB_EXPECTED_RESULT_ROWS,
        )
        covered_weight = NON_WEB_EXPECTED_COVERED_WEIGHT
        excluded_categories = ["web_search_base", "web_search_no_snippet"]
    else:
        expected_results, expected_scores = _build_expected_inventories(
            WEB_COLLECTIONS,
            expected_result_files=WEB_EXPECTED_RESULT_FILES,
            expected_score_files=WEB_EXPECTED_SCORE_FILES,
            expected_result_rows=WEB_EXPECTED_RESULT_ROWS,
        )
        covered_weight = WEB_EXPECTED_COVERED_WEIGHT
        excluded_categories = ["non_live", "live", "multi_turn", "memory"]

    row_count = _validate_result_inventory(result_dir, expected_results)
    scores = _validate_score_inventory(score_dir, expected_scores)
    metrics = (
        _compute_non_web_metrics(scores)
        if args.suite == "non_web"
        else _compute_web_metrics(scores)
    )

    if args.suite == "web":
        assert serpapi_key is not None
        audit_statuses = _validate_serpapi_audit(audit_path)
        quota_after = _get_serpapi_remaining(serpapi_key)
    else:
        audit_statuses = None
        quota_after = None

    summary = {
        "metrics": metrics,
        "metadata": {
            "benchmark": "BFCL v4",
            "bfcl_commit": args.bfcl_commit,
            "scorer_patch_sha256": args.scorer_patch_sha256,
            "suite": args.suite,
            "encoder_revision": args.encoder_revision,
            "covered_weight": covered_weight,
            "excluded_categories": excluded_categories,
            "result_file_count": len(expected_results),
            "score_file_count": len(expected_scores),
            "num_tasks": row_count,
            "provider_model": args.provider_model,
            # BFCL renormalizes missing agentic subsections in its partial-run
            # top-line CSV. The fixed-weight contribution above is therefore
            # computed from the validated native leaf scores, not that partial
            # aggregate.
            "official_partial_overall_used": False,
            "serpapi_audit_statuses": audit_statuses,
            "serpapi_quota_before": quota_before,
            "serpapi_quota_after": quota_after,
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
    run.add_argument("--suite", choices=("non_web", "web"), required=True)
    run.add_argument("--encoder-revision")
    run.add_argument("--serpapi-secret-file", type=Path)
    run.add_argument("--minimum-serpapi-credits", type=int, default=628)
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
        if args.minimum_serpapi_credits < 0:
            raise ValueError("minimum-serpapi-credits must be non-negative")
        if args.suite == "non_web" and not args.encoder_revision:
            raise ValueError("encoder-revision is required for the non-web suite")
        if args.suite == "web" and args.serpapi_secret_file is None:
            raise ValueError("serpapi-secret-file is required for the web suite")
        os.chdir(args.bfcl_root)
        _run(args)


if __name__ == "__main__":
    main()
