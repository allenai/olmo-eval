"""Tests for the BFCL v4 external evaluation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import tomllib
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import requests

from olmo_eval.evals.external.benchmarks.bfcl.eval import (
    BFCL_COMMIT,
    BFCL_ENCODER_REVISION,
    BFCL_SCORER_PATCH_SHA256,
    BFCLV4Args,
    BFCLV4NonWebExternalEval,
)
from olmo_eval.evals.external.benchmarks.bfcl.runner import (
    _classify_serpapi_response,
    _compute_non_web_metrics,
    _compute_web_metrics,
    _get_serpapi_remaining,
    _install_serpapi_audit,
    _validate_result_inventory,
    _validate_score_inventory,
    _validate_serpapi_audit,
)
from olmo_eval.evals.external.benchmarks.bfcl.web import (
    BFCLV4WebArgs,
    BFCLV4WebExternalEval,
)


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(entry)}\n" for entry in entries))


class TestBFCLV4Args(unittest.TestCase):
    def test_defaults_match_official_generation_temperature(self) -> None:
        self.assertEqual(BFCLV4Args.from_dict({}), BFCLV4Args())

    def test_rejects_invalid_runtime_arguments(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature"):
            BFCLV4Args.from_dict({"temperature": -0.1})
        with self.assertRaisesRegex(ValueError, "num_threads"):
            BFCLV4Args.from_dict({"num_threads": 0})

    def test_run_command_quotes_provider_values(self) -> None:
        command = BFCLV4NonWebExternalEval()._build_run_command(
            "model name; false", "http://localhost:8000/v1?x=1&y=2", BFCLV4Args()
        )
        self.assertIn("'model name; false'", command)
        self.assertIn("'http://localhost:8000/v1?x=1&y=2'", command)
        self.assertIn(BFCL_ENCODER_REVISION, command)
        self.assertNotIn("SERPAPI_API_KEY", command)

    def test_setup_pins_bfcl_and_encoder(self) -> None:
        commands = "\n".join(BFCLV4NonWebExternalEval().setup_command)
        self.assertIn(BFCL_COMMIT, commands)
        self.assertIn(BFCL_ENCODER_REVISION, commands)
        self.assertIn(BFCL_SCORER_PATCH_SHA256, commands)
        self.assertIn("soundfile", commands)

        patch_path = (
            Path(__file__).parents[4]
            / "src/olmo_eval/evals/external/benchmarks/bfcl/patches"
            / "0001-check-multi-turn-irrelevance.patch"
        )
        self.assertEqual(
            hashlib.sha256(patch_path.read_bytes()).hexdigest(), BFCL_SCORER_PATCH_SHA256
        )

        pyproject = tomllib.loads((Path(__file__).parents[4] / "pyproject.toml").read_text())
        package_data = pyproject["tool"]["setuptools"]["package-data"]
        self.assertIn(
            "patches/*.patch",
            package_data["olmo_eval.evals.external.benchmarks.bfcl"],
        )

    def test_sandbox_never_forwards_ambient_openai_key(self) -> None:
        evaluator = BFCLV4NonWebExternalEval()
        with (
            tempfile.TemporaryDirectory() as temporary_dir,
            mock.patch.dict("os.environ", {"OPENAI_API_KEY": "real-secret"}),
        ):
            config = evaluator._create_bfcl_sandbox_config("docker", None, Path(temporary_dir))
        self.assertEqual(dict(config.environment)["OPENAI_API_KEY"], "EMPTY")
        self.assertTrue(config.inject_swerex)
        self.assertIsNone(config.log_dir)


class TestBFCLInventoryValidation(unittest.TestCase):
    def test_accepts_complete_result_and_score_inventories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            result_relative = Path("model/non_live/BFCL_v4_simple_python_result.json")
            score_relative = Path("model/non_live/BFCL_v4_simple_python_score.json")
            _write_jsonl(
                root / "result" / result_relative,
                [
                    {"id": "simple_python_0", "result": [{"f": "{}"}]},
                    {"id": "simple_python_1", "result": []},
                ],
            )
            _write_jsonl(
                root / "score" / score_relative,
                [
                    {"accuracy": 0.5, "correct_count": 1, "total_count": 2},
                    {"id": "simple_python_1", "valid": False},
                ],
            )

            self.assertEqual(
                _validate_result_inventory(
                    root / "result",
                    {result_relative: {"simple_python_0", "simple_python_1"}},
                ),
                2,
            )
            scores = _validate_score_inventory(
                root / "score",
                {score_relative: {"simple_python_0", "simple_python_1"}},
            )
            self.assertEqual(scores["simple_python"]["accuracy"], 0.5)

    def test_rejects_missing_duplicate_and_inference_error_results(self) -> None:
        cases = (
            ([{"id": "a_0", "result": []}], {"a_0", "a_1"}, "result-ID mismatch"),
            (
                [{"id": "a_0", "result": []}, {"id": "a_0", "result": []}],
                {"a_0"},
                "Duplicate",
            ),
            (
                [{"id": "a_0", "result": "Error during inference: timeout"}],
                {"a_0"},
                "inference error",
            ),
            (
                [{"id": "a_0", "result": [], "traceback": "boom"}],
                {"a_0"},
                "traceback",
            ),
        )
        for entries, expected_ids, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary_dir:
                root = Path(temporary_dir)
                relative = Path("model/non_live/BFCL_v4_a_result.json")
                _write_jsonl(root / relative, entries)
                with self.assertRaisesRegex(ValueError, message):
                    _validate_result_inventory(root, {relative: expected_ids})

    def test_rejects_inconsistent_score_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            relative = Path("model/non_live/BFCL_v4_a_score.json")
            _write_jsonl(
                root / relative,
                [
                    {"accuracy": 1.0, "correct_count": 2, "total_count": 2},
                    {"id": "a_1", "valid": False},
                ],
            )
            with self.assertRaisesRegex(ValueError, "correct-count mismatch"):
                _validate_score_inventory(root, {relative: {"a_0", "a_1"}})


class TestBFCLMetrics(unittest.TestCase):
    def test_fixed_weight_non_web_contribution(self) -> None:
        categories = {
            "simple_python": (1.0, 400),
            "simple_java": (1.0, 100),
            "simple_javascript": (1.0, 50),
            "multiple": (1.0, 200),
            "parallel": (1.0, 200),
            "parallel_multiple": (1.0, 200),
            "irrelevance": (1.0, 240),
            "live_simple": (1.0, 258),
            "live_multiple": (1.0, 1053),
            "live_parallel": (1.0, 16),
            "live_parallel_multiple": (1.0, 24),
            "live_irrelevance": (1.0, 884),
            "live_relevance": (1.0, 16),
            "multi_turn_base": (1.0, 200),
            "multi_turn_miss_func": (1.0, 200),
            "multi_turn_miss_param": (1.0, 200),
            "multi_turn_long_context": (1.0, 200),
            "memory_kv": (1.0, 155),
            "memory_vector": (1.0, 155),
            "memory_rec_sum": (1.0, 155),
        }
        scores = {
            category: {
                "accuracy": accuracy,
                "correct_count": int(accuracy * count),
                "total_count": count,
            }
            for category, (accuracy, count) in categories.items()
        }
        metrics = _compute_non_web_metrics(scores)
        self.assertAlmostEqual(metrics["non_web_weighted_contribution"], 0.8)
        self.assertLessEqual(metrics["non_web_weighted_contribution"], 0.8)

    def test_fixed_weight_web_contribution(self) -> None:
        scores = {
            "web_search_base": {"accuracy": 1.0, "correct_count": 100, "total_count": 100},
            "web_search_no_snippet": {
                "accuracy": 0.5,
                "correct_count": 50,
                "total_count": 100,
            },
        }
        metrics = _compute_web_metrics(scores)
        self.assertEqual(metrics["web_search_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["web_weighted_contribution"], 0.15)


class TestBFCLWeb(unittest.TestCase):
    def test_web_arguments_are_conservative_and_validated(self) -> None:
        self.assertEqual(BFCLV4WebArgs.from_dict({}).num_threads, 4)
        self.assertEqual(BFCLV4WebArgs.from_dict({}).minimum_serpapi_credits, 628)
        with self.assertRaisesRegex(ValueError, "minimum_serpapi_credits"):
            BFCLV4WebArgs.from_dict({"minimum_serpapi_credits": -1})

    def test_secret_is_file_mounted_and_never_forwarded_in_docker_environment(self) -> None:
        evaluator = BFCLV4WebExternalEval()
        secret = "test-serpapi-secret"
        with (
            tempfile.TemporaryDirectory() as artifact_dir,
            mock.patch.dict("os.environ", {"SERPAPI_API_KEY": secret}, clear=True),
            ExitStack() as stack,
        ):
            volumes = evaluator._create_sensitive_volumes(stack)
            secret_path = Path(volumes[0][0])
            self.assertEqual(secret_path.read_text(), secret)
            self.assertEqual(secret_path.stat().st_mode & 0o077, 0)

            config = evaluator._create_bfcl_sandbox_config(
                "docker", None, Path(artifact_dir), extra_volumes=volumes
            )
            command = evaluator._build_run_command("model", "http://provider/v1", BFCLV4WebArgs())
            serialized_config = repr(config)

        self.assertFalse(secret_path.exists())
        self.assertEqual(evaluator.required_secrets, ("SERPAPI_API_KEY",))
        self.assertNotIn(secret, serialized_config)
        self.assertNotIn(secret, command)
        self.assertNotIn("SERPAPI_API_KEY", dict(config.environment))
        self.assertEqual(evaluator._build_env_vars(("SERPAPI_API_KEY",)), {})
        self.assertIn("--suite web", command)

    def test_missing_secret_fails_before_sandbox_creation(self) -> None:
        evaluator = BFCLV4WebExternalEval()
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            ExitStack() as stack,
            self.assertRaisesRegex(ValueError, "SERPAPI_API_KEY"),
        ):
            evaluator._create_sensitive_volumes(stack)

    def test_classifies_serpapi_payloads_without_preserving_content(self) -> None:
        self.assertEqual(
            _classify_serpapi_response({"organic_results": [{"title": "secret"}]})[0],
            "organic_results",
        )
        self.assertEqual(
            _classify_serpapi_response({"organic_results": []})[0],
            "empty_organic_results",
        )
        status, fields = _classify_serpapi_response({"error": "upstream failure"})
        self.assertEqual(status, "error_payload")
        self.assertNotIn("upstream failure", repr(fields))
        self.assertEqual(_classify_serpapi_response({"error": "429 limit"})[0], "rate_limited")
        self.assertEqual(
            _classify_serpapi_response(
                {"organic_results": [{"title": "partial"}], "error": "429 limit"}
            )[0],
            "rate_limited",
        )
        self.assertEqual(
            _classify_serpapi_response({"search_metadata": {}})[0], "missing_organic_results"
        )

    def test_audits_every_search_and_accepts_only_recoverable_statuses(self) -> None:
        class FakeGoogleSearch:
            responses = [
                {"organic_results": [{"title": "result"}]},
                {"error": "429 throughput limit"},
            ]

            def __init__(self, query: str) -> None:
                self.params_dict = {"q": query, "api_key": "test-secret"}

            def get_dict(self):
                return self.responses.pop(0)

        fake_module = SimpleNamespace(GoogleSearch=FakeGoogleSearch)
        with tempfile.TemporaryDirectory() as temporary_dir:
            audit_path = Path(temporary_dir) / "serpapi_audit.jsonl"
            _install_serpapi_audit(audit_path, fake_module)
            fake_module.GoogleSearch("private query 1").get_dict()
            fake_module.GoogleSearch("private query 2").get_dict()

            statuses = _validate_serpapi_audit(audit_path)
            serialized = audit_path.read_text()

        self.assertEqual(statuses, {"organic_results": 1, "rate_limited": 1})
        self.assertNotIn("private query", serialized)
        self.assertNotIn("test-secret", serialized)

    def test_account_preflight_redacts_request_exception(self) -> None:
        secret = "test-serpapi-secret"
        error = requests.ConnectionError(
            f"request failed for https://serpapi.com/account.json?api_key={secret}"
        )
        with (
            mock.patch("requests.get", side_effect=error),
            self.assertRaisesRegex(RuntimeError, "ConnectionError") as raised,
        ):
            _get_serpapi_remaining(secret)
        self.assertNotIn(secret, str(raised.exception))

    def test_search_exception_redacts_key_and_query(self) -> None:
        secret = "test-serpapi-secret"
        query = "private query"

        class FakeGoogleSearch:
            def __init__(self, query: str) -> None:
                self.params_dict = {"q": query, "api_key": secret}

            def get_dict(self):
                raise RuntimeError(
                    f"request failed for https://serpapi.com/search?q={query}&api_key={secret}"
                )

        fake_module = SimpleNamespace(GoogleSearch=FakeGoogleSearch)
        with tempfile.TemporaryDirectory() as temporary_dir:
            audit_path = Path(temporary_dir) / "serpapi_audit.jsonl"
            _install_serpapi_audit(audit_path, fake_module)
            with self.assertRaisesRegex(RuntimeError, "SerpAPI request failed") as raised:
                fake_module.GoogleSearch(query).get_dict()
            serialized = audit_path.read_text()

        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(query, str(raised.exception))
        self.assertNotIn(secret, serialized)
        self.assertNotIn(query, serialized)

    def test_rate_limit_exception_preserves_retry_signal_without_secrets(self) -> None:
        secret = "test-serpapi-secret"
        query = "private query"

        class FakeGoogleSearch:
            def __init__(self, query: str) -> None:
                self.params_dict = {"q": query, "api_key": secret}

            def get_dict(self):
                raise RuntimeError(f"429 for https://serpapi.com/search?q={query}&api_key={secret}")

        fake_module = SimpleNamespace(GoogleSearch=FakeGoogleSearch)
        with tempfile.TemporaryDirectory() as temporary_dir:
            audit_path = Path(temporary_dir) / "serpapi_audit.jsonl"
            _install_serpapi_audit(audit_path, fake_module)
            with self.assertRaisesRegex(RuntimeError, "429 from SerpAPI") as raised:
                fake_module.GoogleSearch(query).get_dict()
            statuses = _validate_serpapi_audit(audit_path)
            serialized = audit_path.read_text()

        self.assertEqual(statuses, {"rate_limited_exception": 1})
        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(query, str(raised.exception))
        self.assertNotIn(secret, serialized)
        self.assertNotIn(query, serialized)

    def test_rejects_terminal_serpapi_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            audit_path = Path(temporary_dir) / "serpapi_audit.jsonl"
            _write_jsonl(
                audit_path,
                [
                    {
                        "sequence": 1,
                        "query_sha256": "a" * 64,
                        "status": "error_payload",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "terminal failures"):
                _validate_serpapi_audit(audit_path)


if __name__ == "__main__":
    unittest.main()
