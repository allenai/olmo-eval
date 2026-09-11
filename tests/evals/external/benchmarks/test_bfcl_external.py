"""Tests for the BFCL v4 external evaluation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from olmo_eval.evals.external.benchmarks.bfcl.eval import (
    BFCL_COMMIT,
    BFCL_ENCODER_REVISION,
    BFCL_SCORER_PATCH_SHA256,
    BFCLV4Args,
    BFCLV4NonWebExternalEval,
)
from olmo_eval.evals.external.benchmarks.bfcl.runner import (
    _compute_metrics,
    _validate_result_inventory,
    _validate_score_inventory,
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
        metrics = _compute_metrics(scores)
        self.assertAlmostEqual(metrics["non_web_weighted_contribution"], 0.8)
        self.assertLessEqual(metrics["non_web_weighted_contribution"], 0.8)


if __name__ == "__main__":
    unittest.main()
