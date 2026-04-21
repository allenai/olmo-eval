"""Tests for the SciCode benchmark task."""

import unittest

import pytest

from olmo_eval.common.types import LMOutput, Response
from olmo_eval.evals.tasks import scicode as scicode_mod
from olmo_eval.evals.tasks.common import get_task, list_tasks


@pytest.fixture(autouse=True)
def _load_registry():
    import olmo_eval.evals.tasks  # noqa: F401


class TestRegistration(unittest.TestCase):
    def test_tasks_registered(self) -> None:
        names = list_tasks()
        self.assertIn("scicode", names)
        self.assertIn("scicode_with_background", names)

    def test_variants_resolve(self) -> None:
        get_task("scicode:validation")
        get_task("scicode_with_background:validation")
        get_task("scicode:chat")


class TestInstances(unittest.TestCase):
    def test_validation_counts(self) -> None:
        task = get_task("scicode:validation")
        instances = list(task.instances)
        self.assertEqual(len(instances), 15)
        total_substeps = sum(i.metadata["total_steps"] for i in instances)
        self.assertEqual(total_substeps, 50)

    def test_test_counts(self) -> None:
        task = get_task("scicode")
        instances = list(task.instances)
        self.assertEqual(len(instances), 65)
        total_substeps = sum(i.metadata["total_steps"] for i in instances)
        # Matches upstream STEP_NUM=288 (scorable sub-steps only).
        self.assertEqual(total_substeps, 288)

    def test_all_80_problems_present(self) -> None:
        val = list(get_task("scicode:validation").instances)
        test = list(get_task("scicode").instances)
        self.assertEqual(len(val) + len(test), 80)

    def test_hardcoded_problems_present(self) -> None:
        task = get_task("scicode")
        problem_ids = {i.metadata["problem_id"] for i in task.instances}
        for pid in scicode_mod._HARDCODED_SNIPPETS:
            if pid in problem_ids:
                inst = next(i for i in task.instances if i.metadata["problem_id"] == pid)
                self.assertTrue(inst.metadata["hardcoded_prelude"].strip())
                self.assertTrue(
                    any(s.get("_hardcoded") for s in inst.metadata["sub_steps"]),
                    f"problem {pid} should have a _hardcoded sub-step",
                )

    def test_metadata_fields(self) -> None:
        task = get_task("scicode:validation")
        instance = next(iter(task.instances))
        for key in ("id", "problem_id", "sub_steps", "total_steps", "required_dependencies"):
            self.assertIn(key, instance.metadata)
        step = instance.metadata["sub_steps"][0]
        for key in ("step_number", "function_header", "test_cases"):
            self.assertIn(key, step)


class TestPromptBuilding(unittest.TestCase):
    def test_prompt_includes_every_step(self) -> None:
        task = get_task("scicode:validation")
        instance = next(iter(task.instances))
        for step in instance.metadata["sub_steps"]:
            self.assertIn(step["step_number"], instance.question)
            self.assertIn(step["function_header"].split("\n", 1)[0], instance.question)

    def test_with_background_includes_backgrounds(self) -> None:
        task = get_task("scicode_with_background:validation")
        instance = next(iter(task.instances))
        backgrounds = [
            (s.get("step_background") or "").strip() for s in instance.metadata["sub_steps"]
        ]
        non_empty = [b for b in backgrounds if b]
        if non_empty:
            self.assertIn(non_empty[0].splitlines()[0], instance.question)


class TestCodeExtraction(unittest.TestCase):
    def test_extract_strips_imports(self) -> None:
        task = get_task("scicode")
        raw = "```python\nimport numpy as np\nfrom math import pi\n\ndef foo():\n    return 1\n```"
        extracted = task.extract_answer(LMOutput(text=raw))
        self.assertIn("def foo()", extracted)
        self.assertNotIn("import numpy", extracted)
        self.assertNotIn("from math", extracted)

    def test_extract_returns_none_on_empty(self) -> None:
        task = get_task("scicode")
        self.assertIsNone(task.extract_answer(LMOutput(text="")))


class TestScorerScriptAssembly(unittest.TestCase):
    def test_script_contains_helpers_and_targets(self) -> None:
        scorer = scicode_mod.SciCodeExecutionScorer(h5py_file="/tmp/x.h5")
        step = {
            "step_number": "77.1",
            "test_cases": ["assert wrap(1.0, 5.0) == target"],
        }
        script = scorer._build_step_script(
            step,
            deps="import numpy as np",
            code="def wrap(r, L):\n    return r % L",
        )
        self.assertIn("def wrap(r, L)", script)
        self.assertIn("process_hdf5_to_tuple('77.1', 1, '/tmp/x.h5')", script)
        self.assertIn("target = targets[0]", script)
        self.assertIn("assert wrap(1.0, 5.0) == target", script)


class TestMetrics(unittest.TestCase):
    def _response_with_steps(self, passed: int, total: int) -> Response:
        from olmo_eval.common.types import Instance, LMRequest, RequestType

        out = LMOutput(text="")
        out.metadata = {"passed_steps": passed, "total_steps": total}
        return Response(
            instance=Instance(question=""),
            request=LMRequest(request_type=RequestType.COMPLETION, prompt=""),
            outputs=[out],
        )

    def test_sub_step_accuracy_micro_average(self) -> None:
        metric = scicode_mod.SciCodeSubStepAccuracyMetric()
        responses = [
            self._response_with_steps(passed=2, total=4),
            self._response_with_steps(passed=3, total=6),
        ]
        self.assertAlmostEqual(metric.compute(responses), 5 / 10)

    def test_main_problem_accuracy(self) -> None:
        metric = scicode_mod.SciCodeMainProblemAccuracyMetric()
        responses = [
            self._response_with_steps(passed=4, total=4),
            self._response_with_steps(passed=2, total=4),
            self._response_with_steps(passed=3, total=3),
        ]
        self.assertAlmostEqual(metric.compute(responses), 2 / 3)
