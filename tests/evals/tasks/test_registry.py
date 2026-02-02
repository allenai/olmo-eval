"""Tests for olmo_eval.tasks.registry module."""

from collections.abc import Iterator

import pytest

from olmo_eval.core.types import Instance, LMOutput, LMRequest, RequestType
from olmo_eval.evals.tasks import (
    Task,
    TaskConfig,
    clear_registry,
    get_base_task_name,
    get_task,
    get_task_dependencies,
    list_regimes,
    list_tasks,
    parse_overrides,
    register,
    register_regime,
    register_variant,
)
from olmo_eval.evals.tasks.core.registry import _configs, _regimes, _tasks, _variants


class DummyTask(Task):
    """A minimal task implementation for testing."""

    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(question="What is 2+2?", gold_answer="4")

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)

    def extract_answer(self, output: LMOutput) -> str:
        return output.text.strip()


@pytest.fixture(autouse=True)
def clean_registry():
    """Provide an isolated registry for testing.

    Saves the current registry state, clears it for the test,
    then restores it afterward.
    """
    # Save original state
    original_tasks = _tasks.copy()
    original_configs = _configs.copy()
    original_regimes = {k: v.copy() for k, v in _regimes.items()}
    original_variants = {k: v.copy() for k, v in _variants.items()}

    clear_registry()
    yield

    # Restore original state
    clear_registry()
    _tasks.update(original_tasks)
    _configs.update(original_configs)
    _regimes.update(original_regimes)
    _variants.update(original_variants)


class TestRegister:
    """Tests for the @register decorator."""

    def test_register_task(self):
        """Test basic task registration."""

        @register("test_task", lambda: TaskConfig(name="test_task", data_source="test/dataset"))
        class TestTask(DummyTask):
            pass

        assert "test_task" in list_tasks()

    def test_register_duplicate_raises(self):
        """Test that registering duplicate task names raises an error."""

        @register("duplicate", lambda: TaskConfig(name="duplicate", data_source="test/dataset"))
        class FirstTask(DummyTask):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register("duplicate", lambda: TaskConfig(name="duplicate", data_source="test/dataset"))
            class SecondTask(DummyTask):
                pass

    def test_register_preserves_class(self):
        """Test that @register returns the original class."""

        @register("preserved", lambda: TaskConfig(name="preserved", data_source="test/dataset"))
        class PreservedTask(DummyTask):
            pass

        assert PreservedTask.__name__ == "PreservedTask"
        assert issubclass(PreservedTask, Task)


class TestRegisterRegime:
    """Tests for register_regime function."""

    def test_register_regime(self):
        """Test registering a regime for an existing task."""

        @register("base_task", lambda: TaskConfig(name="base_task", data_source="test/dataset"))
        class BaseTask(DummyTask):
            pass

        register_regime("base_task", "custom", num_fewshot=5, limit=100)

        regimes = list_regimes("base_task")
        assert "custom" in regimes["base_task"]

    def test_register_regime_unknown_task_raises(self):
        """Test that registering regime for unknown task raises error."""
        with pytest.raises(ValueError, match="unknown task"):
            register_regime("nonexistent", "regime", num_fewshot=5)

    def test_register_multiple_regimes(self):
        """Test registering multiple regimes for one task."""

        @register(
            "multi_regime", lambda: TaskConfig(name="multi_regime", data_source="test/dataset")
        )
        class MultiRegimeTask(DummyTask):
            pass

        register_regime("multi_regime", "fast", limit=10)
        register_regime("multi_regime", "full", limit=None)
        register_regime("multi_regime", "fewshot", num_fewshot=5)

        regimes = list_regimes("multi_regime")
        assert set(regimes["multi_regime"]) == {"fast", "full", "fewshot"}


class TestGetTask:
    """Tests for get_task function."""

    def test_get_task_by_name(self):
        """Test getting a task by simple name."""

        @register(
            "simple_task",
            lambda: TaskConfig(name="simple_task", data_source="test/dataset", num_fewshot=0),
        )
        class SimpleTask(DummyTask):
            pass

        task = get_task("simple_task")
        assert isinstance(task, SimpleTask)
        assert task.config.name == "simple_task"
        assert task.config.num_fewshot == 0

    def test_get_task_with_regime(self):
        """Test getting a task with regime overrides.

        Regimes are accessed as variants using single colon syntax: task:regime
        """

        @register(
            "regime_task",
            lambda: TaskConfig(name="regime_task", data_source="test/dataset", num_fewshot=0),
        )
        class RegimeTask(DummyTask):
            pass

        register_regime("regime_task", "fewshot", num_fewshot=5)

        # Without regime
        task_base = get_task("regime_task")
        assert task_base.config.num_fewshot == 0

        # With regime (using new variant-style syntax)
        task_regime = get_task("regime_task:fewshot")
        assert task_regime.config.num_fewshot == 5

    def test_get_task_unknown_raises(self):
        """Test that getting unknown task raises KeyError."""
        with pytest.raises(KeyError, match="Unknown task"):
            get_task("nonexistent_task")

    def test_get_task_with_unknown_regime_raises(self):
        """Test that unknown regime/variant raises KeyError."""

        @register(
            "fallback_task",
            lambda: TaskConfig(name="fallback_task", data_source="test/dataset", num_fewshot=3),
        )
        class FallbackTask(DummyTask):
            pass

        # Unknown variant/regime should raise KeyError
        with pytest.raises(KeyError, match="Unknown variant 'unknown_regime'"):
            get_task("fallback_task:unknown_regime")


class TestListTasks:
    """Tests for list_tasks function."""

    def test_list_tasks_empty(self):
        """Test list_tasks with empty registry."""
        assert list_tasks() == []

    def test_list_tasks_returns_sorted(self):
        """Test that list_tasks returns sorted names."""

        @register("zebra", lambda: TaskConfig(name="zebra", data_source="test/dataset"))
        class ZebraTask(DummyTask):
            pass

        @register("alpha", lambda: TaskConfig(name="alpha", data_source="test/dataset"))
        class AlphaTask(DummyTask):
            pass

        @register("middle", lambda: TaskConfig(name="middle", data_source="test/dataset"))
        class MiddleTask(DummyTask):
            pass

        tasks = list_tasks()
        assert tasks == ["alpha", "middle", "zebra"]


class TestListRegimes:
    """Tests for list_regimes function."""

    def test_list_regimes_all(self):
        """Test listing all regimes."""

        @register("task_a", lambda: TaskConfig(name="task_a", data_source="test/dataset"))
        class TaskA(DummyTask):
            pass

        @register("task_b", lambda: TaskConfig(name="task_b", data_source="test/dataset"))
        class TaskB(DummyTask):
            pass

        register_regime("task_a", "regime1")
        register_regime("task_a", "regime2")
        register_regime("task_b", "regime3")

        all_regimes = list_regimes()
        assert "task_a" in all_regimes
        assert "task_b" in all_regimes
        assert set(all_regimes["task_a"]) == {"regime1", "regime2"}
        assert all_regimes["task_b"] == ["regime3"]

    def test_list_regimes_filtered(self):
        """Test listing regimes for specific task."""

        @register("filtered", lambda: TaskConfig(name="filtered", data_source="test/dataset"))
        class FilteredTask(DummyTask):
            pass

        register_regime("filtered", "fast")
        register_regime("filtered", "slow")

        regimes = list_regimes("filtered")
        assert "filtered" in regimes
        assert len(regimes) == 1
        assert set(regimes["filtered"]) == {"fast", "slow"}

    def test_list_regimes_no_regimes(self):
        """Test listing regimes for task with none registered."""

        @register("no_regimes", lambda: TaskConfig(name="no_regimes", data_source="test/dataset"))
        class NoRegimesTask(DummyTask):
            pass

        regimes = list_regimes("no_regimes")
        assert regimes == {"no_regimes": []}


class TestClearRegistry:
    """Tests for clear_registry function."""

    def test_clear_registry(self):
        """Test that clear_registry removes all entries."""

        @register("to_clear", lambda: TaskConfig(name="to_clear", data_source="test/dataset"))
        class ToClearTask(DummyTask):
            pass

        register_regime("to_clear", "regime")

        assert "to_clear" in list_tasks()
        assert list_regimes("to_clear")["to_clear"] == ["regime"]

        clear_registry()

        assert list_tasks() == []
        assert list_regimes() == {}


class TestGetBaseTaskName:
    """Tests for get_base_task_name utility function."""

    def test_simple_task_name(self):
        """Test with a simple task name (no modifiers)."""
        assert get_base_task_name("arc_easy") == "arc_easy"

    def test_task_with_variant(self):
        """Test that variants are preserved."""
        assert get_base_task_name("arc_easy:mc") == "arc_easy:mc"
        assert get_base_task_name("arc_easy:mc:olmes") == "arc_easy:mc:olmes"

    def test_task_with_priority(self):
        """Test that priority suffix is stripped."""
        assert get_base_task_name("arc_easy@high") == "arc_easy"
        assert get_base_task_name("arc_easy@low") == "arc_easy"
        assert get_base_task_name("arc_easy:mc@high") == "arc_easy:mc"


class TestGetTaskDependencies:
    """Tests for get_task_dependencies function."""

    def test_empty_dependencies_returns_empty_list(self):
        """Test that tasks without dependencies return empty list."""

        @register(
            "no_deps_task",
            lambda: TaskConfig(name="no_deps_task", data_source="test/dataset"),
        )
        class NoDepTask(DummyTask):
            pass

        result = get_task_dependencies(["no_deps_task"])
        assert result == []

    def test_single_task_with_dependencies(self):
        """Test extracting dependencies from a single task."""

        @register(
            "deps_task",
            lambda: TaskConfig(
                name="deps_task",
                data_source="test/dataset",
                dependencies=["pkg1==1.0", "pkg2>=2.0"],
            ),
        )
        class DepsTask(DummyTask):
            pass

        result = get_task_dependencies(["deps_task"])
        assert result == ["pkg1==1.0", "pkg2>=2.0"]

    def test_multiple_tasks_merge_dependencies(self):
        """Test that dependencies from multiple tasks are merged."""

        @register(
            "task_a",
            lambda: TaskConfig(
                name="task_a",
                data_source="test/dataset",
                dependencies=["pkg1==1.0"],
            ),
        )
        class TaskA(DummyTask):
            pass

        @register(
            "task_b",
            lambda: TaskConfig(
                name="task_b",
                data_source="test/dataset",
                dependencies=["pkg2==2.0"],
            ),
        )
        class TaskB(DummyTask):
            pass

        result = get_task_dependencies(["task_a", "task_b"])
        assert result == ["pkg1==1.0", "pkg2==2.0"]

    def test_dependencies_deduplicated(self):
        """Test that duplicate dependencies are removed."""

        @register(
            "dup_task_a",
            lambda: TaskConfig(
                name="dup_task_a",
                data_source="test/dataset",
                dependencies=["pkg1==1.0", "pkg2==2.0"],
            ),
        )
        class DupTaskA(DummyTask):
            pass

        @register(
            "dup_task_b",
            lambda: TaskConfig(
                name="dup_task_b",
                data_source="test/dataset",
                dependencies=["pkg1==1.0", "pkg3==3.0"],  # pkg1 is duplicate
            ),
        )
        class DupTaskB(DummyTask):
            pass

        result = get_task_dependencies(["dup_task_a", "dup_task_b"])
        # pkg1==1.0 appears in first task, should only appear once
        assert result == ["pkg1==1.0", "pkg2==2.0", "pkg3==3.0"]

    def test_preserves_order(self):
        """Test that order is preserved (first occurrence wins)."""

        @register(
            "order_task_a",
            lambda: TaskConfig(
                name="order_task_a",
                data_source="test/dataset",
                dependencies=["c-pkg", "a-pkg"],
            ),
        )
        class OrderTaskA(DummyTask):
            pass

        @register(
            "order_task_b",
            lambda: TaskConfig(
                name="order_task_b",
                data_source="test/dataset",
                dependencies=["b-pkg"],
            ),
        )
        class OrderTaskB(DummyTask):
            pass

        result = get_task_dependencies(["order_task_a", "order_task_b"])
        # Order should be preserved: c-pkg, a-pkg from task_a, then b-pkg from task_b
        assert result == ["c-pkg", "a-pkg", "b-pkg"]

    def test_mixed_tasks_with_and_without_deps(self):
        """Test mixing tasks with and without dependencies."""

        @register(
            "with_deps",
            lambda: TaskConfig(
                name="with_deps",
                data_source="test/dataset",
                dependencies=["special-lib"],
            ),
        )
        class WithDeps(DummyTask):
            pass

        @register(
            "without_deps",
            lambda: TaskConfig(name="without_deps", data_source="test/dataset"),
        )
        class WithoutDeps(DummyTask):
            pass

        result = get_task_dependencies(["without_deps", "with_deps"])
        assert result == ["special-lib"]

    def test_empty_task_list(self):
        """Test with empty task list."""
        result = get_task_dependencies([])
        assert result == []

    def test_task_with_variant_inherits_base_dependencies(self):
        """Test that variants can override dependencies."""

        @register(
            "base_deps",
            lambda: TaskConfig(
                name="base_deps",
                data_source="test/dataset",
                dependencies=["base-pkg"],
            ),
        )
        class BaseDepsTask(DummyTask):
            pass

        # Register variant that adds to dependencies
        register_variant("base_deps", "extra", dependencies=["extra-pkg"])

        # Base task should have base dependencies
        base_result = get_task_dependencies(["base_deps"])
        assert base_result == ["base-pkg"]

        # Variant should have overridden dependencies (replace, not merge)
        variant_result = get_task_dependencies(["base_deps:extra"])
        assert variant_result == ["extra-pkg"]

    def test_git_url_dependencies(self):
        """Test tasks with git URL dependencies."""

        @register(
            "git_deps_task",
            lambda: TaskConfig(
                name="git_deps_task",
                data_source="test/dataset",
                dependencies=[
                    "git+https://github.com/user/repo@v1.0",
                    "https://github.com/user/another-repo",
                ],
            ),
        )
        class GitDepsTask(DummyTask):
            pass

        result = get_task_dependencies(["git_deps_task"])
        assert result == [
            "git+https://github.com/user/repo@v1.0",
            "https://github.com/user/another-repo",
        ]


class TestParseOverridesDependencies:
    """Tests for parse_overrides handling of dependencies field."""

    def test_parse_dependencies_json_list(self):
        """Test parsing dependencies as JSON list."""
        result = parse_overrides('dependencies=["pkg1==1.0", "pkg2>=2.0"]')
        assert result == {"dependencies": ["pkg1==1.0", "pkg2>=2.0"]}

    def test_parse_dependencies_single_value(self):
        """Test parsing single dependency value (not JSON) becomes list."""
        result = parse_overrides("dependencies=special-lib")
        assert result == {"dependencies": ["special-lib"]}

    def test_parse_dependencies_with_other_overrides(self):
        """Test parsing dependencies alongside other overrides."""
        result = parse_overrides('num_fewshot=5,dependencies=["pkg1"]')
        assert result == {"num_fewshot": 5, "dependencies": ["pkg1"]}

    def test_parse_dependencies_empty_list(self):
        """Test parsing empty dependencies list."""
        result = parse_overrides("dependencies=[]")
        assert result == {"dependencies": []}

    def test_parse_dependencies_git_urls(self):
        """Test parsing dependencies with git URLs."""
        result = parse_overrides('dependencies=["git+https://github.com/user/repo@v1.0"]')
        assert result == {"dependencies": ["git+https://github.com/user/repo@v1.0"]}
