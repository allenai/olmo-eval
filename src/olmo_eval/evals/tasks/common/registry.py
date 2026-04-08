"""Task registry for registering and retrieving tasks by name.

Task specs follow the format: task_name[:variant1[:variant2...]]

Examples:
    - "arc_easy" - base task
    - "arc_easy:mc" - task with multiple-choice variant
    - "arc_easy:olmes" - task with olmes regime (regimes are now variants)
    - "arc_easy:mc:olmes" - task with variant and regime
    - "mbpp:3shot:bpb:none" - task with stacked variants and regime
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypeVar

from .base import Task, TaskConfig

T = TypeVar("T", bound=type[Task])

# Module-level registries
_tasks: dict[str, type[Task]] = {}
_configs: dict[str, TaskConfig] = {}
_variants: dict[str, dict[str, dict[str, Any]]] = {}
_regimes: dict[str, dict[str, dict[str, Any]]] = {}


def _build_config(name: str, cls: type[Task]) -> TaskConfig:
    """Build TaskConfig from class attributes."""
    kwargs: dict[str, Any] = {}
    for field in TaskConfig.__dataclass_fields__:
        if field == "name":
            continue
        for klass in cls.__mro__:
            if field in klass.__dict__:
                kwargs[field] = klass.__dict__[field]
                break

    return TaskConfig(name=name, **kwargs)


def register(name: str) -> Callable[[T], T]:
    """Register a task class with a name.

    Config is built from class attributes that match TaskConfig fields.
    Dynamically created classes are added to their module's namespace for pickling.

    Usage:
        @register("mmlu")
        class MMLU(Task):
            data_source = DataSource(path="cais/mmlu")
            metrics = (AccuracyMetric(),)
            ...
    """
    import sys

    def decorator(cls: T) -> T:
        if name in _tasks:
            raise ValueError(f"Task '{name}' already registered")
        _tasks[name] = cls
        _configs[name] = _build_config(name, cls)

        # Make dynamically created classes picklable by adding to module namespace
        module = sys.modules.get(cls.__module__)
        if module is not None and not hasattr(module, cls.__name__):
            setattr(module, cls.__name__, cls)

        return cls

    return decorator


def register_variant(task_name: str, variant: str, **overrides: Any) -> None:
    """Register a variant (format modifier) for a task.

    Variants modify how a task is evaluated (e.g., :mc for multiple choice,
    :gen for generation). They are applied before regimes.

    Args:
        task_name: Name of the base task (must already be registered).
        variant: Name of the variant (e.g., "mc", "gen").
        **overrides: TaskConfig field overrides for this variant.

    Raises:
        ValueError: If the task is not registered, or if the resulting task
            spec collides with a registered suite name.
    """
    if task_name not in _tasks:
        raise ValueError(
            f"Cannot register variant '{variant}' for unknown task '{task_name}'. "
            f"Register the task first using @register()."
        )

    from olmo_eval.evals.suites.registry import suite_exists

    spec = f"{task_name}:{variant}"
    if suite_exists(spec):
        raise ValueError(
            f"Task spec {spec!r} collides with a registered suite name. "
            f"Rename the suite to avoid ambiguity."
        )

    _variants.setdefault(task_name, {})[variant] = overrides


def register_regime(task_name: str, regime: str, **overrides: Any) -> None:
    """Register a regime (configuration preset) for a task.

    Regimes are configuration presets that define evaluation settings
    (e.g., :olmes for OLMo-style evaluation). They are applied after variants.

    Args:
        task_name: Name of the base task (must already be registered).
        regime: Name of the regime (e.g., "olmes").
        **overrides: TaskConfig field overrides for this regime.

    Raises:
        ValueError: If the task is not registered.
    """
    if task_name not in _tasks:
        raise ValueError(
            f"Cannot register regime '{regime}' for unknown task '{task_name}'. "
            f"Register the task first using @register()."
        )
    _regimes.setdefault(task_name, {})[regime] = overrides


def parse_overrides(override_str: str) -> dict[str, Any]:
    """Parse 'key=value,key=value' into dict with type coercion.

    Supports JSON values for complex configs (e.g., extra_loader_config={"distributed":true}).

    Args:
        override_str: Override string like "temperature=0.6,max_tokens=512"

    Returns:
        Dict with appropriately typed values.

    Examples:
        >>> parse_overrides("temperature=0.6,max_tokens=512")
        {"temperature": 0.6, "max_tokens": 512}
        >>> parse_overrides("provider=vllm")
        {"provider": "vllm"}
        >>> parse_overrides('extra_loader_config={"distributed":true}')
        {"extra_loader_config": {"distributed": True}}
    """
    import json

    if not override_str:
        return {}

    result: dict[str, Any] = {}
    decoder = json.JSONDecoder()
    i = 0

    while i < len(override_str):
        # Skip commas and whitespace
        while i < len(override_str) and override_str[i] in ", ":
            i += 1
        if i >= len(override_str):
            break

        # Find key=value
        eq_pos = override_str.find("=", i)
        if eq_pos == -1:
            break

        key = override_str[i:eq_pos].strip()
        i = eq_pos + 1

        # Parse value - use raw_decode for JSON, otherwise read until comma
        if i < len(override_str) and override_str[i] in "{[":
            value, end = decoder.raw_decode(override_str, i)
            i = end
        else:
            comma_pos = override_str.find(",", i)
            value_str = (override_str[i:comma_pos] if comma_pos != -1 else override_str[i:]).strip()
            i = comma_pos if comma_pos != -1 else len(override_str)

            # Type coercion
            if key in {
                "num_fewshot",
                "limit",
                "fewshot_seed",
                "seed",
                "max_tokens",
                "max_model_len",
                "top_k",
                "num_samples",
            }:
                value = int(value_str)
            elif key in {"temperature", "top_p"}:
                value = float(value_str)
            elif key == "dependencies":
                # Dependencies should be parsed as JSON list
                import json as json_module

                try:
                    value = json_module.loads(value_str)
                except json_module.JSONDecodeError:
                    # If not valid JSON, treat as single dependency
                    value = [value_str]
            else:
                value = value_str

        result[key] = value

    return result


def parse_task_spec(spec: str) -> tuple[str, list[str], dict[str, Any]]:
    """Parse a task spec into (task_name, variants, overrides).

    Spec format: task_name[:variant1[:variant2...]]

    Note: Regimes are now treated as variants.

    Args:
        spec: Task specification string.

    Returns:
        Tuple of (task_name, variants, overrides). Variants is a list (may be empty).
        Overrides is always an empty dict.

    Examples:
        >>> parse_task_spec("arc_easy")
        ("arc_easy", [], {})
        >>> parse_task_spec("arc_easy:mc")
        ("arc_easy", ["mc"], {})
        >>> parse_task_spec("arc_easy:olmes")
        ("arc_easy", ["olmes"], {})
        >>> parse_task_spec("arc_easy:mc:olmes")
        ("arc_easy", ["mc", "olmes"], {})
    """
    # Split on : to get task name and variants
    parts = spec.split(":")
    task_name = parts[0]
    variants = parts[1:] if len(parts) > 1 else []

    return task_name, variants, {}


def get_base_task_name(spec: str) -> str:
    """Extract the base task name from a spec, stripping priority suffix.

    This is useful for validation when you need to check if a task exists
    without caring about the priority suffix (@high).

    Args:
        spec: Task specification string (e.g., "arc_easy@high")

    Returns:
        Base task name with variants but without priority
        (e.g., "arc_easy" or "arc_easy:mc")

    Examples:
        >>> get_base_task_name("arc_easy")
        "arc_easy"
        >>> get_base_task_name("arc_easy:mc")
        "arc_easy:mc"
        >>> get_base_task_name("arc_easy@high")
        "arc_easy"
        >>> get_base_task_name("arc_easy:mc@high")
        "arc_easy:mc"
    """
    # Strip priority suffix (e.g., "@high")
    base = spec.rsplit("@", 1)[0] if "@" in spec else spec
    return base


def get_task(spec: str, config_overrides: dict[str, Any] | None = None) -> Task:
    """Instantiate a task by spec.

    Spec format: task_name[:variant1[:variant2...]]

    Note: Regimes are now treated as variants. When looking up a variant,
    we check both the variants and regimes registries.

    Task names with colons (e.g., "humaneval:bpb") are checked first before
    parsing as base_task:variant.

    Args:
        spec: Task specification (e.g., "arc_easy", "arc_easy:mc:olmes").
        config_overrides: Additional config overrides to apply (highest priority).

    Returns:
        Instantiated Task with config (and variant if specified).

    Raises:
        KeyError: If task_name is not registered.
    """
    # Try progressively shorter prefixes to find a registered task with colons in its name
    # e.g., for "humaneval:bpb:mc", check "humaneval:bpb:mc", then "humaneval:bpb", then "humaneval"
    parts = spec.split(":")
    task_name = None
    variants: list[str] = []

    for i in range(len(parts), 0, -1):
        candidate = ":".join(parts[:i])
        if candidate in _tasks:
            task_name = candidate
            variants = parts[i:]
            break

    if task_name is None:
        # Fall back to original parsing (first part is task name)
        task_name = parts[0]
        variants = parts[1:] if len(parts) > 1 else []

    # Filter empty variants (from :: separators in task specs like "task:rc::regime")
    variants = [v for v in variants if v]

    if task_name not in _tasks:
        available = ", ".join(sorted(_tasks.keys()))
        raise KeyError(f"Unknown task '{task_name}'. Available: {available}")

    config = _configs[task_name]

    # Apply variant/regime overrides in order (check both registries)
    for variant in variants:
        # First check variants registry
        if task_name in _variants and variant in _variants[task_name]:
            config = replace(config, **_variants[task_name][variant])
        # Then check regimes registry (regimes are now accessed as variants)
        elif task_name in _regimes and variant in _regimes[task_name]:
            config = replace(config, **_regimes[task_name][variant])
        else:
            available_variants = list(_variants.get(task_name, {}).keys())
            available_regimes = list(_regimes.get(task_name, {}).keys())
            available = sorted(set(available_variants + available_regimes))
            raise KeyError(
                f"Unknown variant '{variant}' for task '{task_name}'. "
                f"Available: {', '.join(available) if available else 'none'}"
            )

    # Apply additional config overrides (highest priority)
    if config_overrides:
        config = replace(config, **config_overrides)

    return _tasks[task_name](config)


def list_tasks() -> list[str]:
    """List all registered task names."""
    return sorted(_tasks.keys())


def list_variants(task_name: str | None = None) -> dict[str, list[str]]:
    """List available variants, optionally filtered by task.

    Args:
        task_name: If provided, only return variants for this task.

    Returns:
        Dict mapping task names to their available variants.
    """
    if task_name:
        return {task_name: list(_variants.get(task_name, {}).keys())}
    return {name: list(variants.keys()) for name, variants in _variants.items()}


def list_regimes(task_name: str | None = None) -> dict[str, list[str]]:
    """List available regimes, optionally filtered by task.

    Args:
        task_name: If provided, only return regimes for this task.

    Returns:
        Dict mapping task names to their available regimes.
    """
    if task_name:
        return {task_name: list(_regimes.get(task_name, {}).keys())}
    return {name: list(regimes.keys()) for name, regimes in _regimes.items()}


def task_exists(spec: str) -> bool:
    """Check if a task spec is valid (task exists and variants are registered).

    Handles task names containing colons (e.g., "naturalqs:mc") by trying
    progressively shorter prefixes, matching the logic in get_task().

    Args:
        spec: Task specification string.

    Returns:
        True if the task exists and all variants are valid, False otherwise.
    """
    parts = spec.split(":")
    task_name = None
    variants: list[str] = []

    for i in range(len(parts), 0, -1):
        candidate = ":".join(parts[:i])
        if candidate in _tasks:
            task_name = candidate
            variants = [v for v in parts[i:] if v]
            break

    if task_name is None:
        return False

    # Also validate variants exist
    for variant in variants:
        has_variant = task_name in _variants and variant in _variants[task_name]
        has_regime = task_name in _regimes and variant in _regimes[task_name]
        if not has_variant and not has_regime:
            return False

    return True


def clear_registry() -> None:
    """Clear registry (useful for testing)."""
    _tasks.clear()
    _configs.clear()
    _variants.clear()
    _regimes.clear()


def register_subtasks(
    base_class: type[Task],
    subtasks: list[str],
    *,
    task_prefix: str,
    data_source: str,
    subtask_attr: str = "subset",
    class_attrs: dict[str, Any] | None = None,
    variants: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Register multiple subtasks from a base class.

    This is useful for tasks like MMLU (many subjects) or multilingual MBPP
    (many languages) where you want to generate many similar task registrations.

    Args:
        base_class: The base Task class to subclass.
        subtasks: List of subtask identifiers (e.g., language codes, subject names).
        task_prefix: Prefix for task names (e.g., "mt_mbpp" -> "mt_mbpp_python").
        data_source: HuggingFace dataset path. Each subtask becomes the subset.
        subtask_attr: Class attribute name that receives the subtask identifier.
            Defaults to "subset". Use "language" for multilingual tasks, etc.
        class_attrs: Additional class attributes applied to all generated tasks.
        variants: Dict mapping variant names to their config overrides.

    Example:
        ```python
        register_subtasks(
            base_class=MultilingualMBPPTask,
            subtasks=["python", "java", "rust"],
            task_prefix="mt_mbpp",
            data_source="allenai/multilingual_mbpp",
            subtask_attr="language",
            class_attrs={
                "metrics": (),
                "sampling_params": SamplingParams(max_tokens=1024),
            },
            variants={
                "bpb": {"formatter": PPLFormatter(), "metrics": (BPBMetricByteAvg(),)},
                "3shot": {"num_fewshot": 3},
            },
        )
        # Registers: mt_mbpp_python, mt_mbpp_java, mt_mbpp_rust
        # With variants: mt_mbpp_python:bpb, mt_mbpp_python:3shot, etc.
        ```
    """
    from olmo_eval.data import DataSource

    for subtask in subtasks:
        task_name = f"{task_prefix}_{subtask}"
        class_name = f"{base_class.__name__}_{subtask.title().replace('-', '_')}"

        # Build class attributes
        attrs: dict[str, Any] = {
            subtask_attr: subtask,
            "data_source": DataSource(path=data_source, subset=subtask),
            # Required for pickling: class must be findable via module.class_name
            "__module__": base_class.__module__,
            "__qualname__": class_name,
        }
        if class_attrs:
            attrs.update(class_attrs)

        # Create and register the subclass
        cls = type(class_name, (base_class,), attrs)

        # Make class picklable by adding to the base class's module namespace
        import sys

        setattr(sys.modules[base_class.__module__], class_name, cls)

        register(task_name)(cls)

        # Register variants
        if variants:
            for variant_name, overrides in variants.items():
                register_variant(task_name, variant_name, **overrides)


def get_task_dependencies(specs: list[str]) -> list[str]:
    """Extract and merge dependencies from multiple task specs.

    Collects all runtime dependencies from the specified tasks, merges them,
    and removes duplicates while preserving order.

    Args:
        specs: List of task specifications (e.g., ["my_task", "other_task:variant"]).

    Returns:
        Deduplicated list of package dependencies (preserving order of first occurrence).

    Examples:
        >>> get_task_dependencies(["task_with_deps", "task_without_deps"])
        ["special-lib==1.0", "another-pkg"]
    """
    all_deps: list[str] = []
    for spec in specs:
        task = get_task(spec)
        if task.config.dependencies:
            all_deps.extend(task.config.dependencies)
    # Dedupe preserving order
    return list(dict.fromkeys(all_deps))
