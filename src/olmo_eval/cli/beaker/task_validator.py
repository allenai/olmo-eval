"""Task validation and priority grouping for Beaker launch."""

from __future__ import annotations

from rich.console import Console

console = Console()


class TaskValidator:
    """Validates tasks and groups them by priority."""

    def __init__(
        self,
        task_specs: list[str],
        default_priority: str = "normal",
    ):
        """Initialize the validator.

        Args:
            task_specs: List of task specifications (may include @priority suffixes).
            default_priority: Default priority for tasks without @priority suffix.
        """
        self.task_specs = task_specs
        self.default_priority = default_priority

    def validate_and_group(self) -> tuple[dict[str, list[str]], list[str]]:
        """Validate tasks and group by priority.

        Returns:
            Tuple of (tasks_by_priority, valid_tasks).

        Raises:
            SystemExit: If any tasks are invalid or have no metrics configured.
        """
        from olmo_eval.common.configs import expand_tasks, validate_task_metrics, validate_tasks
        from olmo_eval.launch import validate_priority_configuration

        # Group by priority WITHOUT expanding first
        tasks_by_priority = validate_priority_configuration(
            tasks=self.task_specs,
            default_priority=self.default_priority,
        )

        # Get all specs (without @priority suffix)
        all_task_specs = [t for tasks in tasks_by_priority.values() for t in tasks]

        # Expand for validation only
        expanded_for_validation = expand_tasks(all_task_specs)
        valid_tasks, invalid_tasks = validate_tasks(expanded_for_validation)

        if invalid_tasks:
            console.print("[red]Error:[/red] The following tasks/suites do not exist:")
            for inv in invalid_tasks:
                console.print(f"  - {inv}")
            console.print("\nUse 'olmo-eval tasks' to see available tasks.")
            console.print("Use 'olmo-eval suites' to see available suites.")
            raise SystemExit(1) from None

        # Check for tasks without metrics configured
        _with_metrics, without_metrics = validate_task_metrics(valid_tasks)
        if without_metrics:
            console.print("[red]Error:[/red] The following tasks have no metrics configured:")
            for spec in without_metrics:
                console.print(f"  - {spec}")
            console.print(
                "\n[yellow]Hint:[/yellow] Tasks need metrics to score instances. "
                "Use a variant with metrics (e.g., 'humaneval:bpb') or register "
                "metrics for the base task."
            )
            raise SystemExit(1) from None

        return tasks_by_priority, valid_tasks

    def get_expanded_counts_by_priority(
        self, tasks_by_priority: dict[str, list[str]]
    ) -> dict[str, int]:
        """Get expanded task counts per priority level.

        Args:
            tasks_by_priority: Dict mapping priority -> list of task specs.

        Returns:
            Dict mapping priority -> expanded task count.
        """
        from olmo_eval.common.configs import expand_tasks

        counts: dict[str, int] = {}
        for priority_level, specs in tasks_by_priority.items():
            counts[priority_level] = len(expand_tasks(specs))
        return counts
