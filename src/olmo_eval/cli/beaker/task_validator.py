"""Task validation and priority grouping for Beaker launch."""

from __future__ import annotations

from rich.console import Console

console = Console()


class TaskValidator:
    """Validates tasks and groups them by priority."""

    def __init__(
        self,
        task_specs: list[str],
        cli_priority: str | None,
        default_priority: str,
    ):
        """Initialize the validator.

        Args:
            task_specs: List of task specifications (may include @priority suffixes).
            cli_priority: Priority specified via CLI (if any).
            default_priority: Default priority to use.
        """
        self.task_specs = task_specs
        self.cli_priority = cli_priority
        self.default_priority = default_priority

    def validate_and_group(self) -> tuple[dict[str, list[str]], list[str], set[str]]:
        """Validate tasks and group by priority.

        Returns:
            Tuple of (tasks_by_priority, valid_tasks, agent_task_specs).

        Raises:
            SystemExit: If any tasks are invalid.
        """
        from olmo_eval.core.configs import expand_tasks, validate_tasks
        from olmo_eval.evals.tasks import AgentTask
        from olmo_eval.evals.tasks import get_task as get_task_for_classification
        from olmo_eval.launch import validate_priority_configuration

        # Group by priority WITHOUT expanding first
        try:
            tasks_by_priority = validate_priority_configuration(
                tasks=self.task_specs,
                cli_priority=self.cli_priority,
                default_priority=self.default_priority,
            )
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1) from None

        # Get all specs (without @priority suffix)
        all_task_specs = [t for tasks in tasks_by_priority.values() for t in tasks]

        # Expand for validation only
        expanded_for_validation = expand_tasks(all_task_specs)
        valid_tasks, invalid_tasks = validate_tasks(expanded_for_validation)

        # Detect agent tasks
        agent_task_specs: set[str] = set()
        for task_spec in expanded_for_validation:
            try:
                task_instance = get_task_for_classification(task_spec)
                if isinstance(task_instance, AgentTask):
                    agent_task_specs.add(task_spec)
            except Exception:
                pass

        if invalid_tasks:
            console.print("[red]Error:[/red] The following tasks/suites do not exist:")
            for inv in invalid_tasks:
                console.print(f"  - {inv}")
            console.print("\nUse 'olmo-eval tasks' to see available tasks.")
            console.print("Use 'olmo-eval suites' to see available suites.")
            raise SystemExit(1) from None

        return tasks_by_priority, valid_tasks, agent_task_specs

    def is_agent_spec(self, spec: str, agent_task_specs: set[str]) -> bool:
        """Check if a task spec is an agent task.

        Args:
            spec: Task specification (potentially with priority suffix).
            agent_task_specs: Set of known agent task specs.

        Returns:
            True if all expanded tasks are agent tasks.
        """
        from olmo_eval.core.configs import expand_tasks
        from olmo_eval.evals.tasks import get_base_task_name

        base_spec = get_base_task_name(spec)
        expanded = expand_tasks([base_spec])
        return all(t in agent_task_specs for t in expanded)

    def get_expanded_counts_by_priority(
        self, tasks_by_priority: dict[str, list[str]]
    ) -> dict[str, int]:
        """Get expanded task counts per priority level.

        Args:
            tasks_by_priority: Dict mapping priority -> list of task specs.

        Returns:
            Dict mapping priority -> expanded task count.
        """
        from olmo_eval.core.configs import expand_tasks

        counts: dict[str, int] = {}
        for priority_level, specs in tasks_by_priority.items():
            counts[priority_level] = len(expand_tasks(specs))
        return counts
