"""Reporter registry with lazy loading."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocol import MetricsReporter


class ReporterRegistry:
    """Registry for metrics reporters with lazy loading."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], MetricsReporter]] = {}
        self._register_builtin()

    def _register_builtin(self) -> None:
        """Register built-in reporters."""

        def console_factory() -> MetricsReporter:
            from ..reporters.console import ConsoleReporter

            return ConsoleReporter()

        def jsonl_factory() -> MetricsReporter:
            from ..reporters.jsonl import JSONLReporter

            return JSONLReporter()

        self._factories["console"] = console_factory
        self._factories["jsonl"] = jsonl_factory

    def register(self, name: str, factory: Callable[[], MetricsReporter]) -> None:
        """Register a reporter factory.

        Args:
            name: Reporter name (e.g., "console", "postgres").
            factory: Callable that creates a reporter instance.
        """
        self._factories[name] = factory

    def create(self, name_or_config: str | dict[str, Any]) -> MetricsReporter:
        """Create a reporter instance.

        Args:
            name_or_config: Reporter name or dict with 'name' and config options.

        Returns:
            Configured reporter instance.

        Raises:
            KeyError: If reporter name is not registered.
        """
        if isinstance(name_or_config, str):
            name = name_or_config
            config: dict[str, Any] = {}
        else:
            name = name_or_config.get("name", "console")
            config = {k: v for k, v in name_or_config.items() if k != "name"}

        if name not in self._factories:
            raise KeyError(f"Unknown reporter: {name}. Available: {list(self._factories.keys())}")

        reporter = self._factories[name]()
        if config:
            reporter.configure(**config)
        return reporter

    def available(self) -> list[str]:
        """List available reporter names."""
        return list(self._factories.keys())


# Global registry instance
reporter_registry = ReporterRegistry()
