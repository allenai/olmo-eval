"""Shared Rich console singleton.

Lives in `common` (not `cli`) so non-CLI modules can import it without pulling in
the CLI package and triggering circular imports through `harness.presets`.
"""

from rich.console import Console

console = Console(force_terminal=True, width=120)
