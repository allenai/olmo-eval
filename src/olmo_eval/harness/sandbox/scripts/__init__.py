"""Shell scripts for container monitoring."""

from importlib import resources


def get_script(name: str) -> str:
    """Load a shell script by name.

    Args:
        name: Script name without .sh extension (e.g., "monitor", "diagnostics")

    Returns:
        The script content as a string.
    """
    return resources.files(__package__).joinpath(f"{name}.sh").read_text()
