"""Terminal-Bench specific tools."""

from __future__ import annotations

from .registry import registered_tool


@registered_tool(
    name="submit",
    description="Call this tool when you have completed the task.",
)
async def submit() -> str:
    """Signal that the task is complete.

    Call this tool when you have finished working on the task and are
    ready to submit your solution for verification.

    Returns:
        Confirmation message.
    """
    return "Task submitted successfully. Your work will now be verified."
