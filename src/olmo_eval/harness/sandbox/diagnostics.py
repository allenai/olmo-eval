"""Diagnostics module - starts background monitor inside container."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def start_internal_monitor(
    runtime: Any,
    name: str | None = None,
) -> bool:
    """Start background monitoring process inside the container.

    The monitor writes to /sandbox_logs/ every 5 seconds:
      - stats.log:    Human-readable metrics history (appended)
      - metrics.json: JSON snapshot of latest metrics (overwritten)

    These paths are volume-mounted to {log_dir}/sandboxes/{name}/ on the host.
    When the container becomes unresponsive, read the files directly from
    the host filesystem - no exec needed.

    Args:
        runtime: The swerex runtime instance.
        name: Sandbox name for logging.

    Returns:
        True if monitor started successfully.
    """
    if runtime is None:
        return False

    from swerex.runtime.abstract import Command

    from .scripts import get_script

    monitor_script = get_script("monitor")
    start_cmd = f"""
cat > /tmp/_monitor.sh << 'MONITOR_EOF'
{monitor_script}
MONITOR_EOF
chmod +x /tmp/_monitor.sh
nohup /tmp/_monitor.sh > /dev/null 2>&1 &
echo "Monitor PID: $!"
"""

    prefix = f"[{name}] " if name else ""
    try:
        resp = await runtime.execute(Command(command=["sh", "-c", start_cmd], timeout=10.0))
        output = resp.stdout.strip() if resp.stdout else "OK"
        logger.info(f"{prefix}Started internal monitor: {output}")
        return True
    except Exception as e:
        logger.warning(f"{prefix}Failed to start internal monitor: {e}")
        return False
