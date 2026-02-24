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

    The monitor writes to /sandbox_logs/stats.log every 5 seconds with
    human-readable metrics history.

    This path is volume-mounted to {log_dir}/sandboxes/{name}/ on the host.
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

    prefix = f"[{name}] " if name else ""
    monitor_script = get_script("monitor")

    # Write script, start it, verify it's running
    # Try to fix permissions in case host-side chmod didn't work (user namespace mapping)
    start_cmd = f"""
# Try to make /sandbox_logs writable (may fail if not root, that's OK)
chmod 777 /sandbox_logs 2>/dev/null || true
cat > /tmp/_monitor.sh << 'MONITOR_EOF'
{monitor_script}
MONITOR_EOF
chmod +x /tmp/_monitor.sh
/tmp/_monitor.sh </dev/null >/dev/null 2>&1 &
pid=$!
sleep 0.5
if kill -0 $pid 2>/dev/null; then
    echo "OK:$pid"
else
    echo "FAIL"
    # Show why it failed
    ls -la /sandbox_logs/ 2>&1 || echo "/sandbox_logs/ missing"
    id 2>&1 || true
fi
"""
    try:
        resp = await runtime.execute(Command(command=["sh", "-c", start_cmd], timeout=10.0))
        output = resp.stdout.strip() if resp.stdout else ""
        if output.startswith("OK:"):
            pid = output.split(":")[1]
            logger.info(f"{prefix}Started internal monitor (PID {pid})")
            return True
        else:
            logger.error(f"{prefix}Monitor failed to start: {output}")
            return False
    except Exception as e:
        logger.warning(f"{prefix}Failed to start internal monitor: {e}")
        return False
