"""GPU metrics collection via pynvml.

This module provides optional GPU utilization metrics. If pynvml is not
installed or no NVIDIA GPUs are available, functions return empty results.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from datetime import UTC, datetime

from .schema import GPUSnapshot

logger = logging.getLogger(__name__)

# Flag to track if pynvml is available and initialized
_nvml_initialized = False
_nvml_available = False
_nvml_error: str | None = None


def _ensure_nvml() -> bool:
    """Initialize NVML if not already done.

    Returns:
        True if NVML is available and initialized.
    """
    global _nvml_initialized, _nvml_available, _nvml_error

    if _nvml_initialized:
        return _nvml_available

    _nvml_initialized = True

    try:
        import pynvml

        pynvml.nvmlInit()
        _nvml_available = True
        _nvml_error = None
        logger.debug("NVML initialized successfully")
    except ImportError:
        _nvml_error = "pynvml not installed"
        logger.warning(f"GPU metrics disabled: {_nvml_error}")
        _nvml_available = False
    except Exception as e:
        _nvml_error = str(e)
        logger.warning(f"GPU metrics disabled: NVML init failed: {_nvml_error}")
        _nvml_available = False

    return _nvml_available


class GPUMonitor:
    """Background thread that samples GPU metrics at regular intervals.

    Usage:
        monitor = GPUMonitor(interval_s=1.0)
        monitor.start()
        # ... do inference work ...
        snapshots = monitor.stop()  # Returns all collected snapshots
    """

    def __init__(self, interval_s: float = 1.0) -> None:
        """Initialize the GPU monitor.

        Args:
            interval_s: Sampling interval in seconds.
        """
        self._interval_s = interval_s
        self._snapshots: list[GPUSnapshot] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start background sampling thread."""
        if not _ensure_nvml():
            return

        self._stop_event.clear()
        self._snapshots.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[GPUSnapshot, ...]:
        """Stop sampling and return collected snapshots.

        Returns:
            Tuple of all GPU snapshots collected during monitoring.
        """
        if self._thread is None:
            return ()

        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None

        with self._lock:
            return tuple(self._snapshots)

    def _sample_loop(self) -> None:
        """Background sampling loop."""
        while not self._stop_event.is_set():
            snapshots = collect_gpu_snapshots()
            if snapshots:
                with self._lock:
                    self._snapshots.extend(snapshots)
            # Wait for next sample, but check stop_event frequently
            self._stop_event.wait(timeout=self._interval_s)


def collect_gpu_snapshots() -> tuple[GPUSnapshot, ...]:
    """Collect current GPU utilization snapshots.

    Returns a snapshot for each available NVIDIA GPU with:
    - Device ID and name
    - GPU utilization percentage
    - Memory usage (used/total)
    - Temperature (if available)
    - Power draw (if available)

    Returns:
        Tuple of GPUSnapshot objects, empty if no GPUs or pynvml unavailable.
    """
    if not _ensure_nvml():
        return ()

    try:
        import pynvml

        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return ()

        snapshots: list[GPUSnapshot] = []
        now = datetime.now(UTC)

        for i in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                snapshot = _collect_device_snapshot(handle, i, now)
                snapshots.append(snapshot)
            except Exception as e:
                logger.debug(f"Failed to collect metrics for GPU {i}: {e}")

        return tuple(snapshots)

    except Exception as e:
        logger.debug(f"Failed to collect GPU snapshots: {e}")
        return ()


def _collect_device_snapshot(
    handle: object,
    device_id: int,
    timestamp: datetime,
) -> GPUSnapshot:
    """Collect snapshot for a single GPU device.

    Args:
        handle: NVML device handle.
        device_id: GPU device index.
        timestamp: Timestamp for the snapshot.

    Returns:
        GPUSnapshot with device metrics.
    """
    import pynvml

    # Get device name
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode("utf-8")

    # Get utilization
    try:
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        utilization_pct = float(utilization.gpu)
    except Exception:
        utilization_pct = 0.0

    # Get memory info
    try:
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used_mb = memory.used / (1024 * 1024)
        memory_total_mb = memory.total / (1024 * 1024)
    except Exception:
        memory_used_mb = 0.0
        memory_total_mb = 0.0

    # Get temperature (optional)
    temperature_c: float | None = None
    with contextlib.suppress(Exception):
        temperature_c = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))

    # Get power usage (optional)
    power_watts: float | None = None
    with contextlib.suppress(Exception):
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_watts = power_mw / 1000.0

    return GPUSnapshot(
        device_id=device_id,
        name=name,
        utilization_pct=utilization_pct,
        memory_used_mb=memory_used_mb,
        memory_total_mb=memory_total_mb,
        temperature_c=temperature_c,
        power_watts=power_watts,
        timestamp=timestamp,
    )


def shutdown_nvml() -> None:
    """Shutdown NVML if it was initialized.

    Safe to call even if NVML was never initialized.
    """
    global _nvml_initialized, _nvml_available

    if _nvml_available:
        try:
            import pynvml

            pynvml.nvmlShutdown()
            logger.debug("NVML shutdown successfully")
        except Exception as e:
            logger.debug(f"Failed to shutdown NVML: {e}")

    _nvml_initialized = False
    _nvml_available = False


def is_gpu_available() -> bool:
    """Check if GPU metrics collection is available.

    Returns:
        True if pynvml is available and at least one GPU is present.
    """
    if not _ensure_nvml():
        return False

    try:
        import pynvml

        return pynvml.nvmlDeviceGetCount() > 0
    except Exception:
        return False


def get_gpu_count() -> int:
    """Get the number of available GPUs.

    Returns:
        Number of GPUs, 0 if none or pynvml unavailable.
    """
    if not _ensure_nvml():
        return 0

    try:
        import pynvml

        return pynvml.nvmlDeviceGetCount()
    except Exception:
        return 0
