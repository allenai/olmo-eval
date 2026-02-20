"""GPU metrics collection via pynvml.

This module provides optional GPU utilization metrics. If pynvml is not
installed or no NVIDIA GPUs are available, functions return empty results.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime

from .schema import GPUSnapshot

logger = logging.getLogger(__name__)

# Flag to track if pynvml is available and initialized
_nvml_initialized = False
_nvml_available = False


def _ensure_nvml() -> bool:
    """Initialize NVML if not already done.

    Returns:
        True if NVML is available and initialized.
    """
    global _nvml_initialized, _nvml_available

    if _nvml_initialized:
        return _nvml_available

    _nvml_initialized = True

    try:
        import pynvml

        pynvml.nvmlInit()
        _nvml_available = True
        logger.debug("NVML initialized successfully")
    except ImportError:
        logger.debug("pynvml not installed, GPU metrics disabled")
        _nvml_available = False
    except Exception as e:
        logger.debug(f"Failed to initialize NVML: {e}")
        _nvml_available = False

    return _nvml_available


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
