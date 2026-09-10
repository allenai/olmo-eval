"""Compact per-layer realized-K telemetry for adaptive Qwen routing."""

from __future__ import annotations

import atexit
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


class QwenRealizedKTelemetry:
    """Accumulate GPU histograms and periodically persist compact snapshots."""

    def __init__(self, *, output_dir: Path, interval_seconds: float = 30.0) -> None:
        self.output_dir = output_dir
        self.interval_seconds = interval_seconds
        self._counters: dict[int, torch.Tensor] = {}
        self._lock = threading.Lock()
        self._started = False
        self._policy_description = ""
        atexit.register(self.dump)

    def record(
        self,
        *,
        layer_index: int,
        weights: torch.Tensor,
        policy_description: str,
    ) -> None:
        realized_k = torch.count_nonzero(weights, dim=-1)
        histogram = torch.stack(
            [(realized_k == k).sum(dtype=torch.int64) for k in range(weights.shape[-1] + 1)]
        )
        with self._lock:
            counter = self._counters.get(layer_index)
            if counter is None:
                counter = torch.zeros_like(histogram)
                self._counters[layer_index] = counter
            self._policy_description = policy_description
            if not self._started:
                self._started = True
                threading.Thread(
                    target=self._dump_loop,
                    name="qwen-realized-k-telemetry",
                    daemon=True,
                ).start()
        counter.add_(histogram)

    def _dump_loop(self) -> None:
        while True:
            threading.Event().wait(self.interval_seconds)
            self.dump()

    def dump(self) -> None:
        with self._lock:
            counters = sorted(self._counters.items())
            policy_description = self._policy_description
        if not counters:
            return

        layers: list[dict[str, Any]] = []
        for layer_index, device_histogram in counters:
            histogram = [int(value) for value in device_histogram.detach().cpu().tolist()]
            token_count = sum(histogram)
            weighted_total = sum(k * count for k, count in enumerate(histogram))
            layers.append(
                {
                    "layer_index": layer_index,
                    "histogram": histogram,
                    "token_count": token_count,
                    "mean_k": weighted_total / token_count if token_count else None,
                }
            )

        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "policy": policy_description,
            "includes_server_warmup": True,
            "layers": layers,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.output_dir / f"qwen_realized_k_{os.getpid()}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temporary.replace(destination)


_TELEMETRY: QwenRealizedKTelemetry | None = None


def record_qwen_realized_k(
    *,
    layer_index: int,
    weights: torch.Tensor,
    policy_description: str,
) -> None:
    """Record one router call when telemetry is enabled by the launcher."""
    global _TELEMETRY
    if _TELEMETRY is None:
        result_dir = Path(os.environ.get("OLMO_RESULT_DIR", "/results"))
        interval = float(
            os.environ.get(
                "OLMO_EVAL_VLLM_QWEN_REALIZED_K_INTERVAL_SECONDS",
                "30",
            )
        )
        _TELEMETRY = QwenRealizedKTelemetry(
            output_dir=result_dir / "realized_k",
            interval_seconds=interval,
        )
    _TELEMETRY.record(
        layer_index=layer_index,
        weights=weights,
        policy_description=policy_description,
    )
