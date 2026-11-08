"""Each inference worker scans for vLLM's distributed port from its own base."""

from __future__ import annotations

import os

import pytest

from olmo_eval.runners.asynq.workers import VLLM_PORT_BASE, VLLM_PORTS_PER_GPU, set_vllm_port


def test_each_worker_gets_its_own_vllm_port_base(monkeypatch: pytest.MonkeyPatch) -> None:
    bases = []
    for gpus in ([0], [1], [2, 3], [7]):
        monkeypatch.delenv("VLLM_PORT", raising=False)
        set_vllm_port(gpus)
        bases.append(int(os.environ["VLLM_PORT"]))

    assert bases == [VLLM_PORT_BASE + VLLM_PORTS_PER_GPU * g for g in (0, 1, 2, 7)]
    assert max(bases) + VLLM_PORTS_PER_GPU <= 32768  # below Linux's ephemeral range


def test_a_set_vllm_port_and_cpu_workers_are_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_PORT", "41000")
    set_vllm_port([3])
    assert os.environ["VLLM_PORT"] == "41000"

    monkeypatch.delenv("VLLM_PORT")
    set_vllm_port([])
    assert "VLLM_PORT" not in os.environ
