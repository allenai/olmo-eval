from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[2] / "scripts" / "adaptive_experts" / "router_profile.py"
SPEC = importlib.util.spec_from_file_location("router_profile", SCRIPT)
assert SPEC and SPEC.loader
router_profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router_profile)


def test_uniform_router_metrics() -> None:
    logits = torch.zeros(2, 128)
    values, top_ids, top_probs = router_profile.metrics_from_logits(logits)
    metric = dict(zip(router_profile.METRICS, values[0].tolist(), strict=True))
    assert values.shape == (2, len(router_profile.METRICS))
    assert top_ids.shape == top_probs.shape == (2, 16)
    assert metric["c1"] == 1 / 128
    assert metric["c8"] == 8 / 128
    assert metric["renorm8"] == 16
    assert abs(metric["effective_experts"] - 128) < 1e-4
    assert metric["prob_margin_8_9"] == 0


def test_sample_token_respects_top_k() -> None:
    logits = torch.arange(10, dtype=torch.float32).unsqueeze(0)
    generator = torch.Generator().manual_seed(7)
    draws = {
        int(
            router_profile.sample_token(
                logits, temperature=1.0, top_p=1.0, top_k=2, generator=generator
            ).item()
        )
        for _ in range(100)
    }
    assert draws <= {8, 9}


def test_router_stats_batches_layers_and_merges() -> None:
    first = router_profile.RouterStats(num_layers=2, num_experts=128, sample_size=4, seed=1)
    second = router_profile.RouterStats(num_layers=2, num_experts=128, sample_size=4, seed=2)
    logits = (torch.zeros(3, 128), torch.ones(3, 128))
    first.update(logits, phase="prompt", token_offset=0, example_key="a:0")
    second.update(logits, phase="reasoning", token_offset=0, example_key="a:1")
    first.merge(second)
    assert first.count[0].tolist() == [3, 3]
    assert first.count[1].tolist() == [3, 3]
    assert len(first.samples) == 4
