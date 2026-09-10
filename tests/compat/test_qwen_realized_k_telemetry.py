import json

import torch

from olmo_eval.compat.qwen_realized_k_telemetry import QwenRealizedKTelemetry


def test_realized_k_telemetry_writes_per_layer_histograms(tmp_path) -> None:
    telemetry = QwenRealizedKTelemetry(
        output_dir=tmp_path,
        interval_seconds=3600,
    )
    telemetry.record(
        layer_index=3,
        weights=torch.tensor(
            [
                [0.7, 0.3, 0.0, 0.0],
                [0.4, 0.3, 0.2, 0.1],
            ]
        ),
        policy_description="adaptive_mass(threshold=0.8)",
    )

    telemetry.dump()

    payload = json.loads(next(tmp_path.glob("qwen_realized_k_*.json")).read_text())
    assert payload["policy"] == "adaptive_mass(threshold=0.8)"
    assert payload["includes_server_warmup"] is True
    assert payload["layers"] == [
        {
            "histogram": [0, 0, 1, 0, 1],
            "layer_index": 3,
            "mean_k": 3.0,
            "token_count": 2,
        }
    ]
