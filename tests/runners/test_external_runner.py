"""Tests for external evaluation runner UX."""

import logging

from olmo_eval.evals.external.result import ExternalEvalResult
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.external.runner import ExternalEvalRunner


class _DummyEval:
    async def execute_with_provider(self, **_: object) -> ExternalEvalResult:
        return ExternalEvalResult(
            name="dummy_eval",
            success=False,
            metrics={"resolve_rate": 0.125, "resolved": 1.0, "total": 8.0},
        )


def test_external_runner_logs_useful_failure_reason(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        "olmo_eval.runners.external.runner.get_external_eval",
        lambda name: _DummyEval(),
    )

    runner = ExternalEvalRunner(
        provider_config=ProviderConfig(kind="mock", model="test-model"),
        external_eval_names=["dummy_eval"],
        output_dir=str(tmp_path),
    )

    with caplog.at_level(logging.ERROR):
        results = runner.run()

    assert results["dummy_eval"].success is False
    assert "Failed: Unknown error (metrics: resolve_rate=0.1250, resolved=1, total=8)" in (
        caplog.text
    )
