"""Runner for external black-box evaluations."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.evals.external import (
    ExternalEvalResult,
    get_external_eval,
    list_external_evals,
)
from olmo_eval.inference.providers.config import ProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class ExternalEvalRunner:
    """Runner for executing external black-box evaluations.

    This runner:
    1. Starts a vLLM server to serve the model
    2. Runs each external evaluation in a sandbox container
    3. Collects and saves results

    Attributes:
        provider_config: Configuration for the inference provider.
        external_eval_names: Names of external evaluations to run.
        output_dir: Directory to write results.
        container_runtime: Container runtime to use (docker or podman).
        server_port: Port for the vLLM server.
        eval_args: Arguments to pass to external evaluations.
    """

    provider_config: ProviderConfig
    external_eval_names: list[str] = field(default_factory=list)
    output_dir: str = BEAKER_RESULT_DIR
    container_runtime: str = "podman"
    server_port: int = 8000
    eval_args: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate runner configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not self.provider_config.model:
            raise ValueError("provider_config.model is required")

        if not self.external_eval_names:
            raise ValueError("At least one external_eval_name is required")

        # Validate that all external evals exist
        available = set(list_external_evals())
        for name in self.external_eval_names:
            if name not in available:
                raise ValueError(
                    f"External eval '{name}' not found. Available: {', '.join(sorted(available))}"
                )

    def run(self) -> dict[str, ExternalEvalResult]:
        """Sync wrapper for async execution."""
        return asyncio.run(self.run_async())

    async def run_async(self) -> dict[str, ExternalEvalResult]:
        """Execute all external evaluations.

        Returns:
            Dictionary mapping evaluation names to results.
        """
        start_time = time.time()
        results: dict[str, ExternalEvalResult] = {}

        # Start the vLLM server if needed and get the actual base_url
        server_process = None
        base_url = self.provider_config.base_url

        if self.provider_config.kind == "vllm_server" and self.provider_config.base_url:
            # Server already running externally, use configured base_url
            pass
        elif self.provider_config.kind in ("vllm", "vllm_server"):
            server_process = self._start_server()
            if server_process is None:
                # Failed to start server - save error results and return
                for name in self.external_eval_names:
                    results[name] = ExternalEvalResult.from_error(
                        name, "Failed to start vLLM server"
                    )
                self._save_results(results, time.time() - start_time)
                return results
            # Get the actual base_url from the provider (includes dynamic port)
            base_url = server_process.base_url

        try:
            # Run each evaluation
            for eval_name in self.external_eval_names:
                logger.info(f"Running external evaluation: {eval_name}")

                try:
                    external_eval = get_external_eval(eval_name)
                    result = await external_eval.execute_with_provider(
                        provider_config=self.provider_config.with_overrides(base_url=base_url),
                        args=self.eval_args,
                        output_dir=self.output_dir,
                        container_runtime=self.container_runtime,
                    )
                    results[eval_name] = result

                    if result.success:
                        logger.info(f"[{eval_name}] Completed successfully")
                        for metric, value in result.metrics.items():
                            logger.info(f"  {metric}: {value}")
                    else:
                        logger.error(f"[{eval_name}] Failed: {result.error}")

                except Exception as e:
                    logger.exception(f"[{eval_name}] Unexpected error")
                    results[eval_name] = ExternalEvalResult.from_error(eval_name, str(e))

        finally:
            # Stop the server if we started it
            if server_process is not None:
                self._stop_server(server_process)

        # Calculate total duration
        total_duration = time.time() - start_time

        # Save combined results
        self._save_results(results, total_duration)

        return results

    def _start_server(self) -> Any | None:
        """Start the vLLM server.

        Returns:
            The provider instance (which manages the server) or None if failed.
        """
        try:
            # Set log_dir to persist vLLM server logs
            log_dir = os.path.join(self.output_dir, "logs")

            # Create provider config for the server without a base_url
            # This causes VLLMServerProvider to start its own server
            server_config = self.provider_config.with_overrides(
                kind="vllm_server",
                base_url=None,
                log_dir=log_dir,
            )

            # Create the provider - this starts the server automatically
            provider = server_config.create_provider()
            if hasattr(provider, "base_url"):
                logger.info(f"Provider ready at {provider.base_url}")
            return provider

        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")

        return None

    def _stop_server(self, server: Any) -> None:
        """Stop the vLLM server.

        Args:
            server: Provider instance that manages the server.
        """
        logger.info("Stopping vLLM server")

        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            logger.warning(f"Error stopping server: {e}")

    def _save_results(
        self,
        results: dict[str, ExternalEvalResult],
        total_duration: float,
    ) -> None:
        """Save combined results to the output directory.

        Args:
            results: Dictionary of evaluation results.
            total_duration: Total time taken for all evaluations.
        """
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build combined results
        combined: dict[str, Any] = {
            "model": self.provider_config.model,
            "model_config": self.provider_config.to_dict(),
            "total_duration_seconds": total_duration,
            "evaluations": {},
        }

        # Add individual evaluation results
        all_metrics: dict[str, float] = {}
        for name, result in results.items():
            combined["evaluations"][name] = result.to_dict()
            # Prefix metrics with evaluation name
            for metric, value in result.metrics.items():
                all_metrics[f"{name}/{metric}"] = value

        combined["metrics"] = all_metrics

        # Write combined results
        results_file = output_path / "external_eval_results.json"
        with open(results_file, "w") as f:
            json.dump(combined, f, indent=2)

        logger.info(f"Combined results saved to {results_file}")

        # Also write metrics.json in the standard format
        metrics_file = output_path / "metrics.json"
        metrics_output: dict[str, Any] = {
            "model_name": self.provider_config.model,
            "evaluations": {},
        }

        for name, result in results.items():
            if result.success:
                metrics_output["evaluations"][name] = {
                    "metrics": result.metrics,
                    "success": True,
                }
            else:
                metrics_output["evaluations"][name] = {
                    "error": result.error,
                    "success": False,
                }

        with open(metrics_file, "w") as f:
            json.dump(metrics_output, f, indent=2)

        logger.info(f"Metrics saved to {metrics_file}")
