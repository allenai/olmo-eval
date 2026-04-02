"""Inference manager - owns lifecycle of inference servers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from olmo_eval.inference.providers.vllm_server_utils import VLLMServerProcess

if TYPE_CHECKING:
    from olmo_eval.inference.providers.config import ProviderConfig

logger = logging.getLogger(__name__)

LOCAL_SERVER_KINDS = frozenset({"vllm_server"})


def _create_server(config: ProviderConfig, gpu_ids: list[int] | None = None) -> VLLMServerProcess:
    """Create server process from config."""
    return VLLMServerProcess(
        model_name=config.model,
        gpu_ids=gpu_ids,
        max_model_len=config.max_model_len,
        dtype=config.dtype,
        tokenizer=config.tokenizer,
        **dict(config.kwargs),
    )


@dataclass
class ServerInfo:
    """Info about running server instance(s)."""

    name: str
    resolved_configs: list[ProviderConfig]
    servers: list[VLLMServerProcess | None]


@dataclass
class InferenceManager:
    """Manages lifecycle of inference servers.

    Handles deployment, GPU allocation, and returns resolved ProviderConfigs
    for registry construction.
    """

    configs: dict[str, ProviderConfig] = field(default_factory=dict)
    available_gpu_ids: list[int] = field(default_factory=list)

    _servers: dict[str, ServerInfo] = field(default_factory=dict, init=False)
    _started: bool = field(default=False, init=False)

    def start(self) -> dict[str, list[dict[str, Any]]]:
        """Start all servers with auto GPU allocation.

        Returns:
            Serialized resolved configs: {name: [config_dict, ...]}
        """
        if self._started:
            return self.get_resolved_configs()

        gpu_pool = list(self.available_gpu_ids)

        for name, config in self.configs.items():
            if config.base_url:
                logger.info(f"Using external server for {name!r}: {config.base_url}")
                resolved = replace(config, num_instances=1)
                self._servers[name] = ServerInfo(
                    name=name,
                    resolved_configs=[resolved],
                    servers=[None],
                )

            elif config.kind in LOCAL_SERVER_KINDS:
                num_instances = config.num_instances
                tensor_parallel = config.kwargs.get("tensor_parallel_size", 1)
                gpus_needed = num_instances * tensor_parallel

                if len(gpu_pool) < gpus_needed:
                    raise RuntimeError(
                        f"Not enough GPUs for {name!r}. "
                        f"Needs {gpus_needed} ({num_instances} instances × {tensor_parallel} TP), "
                        f"available: {len(gpu_pool)}"
                    )

                servers: list[VLLMServerProcess | None] = []
                resolved_configs: list[ProviderConfig] = []

                for i in range(num_instances):
                    instance_gpus = [gpu_pool.pop(0) for _ in range(tensor_parallel)]

                    logger.info(
                        f"Starting server {name!r} instance {i + 1}/{num_instances} "
                        f"(model={config.model}, gpus={instance_gpus})"
                    )

                    server = _create_server(config=config, gpu_ids=instance_gpus)
                    base_url = server.start()
                    servers.append(server)

                    resolved = replace(config, base_url=base_url, num_instances=1)
                    resolved_configs.append(resolved)

                    logger.info(f"Server {name!r} instance {i + 1} ready at {base_url}")

                self._servers[name] = ServerInfo(
                    name=name,
                    resolved_configs=resolved_configs,
                    servers=servers,
                )

            else:
                raise ValueError(
                    f"Unsupported provider kind: {config.kind!r}. "
                    f"Use {sorted(LOCAL_SERVER_KINDS)} or provide base_url."
                )

        self._started = True
        return self.get_resolved_configs()

    def get_resolved_configs(self) -> dict[str, list[dict[str, Any]]]:
        """Get resolved configs for ProviderRegistry construction."""
        return {
            name: [cfg.to_dict() for cfg in info.resolved_configs]
            for name, info in self._servers.items()
        }

    def shutdown(self) -> None:
        """Shutdown all managed servers."""
        if not self._started:
            return

        for name, info in self._servers.items():
            for i, server in enumerate(info.servers):
                if server is not None:
                    try:
                        logger.info(f"Stopping server {name!r} instance {i + 1}")
                        server.stop()
                    except Exception as e:
                        logger.warning(f"Error stopping server {name!r} instance {i + 1}: {e}")

        self._servers.clear()
        self._started = False

    def __enter__(self) -> InferenceManager:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()
