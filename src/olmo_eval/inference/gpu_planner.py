"""GPU allocation planning for inference workers and auxiliary providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.inference.providers.config import ProviderConfig


@dataclass(frozen=True)
class GPUAllocation:
    """GPU allocation for a single component."""

    name: str
    gpu_ids: list[int]
    tensor_parallel_size: int = 1
    num_instances: int = 1

    @property
    def total_gpus(self) -> int:
        return len(self.gpu_ids)


@dataclass(frozen=True)
class GPUPlan:
    """GPU allocation plan for main workers and auxiliary providers."""

    main_workers: list[GPUAllocation]
    auxiliary: dict[str, GPUAllocation]

    @property
    def total_gpus_used(self) -> int:
        main = sum(a.total_gpus for a in self.main_workers)
        aux = sum(a.total_gpus for a in self.auxiliary.values())
        return main + aux

    def get_main_worker_gpus(self, worker_idx: int) -> list[int]:
        if worker_idx >= len(self.main_workers):
            raise IndexError(f"Worker {worker_idx} out of range")
        return self.main_workers[worker_idx].gpu_ids

    def get_auxiliary_gpus(self) -> list[int]:
        gpus = []
        for alloc in self.auxiliary.values():
            gpus.extend(alloc.gpu_ids)
        return gpus


@dataclass
class GPUPlanner:
    """Computes GPU allocation plan for main workers and auxiliary providers."""

    total_gpus: int
    num_main_workers: int
    main_tensor_parallel: int = 1
    auxiliary_configs: dict[str, ProviderConfig] = field(default_factory=dict)

    def plan(self) -> GPUPlan:
        """Compute GPU allocation. Main workers first, then auxiliary."""
        main_gpus_per_worker = self.main_tensor_parallel
        main_total_gpus = self.num_main_workers * main_gpus_per_worker

        # Calculate auxiliary requirements
        aux_requirements: dict[str, tuple[int, int]] = {}  # name -> (num_instances, tp)
        aux_total_gpus = 0

        for name, config in self.auxiliary_configs.items():
            if not getattr(config, "requires_local_gpu", True):
                # API-backed or external server - no GPUs needed
                continue
            num_instances = config.num_instances
            tensor_parallel = config.kwargs.get("tensor_parallel_size", 1)
            gpus_needed = num_instances * tensor_parallel
            aux_requirements[name] = (num_instances, tensor_parallel)
            aux_total_gpus += gpus_needed

        total_needed = main_total_gpus + aux_total_gpus

        if total_needed > self.total_gpus:
            raise RuntimeError(
                f"Not enough GPUs. Need {total_needed} "
                f"({main_total_gpus} for {self.num_main_workers} main worker(s) × "
                f"{main_gpus_per_worker} TP, {aux_total_gpus} for auxiliary), "
                f"but only {self.total_gpus} available."
            )

        # Allocate main workers
        gpu_pool = list(range(self.total_gpus))
        main_allocations: list[GPUAllocation] = []

        for i in range(self.num_main_workers):
            worker_gpus = [gpu_pool.pop(0) for _ in range(main_gpus_per_worker)]
            main_allocations.append(
                GPUAllocation(
                    name=f"main-{i}",
                    gpu_ids=worker_gpus,
                    tensor_parallel_size=self.main_tensor_parallel,
                    num_instances=1,
                )
            )

        # Allocate auxiliary providers
        aux_allocations: dict[str, GPUAllocation] = {}

        for name, (num_instances, tensor_parallel) in aux_requirements.items():
            gpus_needed = num_instances * tensor_parallel
            provider_gpus = [gpu_pool.pop(0) for _ in range(gpus_needed)]
            aux_allocations[name] = GPUAllocation(
                name=name,
                gpu_ids=provider_gpus,
                tensor_parallel_size=tensor_parallel,
                num_instances=num_instances,
            )

        return GPUPlan(main_workers=main_allocations, auxiliary=aux_allocations)

    @classmethod
    def from_harness_config(cls, harness_config: object, total_gpus: int) -> GPUPlanner:
        provider = getattr(harness_config, "provider", None)

        # Calculate main provider GPU requirements based on num_instances
        main_tp = 1
        main_instances = 0
        if provider:
            requires_local = getattr(provider, "requires_local_gpu", True)
            if requires_local:
                main_instances = getattr(provider, "num_instances", 1)
                if hasattr(provider, "kwargs"):
                    main_tp = provider.kwargs.get("tensor_parallel_size", 1)

        aux_configs = dict(getattr(harness_config, "auxiliary_providers", {}) or {})

        return cls(
            total_gpus=total_gpus,
            num_main_workers=main_instances,
            main_tensor_parallel=main_tp,
            auxiliary_configs=aux_configs,
        )


def validate_gpu_plan(plan: GPUPlan, total_gpus: int) -> None:
    """Validate that GPU assignments don't overlap or exceed bounds."""
    all_gpus: set[int] = set()

    for alloc in plan.main_workers:
        for gpu in alloc.gpu_ids:
            if gpu >= total_gpus:
                raise ValueError(f"GPU {gpu} out of range (total: {total_gpus})")
            if gpu in all_gpus:
                raise ValueError(f"GPU {gpu} allocated multiple times")
            all_gpus.add(gpu)

    for name, alloc in plan.auxiliary.items():
        for gpu in alloc.gpu_ids:
            if gpu >= total_gpus:
                raise ValueError(f"GPU {gpu} out of range in {name} (total: {total_gpus})")
            if gpu in all_gpus:
                raise ValueError(f"GPU {gpu} allocated to {name} but already used")
            all_gpus.add(gpu)
