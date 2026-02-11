"""Docker registry mirror utilities for Beaker sandbox jobs."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Registry mirror experiment to query for running mirror nodes
REGISTRY_MIRROR_EXPERIMENT = "johannd/registry-mirror"


def get_registry_mirror_url(workspace: str | None = None) -> str | None:
    """Get the URL of running registry mirror nodes from Beaker.

    Queries the registry-mirror experiment for running jobs and extracts
    the BEAKER_NODE_HOSTNAME from each, returning them as comma-separated
    URLs with port 5000.

    Args:
        workspace: Beaker workspace (uses default if None).

    Returns:
        Comma-separated mirror URLs (e.g., "node1:5000,node2:5000"),
        or None if no mirrors are available.
    """
    try:
        from beaker import Beaker
    except ImportError:
        log.warning("Beaker SDK not available, cannot get registry mirror URL")
        return None

    try:
        beaker = Beaker.from_env(default_workspace=workspace)

        # Get the experiment
        experiment = beaker.experiment.get(REGISTRY_MIRROR_EXPERIMENT)

        # Get all jobs for this experiment
        jobs = list(beaker.experiment.jobs(experiment))

        # Filter for running jobs (not finalized, not canceled)
        mirror_hosts: list[str] = []
        for job in jobs:
            status = job.status
            if status.finalized is not None or status.canceled is not None:
                continue

            # Get BEAKER_NODE_HOSTNAME from env vars
            if job.execution and job.execution.spec and job.execution.spec.env_vars:
                for env_var in job.execution.spec.env_vars:
                    if env_var.name == "BEAKER_NODE_HOSTNAME" and env_var.value:
                        mirror_hosts.append(f"{env_var.value}:5000")
                        break

        if not mirror_hosts:
            log.warning("No running registry mirror nodes found")
            return None

        mirror_url = ",".join(mirror_hosts)
        log.info(f"Found registry mirror(s): {mirror_url}")
        return mirror_url

    except Exception as e:
        log.warning(f"Failed to get registry mirror URL: {e}")
        return None
