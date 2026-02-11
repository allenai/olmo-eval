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

        # Get the workload (experiment) by name
        workload = beaker.workload.get(REGISTRY_MIRROR_EXPERIMENT)
        experiment = workload.experiment

        # Find running mirror nodes from experiment tasks
        mirror_hosts: list[str] = []
        for task in experiment.tasks:
            # Skip tasks that don't have a running job
            if not task.idle_job:
                continue

            # Get job details to find the node hostname
            job = beaker.job.get(task.idle_job.id)

            # Extract BEAKER_NODE_HOSTNAME from assigned environment variables
            for env_var in job.assignment_details.assigned_environment_variables:
                if env_var.name == "BEAKER_NODE_HOSTNAME" and env_var.literal:
                    mirror_hosts.append(f"{env_var.literal}:5000")
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
