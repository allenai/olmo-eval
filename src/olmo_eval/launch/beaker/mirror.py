"""Docker registry mirror utilities for Beaker sandbox jobs."""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

# Registry mirror experiment to query for running mirror nodes
REGISTRY_MIRROR_EXPERIMENT = "johannd/registry-mirror"

MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


def get_registry_mirror_url() -> str:
    """Get the URL of running registry mirror nodes from Beaker.

    Queries the registry-mirror experiment for running jobs and extracts
    the BEAKER_NODE_HOSTNAME from each, returning them as comma-separated
    URLs with port 5000.

    Only call this function if registry mirrors are required. If mirrors
    are not needed, don't call this function.

    Returns:
        Comma-separated mirror URLs (e.g., "node1:5000,node2:5000").

    Raises:
        RuntimeError: If registry mirror lookup fails after retries.
    """
    from beaker import Beaker

    last_error: Exception | None = None
    default_workspace = "ai2/sweagent-test"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            beaker = Beaker.from_env(default_workspace=default_workspace)

            # Get the workload (experiment) by name
            workload = beaker.workload.get(REGISTRY_MIRROR_EXPERIMENT)
            experiment = workload.experiment

            # Find running mirror nodes from experiment tasks
            mirror_hosts: list[str] = []
            for task in experiment.tasks:
                job = list(beaker.job.list(task=task))[0]
                # Extract BEAKER_NODE_HOSTNAME from assigned environment variables
                for env_var in job.assignment_details.assigned_environment_variables:
                    if env_var.name == "BEAKER_NODE_HOSTNAME" and env_var.literal:
                        mirror_hosts.append(f"{env_var.literal}:5000")
                        break

            if not mirror_hosts:
                raise RuntimeError("No running registry mirror nodes found")

            mirror_url = ",".join(mirror_hosts)
            log.info(f"Found registry mirror(s): {mirror_url}")
            return mirror_url

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                log.warning(
                    f"Failed to get registry mirror URL (attempt {attempt}/{MAX_RETRIES}): {e}"
                )
                time.sleep(RETRY_DELAY)
            else:
                log.error(f"Failed to get registry mirror URL after {MAX_RETRIES} attempts: {e}")

    raise RuntimeError(
        f"Failed to get registry mirror URL after {MAX_RETRIES} attempts: {last_error}"
    )
