"""Beaker commands for olmo-eval CLI."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import click
from rich.table import Table

from olmo_eval.cli.beaker.group import group
from olmo_eval.cli.beaker.launch import launch
from olmo_eval.cli.beaker.watch import watch
from olmo_eval.cli.utils import console
from olmo_eval.common.constants.infrastructure import BEAKER_KNOWN_CLUSTERS

if TYPE_CHECKING:
    from beaker import Beaker
    from beaker import beaker_pb2 as pb2


@click.group()
def beaker() -> None:
    """Beaker job management commands.

    Commands for launching, monitoring, and managing evaluation jobs on Beaker.
    """
    pass


def _get_tag_value(tags: list[str], prefix: str) -> str | None:
    """Extract value from a tag like 'gpu:h100' -> 'h100'."""
    for tag in tags:
        if tag.startswith(f"{prefix}:"):
            return tag.split(":", 1)[1]
    return None


def _format_memory(memory_bytes: int) -> str:
    """Format memory bytes as human-readable string."""
    gb = memory_bytes / (1024**3)
    return f"{gb:.0f}GB"


def _get_cluster_aliases(cluster_name: str) -> list[str]:
    """Get list of aliases that include this cluster."""
    aliases = []
    for alias, cluster_list in BEAKER_KNOWN_CLUSTERS.items():
        if cluster_name in cluster_list:
            aliases.append(alias)
    return sorted(aliases)


def _fetch_cluster_details(
    b: "Beaker", cluster_name: str
) -> tuple[str, "pb2.Cluster | None", list, str | None, int]:
    """Fetch cluster details and nodes."""
    try:
        # Use get() to fetch full cluster details including queue size
        cluster = b.cluster.get(cluster_name)
        nodes = list(b.node.list(cluster=cluster))
        if nodes:
            res = nodes[0].node_resources
            gpus_per_node = len(res.gpu_ids)
            memory_str = _format_memory(res.memory_bytes) if res.memory_bytes else None
            return cluster_name, cluster, nodes, memory_str, gpus_per_node
        return cluster_name, cluster, [], None, 0
    except Exception:
        return cluster_name, None, [], None, 0


SORT_OPTIONS = ["free", "name", "util", "queue", "total", "used", "jobs"]


@beaker.command()
@click.option("--filter", "-f", default="", help="Filter by alias or cluster name")
@click.option("--aliases", "-a", is_flag=True, help="Only show cluster aliases")
@click.option(
    "--sort",
    "-s",
    type=click.Choice(SORT_OPTIONS),
    default="free",
    help="Sort by column (default: free)",
)
@click.option("--reverse", "-r", is_flag=True, help="Reverse sort order")
def clusters(filter: str, aliases: bool, sort: str, reverse: bool) -> None:
    """List all Beaker clusters with utilization details."""
    # Aliases-only mode: no Beaker API calls needed
    if aliases:
        console.print()
        alias_table = Table(title="Cluster Aliases (for olmo-eval beaker launch)")
        alias_table.add_column("Alias", style="cyan")
        alias_table.add_column("Clusters", style="dim")

        for alias, cluster_list in sorted(BEAKER_KNOWN_CLUSTERS.items()):
            match_alias = filter.lower() in alias.lower()
            match_cluster = any(filter.lower() in c.lower() for c in cluster_list)
            if not filter or match_alias or match_cluster:
                alias_table.add_row(alias, ", ".join(cluster_list))

        console.print(alias_table)
        return

    try:
        from beaker import Beaker
    except ImportError:
        console.print("[red]Error:[/red] beaker package not installed")
        console.print("Install with: [cyan]pip install beaker-py[/cyan]")
        raise SystemExit(1) from None

    b = Beaker.from_env()

    # Data structures
    cluster_total_gpus: dict[str, int] = {}
    cluster_gpus_per_node: dict[str, int] = {}
    cluster_memory: dict[str, str] = {}
    cluster_used_gpus: dict[str, int] = defaultdict(int)
    cluster_running_jobs: dict[str, int] = defaultdict(int)
    node_to_cluster: dict[str, str] = {}  # node_id -> cluster_id mapping

    with console.status("Fetching cluster info from Beaker..."):
        # Get cluster IDs from list(), then fetch full details with get() in parallel
        cluster_ids = [c.id for c in b.cluster.list()]
        cluster_names: dict[str, str] = {}
        all_clusters: dict[str, Any] = {}

        # Fetch full cluster details and nodes in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_fetch_cluster_details, b, cid): cid for cid in cluster_ids}
            for future in as_completed(futures):
                cluster_id, cluster, nodes, memory_str, gpus_per_node = future.result()
                if cluster is None:
                    continue
                all_clusters[cluster_id] = cluster
                cluster_names[cluster_id] = f"{cluster.organization_name}/{cluster.name}"
                cluster_gpus_per_node[cluster_id] = gpus_per_node
                if nodes:
                    cluster_total_gpus[cluster_id] = sum(
                        len(n.node_resources.gpu_ids) for n in nodes
                    )
                    # Build node -> cluster mapping for job lookups
                    for node in nodes:
                        node_to_cluster[node.id] = cluster_id
                else:
                    cluster_total_gpus[cluster_id] = 0
                if memory_str:
                    cluster_memory[cluster_id] = memory_str

        # Fetch jobs and use cached node mapping (no per-job API calls!)
        jobs = list(b.job.list(finalized=False, scheduled=True, limit=2000))
        for job in jobs:
            ad = job.assignment_details
            if ad and ad.node_id:
                cluster_id = node_to_cluster.get(ad.node_id)
                if cluster_id:
                    cluster_running_jobs[cluster_id] += 1
                    if ad.resource_assignment:
                        cluster_used_gpus[cluster_id] += len(ad.resource_assignment.gpus)

    # Print aliases table
    console.print()
    alias_table = Table(title="Cluster Aliases (for olmo-eval beaker launch)")
    alias_table.add_column("Alias", style="cyan")
    alias_table.add_column("Clusters", style="dim")

    for alias, cluster_list in sorted(BEAKER_KNOWN_CLUSTERS.items()):
        match_alias = filter.lower() in alias.lower()
        match_cluster = any(filter.lower() in c.lower() for c in cluster_list)
        if not filter or match_alias or match_cluster:
            alias_table.add_row(alias, ", ".join(cluster_list))

    console.print(alias_table)

    # Collect cluster data for sorting
    cluster_rows: list[dict] = []
    for cluster_id in all_clusters:
        cluster = all_clusters[cluster_id]
        name = cluster_names[cluster_id]

        # Apply filter
        if filter:
            match_name = filter.lower() in name.lower()
            cluster_aliases = _get_cluster_aliases(name)
            match_alias = any(filter.lower() in a.lower() for a in cluster_aliases)
            if not match_name and not match_alias:
                continue

        tags = list(cluster.tags)
        gpu_type = _get_tag_value(tags, "gpu") or "-"
        cluster_aliases = _get_cluster_aliases(name)
        aliases_str = ", ".join(cluster_aliases) if cluster_aliases else "-"

        total = cluster_total_gpus.get(cluster_id, 0)
        gpus_per_node = cluster_gpus_per_node.get(cluster_id, 0)
        memory = cluster_memory.get(cluster_id, "-")
        used = cluster_used_gpus.get(cluster_id, 0)
        free = total - used
        jobs_count = cluster_running_jobs.get(cluster_id, 0)
        queue_size = cluster.cluster_job_queue_size
        util_pct = (used / total * 100) if total > 0 else 0

        if total > 0:
            if util_pct >= 90:
                util_str = f"[red]{util_pct:.0f}%[/red]"
            elif util_pct >= 70:
                util_str = f"[yellow]{util_pct:.0f}%[/yellow]"
            else:
                util_str = f"[green]{util_pct:.0f}%[/green]"
        else:
            util_str = "-"

        cluster_rows.append(
            {
                "name": name,
                "aliases_str": aliases_str,
                "gpu_type": gpu_type.upper(),
                "gpus_per_node": gpus_per_node,
                "node_count": cluster.node_count,
                "total": total,
                "used": used,
                "free": free,
                "util_pct": util_pct,
                "util_str": util_str,
                "queue_size": queue_size,
                "jobs_count": jobs_count,
                "memory": memory,
            }
        )

    # Sort clusters - numeric columns default to descending, name defaults to ascending
    sort_key_map = {
        "free": lambda r: r["free"],
        "name": lambda r: r["name"].lower(),
        "util": lambda r: r["util_pct"],
        "queue": lambda r: r["queue_size"],
        "total": lambda r: r["total"],
        "used": lambda r: r["used"],
        "jobs": lambda r: r["jobs_count"],
    }
    # For numeric sorts, default is descending (highest first); for name, ascending
    default_descending = sort != "name"
    sort_descending = not default_descending if reverse else default_descending
    cluster_rows.sort(key=sort_key_map[sort], reverse=sort_descending)

    # Print cluster details table
    console.print()
    detail_table = Table(title="All Beaker Clusters")
    detail_table.add_column("Cluster", style="cyan")
    detail_table.add_column("Aliases", style="dim")
    detail_table.add_column("GPU", style="yellow")
    detail_table.add_column("GPUs/Node", justify="right")
    detail_table.add_column("Nodes", justify="right")
    detail_table.add_column("Total", justify="right")
    detail_table.add_column("Used", justify="right", style="red")
    detail_table.add_column("Free", justify="right", style="green")
    detail_table.add_column("Util", justify="right")
    detail_table.add_column("Queue", justify="right", style="yellow")
    detail_table.add_column("Jobs", justify="right", style="dim")
    detail_table.add_column("Memory", justify="right", style="dim")

    for row in cluster_rows:
        detail_table.add_row(
            row["name"],
            row["aliases_str"],
            row["gpu_type"],
            str(row["gpus_per_node"]) if row["gpus_per_node"] else "-",
            str(row["node_count"]),
            str(row["total"]) if row["total"] else "-",
            str(row["used"]) if row["used"] else "-",
            str(row["free"]) if row["total"] else "-",
            row["util_str"],
            str(row["queue_size"]),
            str(row["jobs_count"]) if row["jobs_count"] else "-",
            row["memory"],
        )

    console.print(detail_table)


# Register subcommands
beaker.add_command(launch)
beaker.add_command(watch)
beaker.add_command(group)

__all__ = ["beaker", "clusters", "launch", "watch", "group"]
