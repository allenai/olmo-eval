import logging

from olmo_eval.evals.suites.registry import AggregationStrategy, make_suite
from olmo_eval.evals.tasks.state_bench_lc import (
    STATE_BENCH_V1,
    STATE_BENCH_V2,
    StateBenchRelease,
)

logger = logging.getLogger(__name__)


def _all_strata_note(release: StateBenchRelease, span: str) -> str:
    return (
        f" Spans every token stratum {span}; strata longer than the model's context"
        " are left-truncated by the provider, so select stratum suites"
        f" (e.g. {release.task_prefix}:tokens_64k) within the model's context instead."
    )


def _coverage_note(tasks: tuple[str, ...], total: int) -> str:
    """Describe partial config coverage so a narrowed suite is visible, not silent."""
    if len(tasks) == total:
        return ""
    return f" Covers {len(tasks)} of {total} configs; the rest have no split in this tier."


def _make_suite_if_nonempty(
    name: str,
    tasks: tuple[str, ...],
    total: int,
    description: str,
) -> None:
    """Register a suite, skipping (and logging) one whose configs all lack the split."""
    if not tasks:
        logger.info(
            "Skipping state-bench suite %r: none of the %d candidate configs have a "
            "staged split for this token tier.",
            name,
            total,
        )
        return
    make_suite(
        name=name,
        tasks=tasks,
        aggregation=AggregationStrategy.AVERAGE,
        description=description + _coverage_note(tasks, total),
    )


def _register_release_suites(release: StateBenchRelease, title: str, span: str) -> None:
    """Register the full, complexity, formatter and token-tier suites for one release."""
    prefix = release.task_prefix
    sample_prefix = f"{prefix}_10pct"
    note = _all_strata_note(release, span)

    def full_and_sample(
        suffix: str,
        configs: tuple[str, ...],
        description: str,
        sample_description: str,
    ) -> None:
        pairs = tuple(
            (config_name, token_stratum)
            for config_name in configs
            for token_stratum in release.strata_for_config(config_name)
        )
        make_suite(
            name=prefix + suffix,
            tasks=tuple(release.task_name(*pair) for pair in pairs),
            aggregation=AggregationStrategy.AVERAGE,
            description=description + note,
        )
        make_suite(
            name=sample_prefix + suffix,
            tasks=tuple(release.task_name_10pct(*pair) for pair in pairs),
            aggregation=AggregationStrategy.AVERAGE,
            description=sample_description + note,
        )

    def tier(suffix: str, configs: tuple[str, ...], token_stratum: str, what: str) -> None:
        staged = release.configs_for_stratum(token_stratum)
        tier_configs = tuple(config_name for config_name in configs if config_name in staged)
        _make_suite_if_nonempty(
            name=f"{prefix}{suffix}:{token_stratum}",
            tasks=tuple(release.task_name(c, token_stratum) for c in tier_configs),
            total=len(configs),
            description=f"Long-context {title} {what}tasks in the {token_stratum} context tier.",
        )
        _make_suite_if_nonempty(
            name=f"{sample_prefix}{suffix}:{token_stratum}",
            tasks=tuple(release.task_name_10pct(c, token_stratum) for c in tier_configs),
            total=len(configs),
            description=(
                f"Deterministic 10% {title} {what}sample in the {token_stratum} context tier."
            ),
        )

    full_and_sample(
        "",
        release.configs,
        f"Long-context {title} tasks across all formats and complexity classes.",
        f"Deterministic 10% sample of the long-context {title} tasks.",
    )

    for complexity in release.complexities:
        full_and_sample(
            f":{complexity.replace('-', '_')}",
            tuple(c for c in release.configs if c.endswith(f"--{complexity}")),
            f"Long-context {title} tasks in the {complexity} complexity class.",
            f"Deterministic 10% {title} sample in the {complexity} complexity class.",
        )

    for formatter in release.formatters:
        formatter_configs = tuple(c for c in release.configs if c.startswith(f"{formatter}--"))
        full_and_sample(
            f":{formatter.replace('-', '_')}",
            formatter_configs,
            f"Long-context {title} tasks using the {formatter} format.",
            f"Deterministic 10% {title} sample using the {formatter} format.",
        )
        for token_stratum in release.token_strata:
            tier(
                f":{formatter.replace('-', '_')}", formatter_configs, token_stratum, f"{formatter} "
            )

    for token_stratum in release.token_strata:
        tier("", release.configs, token_stratum, "")


_register_release_suites(STATE_BENCH_V1, title="StateBench", span="up to 1m tokens")
_register_release_suites(STATE_BENCH_V2, title="StateBench v2", span="up to and beyond 2m tokens")
