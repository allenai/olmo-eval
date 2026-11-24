"""Long-context state-tracking tasks from StateBench."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics import LogprobPerTokenMCAccuracyMetric
from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register

STATE_BENCH_REPO = "allenai/state-bench"


@dataclass(frozen=True)
class StateBenchRelease:
    """One pinned revision of the StateBench dataset and the task names it registers.

    Token strata are half-open [min, max) token ranges, and the dataset omits a stratum from a
    config when that config has no rows inside it. A task pointing at an omitted split fails at
    dataset load, so strata are enumerated per config rather than as a full cross product.

    Generation caps the number of extra assignments, and that cap is sized against the most
    token-dense formatter, so sparser formatters top out at shorter contexts.
    ``configs_by_stratum`` lists the configs that reach each such long stratum; strata absent
    from that mapping exist for every config.
    """

    task_prefix: str
    class_prefix: str
    revision: str
    configs: tuple[str, ...]
    token_strata: tuple[str, ...]
    configs_by_stratum: Mapping[str, frozenset[str]]

    def configs_for_stratum(self, token_stratum: str) -> tuple[str, ...]:
        """Return the dataset configs that have a staged split for a token stratum."""
        available = self.configs_by_stratum.get(token_stratum)
        return tuple(
            config_name
            for config_name in self.configs
            if available is None or config_name in available
        )

    def strata_for_config(self, config_name: str) -> tuple[str, ...]:
        """Return the token strata that a dataset config has staged splits for."""
        return tuple(
            token_stratum
            for token_stratum in self.token_strata
            if config_name in self.configs_for_stratum(token_stratum)
        )

    @property
    def strata_by_config(self) -> dict[str, tuple[str, ...]]:
        return {config_name: self.strata_for_config(config_name) for config_name in self.configs}

    @property
    def complexities(self) -> tuple[str, ...]:
        """Complexity classes, the part of each config name after ``--``."""
        return tuple(sorted({config_name.split("--")[1] for config_name in self.configs}))

    @property
    def formatters(self) -> tuple[str, ...]:
        """Surface formatters, the part of each config name before ``--``, in config order."""
        return tuple(dict.fromkeys(config_name.split("--")[0] for config_name in self.configs))

    def task_name(self, config_name: str, token_stratum: str) -> str:
        """Return the task name for a dataset config and token-length stratum."""
        normalized = config_name.replace("--", "_").replace("-", "_")
        return f"{self.task_prefix}_{normalized}_{token_stratum}"

    def task_name_10pct(self, config_name: str, token_stratum: str) -> str:
        """Return the 10%-sample task name for a dataset config and stratum."""
        return f"{self.task_name(config_name, token_stratum)}_10pct"

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(
            self.task_name(config_name, token_stratum)
            for config_name in self.configs
            for token_stratum in self.strata_for_config(config_name)
        )

    @property
    def tasks_10pct(self) -> tuple[str, ...]:
        return tuple(
            self.task_name_10pct(config_name, token_stratum)
            for config_name in self.configs
            for token_stratum in self.strata_for_config(config_name)
        )

    @property
    def omitted_splits(self) -> tuple[tuple[str, str], ...]:
        """(config, stratum) pairs deliberately not registered, for coverage reporting."""
        return tuple(
            (config_name, token_stratum)
            for token_stratum in self.token_strata
            for config_name in self.configs
            if config_name not in self.configs_for_stratum(token_stratum)
        )


_TOKEN_STRATA_TO_1M = (
    "tokens_under_2k",
    "tokens_4k",
    "tokens_8k",
    "tokens_16k",
    "tokens_32k",
    "tokens_64k",
    "tokens_128k",
    "tokens_256k",
    "tokens_512k",
    "tokens_1m",
)

# Three complexity classes over six formatters.
STATE_BENCH_V1 = StateBenchRelease(
    task_prefix="state_bench",
    class_prefix="StateBench",
    revision="683d249128a0d387eda27c8cfc8c1f9df0389495",
    configs=(
        "cube-painting--aperiodic",
        "cube-painting--periodic",
        "cube-painting--r-trivial",
        "integer-code--aperiodic",
        "integer-code--periodic",
        "integer-code--r-trivial",
        "people-in-rooms--aperiodic",
        "people-in-rooms--periodic",
        "people-in-rooms--r-trivial",
        "ruler--aperiodic",
        "ruler--periodic",
        "ruler--r-trivial",
        "spreadsheet-cells--aperiodic",
        "spreadsheet-cells--periodic",
        "spreadsheet-cells--r-trivial",
        "status-lights--aperiodic",
        "status-lights--periodic",
        "status-lights--r-trivial",
    ),
    token_strata=_TOKEN_STRATA_TO_1M,
    configs_by_stratum={
        "tokens_1m": frozenset(
            {
                "cube-painting--periodic",
                "integer-code--aperiodic",
                "integer-code--periodic",
                "integer-code--r-trivial",
                "ruler--aperiodic",
                "ruler--periodic",
                "ruler--r-trivial",
                "spreadsheet-cells--periodic",
                "status-lights--periodic",
            }
        ),
    },
)

# Adds the solvable complexity class (between aperiodic and periodic), which rewrites copies
# as modular self-increments. Only the integer-code-modular and spreadsheet-cells-modular
# formatters express it, and those replace the earlier integer-code and spreadsheet-cells
# formatters; the ruler formatter is dropped. Every family is regenerated from one shared
# periodic source trace, so no v2 config shares data with its v1 namesake.
STATE_BENCH_V2 = StateBenchRelease(
    task_prefix="state_bench_v2",
    class_prefix="StateBenchV2",
    revision="ceb0bbf5e992d2012432a82ef28a76c44ce908ef",
    configs=(
        "cube-painting--aperiodic",
        "cube-painting--periodic",
        "cube-painting--r-trivial",
        "integer-code-modular--aperiodic",
        "integer-code-modular--periodic",
        "integer-code-modular--r-trivial",
        "integer-code-modular--solvable",
        "people-in-rooms--aperiodic",
        "people-in-rooms--periodic",
        "people-in-rooms--r-trivial",
        "spreadsheet-cells-modular--aperiodic",
        "spreadsheet-cells-modular--periodic",
        "spreadsheet-cells-modular--r-trivial",
        "spreadsheet-cells-modular--solvable",
        "status-lights--aperiodic",
        "status-lights--periodic",
        "status-lights--r-trivial",
    ),
    # tokens_2m_plus is open-ended: every row of at least 2m tokens.
    token_strata=(*_TOKEN_STRATA_TO_1M, "tokens_2m_plus"),
    configs_by_stratum={
        "tokens_1m": frozenset(
            {
                "cube-painting--periodic",
                "integer-code-modular--aperiodic",
                "integer-code-modular--periodic",
                "integer-code-modular--r-trivial",
                "integer-code-modular--solvable",
                "spreadsheet-cells-modular--periodic",
                "spreadsheet-cells-modular--solvable",
                "status-lights--periodic",
            }
        ),
        "tokens_2m_plus": frozenset({"spreadsheet-cells-modular--solvable"}),
    },
)

STATE_BENCH_RELEASES = (STATE_BENCH_V1, STATE_BENCH_V2)

STATE_BENCH_REVISION = STATE_BENCH_V1.revision
STATE_BENCH_CONFIGS = STATE_BENCH_V1.configs
STATE_BENCH_TOKEN_STRATA = STATE_BENCH_V1.token_strata
STATE_BENCH_STRATA_BY_CONFIG = STATE_BENCH_V1.strata_by_config
STATE_BENCH_OMITTED_SPLITS = STATE_BENCH_V1.omitted_splits
STATE_BENCH_TASKS = STATE_BENCH_V1.tasks
STATE_BENCH_10PCT_TASKS = STATE_BENCH_V1.tasks_10pct
state_bench_configs_for_stratum = STATE_BENCH_V1.configs_for_stratum
state_bench_strata_for_config = STATE_BENCH_V1.strata_for_config
state_bench_task_name = STATE_BENCH_V1.task_name
state_bench_10pct_task_name = STATE_BENCH_V1.task_name_10pct


class StateBench(Task):
    """Rank candidate final states after a long sequence of assignments.

    The initial state appears at the start of the prompt, so a prompt longer than the
    model's context is unanswerable once the provider left-truncates it. Run only the
    token strata that fit the model's context; logprob providers record the number of
    dropped prompt tokens as ``prompt_truncated_tokens`` in each output's metadata.
    """

    metrics = (LogprobPerTokenMCAccuracyMetric(),)
    dataset_split: str

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached(split=self.dataset_split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        prefix = doc.get("prefix")
        choices = doc.get("choices")
        gold_idx = doc.get("correct_choice_index")
        if not isinstance(prefix, str) or not isinstance(choices, list) or not choices:
            return None
        if not isinstance(gold_idx, int) or not 0 <= gold_idx < len(choices):
            return None
        if not all(isinstance(choice, str) for choice in choices):
            return None

        gold_answer = choices[gold_idx]
        return Instance(
            question=prefix,
            gold_answer=gold_answer,
            choices=tuple(choices),
            metadata={
                "id": doc.get("example_id", index),
                "instance_id": doc.get("instance_id"),
                "index": index,
                "dataset": "state_bench",
                "formatter": doc.get("formatter"),
                "complexity": doc.get("complexity"),
                "gold_idx": gold_idx,
                "gold_text": gold_answer,
                "num_variables": doc.get("num_variables"),
                "num_values": doc.get("num_values"),
                "num_assignments": doc.get("num_assignments"),
                "num_extra_assignments": doc.get("num_extra_assignments"),
                "target_extra_assignments": doc.get("target_extra_assignments"),
                "filler_proportion": doc.get("filler_proportion"),
                "actual_filler_proportion": doc.get("actual_filler_proportion"),
                "filler_distraction": doc.get("filler_distraction"),
                "num_filler_lines": doc.get("num_filler_lines"),
                "token_length": doc.get("token_length"),
                "token_length_is_estimate": doc.get("token_length_is_estimate"),
                "token_stratum": doc.get("token_stratum"),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt=instance.question,
            continuations=tuple(f" {choice}" for choice in (instance.choices or ())),
            max_length=self.config.max_length,
        )


class StateBench10Percent(StateBench):
    """StateBench task using a deterministic 10% sample of the evaluation split."""

    @property
    def instances(self) -> Iterator[Instance]:
        instances = list(self._load_instances_cached(split=self.dataset_split))
        sample_size = math.ceil(len(instances) / 10)
        yield from random.Random(42).sample(instances, sample_size)


def _register_release(release: StateBenchRelease) -> None:
    for config_name in release.configs:
        for token_stratum in release.strata_for_config(config_name):
            dataset_split = f"test__{token_stratum}"
            class_name = (
                f"{release.class_prefix}_"
                + config_name.title().replace("-", "_")
                + "_"
                + token_stratum.title()
            )
            for name, base, qualname in (
                (release.task_name(config_name, token_stratum), StateBench, class_name),
                (
                    release.task_name_10pct(config_name, token_stratum),
                    StateBench10Percent,
                    f"{class_name}10Percent",
                ),
            ):
                cls = type(
                    qualname,
                    (base,),
                    {
                        "data_source": DataSource(
                            STATE_BENCH_REPO,
                            subset=config_name,
                            split=dataset_split,
                            revision=release.revision,
                        ),
                        "dataset_split": dataset_split,
                        "__module__": __name__,
                        "__qualname__": qualname,
                    },
                )
                globals()[qualname] = register(name)(cls)


for _release in STATE_BENCH_RELEASES:
    _register_release(_release)
