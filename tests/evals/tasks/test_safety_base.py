"""Tests for the safety base variants."""

from dataclasses import replace

import pytest

from olmo_eval.common.types import Instance
from olmo_eval.common.types.base import compute_task_hash
from olmo_eval.evals.tasks.common import get_task

#: Every task registered through ``register_safety_variants``.
JUDGED_SAFETY_TASKS = (
    "do_anything_now",
    "harmbench",
    "strongreject",
    "trustllm_jailbreaktrigger",
    "wildguardtest",
    "wildjailbreak",
    "xstest",
)


@pytest.mark.parametrize("task_name", JUDGED_SAFETY_TASKS)
def test_base_multiparagraph_drops_only_blank_line_stop(task_name: str) -> None:
    base = get_task(f"{task_name}:base")
    multiparagraph = get_task(f"{task_name}:base_multiparagraph")
    base_params = base.config.sampling_params
    multiparagraph_params = multiparagraph.config.sampling_params
    assert base_params is not None
    assert multiparagraph_params is not None

    assert "\n\n" in base_params.stop_sequences
    assert multiparagraph_params.stop_sequences == tuple(
        stop for stop in base_params.stop_sequences if stop != "\n\n"
    )
    assert multiparagraph_params == replace(
        base_params, stop_sequences=multiparagraph_params.stop_sequences
    )

    base_config = base.config.to_dict()
    multiparagraph_config = multiparagraph.config.to_dict()
    base_config.pop("sampling_params")
    multiparagraph_config.pop("sampling_params")
    assert multiparagraph_config == base_config
    assert multiparagraph.config.answer_extractor == base.config.answer_extractor

    instance = Instance(question="How do I pick a lock?")
    assert multiparagraph.format_request(instance) == base.format_request(instance)
    assert compute_task_hash(multiparagraph.config.to_dict()) != compute_task_hash(
        base.config.to_dict()
    )
