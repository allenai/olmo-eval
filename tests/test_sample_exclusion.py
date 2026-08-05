"""Growing a sample must cost only the part that has not been run.

`OLMO_EVAL_SKIP_SAMPLE_OF_SIZE=40` with `-o limit=100` should run the 60 instances a 40-sample at
the same seed would not have selected.

The case that matters most here is a task whose instance count equals the limit. An earlier
version excluded inside the sampling branch, which such a task never enters, so DeepResearch — 100
instances, limit 100 — re-ran all 40 it had already done while every unit test passed. The tests
below go through `select_instances`, the decision the runner actually makes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from olmo_eval.runners.asynq.preparation import select_instances


@dataclass
class Config:
    limit: int | None
    seed: int = 42


def population(n):
    return [f"inst_{i}" for i in range(n)]


def earlier_sample(pop, size):
    return random.Random(42).sample(pop, size)


@pytest.fixture
def skip_forty(monkeypatch):
    monkeypatch.setenv("OLMO_EVAL_SKIP_SAMPLE_OF_SIZE", "40")


@pytest.mark.parametrize("n", [597, 599, 600, 500, 100])
def test_runs_only_what_the_earlier_sample_missed(n, skip_forty):
    pop = population(n)
    prior = earlier_sample(pop, 40)

    selected = select_instances(pop, Config(limit=100))

    assert len(selected) == 60
    assert not set(selected) & set(prior), "an instance from the earlier run would be paid for twice"
    assert set(selected) | set(prior) == set(random.Random(42).sample(pop, 100))


def test_a_task_with_no_room_to_sample_is_still_excluded(skip_forty):
    """The regression: limit == instance count skips sampling, and used to skip exclusion too."""

    pop = population(100)
    prior = earlier_sample(pop, 40)

    selected = select_instances(pop, Config(limit=100))

    assert len(selected) == 60
    assert not set(selected) & set(prior)


def test_untouched_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv("OLMO_EVAL_SKIP_SAMPLE_OF_SIZE", raising=False)
    pop = population(597)

    assert select_instances(pop, Config(limit=100)) == random.Random(42).sample(pop, 100)


def test_untouched_when_the_earlier_run_was_not_smaller(monkeypatch):
    monkeypatch.setenv("OLMO_EVAL_SKIP_SAMPLE_OF_SIZE", "100")
    pop = population(597)

    assert select_instances(pop, Config(limit=100)) == random.Random(42).sample(pop, 100)


def test_a_nonsense_value_is_ignored_rather_than_fatal(monkeypatch):
    monkeypatch.setenv("OLMO_EVAL_SKIP_SAMPLE_OF_SIZE", "not-a-number")
    pop = population(597)

    assert len(select_instances(pop, Config(limit=100))) == 100


def test_no_limit_means_the_whole_set(monkeypatch):
    monkeypatch.delenv("OLMO_EVAL_SKIP_SAMPLE_OF_SIZE", raising=False)
    pop = population(597)

    assert select_instances(pop, Config(limit=None)) == pop
