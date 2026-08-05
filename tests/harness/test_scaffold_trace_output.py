"""The local span exporter must survive the guard that keeps spans off OpenAI's backend.

`_create_agent` disabled tracing outright, which silenced the FileSpanExporter that
`configure_trace_output` installs, so `{output_dir}/traces/` was never written for any agent run.
The exporter had been dead code. These tests pin the contract that made it live again.

The agents SDK is an optional dependency and is absent from the default dev environment, so it is
stubbed here; the module under test only needs `set_trace_processors` and `BatchTraceProcessor`.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest


@pytest.fixture
def scaffold_tracing(monkeypatch, tmp_path):
    """Import the tracing module against a stubbed agents SDK, freshly each time."""

    installed: list = []

    agents = types.ModuleType("agents")
    agents.set_trace_processors = lambda processors: installed.extend(processors)

    processors_module = types.ModuleType("agents.tracing.processors")

    class BatchTraceProcessor:
        def __init__(self, exporter):
            self.exporter = exporter

    processors_module.BatchTraceProcessor = BatchTraceProcessor
    tracing_pkg = types.ModuleType("agents.tracing")
    tracing_pkg.processors = processors_module

    monkeypatch.setitem(sys.modules, "agents", agents)
    monkeypatch.setitem(sys.modules, "agents.tracing", tracing_pkg)
    monkeypatch.setitem(sys.modules, "agents.tracing.processors", processors_module)
    monkeypatch.delitem(sys.modules, "olmo_eval.harness.scaffolds.tracing", raising=False)

    module = importlib.import_module("olmo_eval.harness.scaffolds.tracing")
    module.installed_processors = installed  # exposed for assertions
    return module


def test_file_output_is_not_configured_before_it_is_asked_for(scaffold_tracing):
    assert scaffold_tracing.file_trace_output_configured() is False


def test_configuring_output_installs_the_exporter_and_records_that_it_did(
    scaffold_tracing, tmp_path
):
    scaffold_tracing.configure_trace_output(str(tmp_path))

    assert scaffold_tracing.file_trace_output_configured() is True, (
        "the scaffold reads this to decide whether it still has to disable tracing; "
        "leaving it false is what silenced the exporter"
    )
    assert len(scaffold_tracing.installed_processors) == 1, (
        "installing exactly one processor is what keeps spans off OpenAI's backend, "
        "so disabling tracing as well is unnecessary"
    )
    assert scaffold_tracing.installed_processors[0].exporter._output_dir == str(
        tmp_path / "traces"
    )


def test_spans_are_written_under_the_output_directory(scaffold_tracing, tmp_path):
    exporter = scaffold_tracing.FileSpanExporter(str(tmp_path))

    class Span:
        trace_id = "trace_abcdef123456"

        def export(self):
            return {"object": "trace.span", "span_data": {"name": "generation"}}

    exporter.export([Span()])

    written = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(written) == 1, "one file per trace id"
    assert "generation" in written[0].read_text()
