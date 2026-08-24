"""Tests for the DeepScholar-Bench task and its export adapter.

The prompt is pinned against a golden copy of lit-agents'
``shared/deepscholar.py::QUERY_TEMPLATE``: the whole point of this task is to be
comparable with the systems prompted that way, and a reworded prompt would
break that silently rather than loudly.
"""

import csv
import json

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.common.types.tools import Function, ToolCall, ToolResult
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.deepscholar_bench import (
    DEEPSCHOLAR_COMMIT,
    DEEPSCHOLAR_QUERY_TEMPLATE,
    SEARCH_TOOL_NAME,
    parse_arxiv_tool_results,
    sources_from_trajectory,
)
from olmo_eval.evals.tasks.deepscholar_export import (
    PAPER_CSV_FIELDS,
    build_paper_rows,
    export_predictions,
)
from olmo_eval.harness.tools.search import _format_arxiv_result

# Golden copy of lit-agents' shared/deepscholar.py::QUERY_TEMPLATE.
GOLDEN_QUERY_TEMPLATE = """Your task is to write a Related Works section for an academic paper given the paper's abstract. Your response should provide the Related Works section and references. Only include references from arXiv that are published before {cutoff_date}. Mention them in a separate, numbered reference list at the end and use the reference numbers to provide in-line citations in the Related Works section for all claims referring to a source (e.g., description of source [3]. Further details [6][7][8][9][10].) Each in-line citation must consist of a single reference number within a pair of brackets. Do not use any other citation format. Do not exceed 600 words for the related works section. Here is the paper abstract: {abstract}"""

CSV_ROW = {
    "arxiv_id": "2506.02838v1",
    "title": "TaxAgent: How Large Language Model Designs Fiscal Policy",
    "abstract": "  Economic inequality is a global challenge.  ",
    "published_date": "2025-06-03T13:06:19+00:00",
}


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


@pytest.fixture
def task():
    return get_task("deepscholar_bench")


def _tool_output(*papers: dict) -> str:
    """Render results the way the search tool does, so the parser is tested
    against the real formatter rather than a hand-written imitation."""
    return "\n\n---\n\n".join(
        _format_arxiv_result(paper, paper.get("externalIds", {}).get("ArXiv", ""))
        for paper in papers
    )


def _paper(
    title: str,
    arxiv_id: str,
    *,
    abstract: str = "An abstract.",
    published: str | None = None,
) -> dict:
    paper = {
        "title": title,
        "abstract": abstract,
        "year": 2024,
        "authors": [{"name": "A. Author"}],
        "externalIds": {"ArXiv": arxiv_id},
    }
    if published is not None:
        paper["publicationDate"] = published
    return paper


def _trajectory(*result_contents: str) -> AgentTrajectory:
    turns = []
    for index, content in enumerate(result_contents):
        call = ToolCall(
            id=f"c{index}",
            function=Function(name=SEARCH_TOOL_NAME, arguments='{"query": "q"}'),
        )
        turns.append(AgentTurn.assistant(tool_calls=[call]))
        turns.append(
            AgentTurn.tool(results=[ToolResult(tool_call_id=f"c{index}", content=content)])
        )
    return AgentTrajectory(turns=tuple(turns))


class TestRegistration:
    def test_registered(self):
        assert "deepscholar_bench" in list_tasks()

    def test_metrics(self, task):
        assert {m.name for m in task.config.metrics} == {
            "exportable_rate",
            "arxiv_citation_rate",
        }
        assert task.config.get_primary_metric().name == "exportable_rate"

    def test_dataset_is_pinned_to_a_commit(self):
        from olmo_eval.evals.tasks.deepscholar_bench import DEEPSCHOLAR_DATASET_URL

        assert DEEPSCHOLAR_COMMIT in DEEPSCHOLAR_DATASET_URL
        assert DEEPSCHOLAR_DATASET_URL.endswith("dataset/papers_with_related_works.csv")


class TestPrompt:
    def test_template_is_verbatim(self):
        assert DEEPSCHOLAR_QUERY_TEMPLATE == GOLDEN_QUERY_TEMPLATE

    def test_request_interpolates_the_cutoff_and_abstract(self, task):
        instance = task.process_doc(dict(CSV_ROW), index=7)
        request = task.format_request(instance)

        assert request.request_type is RequestType.CHAT
        assert request.messages[0]["content"] == GOLDEN_QUERY_TEMPLATE.format(
            cutoff_date="2025-06-03",
            abstract="Economic inequality is a global challenge.",
        )


class TestProcessDoc:
    def test_carries_the_cutoff_to_the_search_tools(self, task):
        instance = task.process_doc(dict(CSV_ROW), index=7)

        assert instance.metadata["retrieval_date_cutoff"] == "2025-06-03"
        assert instance.metadata["cutoff_date"] == "2025-06-03"

    def test_keeps_the_upstream_positional_id(self, task):
        instance = task.process_doc(dict(CSV_ROW), index=7)

        assert instance.metadata["id"] == "7"
        assert instance.metadata["arxiv_id"] == "2506.02838"

    @pytest.mark.parametrize("field", ["abstract", "published_date"])
    def test_rows_missing_a_required_field_are_dropped(self, task, field):
        row = dict(CSV_ROW)
        row[field] = ""

        assert task.process_doc(row, index=0) is None

    def test_unparseable_cutoff_is_dropped(self, task):
        row = dict(CSV_ROW, published_date="not a date")

        assert task.process_doc(row, index=0) is None


class TestResultParsing:
    def test_round_trips_the_tools_rendering(self):
        content = _tool_output(_paper("First", "2401.00001"), _paper("Second", "2401.00002"))

        parsed = parse_arxiv_tool_results(content)

        assert [item["arxiv_id"] for item in parsed] == ["2401.00001", "2401.00002"]
        assert parsed[0]["title"] == "First"
        assert parsed[0]["abstract"] == "An abstract."
        assert parsed[0]["url"] == "https://arxiv.org/abs/2401.00001"
        # No S2 date, so the ID's month is all the tool could claim.
        assert parsed[0]["published_date"] == "2024-01-01"
        assert parsed[0]["date_precision"] == "month"

    def test_a_day_precise_date_round_trips(self):
        content = _tool_output(_paper("Dated", "2401.00001", published="2024-01-15"))

        (parsed,) = parse_arxiv_tool_results(content)

        assert parsed["published_date"] == "2024-01-15"
        assert parsed["date_precision"] == "day"

    def test_multiline_abstracts_survive(self):
        content = _tool_output(_paper("Wrapped", "2401.00003", abstract="Line one.\nLine two."))

        assert parse_arxiv_tool_results(content)[0]["abstract"] == "Line one.\nLine two."

    def test_context_only_results_are_skipped(self):
        content = _format_arxiv_result({"title": "Journal", "abstract": "x"}, "")

        assert parse_arxiv_tool_results(content) == []

    def test_trajectory_sources_are_deduplicated_first_mention_winning(self):
        trajectory = _trajectory(
            _tool_output(_paper("First", "2401.00001")),
            _tool_output(_paper("First again", "2401.00001"), _paper("Second", "2401.00002")),
        )

        sources = sources_from_trajectory(trajectory)

        assert [item["arxiv_id"] for item in sources] == ["2401.00001", "2401.00002"]
        assert sources[0]["title"] == "First"

    def test_no_trajectory_yields_no_sources(self):
        assert sources_from_trajectory(None) == []


class TestScoring:
    def _response(self, answer: str, trajectory: AgentTrajectory | None) -> Response:
        return Response(
            instance=Instance(question="abstract", metadata={"id": "0"}),
            request=LMRequest(request_type=RequestType.CHAT, messages=()),
            outputs=[LMOutput(text=answer)],
            scores={},
            trajectory=trajectory,
        )

    @pytest.mark.anyio
    async def test_exportable_response_scores_one(self, task):
        response = self._response(
            "Related work [Paper](https://arxiv.org/abs/2401.00001).",
            _trajectory(_tool_output(_paper("First", "2401.00001"))),
        )

        (scored,) = await task.score_responses([response])

        assert scored.scores["exportable_rate"] == 1.0
        assert scored.scores["arxiv_citation_rate"] == 1.0
        assert scored.outputs[0].metadata["deepscholar_num_searches"] == 1

    @pytest.mark.anyio
    async def test_response_without_a_trajectory_is_not_exportable(self, task):
        (scored,) = await task.score_responses([self._response("Some text.", None)])

        assert scored.scores["exportable_rate"] == 0.0
        assert scored.scores["arxiv_citation_rate"] == 0.0


class TestExportAdapter:
    def _prediction(self, native_id: str = "7", answer: str = "Related work.") -> dict:
        trajectory = _trajectory(
            _tool_output(
                _paper("First", "2401.00001", published="2024-01-15"),
                _paper("Second", "cs/0501001"),
            )
        )
        return {
            "doc_id": 0,
            "native_id": native_id,
            "final_output": answer,
            "model_output": [{"text": answer}],
            "trajectory": trajectory.to_dict(),
        }

    def _write(self, tmp_path, *predictions: dict):
        path = tmp_path / "deepscholar_bench-predictions.jsonl"
        path.write_text(
            "".join(json.dumps(prediction) + "\n" for prediction in predictions),
            encoding="utf-8",
        )
        return path

    def test_writes_one_folder_per_query(self, tmp_path):
        predictions = self._write(tmp_path, self._prediction())
        output = tmp_path / "export"

        summary = export_predictions(predictions, output)

        assert summary["exported_ids"] == ["7"]
        assert sorted(p.name for p in (output / "7").iterdir()) == [
            "final_report.md",
            "intro.md",
            "paper.csv",
        ]

    def test_intro_and_final_report_are_the_answer(self, tmp_path):
        predictions = self._write(tmp_path, self._prediction(answer="The section."))
        output = tmp_path / "export"

        export_predictions(predictions, output)

        intro = (output / "7" / "intro.md").read_text(encoding="utf-8")
        assert intro == "The section."
        assert (output / "7" / "final_report.md").read_text(encoding="utf-8") == intro

    def test_paper_csv_columns_and_rows(self, tmp_path):
        predictions = self._write(tmp_path, self._prediction())
        output = tmp_path / "export"

        export_predictions(predictions, output)

        with (output / "7" / "paper.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == list(PAPER_CSV_FIELDS)
            rows = list(reader)

        assert [row["id"] for row in rows] == ["2401.00001", "cs/0501001"]
        assert rows[0]["title"] == "First"
        assert rows[0]["snippet"] == "An abstract."
        # S2 dated the first source, so the day survives into the CSV.
        assert rows[0]["published_date"] == "2024-01-15"
        assert rows[0]["date_precision"] == "day"
        assert rows[0]["paper_id"] == "2401.00001"
        # The second was undated, so only its ID's month can be claimed.
        assert rows[1]["published_date"] == "2005-01-01"
        assert rows[1]["date_precision"] == "month"

    def test_predictions_without_sources_are_skipped(self, tmp_path):
        prediction = self._prediction()
        prediction["trajectory"] = AgentTrajectory().to_dict()
        predictions = self._write(tmp_path, prediction)
        output = tmp_path / "export"

        summary = export_predictions(predictions, output)

        assert summary["exported_count"] == 0
        assert summary["skipped"] == [{"idx": "7", "reason": "no arXiv sources in trajectory"}]
        assert not (output / "7").exists()

    def test_predictions_without_an_answer_are_skipped(self, tmp_path):
        prediction = self._prediction(answer="   ")
        predictions = self._write(tmp_path, prediction)

        summary = export_predictions(predictions, tmp_path / "export")

        assert summary["skipped"] == [{"idx": "7", "reason": "no answer text"}]

    def test_undatable_sources_are_dropped(self):
        rows = build_paper_rows(
            [
                {"arxiv_id": "2401.00001", "title": "Datable", "abstract": "a"},
                {"arxiv_id": "not-an-id", "title": "Undatable", "abstract": "b"},
            ]
        )

        assert [row["id"] for row in rows] == ["2401.00001"]

    def test_sources_without_a_published_line_fall_back_to_the_id_month(self):
        # Predictions saved before the tool rendered a Published line.
        (row,) = build_paper_rows([{"arxiv_id": "2401.00001", "title": "Legacy", "abstract": "a"}])

        assert row["published_date"] == "2024-01-01"
        assert row["date_precision"] == "month"

    def test_a_parsed_day_is_written_unchanged(self):
        (row,) = build_paper_rows(
            [
                {
                    "arxiv_id": "2401.00001",
                    "title": "Dated",
                    "abstract": "a",
                    "published_date": "2024-01-15",
                    "date_precision": "day",
                }
            ]
        )

        assert row["published_date"] == "2024-01-15"
        assert row["date_precision"] == "day"
