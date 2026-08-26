"""Tests for the DeepScholar-Bench task and its export adapter.

The prompt is pinned against a golden copy of lit-agents'
``shared/deepscholar.py::QUERY_TEMPLATE``: the point of this task is to be
comparable with the systems prompted that way, and a reworded prompt would break
that silently. The export is pinned against the real upstream parser, copied in
under ``fixtures/deepscholar_c95413b``, because an imitation would only prove
itself self-consistent.
"""

import csv
import json
from datetime import date

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.common.types.tools import Function, ToolCall, ToolResult
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.deepscholar_bench import (
    DEEPSCHOLAR_COMMIT,
    DEEPSCHOLAR_QUERY_TEMPLATE,
    SEARCH_TOOL_NAME,
    build_deepscholar_prompt,
    download_deepscholar_dataset,
    parse_arxiv_tool_results,
    sources_from_trajectory,
)
from olmo_eval.evals.tasks.deepscholar_citations import (
    FINAL_REPORT_MARKER,
    split_final_report,
)
from olmo_eval.evals.tasks.deepscholar_export import (
    PAPER_CSV_FIELDS,
    build_paper_rows,
    export_predictions,
    query_id,
    read_predictions,
)
from olmo_eval.harness.tools.search import _classify_arxiv_hit, _format_arxiv_result
from tests.evals.tasks.fixtures.deepscholar_c95413b.eval.parsers import (
    DeepScholarBaseParser,
    ParserType,
)
from tests.evals.tasks.fixtures.litagents_c95413b import (
    QueryRecord,
    validate_query_export,
    validate_source_rows,
)

# Golden copy of lit-agents' shared/deepscholar.py::QUERY_TEMPLATE.
GOLDEN_QUERY_TEMPLATE = """Your task is to write a Related Works section for an academic paper given the paper's abstract. Your response should provide the Related Works section and references. Only include references from arXiv that are published before {cutoff_date}. Mention them in a separate, numbered reference list at the end and use the reference numbers to provide in-line citations in the Related Works section for all claims referring to a source (e.g., description of source [3]. Further details [6][7][8][9][10].) Each in-line citation must consist of a single reference number within a pair of brackets. Do not use any other citation format. Do not exceed 600 words for the related works section. Here is the paper abstract: {abstract}"""

CSV_ROW = {
    "arxiv_id": "2506.02838v1",
    "title": "TaxAgent: How Large Language Model Designs Fiscal Policy",
    "abstract": "  Economic inequality is a global challenge.  ",
    "published_date": "2025-06-03T13:06:19+00:00",
    "arxiv_link": "http://arxiv.org/abs/2506.02838v1",
}

# The shape the prompt actually mandates: numbered inline citations plus a
# numbered reference list, and not one markdown link anywhere.
NUMBERED_ANSWER = """## Related Works

Early systems retrieved passages [1]. Later work scaled the index [2], and
recent agents ground each claim [3]. Several combine the two [2][3].

## References

[1] A. Author. First Retrieval System. arXiv:2401.00001
[2] B. Author. Scaling The Index. arXiv:2401.00002
[3] C. Author. Grounding Each Claim. arXiv:2401.00003
"""


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


@pytest.fixture
def task():
    return get_task("deepscholar_bench")


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


def _tool_output(*papers: dict) -> str:
    """Render results the way the search tool does, so the parser is tested
    against the real formatter rather than a hand-written imitation."""
    rendered = []
    for paper in papers:
        hit, reason = _classify_arxiv_hit(paper, None)
        assert hit is not None, f"fixture paper is not citable: {reason}"
        rendered.append(_format_arxiv_result(paper, hit))
    return "\n\n---\n\n".join(rendered)


def _trajectory(*result_contents: str, tool_name: str = SEARCH_TOOL_NAME) -> AgentTrajectory:
    turns = []
    for index, content in enumerate(result_contents):
        call = ToolCall(
            id=f"c{index}",
            function=Function(name=tool_name, arguments='{"query": "q"}'),
        )
        turns.append(AgentTurn.assistant(tool_calls=[call]))
        turns.append(
            AgentTurn.tool(results=[ToolResult(tool_call_id=f"c{index}", content=content)])
        )
    return AgentTrajectory(turns=tuple(turns))


def _dataset_csv(tmp_path, *indices: int):
    """A dataset CSV whose rows sit at the given positions."""
    path = tmp_path / "papers.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_ROW))
        writer.writeheader()
        for position in range(max(indices) + 1):
            row = dict(CSV_ROW)
            row["arxiv_id"] = f"2506.0{position:04d}"
            writer.writerow(row)
    return path


class TestRegistration:
    def test_registered(self):
        assert "deepscholar_bench" in list_tasks()

    def test_metrics(self, task):
        assert {m.name for m in task.config.metrics} == {
            "exportable_rate",
            "citation_resolution_rate",
            "unresolved_citation_forms",
            "marker_compliance_rate",
            "marker_misuse_rate",
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
        assert task.process_doc(dict(CSV_ROW, published_date="not a date"), index=0) is None


class TestDataset:
    def test_the_pinned_dataset_yields_63_queries(self, task):
        # Reads the cached CSV; the download only happens on a cold cache.
        download_deepscholar_dataset()
        instances = list(task.instances)

        assert len(instances) == 63
        assert [item.metadata["id"] for item in instances] == [str(i) for i in range(63)]

    def test_positional_ids_do_not_renumber_when_a_row_is_dropped(self, task):
        rows = [dict(CSV_ROW), dict(CSV_ROW, abstract=""), dict(CSV_ROW)]

        kept = [task.process_doc(row, index=index) for index, row in enumerate(rows)]

        assert [item.metadata["id"] for item in kept if item is not None] == ["0", "2"]


class TestResultParsing:
    def test_round_trips_the_tools_rendering(self):
        content = _tool_output(_paper("First", "2401.00001"), _paper("Second", "2401.00002"))

        parsed = parse_arxiv_tool_results(content)

        assert [item["arxiv_id"] for item in parsed] == ["2401.00001", "2401.00002"]
        assert parsed[0]["title"] == "First"
        assert parsed[0]["abstract"] == "An abstract."
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
        content = _format_arxiv_result({"title": "Journal", "abstract": "x"}, None)

        assert parse_arxiv_tool_results(content) == []


class TestTrajectorySources:
    def test_deduplicated_first_mention_winning(self):
        trajectory = _trajectory(
            _tool_output(_paper("First", "2401.00001")),
            _tool_output(_paper("First again", "2401.00001"), _paper("Second", "2401.00002")),
        )

        sources = sources_from_trajectory(trajectory)

        assert [item["arxiv_id"] for item in sources] == ["2401.00001", "2401.00002"]
        assert sources[0]["title"] == "First"

    def test_no_trajectory_yields_no_sources(self):
        assert sources_from_trajectory(None) == []

    def test_only_the_search_tools_results_are_read(self):
        # Another tool naming an arXiv ID was never shown to the agent as a
        # citable search result, so it must not become citable.
        trajectory = _trajectory(
            _tool_output(_paper("Fetched", "2401.00009")), tool_name="browse_webpage"
        )

        assert sources_from_trajectory(trajectory) == []

    @pytest.mark.parametrize(
        "abstract",
        [
            "Contains the block separator\n\n---\n\nmid-abstract.",
            "A continuation line that starts URL: https://example.com/x",
            "A continuation that starts Abstract: again, and arXiv: 9999.99999 too.",
        ],
    )
    def test_adversarial_abstracts_still_yield_the_id(self, abstract):
        # The ID is read from its own line, so a block the field parser cannot
        # make sense of still contributes the one field the export needs.
        trajectory = _trajectory(_tool_output(_paper("Tricky", "2401.00007", abstract=abstract)))

        assert [item["arxiv_id"] for item in sources_from_trajectory(trajectory)] == ["2401.00007"]

    def test_a_newline_in_a_title_still_yields_the_id(self):
        trajectory = _trajectory(_tool_output(_paper("Broken\nTitle", "2401.00008")))

        assert [item["arxiv_id"] for item in sources_from_trajectory(trajectory)] == ["2401.00008"]


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
    async def test_a_prompt_compliant_answer_is_exportable(self, task):
        # No markdown link anywhere in the answer: it scores 1.0 because the
        # bridge resolves its numbered citations, not because a URL is present.
        trajectory = _trajectory(
            _tool_output(
                _paper("First Retrieval System", "2401.00001"),
                _paper("Scaling The Index", "2401.00002"),
                _paper("Grounding Each Claim", "2401.00003"),
            )
        )
        assert "](http" not in NUMBERED_ANSWER

        (scored,) = await task.score_responses([self._response(NUMBERED_ANSWER, trajectory)])

        assert scored.scores["exportable_rate"] == 1.0
        assert scored.scores["citation_resolution_rate"] == 1.0
        assert scored.outputs[0].metadata["score:exportable_rate"] == 1.0

    @pytest.mark.anyio
    async def test_citing_only_unretrieved_papers_is_not_exportable(self, task):
        trajectory = _trajectory(_tool_output(_paper("Unrelated", "2401.09999")))

        (scored,) = await task.score_responses([self._response(NUMBERED_ANSWER, trajectory)])

        assert scored.scores["exportable_rate"] == 0.0
        assert scored.scores["citation_resolution_rate"] == 0.0

    @pytest.mark.anyio
    async def test_a_bare_url_in_prose_is_not_a_citation(self, task):
        # The old metric counted any arxiv.org URL anywhere in the text; the
        # parser counts only markdown links, so this must score zero.
        answer = "See https://arxiv.org/abs/2401.00001 for details."
        trajectory = _trajectory(_tool_output(_paper("First", "2401.00001")))

        (scored,) = await task.score_responses([self._response(answer, trajectory)])

        assert scored.scores["exportable_rate"] == 0.0

    @pytest.mark.anyio
    async def test_response_without_a_trajectory_is_not_exportable(self, task):
        (scored,) = await task.score_responses([self._response(NUMBERED_ANSWER, None)])

        assert scored.scores["exportable_rate"] == 0.0


class TestExportAdapter:
    def _prediction(self, native_id: str = "7", answer: str = NUMBERED_ANSWER) -> dict:
        trajectory = _trajectory(
            _tool_output(
                _paper("First Retrieval System", "2401.00001", published="2024-01-15"),
                _paper("Scaling The Index", "2401.00002"),
                _paper("Grounding Each Claim", "cs/0501001"),
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

    def _export(self, tmp_path, *predictions: dict, **kwargs):
        return export_predictions(
            self._write(tmp_path, *predictions),
            tmp_path / "export",
            dataset_path=_dataset_csv(tmp_path, 7),
            fetch_abstracts=False,
            **kwargs,
        )

    def test_writes_one_folder_per_query(self, tmp_path):
        summary = self._export(tmp_path, self._prediction())

        assert summary["exported_ids"] == [7]
        assert sorted(p.name for p in (tmp_path / "export" / "7").iterdir()) == [
            "export_manifest.json",
            "final_report.md",
            "intro.md",
            "paper.csv",
        ]

    def test_intro_is_the_rewritten_answer_and_final_report_matches(self, tmp_path):
        self._export(tmp_path, self._prediction())
        folder = tmp_path / "export" / "7"

        intro = (folder / "intro.md").read_text(encoding="utf-8")

        # The raw answer has no markdown link; the exported intro must, or the
        # upstream parser finds no documents at all.
        assert "[First Retrieval System](https://arxiv.org/abs/2401.00001)" in intro
        assert "## References" not in intro
        assert intro != NUMBERED_ANSWER
        assert (folder / "final_report.md").read_text(encoding="utf-8") == intro

    def test_the_real_upstream_parser_finds_documents(self, tmp_path):
        self._export(tmp_path, self._prediction())
        folder = tmp_path / "export" / "7"

        parser = DeepScholarBaseParser(
            str(folder),
            {
                "mode": ParserType.DEEPSCHOLAR_BASE,
                "file_id": "7",
                "s_map_groundtruth": {
                    "title": "T",
                    "abstract": "A",
                    "arxiv_link": "https://arxiv.org/abs/2506.02838",
                    "related_works_section": "",
                    "arxiv_id": "2506.02838",
                },
            },
        )

        assert parser.docs, "the pinned parser found no documents in the export"
        assert all(doc["title"].strip() for doc in parser.docs)
        assert all(doc["sent"].strip() for doc in parser.docs)
        # Every citation the parser renumbered points at a row it resolved.
        assert "[1]" in parser.clean_text

    def test_paper_csv_holds_only_the_cited_sources(self, tmp_path):
        prediction = self._prediction()
        # A retrieved paper the answer never cites must not appear.
        trajectory = AgentTrajectory.from_dict(prediction["trajectory"])
        extra = _tool_output(_paper("Never Cited", "2401.05555"))
        prediction["trajectory"] = _trajectory(
            *[r.content for r in trajectory.tool_result_sequence], extra
        ).to_dict()

        self._export(tmp_path, prediction)

        rows = self._rows(tmp_path / "export" / "7" / "paper.csv")
        assert "2401.05555" not in {row["id"] for row in rows}
        assert {row["id"] for row in rows} == {"2401.00001", "2401.00002", "cs/0501001"}

    def test_paper_csv_columns_and_dates(self, tmp_path):
        self._export(tmp_path, self._prediction())

        rows = self._rows(tmp_path / "export" / "7" / "paper.csv")
        by_id = {row["id"]: row for row in rows}

        assert list(by_id["2401.00001"]) == list(PAPER_CSV_FIELDS)
        # S2 dated the first source, so the day survives into the CSV.
        assert by_id["2401.00001"]["published_date"] == "2024-01-15"
        assert by_id["2401.00001"]["date_precision"] == "day"
        # The others were undated, so only their ID's month can be claimed.
        assert by_id["2401.00002"]["published_date"] == "2024-01-01"
        assert by_id["cs/0501001"]["date_precision"] == "month"

    def test_export_manifest_checksums_every_scored_file(self, tmp_path):
        import hashlib

        self._export(tmp_path, self._prediction())
        folder = tmp_path / "export" / "7"

        manifest = json.loads((folder / "export_manifest.json").read_text(encoding="utf-8"))

        assert set(manifest) == {
            "schema_version",
            "system",
            "idx",
            "query_fingerprint",
            "num_papers",
            "files",
        }
        assert manifest["idx"] == 7
        assert manifest["num_papers"] == 3
        for name, checksum in manifest["files"].items():
            assert checksum == hashlib.sha256((folder / name).read_bytes()).hexdigest()

    def test_summary_and_generation_manifest_are_written(self, tmp_path):
        self._export(tmp_path, self._prediction())
        root = tmp_path / "export"

        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        generation = json.loads((root / "generation_manifest.json").read_text(encoding="utf-8"))

        assert summary == [
            {
                "idx": 7,
                "arxiv_id": "2506.00007",
                "status": "success",
                "num_papers": 3,
                # Every row carries the flag, so the compliance rate stays
                # recomputable from the export on disk.
                "marker_found": False,
                "marker_misused": False,
            }
        ]
        assert generation["system"] == "single_agent_1"
        assert [item["idx"] for item in generation["queries"]] == [7]

    def test_an_answer_citing_nothing_retrieved_is_recorded_not_written(self, tmp_path):
        answer = "Body [1].\n\nReferences\n[1] Someone. Never Retrieved. arXiv:2999.99999\n"

        summary = self._export(tmp_path, self._prediction(answer=answer))

        assert summary["exported_count"] == 0
        assert summary["skipped"][0]["status"] == "no_eligible_source"
        assert not (tmp_path / "export" / "7").exists()

    def test_an_answer_whose_only_url_is_bare_prose_is_not_exported(self, tmp_path):
        # The parser reads markdown links only, so this answer would yield no
        # documents; writing a folder for it would manufacture a scoreable query.
        answer = "See https://arxiv.org/abs/2401.00001 for details."

        summary = self._export(tmp_path, self._prediction(answer=answer))

        assert summary["exported_count"] == 0
        assert summary["skipped"][0]["status"] == "no_eligible_source"

    def test_predictions_without_an_answer_are_recorded_as_no_answer(self, tmp_path):
        summary = self._export(tmp_path, self._prediction(answer="   "))

        assert summary["skipped"][0]["status"] == "no_answer"

    @pytest.mark.parametrize("native_id", [None, "abc", ""])
    def test_a_prediction_without_a_positional_id_fails_loudly(self, native_id):
        # doc_id is a within-run counter, so falling back to it would score one
        # paper's answer against another paper's ground truth.
        prediction = {"doc_id": 3}
        if native_id is not None:
            prediction["native_id"] = native_id

        with pytest.raises(ValueError, match="positional native_id"):
            query_id(prediction)

    @staticmethod
    def _rows(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))


class TestPaperRows:
    SOURCE = {
        "arxiv_id": "2401.00001",
        "title": "Truncated Title",
        "abstract": "A truncated preview...",
        "published_date": "2024-01-01",
        "date_precision": "month",
    }

    def test_snippet_is_the_full_refetched_abstract(self):
        full = "A full abstract. " * 60
        fetched = {"2401.00001": {"title": "Full Title", "abstract": full}}

        (row,) = build_paper_rows([self.SOURCE], fetched)

        assert row["snippet"] == full.strip()
        assert len(row["snippet"]) > 500
        assert row["title"] == "Full Title"
        assert row["snippet_source"] == "s2_batch"

    def test_a_failed_lookup_falls_back_to_the_preview_and_says_so(self):
        (row,) = build_paper_rows([self.SOURCE], {})

        assert row["snippet"] == "A truncated preview..."
        assert row["snippet_source"] == "tool_output"

    def test_a_refetched_publication_date_wins(self):
        fetched = {
            "2401.00001": {
                "title": "T",
                "abstract": "A",
                "published_date": "2024-01-22",
            }
        }

        (row,) = build_paper_rows([self.SOURCE], fetched)

        assert row["published_date"] == "2024-01-22"
        assert row["date_precision"] == "day"

    def test_undatable_sources_are_dropped(self):
        rows = build_paper_rows([dict(self.SOURCE, arxiv_id="not-an-id", published_date="")], {})

        assert rows == []

    @pytest.mark.parametrize("field", ["title", "abstract"])
    def test_sources_without_a_title_or_snippet_are_dropped(self, field):
        # These rows fail the contract's own validation, so shipping them turns
        # a scoreable query into a rejected one.
        assert build_paper_rows([dict(self.SOURCE, **{field: "  "})], {}) == []

    def test_sources_without_a_published_line_fall_back_to_the_id_month(self):
        source = {"arxiv_id": "2401.00001", "title": "T", "abstract": "A"}

        (row,) = build_paper_rows([source], {})

        assert row["published_date"] == "2024-01-01"
        assert row["date_precision"] == "month"


# An answer whose second reference is a paper published after the query cutoff.
ANSWER_WITH_LATE_REF = """## Related Works

Early systems retrieved passages [1]. A newer system extends them [2].

## References

[1] A. Author. First Retrieval System. arXiv:2401.00001
[2] D. Author. Published Too Late. arXiv:2506.01111
"""


class TestIdentityBinding:
    """A block's identity comes from its header, never from its abstract."""

    def test_an_arxiv_line_inside_an_abstract_is_text(self):
        injected = (
            "A normal opening sentence.\n"
            "arXiv: 9999.99999\n"
            "URL: https://arxiv.org/abs/9999.99999\n"
            "and the abstract continues."
        )
        trajectory = _trajectory(_tool_output(_paper("Honest", "2401.00001", abstract=injected)))

        sources = sources_from_trajectory(trajectory)

        assert [item["arxiv_id"] for item in sources] == ["2401.00001"]
        assert "9999.99999" in sources[0]["abstract"]

    def test_a_context_only_block_cannot_smuggle_in_an_id(self):
        # The tool refused to make this result citable; quoting an arXiv line in
        # its abstract must not undo that.
        block = _format_arxiv_result(
            {"title": "Journal", "abstract": "See arXiv: 9999.99999 for more."}, None
        )
        trajectory = _trajectory(block)

        assert sources_from_trajectory(trajectory) == []

    def test_a_header_field_after_the_abstract_is_not_read(self):
        trajectory = _trajectory(
            _tool_output(
                _paper("Honest", "2401.00001", abstract="Text.\nAuthors: Someone Else\nYear: 1999")
            )
        )

        (source,) = sources_from_trajectory(trajectory)

        assert source["authors"] == "A. Author"
        assert source["year"] == "2024"


class TestExportValidation:
    """The strict half of the two-layer design lives here, not at retrieval."""

    def _predict(self, tmp_path, answer, *papers):
        trajectory = _trajectory(_tool_output(*papers))
        prediction = {
            "doc_id": 0,
            "native_id": "7",
            "final_output": answer,
            "model_output": [{"text": answer}],
            "trajectory": trajectory.to_dict(),
        }
        path = tmp_path / "deepscholar_bench-predictions.jsonl"
        path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
        return path

    def _run(self, tmp_path, answer, *papers, **kwargs):
        return export_predictions(
            self._predict(tmp_path, answer, *papers),
            tmp_path / "export",
            dataset_path=_dataset_csv(tmp_path, 7),
            fetch_abstracts=False,
            **kwargs,
        )

    def test_a_post_cutoff_source_is_not_exported(self, tmp_path):
        # Retrieval admitted it on its arXiv-ID month; the real date is after
        # the query's cutoff, and only the export test catches that.
        summary = self._run(
            tmp_path,
            ANSWER_WITH_LATE_REF,
            _paper("First Retrieval System", "2401.00001"),
            _paper("Published Too Late", "2506.01111", published="2025-07-01"),
        )

        rows = self._rows(tmp_path / "export" / "7" / "paper.csv")
        assert summary["exported_ids"] == [7]
        assert [row["id"] for row in rows] == ["2401.00001"]

    def test_a_dropped_source_loses_its_citation_too(self, tmp_path):
        # intro.md and paper.csv must never disagree: the contract checks that
        # every cited ID has a row, and the parser would render an empty one.
        self._run(
            tmp_path,
            ANSWER_WITH_LATE_REF,
            _paper("First Retrieval System", "2401.00001"),
            _paper("Published Too Late", "2506.01111", published="2025-07-01"),
        )

        intro = (tmp_path / "export" / "7" / "intro.md").read_text(encoding="utf-8")

        assert "2506.01111" not in intro
        assert "Published Too Late" not in intro
        assert "2401.00001" in intro

    def test_an_answer_whose_every_source_fails_is_recorded_not_written(self, tmp_path):
        answer = "Only this [1].\n\n## References\n[1] D. Published Too Late. arXiv:2506.01111\n"

        summary = self._run(
            tmp_path, answer, _paper("Published Too Late", "2506.01111", published="2025-07-01")
        )

        assert summary["exported_count"] == 0
        assert summary["skipped"][0]["status"] == "no_eligible_source"
        assert not (tmp_path / "export" / "7").exists()

    def test_the_reference_validators_accept_the_export(self, tmp_path):
        # The export is written to satisfy these functions; testing it against
        # our own reimplementation of them would prove nothing.
        self._run(
            tmp_path,
            NUMBERED_ANSWER,
            _paper("First Retrieval System", "2401.00001"),
            _paper("Scaling The Index", "2401.00002"),
            _paper("Grounding Each Claim", "2401.00003"),
        )

        record = QueryRecord(
            query_id=7,
            query=build_deepscholar_prompt(
                "2025-06-03", "Economic inequality is a global challenge."
            ),
            cutoff_date=date(2025, 6, 3),
            arxiv_id="2506.00007",
            title=CSV_ROW["title"],
            abstract="Economic inequality is a global challenge.",
        )

        manifest = validate_query_export(tmp_path / "export" / "7", record, "single_agent_1")

        assert manifest["num_papers"] == 3

    def test_the_reference_row_validator_accepts_paper_csv(self, tmp_path):
        self._run(
            tmp_path,
            NUMBERED_ANSWER,
            _paper("First Retrieval System", "2401.00001"),
            _paper("Scaling The Index", "2401.00002"),
            _paper("Grounding Each Claim", "2401.00003"),
        )

        rows = self._rows(tmp_path / "export" / "7" / "paper.csv")

        # Raises DeepScholarContractError if any row fails.
        validate_source_rows(rows, date(2025, 6, 3), 7)

    @staticmethod
    def _rows(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))


class TestExportRefusals:
    def _predictions(self, tmp_path, text):
        path = tmp_path / "deepscholar_bench-predictions.jsonl"
        path.write_text(text, encoding="utf-8")
        return path

    def test_an_unreadable_prediction_row_stops_the_export(self, tmp_path):
        # Skipping it would shrink the export and the manifests built from it,
        # so a truncated run would produce a folder that looks complete.
        path = self._predictions(
            tmp_path, json.dumps({"native_id": "7"}) + '\n{"native_id": "8", trunc\n'
        )

        with pytest.raises(ValueError, match="line\\(s\\) 2"):
            read_predictions(path)

    def test_blank_lines_are_not_unreadable(self, tmp_path):
        path = self._predictions(tmp_path, json.dumps({"native_id": "7"}) + "\n\n")

        assert len(read_predictions(path)) == 1

    def test_a_non_empty_output_directory_is_refused(self, tmp_path):
        # A stale numeric folder survives into the new run and breaks preflight
        # with an error naming the run that is not at fault.
        output = tmp_path / "export"
        (output / "99").mkdir(parents=True)
        path = self._predictions(tmp_path, "")

        with pytest.raises(FileExistsError, match="already holds"):
            export_predictions(path, output, dataset_path=_dataset_csv(tmp_path, 7))

    def test_force_replaces_a_previous_run(self, tmp_path):
        output = tmp_path / "export"
        (output / "99").mkdir(parents=True)
        path = self._predictions(tmp_path, "")

        export_predictions(
            path, output, dataset_path=_dataset_csv(tmp_path, 7), fetch_abstracts=False, force=True
        )

        assert not (output / "99").exists()
        assert (output / "summary.json").is_file()


class TestFinalReportMarkerExport:
    """The delimiter contract as the export applies it.

    The 9B run put a median 9.8k characters of planning above its Related Works
    heading and the scorer read all of it as the report. The preset's system
    prompt now asks for a marker line instead of banning the planning; these pin
    what the export does with an answer that honours it and with one that does
    not.
    """

    DELIBERATION = "First I will search for retrieval papers, then for scaling.\n"

    def _prediction(self, answer: str, native_id: str = "7") -> dict:
        trajectory = _trajectory(
            _tool_output(
                _paper("First Retrieval System", "2401.00001", published="2024-01-15"),
                _paper("Scaling The Index", "2401.00002"),
                _paper("Grounding Each Claim", "cs/0501001"),
            )
        )
        return {
            "doc_id": 0,
            "native_id": native_id,
            "final_output": answer,
            "model_output": [{"text": answer}],
            "trajectory": trajectory.to_dict(),
        }

    def _export(self, tmp_path, answer: str, name: str = "export"):
        path = tmp_path / (name + "-predictions.jsonl")
        path.write_text(json.dumps(self._prediction(answer)) + "\n", encoding="utf-8")
        summary = export_predictions(
            path,
            tmp_path / name,
            dataset_path=_dataset_csv(tmp_path, 7),
            fetch_abstracts=False,
        )
        return summary, tmp_path / name

    def _summary_rows(self, root):
        return json.loads((root / "summary.json").read_text(encoding="utf-8"))

    def test_the_marker_splits_the_deliberation_out_of_intro_md(self, tmp_path):
        answer = self.DELIBERATION + "\n" + FINAL_REPORT_MARKER + "\n\n" + NUMBERED_ANSWER

        summary, root = self._export(tmp_path, answer)

        assert summary["exported_ids"] == [7]
        intro = (root / "7" / "intro.md").read_text(encoding="utf-8")
        assert "First I will search" not in intro
        # No residue of the delimiter itself in the scored bytes.
        assert "FINAL REPORT" not in intro
        assert "===" not in intro
        # And the bridge still ran on what was kept.
        assert "[First Retrieval System](https://arxiv.org/abs/2401.00001)" in intro

    def test_a_compliant_answer_scores_its_marker_compliance(self, tmp_path):
        answer = self.DELIBERATION + "\n" + FINAL_REPORT_MARKER + "\n\n" + NUMBERED_ANSWER

        summary, root = self._export(tmp_path, answer)

        assert summary["marker_compliance_rate"] == 1.0
        assert summary["markers_found"] == 1
        assert summary["answers_with_text"] == 1
        assert self._summary_rows(root)[0]["marker_found"] is True

    def test_an_answer_without_the_marker_is_kept_whole_and_counted(self, tmp_path):
        # The fallback must not cost the answer its export -- only its place in
        # the compliance rate.
        answer = self.DELIBERATION + NUMBERED_ANSWER

        summary, root = self._export(tmp_path, answer)

        assert summary["exported_ids"] == [7]
        assert summary["marker_compliance_rate"] == 0.0
        assert summary["markers_found"] == 0
        assert self._summary_rows(root)[0]["marker_found"] is False
        assert "First I will search" in (root / "7" / "intro.md").read_text(encoding="utf-8")

    def test_the_last_marker_wins_in_the_export(self, tmp_path):
        answer = (
            FINAL_REPORT_MARKER
            + "\n## Related Works\n\nAn abandoned draft citing nothing.\n\n"
            + FINAL_REPORT_MARKER
            + "\n"
            + NUMBERED_ANSWER
        )

        summary, root = self._export(tmp_path, answer)

        intro = (root / "7" / "intro.md").read_text(encoding="utf-8")
        assert summary["marker_compliance_rate"] == 1.0
        assert "abandoned draft" not in intro
        assert "[First Retrieval System](https://arxiv.org/abs/2401.00001)" in intro

    def test_the_sentinel_in_prose_does_not_split_the_export(self, tmp_path):
        answer = "I will write " + FINAL_REPORT_MARKER + " when ready.\n\n" + NUMBERED_ANSWER

        summary, root = self._export(tmp_path, answer)

        assert summary["marker_compliance_rate"] == 0.0
        assert "I will write" in (root / "7" / "intro.md").read_text(encoding="utf-8")

    def test_a_marker_with_nothing_above_it_changes_nothing(self, tmp_path):
        with_marker, marked_root = self._export(
            tmp_path, FINAL_REPORT_MARKER + "\n" + NUMBERED_ANSWER, name="marked"
        )
        without, plain_root = self._export(tmp_path, NUMBERED_ANSWER, name="plain")

        assert with_marker["exported_ids"] == without["exported_ids"] == [7]
        assert (marked_root / "7" / "intro.md").read_text(encoding="utf-8") == (
            plain_root / "7" / "intro.md"
        ).read_text(encoding="utf-8")
        # Same bytes, different bookkeeping: one followed the instruction.
        assert with_marker["marker_compliance_rate"] == 1.0
        assert without["marker_compliance_rate"] == 0.0

    def test_a_reference_list_with_no_prose_is_unscoreable_but_still_flagged(self, tmp_path):
        # Nothing cites anything once the reference tail is stripped, so there is
        # no document for the upstream parser to find. That is a real result, not
        # a marker failure, and the two must stay distinguishable.
        answer = (
            self.DELIBERATION
            + "\n"
            + FINAL_REPORT_MARKER
            + "\n\n## References\n\n[1] A. Author. First Retrieval System. arXiv:2401.00001\n"
        )

        summary, root = self._export(tmp_path, answer)

        assert summary["exported_ids"] == []
        (row,) = self._summary_rows(root)
        assert row["status"] == "no_eligible_source"
        assert row["marker_found"] is True
        assert summary["marker_compliance_rate"] == 1.0

    def test_a_marker_followed_by_nothing_is_named_for_what_it_is(self, tmp_path):
        # An empty tail is compliance without a deliverable; reporting it as "the
        # run produced no answer text" would blame the wrong thing.
        summary, root = self._export(tmp_path, self.DELIBERATION + FINAL_REPORT_MARKER + "\n")

        (row,) = self._summary_rows(root)
        assert row["status"] == "no_answer"
        assert "marker was followed by no text" in row["reason"]
        assert row["marker_found"] is True

    def test_an_empty_answer_stays_out_of_the_compliance_denominator(self, tmp_path):
        # A query that generated nothing can neither honour the contract nor
        # break it; counting it as non-compliant would blame the prompt.
        summary, root = self._export(tmp_path, "")

        assert summary["answers_with_text"] == 0
        assert summary["marker_compliance_rate"] == 0.0
        assert self._summary_rows(root)[0]["reason"] == "the run produced no answer text"

    def test_the_preset_prompt_and_the_export_agree_on_the_marker(self):
        # The contract lives in two places -- the sentence the model reads and
        # the regex the export applies -- and a silent drift between them would
        # discard every report. So the marker is read back out of the prompt and
        # run through the splitter.
        from olmo_eval.harness.presets import get_harness_preset

        prompt = get_harness_preset("arxiv_paper_search_agent").system_prompt
        assert prompt is not None
        mandated = [line for line in prompt.splitlines() if line.strip().startswith("===")]

        assert mandated == [FINAL_REPORT_MARKER]
        body, found = split_final_report("Planning.\n" + mandated[0] + "\nThe report.\n")
        assert found is True
        assert body == "The report.\n"

    def test_the_preset_prompt_still_states_the_whole_contract(self):
        # Three parts, all load-bearing: deliberation is allowed, the marker is
        # exact, and what comes before it is thrown away. Dropping any one of
        # them changes what the model does.
        from olmo_eval.harness.presets import get_harness_preset

        prompt = (get_harness_preset("arxiv_paper_search_agent").system_prompt or "").lower()

        assert "deliberate" in prompt
        assert "exactly" in prompt
        assert "discarded" in prompt


class TestMarkerMisuseFuse:
    """A marker honoured but misplaced must not cost the query its export.

    Qwen3.5-9B wrote its Related Works section above the line and only the
    numbered list below it. On the first smoke that took query idx=1 from
    exportable to unscoreable, which is an instrument regression rather than a
    model one, so the export now falls back to the full text and records why.
    """

    def _prediction(self, answer: str, native_id: str = "7") -> dict:
        trajectory = _trajectory(
            _tool_output(
                _paper("First Retrieval System", "2401.00001", published="2024-01-15"),
                _paper("Scaling The Index", "2401.00002"),
                _paper("Grounding Each Claim", "cs/0501001"),
            )
        )
        return {
            "doc_id": 0,
            "native_id": native_id,
            "final_output": answer,
            "model_output": [{"text": answer}],
            "trajectory": trajectory.to_dict(),
        }

    def _export(self, tmp_path, answer: str, name: str = "export"):
        path = tmp_path / (name + "-predictions.jsonl")
        path.write_text(json.dumps(self._prediction(answer)) + "\n", encoding="utf-8")
        summary = export_predictions(
            path,
            tmp_path / name,
            dataset_path=_dataset_csv(tmp_path, 7),
            fetch_abstracts=False,
        )
        return summary, tmp_path / name

    # The 9B shape: report above the marker, bare numbered list below it.
    def _misused(self) -> str:
        return (
            NUMBERED_ANSWER
            + "\n=== FINAL REPORT ===\n\n"
            + "1. A. Author. First Retrieval System. arXiv:2401.00001\n"
            + "2. B. Author. Scaling The Index. arXiv:2401.00002\n"
        )

    def test_a_misplaced_report_still_exports(self, tmp_path):
        summary, root = self._export(tmp_path, self._misused())

        assert summary["exported_ids"] == [7]
        intro = (root / "7" / "intro.md").read_text(encoding="utf-8")
        assert "[First Retrieval System](https://arxiv.org/abs/2401.00001)" in intro

    def test_the_misuse_is_counted_without_costing_compliance(self, tmp_path):
        summary, root = self._export(tmp_path, self._misused())

        # The instruction WAS followed -- the marker is there.
        assert summary["marker_compliance_rate"] == 1.0
        # And it was followed wrongly, which is a different fact.
        assert summary["marker_misuse_rate"] == 1.0
        assert summary["markers_misused"] == 1
        row = json.loads((root / "summary.json").read_text(encoding="utf-8"))[0]
        assert row["marker_found"] is True
        assert row["marker_misused"] is True

    def test_the_fuse_matches_the_no_marker_export_byte_for_byte(self, tmp_path):
        # "Falls back to the full text" has to mean exactly the marker-absent
        # path, or the fuse is its own third behaviour.
        fused, fused_root = self._export(tmp_path, self._misused(), name="fused")
        plain, plain_root = self._export(tmp_path, NUMBERED_ANSWER, name="plain")

        assert fused["exported_ids"] == plain["exported_ids"] == [7]
        assert (fused_root / "7" / "paper.csv").read_text(encoding="utf-8") == (
            plain_root / "7" / "paper.csv"
        ).read_text(encoding="utf-8")

    def test_a_well_placed_report_is_not_misuse(self, tmp_path):
        answer = "Planning first.\n\n=== FINAL REPORT ===\n\n" + NUMBERED_ANSWER

        summary, _ = self._export(tmp_path, answer)

        assert summary["marker_compliance_rate"] == 1.0
        assert summary["marker_misuse_rate"] == 0.0
        assert summary["exported_ids"] == [7]

    def test_splitting_never_exports_less_than_not_splitting(self, tmp_path):
        # The gate the researcher set, as a unit test.
        for index, answer in enumerate(
            (
                "Planning.\n\n=== FINAL REPORT ===\n\n" + NUMBERED_ANSWER,
                self._misused(),
                NUMBERED_ANSWER,
            )
        ):
            split, _ = self._export(tmp_path, answer, name=f"split-{index}")
            stripped, _ = self._export(
                tmp_path,
                answer.replace("=== FINAL REPORT ===", ""),
                name=f"nosplit-{index}",
            )

            assert split["exported_count"] >= stripped["exported_count"], answer


class TestHardenedContractWording:
    """The prompt has to say where the prose goes, not only where it ends up."""

    def _prompt(self) -> str:
        from olmo_eval.harness.presets import get_harness_preset

        prompt = get_harness_preset("arxiv_paper_search_agent").system_prompt
        assert prompt is not None
        return prompt

    def _flat(self) -> str:
        """The prompt as one lowercase line; it wraps, the sentences do not."""
        return " ".join(self._prompt().lower().split())

    def test_the_three_original_elements_survive(self):
        prompt = self._prompt().lower()

        assert "deliberate" in prompt
        assert "exactly" in prompt
        assert "discarded" in prompt

    def test_it_forbids_writing_the_section_above_the_line(self):
        # The exact ambiguity 9B took the other way.
        prompt = self._flat()

        assert "do not write any part of the related works section before the marker" in prompt

    def test_it_puts_the_whole_report_after_the_marker(self):
        prompt = self._flat()

        assert "entire report goes after that line" in prompt
        assert "reference list" in prompt

    def test_the_marker_still_round_trips_through_the_splitter(self):
        mandated = [line for line in self._prompt().splitlines() if line.strip().startswith("===")]

        assert mandated == [FINAL_REPORT_MARKER]
        body, found = split_final_report("Planning.\n" + mandated[0] + "\nThe report.\n")
        assert found is True
        assert body == "The report.\n"
