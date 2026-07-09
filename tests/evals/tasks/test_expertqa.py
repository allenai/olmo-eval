"""Tests for the ExpertQA attributed long-form QA task."""

import json
import logging

import pytest

from olmo_eval.common.types import (
    AgentTrajectory,
    AgentTurn,
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    ToolCall,
    ToolResult,
)
from olmo_eval.evals.suites import get_suite
from olmo_eval.evals.tasks import expertqa as expertqa_module
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.expertqa import EXPERTQA_OUTPUT_LABELS


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


@pytest.fixture
def task():
    return get_task("expertqa")


@pytest.fixture
def cite_task():
    return get_task("expertqa:cite")


class TestRegistration:
    def test_task_registered(self):
        assert "expertqa" in list_tasks()

    def test_cite_task_registered(self):
        assert "expertqa:cite" in list_tasks()

    def test_metrics_exclude_ingredient_recall(self, task):
        metric_names = {m.name for m in task.config.metrics}
        assert metric_names == {
            "global_avg",
            "citation_precision",
            "citation_recall",
            "answer_precision",
            "snippet_grounding_rate",
        }

    def test_primary_metric(self, task):
        # Primary metric is a citation tier, not the global_avg aggregate.
        assert task.config.get_primary_metric().name == "citation_recall"

    def test_declares_judge_secret(self, task):
        assert task.config.required_secrets == ("OPENAI_API_KEY",)

    def test_data_source(self, task):
        source = task.config.data_source
        assert source.path == "cmalaviya/expertqa"
        assert source.subset == "main"


class TestProcessDoc:
    def test_keeps_field_metadata(self, task):
        doc = {
            "question": "What causes treatment resistance in oncology?",
            "metadata": {
                "field": "Healthcare / Medicine",
                "specific_field": "Oncology",
                "question_type": "Directed question",
            },
        }
        instance = task.process_doc(doc, index=3)
        assert instance is not None
        assert instance.question.startswith("What causes")
        assert instance.metadata["field"] == "Healthcare / Medicine"
        assert instance.metadata["specific_field"] == "Oncology"
        assert instance.metadata["case_id"] == "expertqa_3"
        assert instance.metadata["index"] == 3

    def test_missing_question_skipped(self, task):
        assert task.process_doc({"question": ""}, index=0) is None

    def test_missing_metadata_defaults_empty(self, task):
        instance = task.process_doc({"question": "Q?"}, index=0)
        assert instance is not None
        assert instance.metadata["field"] == ""
        assert instance.metadata["specific_field"] == ""


class TestExtractAnswer:
    def test_valid_json_response(self, task):
        output = LMOutput(text='{"sections": [{"text": "hello", "citations": []}]}')
        result = task.extract_answer(output)
        assert result is not None
        assert "sections" in result
        assert output.metadata["parsed_response"] == result

    def test_invalid_json(self, task):
        output = LMOutput(text="not json at all")
        assert task.extract_answer(output) is None

    def test_strips_think_block(self, task):
        text = (
            '<think>\nplan: {"junk": 1}\n</think>\n{"sections": [{"text": "hi", "citations": []}]}'
        )
        output = LMOutput(text=text)
        result = task.extract_answer(output)
        assert result is not None
        assert "sections" in result

    def test_cite_extract_answer_parses_tags(self, cite_task):
        text = (
            "<think>\nplan\n</think>\n"
            'Intro <cite url="https://example.com/source">Grounded claim.</cite>'
        )
        output = LMOutput(text=text)
        result = cite_task.extract_answer(output)
        assert result is not None
        assert result["sections"][0]["text"] == "Intro Grounded claim. [1]"
        assert result["sections"][0]["citations"] == [
            {
                "id": "[1]",
                "url": "https://example.com/source",
                "title": "",
                "snippets": [],
            }
        ]
        assert output.metadata["parsed_response"] == result


class TestFormatRequest:
    def test_chat_request_embeds_question(self, task):
        instance = Instance(question="What is attention?", metadata={})
        request = task.format_request(instance)
        assert request.request_type == RequestType.CHAT
        assert len(request.messages) == 1
        content = request.messages[0]["content"]
        assert "well-cited report" in content
        assert "serper_google_webpage_search" in content
        assert "serper_fetch_webpage_content" in content
        assert "browse_webpage" in content
        assert "serper_fetch_webpage_content or browse_webpage" in content
        assert "`url` is the source page" in content
        assert "Your final answer must consist of ONLY a single JSON object" in content
        assert "The first character of your final answer must be '{'" in content
        assert "Do not use Markdown, code fences, headings" in content
        assert "Do not create a References section" in content
        assert "Do not answer from memory. You must search before answering" in content
        workflow_start = content.index("Workflow (follow in order):")
        assert workflow_start > content.index("Your final answer must consist")
        assert content.endswith("Question: What is attention?")

    def test_cite_request_uses_cite_prompt(self, cite_task):
        instance = Instance(question="What is attention?", metadata={})
        request = cite_task.format_request(instance)
        assert request.request_type == RequestType.CHAT
        content = request.messages[0]["content"]
        assert "Markdown is allowed" in content
        assert "Do not return JSON." in content
        assert '<cite url="https://example.com/page">claim text</cite>' in content
        assert "serper_google_webpage_search" in content
        assert "serper_fetch_webpage_content" in content
        assert "browse_webpage" in content
        assert "serper_fetch_webpage_content or browse_webpage" in content
        assert "Do not answer from memory. You must search before answering" in content
        workflow_start = content.index("Workflow (follow in order):")
        assert workflow_start > content.index("Do not return JSON.")
        assert "Return valid JSON" not in content
        assert content.endswith("Question: What is attention?")


class TestScoreResponses:
    def _response(self, parsed, trajectory):
        instance = Instance(question="What causes X?", metadata={})
        text = json.dumps(parsed) if parsed is not None else "not json at all"
        output = LMOutput(text=text)
        return Response(
            instance=instance,
            request=LMRequest(request_type=RequestType.CHAT, messages=()),
            outputs=[output],
            scores={},
            trajectory=trajectory,
        )

    @pytest.mark.anyio
    async def test_no_parsed_response_scores_zero(self, task, monkeypatch):
        monkeypatch.setattr(expertqa_module, "_build_judge_fn", lambda: _unused_judge)
        response = self._response(None, _trajectory_with_source(""))

        await task.score_responses([response])

        assert response.scores == {
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "answer_precision": 0.0,
            "snippet_grounding_rate": 0.0,
            "global_avg": 0.0,
        }
        assert response.outputs[0].metadata["grounding_stats"]["snippet_grounding_rate"] == 0.0

    @pytest.mark.anyio
    async def test_global_avg_is_mean_of_three_axes(self, task, monkeypatch):
        monkeypatch.setattr(expertqa_module, "_build_judge_fn", lambda: _citation_split_judge)
        snippet = "Evidence from a fetched source supports the first claim."
        parsed = {
            "sections": [
                {
                    "text": "Claim A [1]. Claim B [1].",
                    "citations": [{"id": "[1]", "snippets": [snippet], "title": "Paper"}],
                }
            ]
        }
        response = self._response(parsed, _trajectory_with_source(snippet))

        await task.score_responses([response])

        scores = response.scores
        # One of two claims attributable -> recall 0.5; precision averages 0.5;
        # no irrelevant paragraphs -> answer_precision 1.0.
        assert scores["citation_recall"] == pytest.approx(0.5)
        assert scores["citation_precision"] == pytest.approx(0.5)
        assert scores["answer_precision"] == pytest.approx(1.0)
        assert scores["snippet_grounding_rate"] == pytest.approx(1.0)
        assert scores["global_avg"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)

    @pytest.mark.anyio
    async def test_grounded_snippet_reaches_judge(self, task, monkeypatch):
        snippet = "This fetched passage exactly supports the grounded ExpertQA answer."
        judge = _RecordingJudge()
        monkeypatch.setattr(expertqa_module, "_build_judge_fn", lambda: judge)
        parsed = {
            "sections": [
                {
                    "text": "Grounded claim [1].",
                    "citations": [
                        {
                            "id": "[1]",
                            "url": "https://example.com/source",
                            "title": "Fetched Source",
                            "snippets": [snippet],
                        }
                    ],
                }
            ]
        }
        response = self._response(parsed, _trajectory_with_source(f"Fetched text: {snippet}"))

        await task.score_responses([response])

        citation_prompts = [p for p in judge.prompts if "References:" in p]
        assert citation_prompts
        assert snippet in citation_prompts[0]
        assert response.outputs[0].metadata["grounding_stats"] == {
            "n_snippets": 1.0,
            "n_grounded": 1.0,
            "snippet_grounding_rate": 1.0,
        }
        assert response.outputs[0].metadata["score:snippet_grounding_rate"] == pytest.approx(1.0)
        assert response.scores["snippet_grounding_rate"] == pytest.approx(1.0)

    @pytest.mark.anyio
    async def test_browse_webpage_grounded_snippet_reaches_judge(self, task, monkeypatch):
        url = "https://example.com/crawl4ai-source"
        snippet = "Crawl4ai fetched passage exactly supports the grounded ExpertQA answer."
        page_content = f"Fetched text: {snippet}"
        judge = _RecordingJudge()
        monkeypatch.setattr(expertqa_module, "_build_judge_fn", lambda: judge)
        parsed = {
            "sections": [
                {
                    "text": "Grounded crawl4ai claim [1].",
                    "citations": [
                        {
                            "id": "[1]",
                            "url": url,
                            "title": "Fetched Source",
                            "snippets": [snippet],
                        }
                    ],
                }
            ]
        }
        response = self._response(
            parsed,
            _trajectory_with_search_and_fetch(
                fetch_url=url,
                fetch_content=page_content,
                search_content=f"URL: {url}",
                fetch_tool_name="browse_webpage",
            ),
        )

        assert expertqa_module._trajectory_source_text(response) == page_content

        await task.score_responses([response])

        citation_prompts = [p for p in judge.prompts if "References:" in p]
        assert citation_prompts
        assert snippet in citation_prompts[0]
        assert response.outputs[0].metadata["grounding_stats"] == {
            "n_snippets": 1.0,
            "n_grounded": 1.0,
            "snippet_grounding_rate": 1.0,
        }
        assert response.scores["snippet_grounding_rate"] == pytest.approx(1.0)

    @pytest.mark.anyio
    async def test_fabricated_snippet_removed_before_judging(self, task, monkeypatch):
        grounded_snippet = "Fetched evidence appears verbatim in the trajectory source text."
        fabricated_snippet = "Fabricated evidence never appeared in any fetched page."
        judge = _RecordingJudge()
        monkeypatch.setattr(expertqa_module, "_build_judge_fn", lambda: judge)
        parsed = {
            "sections": [
                {
                    "text": "One grounded claim [1]. One fabricated citation [2].",
                    "citations": [
                        {"id": "[1]", "snippets": [grounded_snippet], "title": "Fetched"},
                        {"id": "[2]", "snippets": [fabricated_snippet], "title": "Missing"},
                    ],
                }
            ]
        }
        response = self._response(parsed, _trajectory_with_source(grounded_snippet))

        await task.score_responses([response])

        citation_prompts = [p for p in judge.prompts if "References:" in p]
        assert citation_prompts
        assert grounded_snippet in citation_prompts[0]
        assert fabricated_snippet not in citation_prompts[0]
        assert response.outputs[0].metadata["grounding_stats"]["n_snippets"] == 2.0
        assert response.outputs[0].metadata["grounding_stats"]["n_grounded"] == 1.0
        assert response.scores["snippet_grounding_rate"] == pytest.approx(0.5)

    @pytest.mark.anyio
    async def test_fully_fabricated_citation_scores_zero_with_supporting_judge(
        self, task, monkeypatch
    ):
        fabricated_snippet = "Fabricated ExpertQA evidence never appeared in retrieved sources."
        judge = _EverythingSupportingJudge()
        monkeypatch.setattr(expertqa_module, "_build_judge_fn", lambda: judge)
        parsed = {
            "sections": [
                {
                    "text": "A fully fabricated citation should not receive credit [1].",
                    "citations": [
                        {
                            "id": "[1]",
                            "snippets": [fabricated_snippet],
                            "title": "Invented Source Title",
                        }
                    ],
                }
            ]
        }
        response = self._response(
            parsed, _trajectory_with_source("Retrieved unrelated source text.")
        )

        await task.score_responses([response])

        assert response.scores["citation_precision"] == pytest.approx(0.0)
        assert response.scores["citation_recall"] == pytest.approx(0.0)
        assert response.scores["snippet_grounding_rate"] == pytest.approx(0.0)
        assert response.outputs[0].metadata["grounding_stats"] == {
            "n_snippets": 1.0,
            "n_grounded": 0.0,
            "snippet_grounding_rate": 0.0,
        }
        assert [p for p in judge.prompts if "References:" in p] == []

    @pytest.mark.anyio
    async def test_missing_trajectory_warns_and_ungrounds_snippets(self, task, monkeypatch, caplog):
        monkeypatch.setattr(expertqa_module, "_build_judge_fn", lambda: _citation_split_judge)
        caplog.set_level(logging.WARNING, logger=expertqa_module.__name__)
        parsed = {
            "sections": [
                {
                    "text": "Claim with missing trajectory [1].",
                    "citations": [
                        {
                            "id": "[1]",
                            "snippets": ["This snippet is long enough but has no source text."],
                            "title": "Missing",
                        }
                    ],
                }
            ]
        }
        response = self._response(parsed, trajectory=None)

        await task.score_responses([response])

        assert response.scores["snippet_grounding_rate"] == pytest.approx(0.0)
        assert "web_search_agent" in caplog.text
        assert "no trajectory" in caplog.text

    @pytest.mark.anyio
    async def test_multi_output_scores_metadata_and_uses_configured_aggregation(self, monkeypatch):
        task = get_task("expertqa", config_overrides={"output_score_aggregation": "first"})
        monkeypatch.setattr(expertqa_module, "_build_judge_fn", lambda: _citation_split_judge)
        grounded_snippet = "Grounded multi-output evidence appears in fetched content."
        fabricated = {
            "sections": [
                {
                    "text": "Ungrounded first output [1].",
                    "citations": [
                        {
                            "id": "[1]",
                            "snippets": ["This first output quote is not in trajectory text."],
                        }
                    ],
                }
            ]
        }
        grounded = {
            "sections": [
                {
                    "text": "Grounded second output [1].",
                    "citations": [{"id": "[1]", "snippets": [grounded_snippet]}],
                }
            ]
        }
        response = Response(
            instance=Instance(question="What causes X?", metadata={}),
            request=LMRequest(request_type=RequestType.CHAT, messages=()),
            outputs=[LMOutput(text=json.dumps(fabricated)), LMOutput(text=json.dumps(grounded))],
            scores={},
            trajectory=_trajectory_with_source(grounded_snippet),
        )

        await task.score_responses([response])

        for output in response.outputs:
            for label in EXPERTQA_OUTPUT_LABELS:
                assert f"score:{label}" in output.metadata
        assert response.outputs[0].metadata["score:snippet_grounding_rate"] == pytest.approx(0.0)
        assert response.outputs[1].metadata["score:snippet_grounding_rate"] == pytest.approx(1.0)
        assert response.scores["snippet_grounding_rate"] == pytest.approx(0.0)

    @pytest.mark.anyio
    async def test_cite_grounded_fetched_url_reaches_judge(self, cite_task, monkeypatch):
        page_excerpt = "Fetched page evidence supports the cite-tag ExpertQA claim."
        judge = _RecordingJudge()
        monkeypatch.setattr(expertqa_module, "_build_judge_fn", lambda: judge)
        response = _cite_response(
            '<cite url="https://example.com/source">Grounded cite claim.</cite>',
            _trajectory_with_search_and_fetch(
                fetch_url="https://example.com/source",
                fetch_content=page_excerpt,
                search_content="URL: https://example.com/source\nURL: https://example.com/other",
            ),
        )

        await cite_task.score_responses([response])

        citation_prompts = [p for p in judge.prompts if "References:" in p]
        assert citation_prompts
        assert page_excerpt in citation_prompts[0]
        assert response.outputs[0].metadata["grounding_stats"] == {
            "n_citations": 1.0,
            "n_grounded": 1.0,
            "n_half": 0.0,
            "snippet_grounding_rate": 1.0,
        }
        assert response.scores["snippet_grounding_rate"] == pytest.approx(1.0)

    @pytest.mark.anyio
    async def test_cite_invented_url_scores_zero_with_supporting_judge(
        self, cite_task, monkeypatch
    ):
        judge = _EverythingSupportingJudge()
        monkeypatch.setattr(expertqa_module, "_build_judge_fn", lambda: judge)
        response = _cite_response(
            '<cite url="https://invented.example/missing">Invented cite claim.</cite>',
            _trajectory_with_search_and_fetch(
                fetch_url="https://example.com/source",
                fetch_content="Fetched content for a different source.",
                search_content="URL: https://example.com/source",
            ),
        )

        await cite_task.score_responses([response])

        assert response.scores["citation_precision"] == pytest.approx(0.0)
        assert response.scores["citation_recall"] == pytest.approx(0.0)
        assert response.scores["snippet_grounding_rate"] == pytest.approx(0.0)
        assert response.outputs[0].metadata["grounding_stats"] == {
            "n_citations": 1.0,
            "n_grounded": 0.0,
            "n_half": 0.0,
            "snippet_grounding_rate": 0.0,
        }
        assert [p for p in judge.prompts if "References:" in p] == []

    @pytest.mark.anyio
    async def test_cite_empty_url_scores_zero_with_supporting_judge(self, cite_task, monkeypatch):
        judge = _EverythingSupportingJudge()
        monkeypatch.setattr(expertqa_module, "_build_judge_fn", lambda: judge)
        response = _cite_response(
            '<cite url="">Empty-url cite claim.</cite>',
            _trajectory_with_search_and_fetch(
                fetch_url="https://example.com/source",
                fetch_content="Fetched content for a source.",
                search_content="URL: https://example.com/source",
            ),
        )

        await cite_task.score_responses([response])

        assert response.outputs[0].metadata["parsed_response"] is None
        assert response.scores["citation_precision"] == pytest.approx(0.0)
        assert response.scores["citation_recall"] == pytest.approx(0.0)
        assert response.scores["snippet_grounding_rate"] == pytest.approx(0.0)
        assert response.outputs[0].metadata["grounding_stats"] == {
            "n_snippets": 0.0,
            "n_grounded": 0.0,
            "snippet_grounding_rate": 0.0,
        }
        assert [p for p in judge.prompts if "References:" in p] == []

    @pytest.mark.anyio
    async def test_cite_fetch_error_routes_to_half_credit(self, cite_task, monkeypatch):
        judge = _RecordingJudge()
        monkeypatch.setattr(expertqa_module, "_build_judge_fn", lambda: judge)
        response = _cite_response(
            '<cite url="https://example.com/source">Error-fetch cite claim.</cite>',
            _trajectory_with_search_and_fetch(
                fetch_url="https://example.com/source",
                fetch_content="Error fetching webpage: timed out",
                search_content="URL: https://example.com/source",
                fetch_is_error=True,
            ),
        )

        await cite_task.score_responses([response])

        citation_prompts = [p for p in judge.prompts if "References:" in p]
        assert citation_prompts
        assert "Paper content unavailable" in citation_prompts[0]
        assert "The paper's title is: https://example.com/source" in citation_prompts[0]
        assert "Error fetching webpage" not in citation_prompts[0]
        assert response.outputs[0].metadata["grounding_stats"] == {
            "n_citations": 1.0,
            "n_grounded": 0.0,
            "n_half": 1.0,
            "snippet_grounding_rate": 0.0,
        }


class TestSuiteMembership:
    def test_expertqa_research_only_for_science_execution_split(self):
        assert "expertqa" in get_suite("science:research").expand()
        assert "expertqa" not in get_suite("science:judge").expand()


class TestTrajectoryUrlContent:
    def test_browse_webpage_fetch_content_maps_url(self):
        url = "https://example.com/crawl4ai-source"
        fetch_content = "Crawl4ai fetched page evidence belongs to the cited URL."
        trajectory = _trajectory_with_search_and_fetch(
            fetch_url=url,
            fetch_content=fetch_content,
            search_content=f"URL: {url}",
            fetch_tool_name="browse_webpage",
        )

        url_to_content = expertqa_module._trajectory_url_content(_cite_response("", trajectory))

        assert url_to_content[url] == fetch_content

    def test_good_fetch_content_survives_later_error_refetch(self):
        url = "https://example.com/source"
        good_content = "Fetched page evidence that should remain mapped."
        trajectory = AgentTrajectory(
            turns=(
                AgentTurn.assistant(
                    tool_calls=[
                        ToolCall.create(
                            "fetch_1",
                            "serper_fetch_webpage_content",
                            {"url": url},
                        )
                    ]
                ),
                AgentTurn.tool(
                    [
                        ToolResult(
                            tool_call_id="fetch_1",
                            content=good_content,
                        )
                    ]
                ),
                AgentTurn.assistant(
                    tool_calls=[
                        ToolCall.create(
                            "fetch_2",
                            "serper_fetch_webpage_content",
                            {"url": url},
                        )
                    ]
                ),
                AgentTurn.tool(
                    [
                        ToolResult(
                            tool_call_id="fetch_2",
                            content="Error fetching webpage: timed out",
                            is_error=True,
                        )
                    ]
                ),
            )
        )

        url_to_content = expertqa_module._trajectory_url_content(_cite_response("", trajectory))

        assert url_to_content[url] == good_content

    def test_two_tool_calls_pair_results_by_order_when_ids_empty(self):
        fetch_url = "https://example.com/source"
        search_content = f"URL: {fetch_url}\nSearch result text should not be fetch evidence."
        fetch_content = "Fetched page evidence belongs to the fetch call."
        trajectory = AgentTrajectory(
            turns=(
                AgentTurn.assistant(
                    tool_calls=[
                        ToolCall.create(
                            "",
                            "serper_google_webpage_search",
                            {"query": "expertqa source"},
                        ),
                        ToolCall.create(
                            "",
                            "serper_fetch_webpage_content",
                            {"url": fetch_url},
                        ),
                    ]
                ),
                AgentTurn.tool(
                    [
                        ToolResult(tool_call_id="", content=search_content),
                        ToolResult(tool_call_id="", content=fetch_content),
                    ]
                ),
            )
        )

        url_to_content = expertqa_module._trajectory_url_content(_cite_response("", trajectory))

        assert url_to_content[fetch_url] == fetch_content
        assert url_to_content[fetch_url] != search_content


async def _unused_judge(prompt, **kwargs):
    raise AssertionError("judge should not be called")


async def _citation_split_judge(prompt, **kwargs):
    """Stub judge: no irrelevant paragraphs, one supported and one unsupported claim."""
    if "irrelevant paragraphs" in prompt:
        return json.dumps({"irrelevant_paragraphs": []})
    return json.dumps(
        {
            "claims": [
                {
                    "text": "Claim A",
                    "supporting": ["[1]"],
                    "non_supporting": [],
                    "is_fully_supported": True,
                },
                {
                    "text": "Claim B",
                    "supporting": [],
                    "non_supporting": ["[1]"],
                    "is_fully_supported": False,
                },
            ]
        }
    )


class _RecordingJudge:
    def __init__(self):
        self.prompts = []

    async def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if "irrelevant paragraphs" in prompt:
            return json.dumps({"irrelevant_paragraphs": []})
        return json.dumps(
            {
                "claims": [
                    {
                        "text": "Claim",
                        "supporting": ["[1]"],
                        "non_supporting": [],
                        "is_fully_supported": True,
                    }
                ]
            }
        )


class _EverythingSupportingJudge:
    def __init__(self):
        self.prompts = []

    async def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if "irrelevant paragraphs" in prompt:
            return json.dumps({"irrelevant_paragraphs": []})
        return json.dumps(
            {
                "claims": [
                    {
                        "text": "Claim",
                        "supporting": ["[1]"],
                        "non_supporting": [],
                        "is_fully_supported": True,
                    }
                ]
            }
        )


def _trajectory_with_source(source_text):
    return AgentTrajectory(
        turns=(
            AgentTurn.tool(
                [
                    ToolResult(
                        tool_call_id="call_1",
                        content=source_text,
                    )
                ]
            ),
        )
    )


def _cite_response(text, trajectory):
    return Response(
        instance=Instance(question="What causes X?", metadata={}),
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text=text)],
        scores={},
        trajectory=trajectory,
    )


def _trajectory_with_search_and_fetch(
    fetch_url,
    fetch_content,
    search_content,
    fetch_is_error=False,
    fetch_tool_name="serper_fetch_webpage_content",
):
    return AgentTrajectory(
        turns=(
            AgentTurn.assistant(
                tool_calls=[
                    ToolCall.create(
                        "",
                        "serper_google_webpage_search",
                        {"query": "expertqa source"},
                    )
                ]
            ),
            AgentTurn.tool(
                [
                    ToolResult(
                        tool_call_id="",
                        content=search_content,
                    )
                ]
            ),
            AgentTurn.assistant(
                tool_calls=[
                    ToolCall.create(
                        "",
                        fetch_tool_name,
                        {"url": fetch_url},
                    )
                ]
            ),
            AgentTurn.tool(
                [
                    ToolResult(
                        tool_call_id="",
                        content=fetch_content,
                        is_error=fetch_is_error,
                    )
                ]
            ),
        )
    )
