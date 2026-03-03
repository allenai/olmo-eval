"""Tests for AstaBench ScholArQA task logic."""

import json

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.astabench_sqa import (
    JUST_HAS_A_TITLE,
    AnswerPrecisionMetric,
    CitationPrecisionMetric,
    CitationRecallMetric,
    GlobalAvgMetric,
    IngredientRecallMetric,
    _filter_citation,
    compute_citation_scores_from_groups,
    compute_ingredient_score,
    compute_precision_score,
    extract_json_from_response,
    format_report,
    normalize_agent_response_dict,
    score_citation_group,
)
from olmo_eval.evals.tasks.common import get_task, list_tasks


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


# =============================================================================
# Registration
# =============================================================================


class TestRegistration:
    def test_task_registered(self):
        assert "astabench_scholarqa" in list_tasks()

    def test_get_task(self):
        task = get_task("astabench_scholarqa")
        assert task.config.name == "astabench_scholarqa"

    def test_test_variant(self):
        task = get_task("astabench_scholarqa:test")
        assert task.config.data_source.data_files == "tasks/sqa/rubrics_v2_recomputed.json"

    def test_has_all_metrics(self):
        task = get_task("astabench_scholarqa")
        metric_names = {m.name for m in task.config.metrics}
        assert metric_names == {
            "global_avg",
            "ingredient_recall",
            "answer_precision",
            "citation_precision",
            "citation_recall",
        }


# =============================================================================
# JSON Parsing
# =============================================================================


class TestExtractJson:
    def test_valid_json(self):
        result = extract_json_from_response('{"sections": []}')
        assert result == {"sections": []}

    def test_json_with_prefix(self):
        result = extract_json_from_response('Here is the result: {"key": "value"}')
        assert result == {"key": "value"}

    def test_no_json(self):
        assert extract_json_from_response("no json here") is None

    def test_empty_string(self):
        assert extract_json_from_response("") is None

    def test_malformed_json(self):
        assert extract_json_from_response("{malformed}") is None


class TestNormalizeResponse:
    def test_direct_sections(self):
        d = {"sections": [{"text": "hello"}]}
        assert normalize_agent_response_dict(d) == d

    def test_nested_response(self):
        d = {"response": {"sections": [{"text": "hello"}]}}
        result = normalize_agent_response_dict(d)
        assert result == {"sections": [{"text": "hello"}]}

    def test_empty_sections_with_response(self):
        d = {"sections": [], "response": {"sections": [{"text": "hi"}]}}
        # sections is empty (falsy list), but it exists, so don't unwrap
        # Actually [] is falsy, so it will return response
        result = normalize_agent_response_dict(d)
        assert result == {"sections": [{"text": "hi"}]}


# =============================================================================
# format_report
# =============================================================================


class TestFormatReport:
    def test_basic(self):
        response = {
            "sections": [
                {"text": "First section text."},
                {"text": "Second section text."},
            ]
        }
        result = format_report(response)
        assert "First section text." in result
        assert "Second section text." in result

    def test_with_table(self):
        response = {
            "sections": [
                {
                    "text": "Main text.",
                    "table": {"text": "Table content."},
                }
            ]
        }
        result = format_report(response)
        assert "Main text." in result
        assert "Table content." in result

    def test_empty_sections(self):
        assert format_report({"sections": []}) == ""

    def test_missing_text(self):
        response = {"sections": [{"title": "Title only"}]}
        assert format_report(response) == ""


# =============================================================================
# process_doc
# =============================================================================


class TestProcessDoc:
    @pytest.fixture
    def task(self):
        return get_task("astabench_scholarqa")

    def test_basic_conversion(self, task):
        doc = {
            "case_id": "test_001",
            "question": "What is the impact of attention mechanisms?",
            "annotator": "expert_1",
            "ingredients": [
                {
                    "name": "attention_types",
                    "criterion": "Discuss different types of attention",
                    "weight": 0.5,
                    "examples": ["self-attention", "cross-attention"],
                }
            ],
        }
        instance = task.process_doc(doc, index=0)
        assert instance is not None
        assert instance.question == "What is the impact of attention mechanisms?"
        assert instance.metadata["case_id"] == "test_001"
        assert instance.metadata["annotator"] == "expert_1"
        assert "ingredients" in instance.metadata["rubric_config"]
        assert len(instance.metadata["rubric_config"]["ingredients"]) == 1

    def test_missing_question_skipped(self, task):
        doc = {"case_id": "empty", "question": ""}
        assert task.process_doc(doc, index=0) is None

    def test_empty_ingredients(self, task):
        doc = {
            "question": "Test question?",
            "ingredients": [],
        }
        instance = task.process_doc(doc, index=0)
        assert instance is not None
        assert isinstance(instance.metadata["rubric_config"], dict)
        assert instance.metadata["rubric_config"]["ingredients"] == []

    def test_missing_ingredients(self, task):
        doc = {
            "question": "Test question?",
        }
        instance = task.process_doc(doc, index=0)
        assert instance is not None
        assert instance.metadata["rubric_config"]["ingredients"] == []


# =============================================================================
# extract_answer
# =============================================================================


class TestExtractAnswer:
    @pytest.fixture
    def task(self):
        return get_task("astabench_scholarqa")

    def test_valid_json_response(self, task):
        output = LMOutput(text='{"sections": [{"text": "hello", "citations": []}]}')
        result = task.extract_answer(output)
        assert result is not None
        assert "sections" in result

    def test_nested_response_format(self, task):
        output = LMOutput(text='{"response": {"sections": [{"text": "hello", "citations": []}]}}')
        result = task.extract_answer(output)
        assert result is not None
        assert "sections" in result

    def test_invalid_json(self, task):
        output = LMOutput(text="This is not valid JSON at all")
        result = task.extract_answer(output)
        assert result is None

    def test_stores_in_metadata(self, task):
        output = LMOutput(text='{"sections": []}')
        task.extract_answer(output)
        assert "parsed_response" in output.metadata

    def test_strips_think_block(self, task):
        text = (
            '<think>\nI should format as {"key": "val"}\n</think>\n'
            '{"sections": [{"text": "hello", "citations": []}]}'
        )
        output = LMOutput(text=text)
        result = task.extract_answer(output)
        assert result is not None
        assert "sections" in result


# =============================================================================
# format_request
# =============================================================================


class TestFormatRequest:
    def test_chat_request(self):
        task = get_task("astabench_scholarqa")
        instance = Instance(
            question="What is attention?",
            metadata={"case_id": "test", "rubric_config": {}},
        )
        request = task.format_request(instance)
        assert request.request_type == RequestType.CHAT
        assert len(request.messages) == 1
        assert "What is attention?" in request.messages[0]["content"]
        assert "Generate a report" in request.messages[0]["content"]


# =============================================================================
# Precision Scoring
# =============================================================================


class TestComputePrecisionScore:
    def test_no_irrelevant(self):
        parsed = {"irrelevant_paragraphs": []}
        answer = "Para 1.\n\nPara 2.\n\nPara 3."
        score, meta = compute_precision_score(parsed, answer)
        assert score == pytest.approx(1.0)

    def test_one_irrelevant_of_three(self):
        parsed = {"irrelevant_paragraphs": [{"reason": "off topic", "answer_text": "Para 2."}]}
        answer = "Para 1.\n\nPara 2.\n\nPara 3."
        score, meta = compute_precision_score(parsed, answer)
        assert score == pytest.approx(2 / 3)

    def test_all_irrelevant(self):
        text_a = "This paragraph discusses cats and their behavior."
        text_b = "Another paragraph about the history of ancient Rome."
        parsed = {
            "irrelevant_paragraphs": [
                {"reason": "off topic", "answer_text": text_a},
                {"reason": "off topic", "answer_text": text_b},
            ]
        }
        answer = f"{text_a}\n\n{text_b}"
        score, meta = compute_precision_score(parsed, answer)
        assert score == pytest.approx(0.0)

    def test_fuzzy_matching(self):
        # Slightly different text should still match with 85% threshold
        parsed = {
            "irrelevant_paragraphs": [
                {"reason": "off topic", "answer_text": "This is a paragraph about something."}
            ]
        }
        answer = "This is a paragraph about something.\n\nAnother paragraph."
        score, meta = compute_precision_score(parsed, answer)
        assert score == pytest.approx(0.5)

    def test_empty_answer(self):
        parsed = {"irrelevant_paragraphs": []}
        score, meta = compute_precision_score(parsed, "")
        assert score == pytest.approx(1.0)


# =============================================================================
# Ingredient Scoring
# =============================================================================


class TestComputeIngredientScore:
    def test_all_perfect(self):
        parsed = {
            "scores": [
                {"criteria_idx": 1, "score": 2, "reasoning": "good", "evidence": ""},
                {"criteria_idx": 2, "score": 2, "reasoning": "good", "evidence": ""},
            ]
        }
        ingredients = [
            {"name": "a", "criterion": "criterion A", "weight": 0.5, "examples": []},
            {"name": "b", "criterion": "criterion B", "weight": 0.5, "examples": []},
        ]
        assert compute_ingredient_score(parsed, ingredients) == pytest.approx(1.0)

    def test_all_zero(self):
        parsed = {
            "scores": [
                {"criteria_idx": 1, "score": 0, "reasoning": "bad", "evidence": ""},
            ]
        }
        ingredients = [
            {"name": "a", "criterion": "criterion A", "weight": 1.0, "examples": []},
        ]
        assert compute_ingredient_score(parsed, ingredients) == pytest.approx(0.0)

    def test_partial(self):
        parsed = {
            "scores": [
                {"criteria_idx": 1, "score": 1, "reasoning": "ok", "evidence": ""},
            ]
        }
        ingredients = [
            {"name": "a", "criterion": "criterion A", "weight": 1.0, "examples": []},
        ]
        assert compute_ingredient_score(parsed, ingredients) == pytest.approx(0.5)

    def test_unequal_weights(self):
        parsed = {
            "scores": [
                {"criteria_idx": 1, "score": 2, "reasoning": "", "evidence": ""},
                {"criteria_idx": 2, "score": 0, "reasoning": "", "evidence": ""},
            ]
        }
        ingredients = [
            {"name": "a", "criterion": "crit A", "weight": 0.75, "examples": []},
            {"name": "b", "criterion": "crit B", "weight": 0.25, "examples": []},
        ]
        # a: 1.0 * 0.75, b: 0.0 * 0.25 => 0.75
        assert compute_ingredient_score(parsed, ingredients) == pytest.approx(0.75)

    def test_no_criteria(self):
        parsed = {"scores": []}
        ingredients = [{"name": "a", "weight": 0.5, "examples": []}]
        assert compute_ingredient_score(parsed, ingredients) == pytest.approx(0.0)

    def test_skips_ingredients_without_criterion(self):
        parsed = {
            "scores": [
                {"criteria_idx": 1, "score": 2, "reasoning": "", "evidence": ""},
            ]
        }
        ingredients = [
            {"name": "a", "criterion": "crit A", "weight": 0.5, "examples": []},
            {"name": "b", "weight": 0.5, "examples": []},  # No criterion
        ]
        # Only "a" has criterion, so weight gets normalized to 1.0
        assert compute_ingredient_score(parsed, ingredients) == pytest.approx(1.0)


# =============================================================================
# Citation Scoring
# =============================================================================


class TestComputeCitationScoresFromGroups:
    def test_all_supported(self):
        results = [
            {
                "n_attributable": 3,
                "n_extrapolatory": 0,
                "n_half_credit_claims": 0,
                "supporting_counts": [2, 1, 1],
                "non_supporting_counts": [0, 0, 0],
                "n_half_credit_citations": [0, 0, 0],
            }
        ]
        scores = compute_citation_scores_from_groups(results)
        assert scores["citation_recall"] == pytest.approx(1.0)
        assert scores["citation_precision"] == pytest.approx(1.0)

    def test_none_supported(self):
        results = [
            {
                "n_attributable": 0,
                "n_extrapolatory": 3,
                "n_half_credit_claims": 0,
                "supporting_counts": [],
                "non_supporting_counts": [],
                "n_half_credit_citations": [],
            }
        ]
        scores = compute_citation_scores_from_groups(results)
        assert scores["citation_recall"] == pytest.approx(0.0)
        assert scores["citation_precision"] == pytest.approx(0.0)

    def test_half_credit_reduces_recall(self):
        results = [
            {
                "n_attributable": 2,
                "n_extrapolatory": 0,
                "n_half_credit_claims": 2,
                "supporting_counts": [1, 1],
                "non_supporting_counts": [0, 0],
                "n_half_credit_citations": [1, 1],
            }
        ]
        scores = compute_citation_scores_from_groups(results)
        # recall = (2 - 0.5*2) / (2 + 0) = 1/2 = 0.5
        assert scores["citation_recall"] == pytest.approx(0.5)

    def test_mixed_precision(self):
        results = [
            {
                "n_attributable": 1,
                "n_extrapolatory": 1,
                "n_half_credit_claims": 0,
                "supporting_counts": [2, 0],
                "non_supporting_counts": [1, 1],
                "n_half_credit_citations": [0, 0],
            }
        ]
        scores = compute_citation_scores_from_groups(results)
        # precision for claim 1: (2 - 0) / (2 + 1) = 2/3
        # claim 2: supporting=0 but non_supporting=1, so 0+1=1 > 0 => (0-0)/1 = 0
        # mean precision: (2/3 + 0) / 2 = 1/3
        assert scores["citation_precision"] == pytest.approx(1 / 3)

    def test_empty_groups(self):
        scores = compute_citation_scores_from_groups([])
        assert scores["citation_recall"] == pytest.approx(0.0)
        assert scores["citation_precision"] == pytest.approx(0.0)


# =============================================================================
# score_citation_group (with stub judge)
# =============================================================================


class TestScoreCitationGroup:
    def test_no_citations_uses_sentence_count(self):
        """With no citations, n_extrapolatory should reflect sentence count."""

        def stub_judge(prompt, **kwargs):
            raise AssertionError("should not be called")

        result = score_citation_group(
            stub_judge,
            "First sentence. Second sentence. Third sentence.",
            [],
        )
        assert result["n_attributable"] == 0
        assert result["n_extrapolatory"] == 3

    def test_invalid_json_fallback_uses_sentence_count(self):
        """When judge returns garbage, fallback should count sentences."""

        def stub_judge(prompt, **kwargs):
            return "this is not json at all"

        result = score_citation_group(
            stub_judge,
            "Sentence one. Sentence two.",
            [{"id": "[1]", "snippets": "some snippet"}],
        )
        assert result["n_attributable"] == 0
        assert result["n_extrapolatory"] == 2

    def test_valid_judge_response(self):
        """Normal judge response should produce correct counts."""
        judge_response = json.dumps(
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

        def stub_judge(prompt, **kwargs):
            return judge_response

        result = score_citation_group(
            stub_judge,
            "Claim A [1]. Claim B [1].",
            [{"id": "[1]", "snippets": "evidence text"}],
        )
        assert result["n_attributable"] == 1
        assert result["n_extrapolatory"] == 1
        assert result["supporting_counts"] == [1, 0]
        assert result["non_supporting_counts"] == [0, 1]

    def test_half_credit_title_only(self):
        """Citations with only a title should get half credit."""
        judge_response = json.dumps(
            {
                "claims": [
                    {
                        "text": "Claim A",
                        "supporting": ["[1]"],
                        "non_supporting": [],
                        "is_fully_supported": True,
                    },
                ]
            }
        )

        def stub_judge(prompt, **kwargs):
            return judge_response

        result = score_citation_group(
            stub_judge,
            "Claim A [1].",
            [{"id": "[1]", "snippets": f"{JUST_HAS_A_TITLE}Some Paper"}],
        )
        assert result["n_attributable"] == 1
        assert result["n_half_credit_claims"] == 1
        assert result["n_half_credit_citations"] == [1]


# =============================================================================
# _filter_citation
# =============================================================================


class TestFilterCitation:
    def test_usable_snippets_pass(self):
        """Citation with real snippets not in the text should pass."""
        citation = {
            "snippets": ["This is evidence from the paper."],
            "title": "Paper Title",
        }
        assert _filter_citation(citation, "The section discusses CNNs.") is True

    def test_snippet_in_text_fails(self):
        """Citation whose snippet appears verbatim in the text should fail."""
        citation = {
            "snippets": ["CNNs are great"],
            "title": "Paper Title",
        }
        # Alpha-only comparison: "cnnsaregreat" in "thesectiondiscussescnnsaregreat"
        assert _filter_citation(citation, "The section discusses CNNs are great.") is False

    def test_empty_snippets_fails(self):
        """Citation with no snippets should fail."""
        citation = {"snippets": [], "title": "Paper Title"}
        assert _filter_citation(citation, "Some text.") is False

    def test_snippet_matches_title_fails(self):
        """Citation whose snippet is just the title should fail."""
        citation = {
            "snippets": ["Paper Title"],
            "title": "Paper Title",
        }
        assert _filter_citation(citation, "Some unrelated text.") is False

    def test_no_title_with_real_snippets_passes(self):
        """Citation without a title but with real snippets should pass."""
        citation = {"snippets": ["Unique evidence text."]}
        assert _filter_citation(citation, "Different section text.") is True

    def test_string_snippets_treated_as_single_item(self):
        """String snippets should be normalized to a list, not iterated as chars."""
        citation = {"snippets": "Unique evidence text.", "title": "Paper Title"}
        # Should pass: the snippet is not in the section text and differs from title
        assert _filter_citation(citation, "Different section text.") is True

    def test_string_snippets_in_text_fails(self):
        """String snippet that matches section text should fail."""
        citation = {"snippets": "CNNs are great", "title": "Paper Title"}
        assert _filter_citation(citation, "The section discusses CNNs are great.") is False


# =============================================================================
# _score_citations: title-only half-credit
# =============================================================================


class TestScoreCitationsHalfCredit:
    def _build_citations(self, sec_text, raw_citations):
        """Replicate the citation-building logic from _score_citations."""
        bad_snippet = "Please click on the paper title to read the abstract on Semantic Scholar."
        citations = []
        for c in raw_citations:
            cit_id = c.get("id")
            if not cit_id:
                continue
            snippets = c.get("snippets", [])
            if isinstance(snippets, list):
                snippet_text = "... ".join(str(s) for s in snippets)
            else:
                snippet_text = str(snippets)

            if _filter_citation(c, sec_text) and bad_snippet not in snippet_text:
                citations.append({"id": cit_id, "snippets": snippet_text})
            else:
                title = c.get("title", "")
                if title:
                    citations.append({"id": cit_id, "snippets": f"{JUST_HAS_A_TITLE}{title}"})
                else:
                    citations.append({"id": cit_id, "snippets": ""})
        return citations

    def test_no_title_no_snippets_gets_zero_credit(self):
        """Citation with no valid snippets AND no title should not get half-credit."""
        citations = self._build_citations(
            "Claim A [1].",
            [{"id": "[1]", "snippets": [], "title": ""}],
        )
        half_credit_ids = [c["id"] for c in citations if c["snippets"].startswith(JUST_HAS_A_TITLE)]
        assert "[1]" not in half_credit_ids
        assert citations[0]["snippets"] == ""

    def test_with_title_gets_half_credit_marker(self):
        """Citation with a title but no valid snippets should get the half-credit marker."""
        citations = self._build_citations(
            "Claim A [1].",
            [{"id": "[1]", "snippets": [], "title": "Real Paper Title"}],
        )
        half_credit_ids = [c["id"] for c in citations if c["snippets"].startswith(JUST_HAS_A_TITLE)]
        assert "[1]" in half_credit_ids


# =============================================================================
# Metric Computation
# =============================================================================


class TestMetrics:
    def _make_response(self, scores: dict[str, float]) -> Response:
        return Response(
            instance=Instance(question="Q?", metadata={}),
            request=LMRequest(request_type=RequestType.CHAT, messages=()),
            outputs=[LMOutput(text="")],
            scores=scores,
        )

    def test_global_avg_metric(self):
        metric = GlobalAvgMetric()
        responses = [
            self._make_response({"global_avg": 0.8}),
            self._make_response({"global_avg": 0.6}),
        ]
        assert metric.compute(responses) == pytest.approx(0.7)

    def test_ingredient_recall_metric(self):
        metric = IngredientRecallMetric()
        responses = [
            self._make_response({"ingredient_recall": 1.0}),
            self._make_response({"ingredient_recall": 0.5}),
        ]
        assert metric.compute(responses) == pytest.approx(0.75)

    def test_empty_responses(self):
        assert GlobalAvgMetric().compute([]) == pytest.approx(0.0)
        assert AnswerPrecisionMetric().compute([]) == pytest.approx(0.0)
        assert CitationPrecisionMetric().compute([]) == pytest.approx(0.0)
        assert CitationRecallMetric().compute([]) == pytest.approx(0.0)
