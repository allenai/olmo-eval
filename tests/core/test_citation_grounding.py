"""Tests for trajectory-source grounding of citation snippets."""

from __future__ import annotations

import copy

import pytest

from olmo_eval.common.scorers.citation import ground_citations_in_sources


def test_grounded_snippet_kept():
    snippet = "The Eiffel Tower opened in 1889 for the world's fair."
    response = {
        "sections": [
            {
                "text": "The Eiffel Tower opened in 1889 [1].",
                "citations": [{"id": "[1]", "snippets": [snippet], "title": "Tower"}],
            }
        ]
    }

    grounded, stats = ground_citations_in_sources(response, f"Source says: {snippet}")

    assert grounded["sections"][0]["citations"][0]["snippets"] == [snippet]
    assert stats["n_snippets"] == 1
    assert stats["n_grounded"] == 1
    assert stats["snippet_grounding_rate"] == pytest.approx(1.0)


def test_fabricated_only_citation_removed():
    response = {
        "sections": [
            {
                "text": "The answer cites a fabricated quote [1].",
                "citations": [
                    {
                        "id": "[1]",
                        "snippets": ["This fabricated passage is not present in sources."],
                        "title": "Missing Title",
                    }
                ],
            }
        ]
    }

    grounded, stats = ground_citations_in_sources(response, "The source has unrelated text.")

    assert grounded["sections"][0]["citations"] == []
    assert stats["n_snippets"] == 1
    assert stats["n_grounded"] == 0
    assert stats["snippet_grounding_rate"] == pytest.approx(0.0)


def test_mixed_citation_keeps_only_grounded_snippets():
    grounded_snippet = "Grounded evidence appears verbatim in the fetched page."
    fabricated_snippet = "Fabricated evidence does not appear anywhere."
    response = {
        "sections": [
            {
                "text": "Only one quote is grounded [1].",
                "citations": [
                    {
                        "id": "[1]",
                        "snippets": [grounded_snippet, fabricated_snippet],
                        "title": "Fetched Page",
                    }
                ],
            }
        ]
    }

    grounded, stats = ground_citations_in_sources(response, grounded_snippet)

    assert grounded["sections"][0]["citations"][0]["snippets"] == [grounded_snippet]
    assert stats["n_snippets"] == 2
    assert stats["n_grounded"] == 1
    assert stats["snippet_grounding_rate"] == pytest.approx(0.5)


def test_normalization_allows_case_punctuation_and_whitespace_differences():
    response = {
        "sections": [
            {
                "text": "Normalization should ground this [1].",
                "citations": [
                    {
                        "id": "[1]",
                        "snippets": ["Alpha beta gamma delta evidence string"],
                    }
                ],
            }
        ]
    }
    source = "ALPHA-beta,\ngamma    delta evidence string!"

    grounded, stats = ground_citations_in_sources(response, source)

    assert grounded["sections"][0]["citations"][0]["snippets"] == [
        "Alpha beta gamma delta evidence string"
    ]
    assert stats["snippet_grounding_rate"] == pytest.approx(1.0)


def test_short_snippet_removed_and_counted():
    response = {
        "sections": [
            {
                "text": "The city is Paris [1].",
                "citations": [{"id": "[1]", "snippets": ["Paris"], "title": "Paris"}],
            }
        ]
    }

    grounded, stats = ground_citations_in_sources(response, "Paris")

    assert grounded["sections"][0]["citations"][0]["snippets"] == []
    assert grounded["sections"][0]["citations"][0]["title"] == "Paris"
    assert stats["n_snippets"] == 1
    assert stats["n_grounded"] == 0
    assert stats["snippet_grounding_rate"] == pytest.approx(0.0)


def test_grounded_title_only_citation_retained():
    response = {
        "sections": [
            {
                "text": "Title-only citations get filtered [1][2].",
                "citations": [
                    {"id": "[1]", "snippets": [], "title": "Important Study Title"},
                    {"id": "[2]", "snippets": [], "title": "Absent Study Title"},
                ],
            }
        ]
    }

    grounded, stats = ground_citations_in_sources(response, "Important Study Title appears here.")

    citations = grounded["sections"][0]["citations"]
    assert len(citations) == 1
    assert citations[0]["id"] == "[1]"
    assert citations[0]["title"] == "Important Study Title"
    assert stats["n_snippets"] == 0
    assert stats["snippet_grounding_rate"] == pytest.approx(0.0)


def test_idless_citation_left_untouched_and_not_counted():
    idless_snippet = "Idless evidence appears verbatim in source text."
    grounded_snippet = "Grounded cited evidence appears verbatim in source text."
    response = {
        "sections": [
            {
                "text": "Only id-bearing snippets count [1].",
                "citations": [
                    {"snippets": [idless_snippet], "title": "Idless Title"},
                    {"id": "[1]", "snippets": [grounded_snippet], "title": "Cited Title"},
                ],
            }
        ]
    }

    grounded, stats = ground_citations_in_sources(response, f"{idless_snippet} {grounded_snippet}")

    citations = grounded["sections"][0]["citations"]
    assert citations[0] == {"snippets": [idless_snippet], "title": "Idless Title"}
    assert citations[1]["snippets"] == [grounded_snippet]
    assert stats["n_snippets"] == 1
    assert stats["n_grounded"] == 1
    assert stats["snippet_grounding_rate"] == pytest.approx(1.0)


def test_table_sub_section_is_grounded():
    snippet = "Table evidence value was copied from fetched content."
    response = {
        "sections": [
            {
                "text": "Main text.",
                "citations": [],
                "table": {
                    "text": "A table claim [1].",
                    "citations": [{"id": "[1]", "snippets": [snippet], "title": "Table"}],
                },
            }
        ]
    }

    grounded, stats = ground_citations_in_sources(response, snippet)

    table_citation = grounded["sections"][0]["table"]["citations"][0]
    assert table_citation["snippets"] == [snippet]
    assert stats["snippet_grounding_rate"] == pytest.approx(1.0)


def test_empty_sections_rate_zero():
    grounded, stats = ground_citations_in_sources({"sections": []}, "source")

    assert grounded == {"sections": []}
    assert stats == {
        "n_snippets": 0.0,
        "n_grounded": 0.0,
        "snippet_grounding_rate": 0.0,
    }


def test_input_dict_not_mutated():
    response = {
        "sections": [
            {
                "text": "Fabricated citation [1].",
                "citations": [
                    {
                        "id": "[1]",
                        "snippets": ["This fabricated passage is not in sources."],
                        "title": "Missing Title",
                    }
                ],
            }
        ]
    }
    original = copy.deepcopy(response)

    ground_citations_in_sources(response, "unrelated source")

    assert response == original
