"""Tests for trajectory-source grounding of citation snippets."""

from __future__ import annotations

import copy

import pytest

from olmo_eval.common.scorers.citation import (
    ground_citations_by_url,
    ground_citations_in_sources,
    parse_cite_tag_response,
)


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


class TestParseCiteTagResponse:
    def test_parse_well_formed_tag(self):
        parsed = parse_cite_tag_response(
            'Intro <cite url="https://example.com/page">Alpha claim.</cite> Outro.'
        )

        assert parsed == {
            "sections": [
                {
                    "title": "",
                    "text": "Intro Alpha claim. [1] Outro.",
                    "citations": [
                        {
                            "id": "[1]",
                            "url": "https://example.com/page",
                            "title": "",
                            "snippets": [],
                        }
                    ],
                }
            ]
        }

    @pytest.mark.parametrize(
        "text",
        [
            "No citation tags here.",
            '<cite url="https://example.com/page">Missing close tag.',
            "<cite>Missing URL.</cite>",
            '<cite data-url="https://example.com/page">Wrong attr.</cite>',
        ],
    )
    def test_parse_none_or_malformed_returns_none(self, text):
        assert parse_cite_tag_response(text) is None

    @pytest.mark.parametrize(
        "text",
        [
            '<cite url="">Empty URL claim.</cite>',
            '<cite url="   ">Blank URL claim.</cite>',
        ],
    )
    def test_empty_url_tag_returns_none(self, text):
        assert parse_cite_tag_response(text) is None

    def test_single_and_double_quotes_with_whitespace(self):
        parsed = parse_cite_tag_response(
            "<cite  url = 'https://example.com/a' >Claim A.</cite> "
            '<cite url="https://example.com/b">Claim B.</cite>'
        )

        assert parsed is not None
        section = parsed["sections"][0]
        assert section["text"] == "Claim A. [1] Claim B. [2]"
        assert [c["url"] for c in section["citations"]] == [
            "https://example.com/a",
            "https://example.com/b",
        ]

    def test_duplicate_urls_share_id(self):
        parsed = parse_cite_tag_response(
            '<cite url="https://example.com/a">Claim A.</cite> '
            '<cite url="https://example.com/a">Claim B.</cite>'
        )

        assert parsed is not None
        section = parsed["sections"][0]
        assert section["text"] == "Claim A. [1] Claim B. [1]"
        assert section["citations"] == [
            {
                "id": "[1]",
                "url": "https://example.com/a",
                "title": "",
                "snippets": [],
            }
        ]


class TestGroundCitationsByUrl:
    def test_content_grounded_citation_gets_excerpt(self):
        response = {
            "sections": [
                {
                    "text": "Grounded claim [1].",
                    "citations": [{"id": "[1]", "url": "https://example.com/a"}],
                }
            ]
        }

        grounded, stats = ground_citations_by_url(
            response,
            {"https://example.com/a": "Fetched content supports the claim."},
            max_evidence_chars=15,
        )

        citation = grounded["sections"][0]["citations"][0]
        assert citation["snippets"] == ["Fetched content"]
        assert stats == {
            "n_citations": 1.0,
            "n_grounded": 1.0,
            "n_half": 0.0,
            "snippet_grounding_rate": 1.0,
        }

    def test_seen_no_content_citation_kept_for_half_credit(self):
        response = {
            "sections": [
                {
                    "text": "Search-result claim [1].",
                    "citations": [{"id": "[1]", "url": "https://example.com/a"}],
                }
            ]
        }

        grounded, stats = ground_citations_by_url(response, {"https://example.com/a": ""})

        citation = grounded["sections"][0]["citations"][0]
        assert citation["snippets"] == []
        assert citation["title"] == "https://example.com/a"
        assert stats["n_citations"] == 1
        assert stats["n_grounded"] == 0
        assert stats["n_half"] == 1
        assert stats["snippet_grounding_rate"] == pytest.approx(0.0)

    def test_url_less_citations_dropped(self):
        response = {
            "sections": [
                {
                    "text": "Missing-url claims [1][2].",
                    "citations": [{"id": "[1]", "url": ""}, {"id": "[2]"}],
                }
            ]
        }

        grounded, stats = ground_citations_by_url(
            response,
            {"https://example.com/a": "Fetched content."},
        )

        assert grounded["sections"][0]["citations"] == []
        assert stats == {
            "n_citations": 0.0,
            "n_grounded": 0.0,
            "n_half": 0.0,
            "snippet_grounding_rate": 0.0,
        }

    def test_never_seen_citation_dropped(self):
        response = {
            "sections": [
                {
                    "text": "Seen claim [1]. Invented claim [2].",
                    "citations": [
                        {"id": "[1]", "url": "https://example.com/a"},
                        {"id": "[2]", "url": "https://invented.example/missing"},
                    ],
                }
            ]
        }

        grounded, stats = ground_citations_by_url(
            response, {"https://example.com/a": "Fetched content."}
        )

        citations = grounded["sections"][0]["citations"]
        assert [c["id"] for c in citations] == ["[1]"]
        assert stats["n_citations"] == 2
        assert stats["n_grounded"] == 1
        assert stats["n_half"] == 0
        assert stats["snippet_grounding_rate"] == pytest.approx(0.5)

    def test_rate_math_counts_grounded_half_and_dropped_urls(self):
        response = {
            "sections": [
                {
                    "text": "Grounded [1]. Half [2]. Dropped [3].",
                    "citations": [
                        {"id": "[1]", "url": "https://example.com/a/"},
                        {"id": "[2]", "url": "https://example.com/b."},
                        {"id": "[3]", "url": "https://example.com/c"},
                    ],
                }
            ]
        }

        _, stats = ground_citations_by_url(
            response,
            {
                "https://example.com/a": "Fetched content.",
                "https://example.com/b": "",
            },
        )

        assert stats["n_citations"] == 3
        assert stats["n_grounded"] == 1
        assert stats["n_half"] == 1
        assert stats["snippet_grounding_rate"] == pytest.approx(1 / 3)

    def test_input_dict_not_mutated(self):
        response = {
            "sections": [
                {
                    "text": "Invented claim [1].",
                    "citations": [{"id": "[1]", "url": "https://invented.example"}],
                }
            ]
        }
        original = copy.deepcopy(response)

        ground_citations_by_url(response, {})

        assert response == original
