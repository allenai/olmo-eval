"""Tests for the citation bridge between the prompt's form and the scorer's.

The prompt mandates "[3]"-style citations; the upstream parser credits only
markdown links to arxiv.org/abs. These pin the translation, including the cases
where it must refuse to translate rather than point a citation at the wrong
paper.
"""

import pytest

from olmo_eval.evals.tasks.deepscholar_citations import (
    abs_url,
    reference_list,
    resolve_numbering,
    rewrite_intro,
    strip_references,
    unresolved_citation_forms,
)

SOURCES = [
    {"arxiv_id": "2401.00001", "title": "Retrieval Augmented Generation for Science"},
    {"arxiv_id": "2401.00002", "title": "Scaling Laws for Literature Search"},
    {"arxiv_id": "2401.00003", "title": "Agentic Citation Grounding"},
]

NUMBERED_ANSWER = """## Related Works

Retrieval augmentation is well studied [1]. Later work scaled it [2] and grounded
citations directly [3]. Some combine both [2, 3].

## References

[1] A. Author. Retrieval Augmented Generation for Science. arXiv:2401.00001
[2] B. Author. Scaling Laws for Literature Search. arXiv:2401.00002v2
[3] C. Author. Agentic Citation Grounding. https://arxiv.org/abs/2401.00003
"""


class TestReferenceList:
    def test_bracket_numbering(self):
        assert set(reference_list(NUMBERED_ANSWER)) == {"1", "2", "3"}

    def test_dotted_and_parenthesised_numbering(self):
        report = (
            "Body [1].\n\nReferences\n1. First entry arXiv:2401.00001\n2) Second arXiv:2401.00002\n"
        )

        assert set(reference_list(report)) == {"1", "2"}

    def test_wrapped_entries_stay_whole(self):
        report = (
            "Body [1].\n\nReferences\n[1] A. Author. A Very Long Title That\n"
            "wraps onto another line. arXiv:2401.00001\n"
        )

        assert "wraps onto another line" in reference_list(report)["1"]

    def test_no_reference_section_yields_nothing_numbered(self):
        assert reference_list("Just prose with no list.") == {}


class TestStripReferences:
    def test_reference_section_is_removed(self):
        assert "arXiv:2401.00001" not in strip_references(NUMBERED_ANSWER)
        assert "Retrieval augmentation is well studied" in strip_references(NUMBERED_ANSWER)


class TestResolveNumbering:
    def test_numbers_resolve_by_arxiv_id(self):
        resolved = resolve_numbering(NUMBERED_ANSWER, SOURCES)

        assert {number: source["arxiv_id"] for number, source in resolved.items()} == {
            "1": "2401.00001",
            "2": "2401.00002",
            "3": "2401.00003",
        }

    def test_version_suffixes_in_the_reference_list_still_match(self):
        # Entry [2] is written "arXiv:2401.00002v2" while the source is normalised.
        assert resolve_numbering(NUMBERED_ANSWER, SOURCES)["2"]["arxiv_id"] == "2401.00002"

    def test_numbers_resolve_by_title_when_no_id_is_given(self):
        report = (
            "Body [1].\n\nReferences\n[1] A. Author. Agentic Citation Grounding. In Proc. 2024.\n"
        )

        assert resolve_numbering(report, SOURCES)["1"]["arxiv_id"] == "2401.00003"

    def test_an_entry_naming_no_retrieved_paper_stays_unresolved(self):
        report = "Body [1].\n\nReferences\n[1] Someone. A Paper Never Retrieved. arXiv:2999.99999\n"

        assert resolve_numbering(report, SOURCES) == {}


class TestRewriteIntro:
    def test_numbered_citations_become_arxiv_links(self):
        intro, cited = rewrite_intro(NUMBERED_ANSWER, SOURCES)

        assert f"[Retrieval Augmented Generation for Science]({abs_url('2401.00001')})" in intro
        assert f"[Agentic Citation Grounding]({abs_url('2401.00003')})" in intro
        assert {source["arxiv_id"] for source in cited} == {
            "2401.00001",
            "2401.00002",
            "2401.00003",
        }

    def test_the_reference_list_is_not_carried_into_the_intro(self):
        intro, _ = rewrite_intro(NUMBERED_ANSWER, SOURCES)

        assert "## References" not in intro
        assert "B. Author" not in intro

    def test_grouped_citations_expand_to_one_link_each(self):
        intro, _ = rewrite_intro(NUMBERED_ANSWER, SOURCES)

        # "[2, 3]" is two citations, not one label.
        assert intro.count(abs_url("2401.00002")) == 2
        assert intro.count(abs_url("2401.00003")) == 2

    def test_generated_urls_carry_no_version_suffix(self):
        # The upstream parser keys its reference map on the raw URL text, so a
        # "v2" here would never match the normalised paper.csv row.
        intro, _ = rewrite_intro(NUMBERED_ANSWER, SOURCES)

        assert "v2)" not in intro
        assert abs_url("2401.00002") in intro

    def test_unresolvable_numbers_are_deleted_not_left_behind(self):
        report = (
            "Grounded work [1]. Invented work [7].\n\n"
            "References\n[1] A. Author. Agentic Citation Grounding. arXiv:2401.00003\n"
        )

        intro, cited = rewrite_intro(report, SOURCES)

        # Leaving "[7]" would make the parser number a document that has no row.
        assert "[7]" not in intro
        assert len(cited) == 1

    def test_an_answer_citing_only_unretrieved_papers_yields_nothing(self):
        report = "Body [1].\n\nReferences\n[1] Someone. Never Retrieved. arXiv:2999.99999\n"

        assert rewrite_intro(report, SOURCES) == ("", [])

    def test_existing_markdown_links_are_normalised(self):
        report = "See [this paper](https://arxiv.org/abs/2401.00002v3) for details."

        intro, cited = rewrite_intro(report, SOURCES)

        assert f"[Scaling Laws for Literature Search]({abs_url('2401.00002')})" in intro
        assert [source["arxiv_id"] for source in cited] == ["2401.00002"]

    def test_a_link_to_an_unretrieved_paper_keeps_its_prose_label(self):
        report = "See [some other work](https://arxiv.org/abs/2999.99999) and [1].\n\n"
        report += "References\n[1] A. Author. Agentic Citation Grounding. arXiv:2401.00003\n"

        intro, _ = rewrite_intro(report, SOURCES)

        assert "some other work" in intro
        assert "2999.99999" not in intro

    @pytest.mark.parametrize("report", ["", "   \n\n"])
    def test_an_empty_answer_yields_nothing(self, report):
        assert rewrite_intro(report, SOURCES) == ("", [])

    def test_no_sources_yields_nothing(self):
        assert rewrite_intro(NUMBERED_ANSWER, []) == ("", [])


class TestAlternateReferenceHeadings:
    """lit-agents strips only "References"; models write three other things."""

    @pytest.mark.parametrize(
        "heading",
        ["References", "## Bibliography", "Works Cited", "### SOURCES", "references:"],
    )
    def test_the_tail_is_stripped(self, heading):
        report = (
            f"Grounded work [1].\n\n{heading}\n"
            "[1] A. Author. Agentic Citation Grounding. arXiv:2401.00003\n"
        )

        intro, cited = rewrite_intro(report, SOURCES)

        assert "A. Author" not in intro
        assert [source["arxiv_id"] for source in cited] == ["2401.00003"]

    def test_an_unstripped_bibliography_would_fabricate_citations(self):
        # The failure this guards: entries left in the prose have their own
        # "[1]" markers, and every one becomes an inline citation.
        report = (
            "Body with no citations at all.\n\n## Bibliography\n"
            "[1] A. Author. Agentic Citation Grounding. arXiv:2401.00003\n"
        )

        intro, cited = rewrite_intro(report, SOURCES)

        assert cited == []
        assert intro == ""

    def test_a_non_reference_heading_is_logged(self, caplog):
        report = (
            "Body [1].\n\n## Bibliography\n[1] A. Agentic Citation Grounding. arXiv:2401.00003\n"
        )

        with caplog.at_level("WARNING"):
            strip_references(report)

        assert "Bibliography" in caplog.text

    def test_the_canonical_heading_is_not_logged(self, caplog):
        with caplog.at_level("WARNING"):
            strip_references(NUMBERED_ANSWER)

        assert caplog.text == ""


class TestExistingLinksAreNotReprocessed:
    def test_a_rendered_link_is_not_wrapped_again(self):
        # The link pass emits "[Title](url)"; the bracket pass would match its
        # "[Title]" -- titles are aliases -- and produce "[Title](url)(url)".
        report = "See [Agentic Citation Grounding](https://arxiv.org/abs/2401.00003) here."

        intro, _ = rewrite_intro(report, SOURCES)

        assert intro.strip() == (
            "See [Agentic Citation Grounding](https://arxiv.org/abs/2401.00003) here."
        )
        assert ")(" not in intro

    def test_a_title_written_as_a_bare_bracket_still_resolves(self):
        report = "See [Agentic Citation Grounding] here."

        intro, _ = rewrite_intro(report, SOURCES)

        assert f"[Agentic Citation Grounding]({abs_url('2401.00003')})" in intro
        assert ")(" not in intro


class TestCitationForms:
    def test_numeric_ranges_expand(self):
        report = "Several systems [1-3] agree.\n\n## References\n" + (
            "[1] A. First Retrieval System. arXiv:2401.00001\n"
            "[2] B. Scaling. arXiv:2401.00002\n"
            "[3] C. Grounding. arXiv:2401.00003\n"
        )

        intro, cited = rewrite_intro(report, SOURCES)

        assert {source["arxiv_id"] for source in cited} == {
            "2401.00001",
            "2401.00002",
            "2401.00003",
        }
        assert "[1-3]" not in intro

    def test_a_reversed_range_is_left_alone(self):
        # At that width it is a page range, not a citation, and expanding it
        # would invent 300 citation markers out of one bracket.
        report = "Pages [300-1] here [1].\n\n## References\n[1] A. Grounding. arXiv:2401.00003\n"

        intro, cited = rewrite_intro(report, SOURCES)

        assert [source["arxiv_id"] for source in cited] == ["2401.00003"]
        assert "[300-1]" in intro

    def test_footnote_markers_resolve_like_plain_numbers(self):
        report = "Grounded [^1].\n\n## References\n[1] A. Grounding. arXiv:2401.00003\n"

        intro, cited = rewrite_intro(report, SOURCES)

        assert [source["arxiv_id"] for source in cited] == ["2401.00003"]

    @pytest.mark.parametrize(
        "body",
        [
            "Superscript citation<sup>1</sup> here.",
            "Unicode superscript\u00b9 here.",
            "Author-year style [Smith, 2023] here.",
            "A footnote the list never defined [^7] here.",
        ],
    )
    def test_unhandled_forms_are_counted_not_ignored(self, body):
        # An answer that cited carefully in a style this bridge cannot read
        # scores zero, and without this count it looks like one that never cited.
        assert unresolved_citation_forms(body) >= 1

    def test_a_clean_intro_counts_zero(self):
        intro, _ = rewrite_intro(NUMBERED_ANSWER, SOURCES)

        assert unresolved_citation_forms(intro) == 0
