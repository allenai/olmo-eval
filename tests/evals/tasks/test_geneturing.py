"""Tests for GeneTuring task scorers and extraction functions."""

import pytest

from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import ExactMatchScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.geneturing import (
    _AMINO_ACID_PATTERN,
    _DEFAULT_SAMPLING,
    _DNA_PATTERN,
    _SHORT_ANSWER_SAMPLING,
    ChromosomeScorer,
    ContainmentScorer,
    GeneRecallScorer,
    JaccardScorer,
    MeanScoreMetric,
    SetRecallScorer,
    SpeciesScorer,
    _extract_activation,
    _extract_boolean,
    _extract_chromosome,
    _extract_first_line_answer,
    _extract_gene_names,
    _extract_gene_symbols,
    _extract_go_term,
    _extract_sequence,
    _extract_species,
    _normalize_chromosome,
    _parse_name_set,
    _strip_thinking,
)


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


# =============================================================================
# Extraction Functions
# =============================================================================


class TestStripThinking:
    def test_removes_think_block(self):
        text = "<think>reasoning here</think>\nThe answer is 42."
        assert _strip_thinking(text) == "The answer is 42."

    def test_removes_multiline_think_block(self):
        text = "<think>\nline 1\nline 2\n</think>\nAnswer"
        assert _strip_thinking(text) == "Answer"

    def test_no_think_block(self):
        assert _strip_thinking("plain text") == "plain text"

    def test_empty_string(self):
        assert _strip_thinking("") == ""

    def test_multiple_think_blocks(self):
        text = "<think>first</think> middle <think>second</think> end"
        assert _strip_thinking(text) == "middle end"


class TestNormalizeChromosome:
    def test_chr_format(self):
        assert _normalize_chromosome("chr18") == "chr18"

    def test_chromosome_word(self):
        assert _normalize_chromosome("chromosome 7") == "chr7"

    def test_bare_number(self):
        assert _normalize_chromosome("18") == "chr18"

    def test_bare_x(self):
        assert _normalize_chromosome("X") == "chrX"

    def test_bare_y(self):
        assert _normalize_chromosome("Y") == "chrY"

    def test_lowercase_x(self):
        assert _normalize_chromosome("chrx") == "chrX"

    def test_no_match(self):
        assert _normalize_chromosome("not a chromosome") is None

    def test_embedded_in_text(self):
        assert _normalize_chromosome("located on chr3p21") == "chr3"


class TestExtractChromosome:
    def test_with_thinking(self):
        text = "<think>let me think</think>\nchromosome 5"
        assert _extract_chromosome(text) == "chr5"

    def test_plain(self):
        assert _extract_chromosome("chr12") == "chr12"


class TestExtractBoolean:
    def test_true(self):
        assert _extract_boolean("TRUE") == "TRUE"

    def test_yes(self):
        assert _extract_boolean("Yes, it is protein-coding.") == "TRUE"

    def test_na(self):
        assert _extract_boolean("NA") == "NA"

    def test_no(self):
        assert _extract_boolean("No, it is not protein-coding.") == "NA"

    def test_non_coding_phrase(self):
        assert _extract_boolean("This is a non-coding RNA gene.") == "NA"

    def test_not_a_protein(self):
        assert _extract_boolean("It is not a protein-coding gene.") == "NA"

    def test_with_thinking(self):
        text = "<think>hmm</think>\nYes"
        assert _extract_boolean(text) == "TRUE"

    def test_false(self):
        assert _extract_boolean("False") == "NA"

    def test_false_in_sentence(self):
        assert _extract_boolean("It is false that this gene is protein-coding.") == "NA"

    def test_n_slash_a(self):
        assert _extract_boolean("N/A") == "NA"

    def test_not_protein_coding(self):
        assert _extract_boolean("This gene is not protein-coding.") == "NA"

    def test_ambiguous(self):
        assert _extract_boolean("I'm not sure about this gene.") is None


class TestExtractActivation:
    def test_activation(self):
        assert _extract_activation("Activation") == "Activation"

    def test_repression(self):
        assert _extract_activation("Repression") == "Repression"

    def test_activates_keyword(self):
        assert _extract_activation("The TF activates the gene.") == "Activation"

    def test_represses_keyword(self):
        assert _extract_activation("The TF represses the gene.") == "Repression"

    def test_both_activation_first(self):
        text = "It activates some genes but represses this one."
        assert _extract_activation(text) == "Activation"

    def test_both_repression_first(self):
        text = "It represses this gene, not activates it."
        assert _extract_activation(text) == "Repression"

    def test_neither(self):
        assert _extract_activation("This gene does something.") is None

    def test_with_thinking(self):
        text = "<think>reasoning about repression</think>\nActivation"
        assert _extract_activation(text) == "Activation"


class TestExtractSequence:
    def test_amino_acid_sequence(self):
        text = "The protein sequence is MKWVTFISLLFLFSSAYS"
        result = _extract_sequence(text, _AMINO_ACID_PATTERN)
        assert result == "MKWVTFISLLFLFSSAYS"

    def test_dna_sequence(self):
        text = "The DNA is ACGTACGTACGT"
        result = _extract_sequence(text, _DNA_PATTERN)
        assert result == "ACGTACGTACGT"

    def test_longest_match(self):
        text = "Short ACGTACGTAC and longer ACGTACGTACGTACGTACGT"
        result = _extract_sequence(text, _DNA_PATTERN)
        assert result == "ACGTACGTACGTACGTACGT"

    def test_no_match(self):
        assert _extract_sequence("no sequences here", _DNA_PATTERN) is None

    def test_uppercases_result(self):
        text = "The sequence is acgtacgtacgt"
        result = _extract_sequence(text, _DNA_PATTERN)
        assert result == "ACGTACGTACGT"

    def test_with_thinking(self):
        text = "<think>ACGTACGTACGTACGTACGTACGT</think>\nacgtacgtacgt"
        # Think block stripped, so only the shorter one outside is found
        result = _extract_sequence(text, _DNA_PATTERN)
        assert result == "ACGTACGTACGT"


class TestExtractFirstLineAnswer:
    def test_simple_answer(self):
        assert _extract_first_line_answer("BRCA1") == "BRCA1"

    def test_strips_bold(self):
        assert _extract_first_line_answer("**BRCA1**") == "BRCA1"

    def test_strips_verbose_prefix(self):
        text = "The gene symbol for ENSG00000141510 is BRCA1."
        assert _extract_first_line_answer(text) == "BRCA1"

    def test_takes_first_sentence(self):
        text = "BRCA1. This gene encodes a tumor suppressor."
        assert _extract_first_line_answer(text) == "BRCA1"

    def test_strips_parenthetical(self):
        text = "PTK7 (Protein Tyrosine Kinase 7)"
        assert _extract_first_line_answer(text) == "PTK7"

    def test_strips_thinking(self):
        text = "<think>thinking</think>\nTP53"
        assert _extract_first_line_answer(text) == "TP53"

    def test_skips_blank_lines(self):
        text = "\n\n  \nBRCA2"
        assert _extract_first_line_answer(text) == "BRCA2"

    def test_strips_ensembl_prefix(self):
        text = "The Ensembl gene identifier ENSG00000141510 corresponds to TP53"
        assert _extract_first_line_answer(text) == "TP53"

    def test_strips_snp_association_prefix(self):
        text = "The SNP rs1234 is associated with the gene BRCA1"
        assert _extract_first_line_answer(text) == "BRCA1"

    def test_based_on_prefix(self):
        text = "Based on current databases, BRCA1 is the gene."
        assert _extract_first_line_answer(text) == "BRCA1 is the gene"

    def test_empty_string(self):
        assert _extract_first_line_answer("") is None

    def test_whitespace_only(self):
        assert _extract_first_line_answer("   ") is None


class TestExtractSpecies:
    def test_simple_species(self):
        assert _extract_species("human") == "human"

    def test_species_in_sentence(self):
        assert _extract_species("This sequence is from a mouse.") == "mouse"

    def test_scientific_name(self):
        assert _extract_species("Homo sapiens genome") == "human"

    def test_abbreviated_scientific_name(self):
        assert _extract_species("Found in C. elegans") == "worm"

    def test_chicken(self):
        assert _extract_species("Gallus gallus domesticus") == "chicken"

    def test_last_mention_wins(self):
        text = "Could be human but is actually mouse"
        assert _extract_species(text) == "mouse"

    def test_drosophila_alias(self):
        assert _extract_species("Drosophila melanogaster") == "fruit fly"

    def test_yeast_scientific(self):
        assert _extract_species("Saccharomyces cerevisiae") == "yeast"

    def test_no_species(self):
        assert _extract_species("no species mentioned here") is None

    def test_with_thinking(self):
        text = "<think>could be human</think>\nmouse"
        assert _extract_species(text) == "mouse"


class TestParseNameSet:
    def test_comma_separated(self):
        assert _parse_name_set("BRCA1, TP53, EGFR") == {"brca1", "tp53", "egfr"}

    def test_semicolon_separated(self):
        assert _parse_name_set("BRCA1; TP53") == {"brca1", "tp53"}

    def test_and_separator(self):
        assert _parse_name_set("BRCA1 and TP53") == {"brca1", "tp53"}

    def test_filters_none(self):
        assert _parse_name_set("None") == set()

    def test_filters_no_gene(self):
        assert _parse_name_set("No gene") == set()

    def test_empty_string(self):
        assert _parse_name_set("") == set()

    def test_strips_periods(self):
        assert _parse_name_set("BRCA1.") == {"brca1"}


class TestExtractGeneNames:
    def test_no_gene(self):
        assert _extract_gene_names("No gene or protein mentioned.") == "No gene"

    def test_no_protein(self):
        assert _extract_gene_names("No protein is referenced.") == "No gene"

    def test_none_answer(self):
        assert _extract_gene_names("None") == "No gene"

    def test_extracts_gene(self):
        result = _extract_gene_names("BRCA1")
        assert result is not None
        assert "BRCA1" in result


class TestExtractGeneSymbols:
    def test_bold_genes(self):
        text = "The associated genes are **BRCA1**, **TP53**, and **EGFR**."
        result = _extract_gene_symbols(text)
        assert result == "BRCA1, TP53, EGFR"

    def test_uppercase_tokens(self):
        text = "Genes include BRCA1 and TP53 in this pathway."
        result = _extract_gene_symbols(text)
        assert result is not None
        assert "BRCA1" in result
        assert "TP53" in result

    def test_filters_common_words(self):
        text = "THE GENE BRCA1 AND TP53 ARE ASSOCIATED WITH DNA REPAIR."
        result = _extract_gene_symbols(text)
        assert result is not None
        assert "BRCA1" in result
        assert "TP53" in result
        assert "THE" not in result
        assert "AND" not in result
        assert "DNA" not in result
        assert "ARE" not in result

    def test_prefers_bold(self):
        # When bold symbols exist, plain uppercase tokens are ignored
        text = "EGFR is mentioned but **BRCA1** is the answer."
        result = _extract_gene_symbols(text)
        assert result == "BRCA1"

    def test_no_genes(self):
        assert _extract_gene_symbols("this has no gene symbols") is None

    def test_with_thinking(self):
        text = "<think>**TP53** is relevant</think>\n**BRCA1** is the gene."
        result = _extract_gene_symbols(text)
        assert result == "BRCA1"

    def test_deduplicates(self):
        text = "BRCA1 is mentioned, and BRCA1 again, also TP53."
        result = _extract_gene_symbols(text)
        assert result is not None
        assert result.count("BRCA1") == 1


class TestExtractGoTerm:
    def test_quoted_after_is(self):
        text = 'The enriched GO term is "lipid metabolic process".'
        assert _extract_go_term(text) == "lipid metabolic process"

    def test_unicode_quotes(self):
        text = "The enriched GO term is \u201clipid metabolic process\u201d."
        assert _extract_go_term(text) == "lipid metabolic process"

    def test_bold_after_is(self):
        text = "The enriched GO term is **lipid metabolic process**."
        assert _extract_go_term(text) == "lipid metabolic process"

    def test_bold_strips_go_id(self):
        text = "The enriched GO term is **lipid metabolic process (GO:0006629)**."
        assert _extract_go_term(text) == "lipid metabolic process"

    def test_quoted_string_fallback(self):
        text = 'These genes share "signal transduction" as a common function.'
        assert _extract_go_term(text) == "signal transduction"

    def test_is_likely_quoted(self):
        text = 'The enriched GO term is likely "cell adhesion".'
        assert _extract_go_term(text) == "cell adhesion"

    def test_falls_back_to_first_line(self):
        text = "lipid metabolic process"
        assert _extract_go_term(text) == "lipid metabolic process"

    def test_with_thinking(self):
        text = '<think>"thinking term"</think>\nThe term is "signal transduction".'
        assert _extract_go_term(text) == "signal transduction"

    def test_skips_go_id_only_quotes(self):
        text = '"GO:0006629" is the GO ID. The term is "lipid metabolic process".'
        assert _extract_go_term(text) == "lipid metabolic process"


# =============================================================================
# Scorers
# =============================================================================


class TestChromosomeScorer:
    def setup_method(self):
        self.scorer = ChromosomeScorer()

    def _score(self, gold: str, extracted: str) -> float:
        instance = Instance(question="Q", gold_answer=gold)
        output = LMOutput(text="")
        output.extracted_answer = extracted
        return self.scorer.score(instance, output)

    def test_exact_match(self):
        assert self._score("chr18", "chr18") == 1.0

    def test_normalized_match(self):
        assert self._score("18", "chromosome 18") == 1.0

    def test_mismatch(self):
        assert self._score("chr18", "chr7") == 0.0

    def test_x_chromosome(self):
        assert self._score("chrX", "X") == 1.0

    def test_none_gold(self):
        instance = Instance(question="Q", gold_answer=None)
        output = LMOutput(text="")
        output.extracted_answer = "chr1"
        assert self.scorer.score(instance, output) == 0.0

    def test_none_extracted(self):
        instance = Instance(question="Q", gold_answer="chr1")
        output = LMOutput(text="")
        output.extracted_answer = None
        assert self.scorer.score(instance, output) == 0.0

    def test_unparseable_gold(self):
        assert self._score("not a chromosome", "chr1") == 0.0

    def test_name(self):
        assert self.scorer.name == "chromosome"


class TestJaccardScorer:
    def setup_method(self):
        self.scorer = JaccardScorer()

    def _score(self, gold: str, extracted: str) -> float:
        instance = Instance(question="Q", gold_answer=gold)
        output = LMOutput(text="")
        output.extracted_answer = extracted
        return self.scorer.score(instance, output)

    def test_exact_match(self):
        assert self._score("BRCA1, TP53", "BRCA1, TP53") == 1.0

    def test_partial_overlap(self):
        # gold={brca1, tp53}, pred={brca1, egfr} → intersection=1, union=3
        assert self._score("BRCA1, TP53", "BRCA1, EGFR") == pytest.approx(1 / 3)

    def test_no_overlap(self):
        assert self._score("BRCA1, TP53", "EGFR, KRAS") == 0.0

    def test_both_empty(self):
        assert self._score("", "") == 1.0

    def test_gold_empty_pred_nonempty(self):
        assert self._score("", "BRCA1") == 0.0

    def test_none_gold(self):
        instance = Instance(question="Q", gold_answer=None)
        output = LMOutput(text="")
        output.extracted_answer = "BRCA1"
        assert self.scorer.score(instance, output) == 0.0

    def test_none_extracted(self):
        instance = Instance(question="Q", gold_answer="BRCA1")
        output = LMOutput(text="")
        output.extracted_answer = None
        assert self.scorer.score(instance, output) == 0.0

    def test_name(self):
        assert self.scorer.name == "jaccard"


class TestSetRecallScorer:
    def setup_method(self):
        self.scorer = SetRecallScorer()

    def _score(self, gold: str, extracted: str) -> float:
        instance = Instance(question="Q", gold_answer=gold)
        output = LMOutput(text="")
        output.extracted_answer = extracted
        return self.scorer.score(instance, output)

    def test_full_recall(self):
        assert self._score("BRCA1, TP53", "BRCA1, TP53, EGFR") == 1.0

    def test_partial_recall(self):
        assert self._score("BRCA1, TP53", "BRCA1") == pytest.approx(0.5)

    def test_zero_recall(self):
        assert self._score("BRCA1, TP53", "EGFR") == 0.0

    def test_empty_gold_empty_pred(self):
        assert self._score("", "") == 1.0

    def test_empty_gold_nonempty_pred(self):
        assert self._score("", "BRCA1") == 0.0

    def test_none_gold(self):
        instance = Instance(question="Q", gold_answer=None)
        output = LMOutput(text="")
        output.extracted_answer = "BRCA1"
        assert self.scorer.score(instance, output) == 0.0

    def test_none_extracted(self):
        instance = Instance(question="Q", gold_answer="BRCA1")
        output = LMOutput(text="")
        output.extracted_answer = None
        assert self.scorer.score(instance, output) == 0.0

    def test_name(self):
        assert self.scorer.name == "set_recall"


class TestGeneRecallScorer:
    def setup_method(self):
        self.scorer = GeneRecallScorer()

    def _score(self, gold: str, raw_text: str) -> float:
        instance = Instance(question="Q", gold_answer=gold)
        output = LMOutput(text=raw_text)
        return self.scorer.score(instance, output)

    def test_all_found(self):
        assert self._score("BRCA1, TP53", "The genes BRCA1 and TP53 are relevant.") == 1.0

    def test_partial_found(self):
        assert self._score("BRCA1, TP53", "Only BRCA1 is relevant.") == pytest.approx(0.5)

    def test_none_found(self):
        assert self._score("BRCA1, TP53", "No relevant genes.") == 0.0

    def test_case_insensitive(self):
        assert self._score("BRCA1", "the gene brca1 is mentioned") == 1.0

    def test_strips_thinking(self):
        text = "<think>BRCA1 in thinking</think>\nNo genes mentioned."
        # BRCA1 is only in the think block, which gets stripped
        assert self._score("BRCA1", text) == 0.0

    def test_checks_raw_text_not_extracted(self):
        """Scorer checks raw output text, not extracted_answer."""
        instance = Instance(question="Q", gold_answer="BRCA1")
        output = LMOutput(text="BRCA1 is the gene.")
        output.extracted_answer = "something else"
        assert self.scorer.score(instance, output) == 1.0

    def test_none_gold(self):
        instance = Instance(question="Q", gold_answer=None)
        output = LMOutput(text="BRCA1")
        assert self.scorer.score(instance, output) == 0.0

    def test_semicolon_separated_gold(self):
        assert self._score("BRCA1; TP53", "BRCA1 and TP53") == 1.0

    def test_word_boundary_no_false_positive(self):
        """BRCA1 should not match BRCA10 or BRCA1P."""
        assert self._score("BRCA1", "The gene BRCA10 was found.") == 0.0

    def test_word_boundary_correct_match(self):
        """BRCA1 should still match when bounded by non-word chars."""
        assert self._score("BRCA1", "Found BRCA1, TP53.") == 1.0

    def test_name(self):
        assert self.scorer.name == "gene_recall"


class TestContainmentScorer:
    def setup_method(self):
        self.scorer = ContainmentScorer()

    def _score(self, gold: str, extracted: str) -> float:
        instance = Instance(question="Q", gold_answer=gold)
        output = LMOutput(text="")
        output.extracted_answer = extracted
        return self.scorer.score(instance, output)

    def test_exact_match(self):
        assert self._score("lipid metabolic process", "lipid metabolic process") == 1.0

    def test_case_insensitive_exact(self):
        assert self._score("Lipid Metabolic Process", "lipid metabolic process") == 1.0

    def test_gold_in_pred(self):
        assert self._score("cell adhesion", "regulation of cell adhesion") == 0.5

    def test_pred_in_gold(self):
        assert self._score("regulation of cell adhesion", "cell adhesion") == 0.5

    def test_no_match(self):
        assert self._score("lipid metabolic process", "DNA repair") == 0.0

    def test_none_gold(self):
        instance = Instance(question="Q", gold_answer=None)
        output = LMOutput(text="")
        output.extracted_answer = "something"
        assert self.scorer.score(instance, output) == 0.0

    def test_none_extracted(self):
        instance = Instance(question="Q", gold_answer="something")
        output = LMOutput(text="")
        output.extracted_answer = None
        assert self.scorer.score(instance, output) == 0.0

    def test_name(self):
        assert self.scorer.name == "containment"


class TestSpeciesScorer:
    def setup_method(self):
        self.scorer = SpeciesScorer()

    def test_exact_match(self):
        instance = Instance(question="Q", gold_answer="human")
        output = LMOutput(text="human")
        output.extracted_answer = "human"
        assert self.scorer.score(instance, output) == 1.0

    def test_partial_credit_in_raw(self):
        instance = Instance(question="Q", gold_answer="human")
        output = LMOutput(text="This sequence is from a human genome.")
        output.extracted_answer = "mouse"  # wrong extraction
        assert self.scorer.score(instance, output) == 0.5

    def test_alias_partial_credit(self):
        """Gold is 'worm', raw text says 'C. elegans' → 0.5."""
        instance = Instance(question="Q", gold_answer="worm")
        output = LMOutput(text="The sequence is from C. elegans.")
        output.extracted_answer = "nematode"
        assert self.scorer.score(instance, output) == 0.5

    def test_no_match(self):
        instance = Instance(question="Q", gold_answer="human")
        output = LMOutput(text="This is from mouse genome.")
        output.extracted_answer = "mouse"
        assert self.scorer.score(instance, output) == 0.0

    def test_none_gold(self):
        instance = Instance(question="Q", gold_answer=None)
        output = LMOutput(text="human")
        assert self.scorer.score(instance, output) == 0.0

    def test_strips_thinking_for_raw_check(self):
        instance = Instance(question="Q", gold_answer="human")
        output = LMOutput(text="<think>human is likely</think>\nmouse")
        output.extracted_answer = "mouse"
        # "human" is only in think block, should be stripped
        assert self.scorer.score(instance, output) == 0.0

    def test_name(self):
        assert self.scorer.name == "species"


# =============================================================================
# MeanScoreMetric
# =============================================================================


class TestMeanScoreMetric:
    def _make_response(self, score: float, scorer_name: str = "jaccard") -> Response:
        return Response(
            instance=Instance(question="Q", gold_answer="A"),
            request=LMRequest(request_type=RequestType.CHAT, messages=()),
            outputs=[LMOutput(text="")],
            scores={scorer_name: score},
        )

    def test_mean_of_scores(self):
        metric = MeanScoreMetric(name="jaccard", scorer=JaccardScorer)
        responses = [
            self._make_response(1.0),
            self._make_response(0.5),
            self._make_response(0.0),
        ]
        assert metric.compute(responses) == pytest.approx(0.5)

    def test_empty_responses(self):
        metric = MeanScoreMetric(name="jaccard", scorer=JaccardScorer)
        assert metric.compute([]) == 0.0

    def test_missing_scorer_key_defaults_zero(self):
        metric = MeanScoreMetric(name="jaccard", scorer=JaccardScorer)
        responses = [self._make_response(0.0, scorer_name="other_scorer")]
        assert metric.compute(responses) == 0.0


# =============================================================================
# Task Registration and Wiring
# =============================================================================

_ALL_TASKS = (
    "geneturing_gene_name_conversion",
    "geneturing_gene_location",
    "geneturing_snp_location",
    "geneturing_gene_snp_association",
    "geneturing_protein_coding_genes",
    "geneturing_tf_regulation",
    "geneturing_human_genome_dna_alignment",
    "geneturing_amino_acid_translation",
    "geneturing_dna_sequence_extraction",
    "geneturing_gene_name_extraction",
    "geneturing_gene_alias",
    "geneturing_gene_disease_association",
    "geneturing_gene_ontology",
    "geneturing_multi_species_dna_alignment",
)


class TestTaskRegistration:
    @pytest.mark.parametrize("task_name", _ALL_TASKS)
    def test_task_registered(self, task_name):
        assert task_name in list_tasks()

    @pytest.mark.parametrize("task_name", _ALL_TASKS)
    def test_get_task(self, task_name):
        task = get_task(task_name)
        assert task.config.name == task_name


class TestTaskWiring:
    """Verify each task has the correct scorer, metric, formatter, and sampling_params."""

    @pytest.mark.parametrize(
        "task_name, expected_scorer",
        [
            ("geneturing_gene_name_conversion", ExactMatchScorer),
            ("geneturing_gene_location", ChromosomeScorer),
            ("geneturing_snp_location", ChromosomeScorer),
            ("geneturing_gene_snp_association", ExactMatchScorer),
            ("geneturing_protein_coding_genes", ExactMatchScorer),
            ("geneturing_tf_regulation", ExactMatchScorer),
            ("geneturing_human_genome_dna_alignment", ChromosomeScorer),
            ("geneturing_amino_acid_translation", ExactMatchScorer),
            ("geneturing_dna_sequence_extraction", ExactMatchScorer),
            ("geneturing_gene_name_extraction", JaccardScorer),
            ("geneturing_gene_alias", JaccardScorer),
            ("geneturing_gene_disease_association", GeneRecallScorer),
            ("geneturing_gene_ontology", ContainmentScorer),
            ("geneturing_multi_species_dna_alignment", SpeciesScorer),
        ],
    )
    def test_scorer_type(self, task_name, expected_scorer):
        task = get_task(task_name)
        metric = task.config.primary_metric
        if isinstance(metric, AccuracyMetric):
            assert metric.scorer == expected_scorer
        else:
            assert isinstance(metric, MeanScoreMetric)
            assert metric.scorer == expected_scorer

    @pytest.mark.parametrize(
        "task_name, expected_metric_name",
        [
            ("geneturing_gene_name_conversion", "accuracy"),
            ("geneturing_gene_location", "accuracy"),
            ("geneturing_gene_name_extraction", "jaccard"),
            ("geneturing_gene_alias", "jaccard"),
            ("geneturing_gene_disease_association", "recall"),
            ("geneturing_gene_ontology", "containment"),
            ("geneturing_multi_species_dna_alignment", "species"),
        ],
    )
    def test_metric_name(self, task_name, expected_metric_name):
        task = get_task(task_name)
        assert task.config.primary_metric.name == expected_metric_name

    @pytest.mark.parametrize(
        "task_name",
        [
            "geneturing_amino_acid_translation",
            "geneturing_multi_species_dna_alignment",
        ],
    )
    def test_short_answer_sampling(self, task_name):
        task = get_task(task_name)
        assert task.config.sampling_params.max_tokens == _SHORT_ANSWER_SAMPLING.max_tokens

    @pytest.mark.parametrize(
        "task_name",
        [
            "geneturing_gene_name_conversion",
            "geneturing_gene_location",
            "geneturing_dna_sequence_extraction",
        ],
    )
    def test_default_sampling(self, task_name):
        task = get_task(task_name)
        assert task.config.sampling_params.max_tokens == _DEFAULT_SAMPLING.max_tokens

    @pytest.mark.parametrize(
        "task_name",
        [
            "geneturing_protein_coding_genes",
            "geneturing_tf_regulation",
            "geneturing_gene_disease_association",
            "geneturing_gene_alias",
            "geneturing_gene_ontology",
            "geneturing_multi_species_dna_alignment",
        ],
    )
    def test_custom_formatter(self, task_name):
        """Tasks with custom formatters should not use the default system prompt."""
        task = get_task(task_name)
        default_prompt = (
            "You are a genomics expert. Answer the following question "
            "concisely and accurately.\n"
            "Give only the answer with no explanation unless asked."
        )
        assert task.config.formatter.system_prompt != default_prompt
