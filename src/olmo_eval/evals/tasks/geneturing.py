"""GeneTuring genomics Q&A benchmark.

GeneTuring evaluates LLMs on 14 genomics tasks (1,400 questions total) spanning
gene nomenclature, genomic location, functional analysis, sequence alignment,
and more. Each task has 100 questions with gold-standard answers.

Dataset: allenai/geneturing (derived from Hou & Ji, 2024 supplementary data)

Tasks:
    geneturing_gene_name_conversion      Ensembl ID → gene symbol
    geneturing_gene_location             Gene → chromosome
    geneturing_snp_location              SNP → chromosome
    geneturing_gene_snp_association      SNP → associated gene
    geneturing_protein_coding_genes      Is gene protein-coding? (TRUE/NA)
    geneturing_tf_regulation             TF activates or represses gene?
    geneturing_human_genome_dna_alignment  DNA → genomic coordinates
    geneturing_amino_acid_translation    Nucleotide → amino acid sequence
    geneturing_dna_sequence_extraction   Genomic coordinates → DNA sequence
    geneturing_gene_name_extraction      Sentence → gene/protein names
    geneturing_gene_alias               Alias → official gene symbol
    geneturing_gene_disease_association  Disease → associated genes
    geneturing_gene_ontology            Gene set → GO term
    geneturing_multi_species_dna_alignment  DNA → species of origin
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import ChatFormatter
from olmo_eval.common.metrics import AccuracyMetric, Metric
from olmo_eval.common.scorers import ExactMatchScorer, Scorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register

# =============================================================================
# Custom Scorers
# =============================================================================


@dataclass(frozen=True, slots=True)
class ChromosomeScorer(Scorer):
    """Normalize both gold and predicted to ``chrN`` format, then exact-match."""

    name: str = "chromosome"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.extracted_answer is None:
            return 0.0
        gold = _normalize_chromosome(instance.gold_answer)
        pred = _normalize_chromosome(str(output.extracted_answer))
        if gold is None or pred is None:
            return 0.0
        return 1.0 if gold == pred else 0.0


@dataclass(frozen=True, slots=True)
class JaccardScorer(Scorer):
    """Jaccard index (|A∩B|/|A∪B|) over comma-separated name sets."""

    name: str = "jaccard"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.extracted_answer is None:
            return 0.0
        gold_set = _parse_name_set(instance.gold_answer)
        pred_set = _parse_name_set(str(output.extracted_answer))
        if not gold_set and not pred_set:
            return 1.0
        if not gold_set or not pred_set:
            return 0.0
        intersection = gold_set & pred_set
        union = gold_set | pred_set
        return len(intersection) / len(union)


@dataclass(frozen=True, slots=True)
class SetRecallScorer(Scorer):
    """Recall: fraction of gold set items found in predicted set."""

    name: str = "set_recall"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.extracted_answer is None:
            return 0.0
        gold_set = _parse_name_set(instance.gold_answer)
        pred_set = _parse_name_set(str(output.extracted_answer))
        if not gold_set:
            return 1.0 if not pred_set else 0.0
        found = sum(1 for g in gold_set if g in pred_set)
        return found / len(gold_set)


@dataclass(frozen=True, slots=True)
class GeneRecallScorer(Scorer):
    """Fraction of gold gene symbols mentioned anywhere in the model response.

    Follows the GeneTuring paper: "The answer provided by an LLM was scored
    based on the proportion of gold standard genes mentioned in the response."
    Checks the full (think-stripped) output text rather than the extracted answer.
    """

    name: str = "gene_recall"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None:
            return 0.0
        gold_genes = [
            g.strip().upper() for g in re.split(r"[,;]", instance.gold_answer) if g.strip()
        ]
        if not gold_genes:
            return 0.0
        response = _strip_thinking(output.text).upper()
        found = sum(1 for g in gold_genes if re.search(r"\b" + re.escape(g) + r"\b", response))
        return found / len(gold_genes)


@dataclass(frozen=True, slots=True)
class ContainmentScorer(Scorer):
    """1.0 for exact match, 0.5 for partial match (either direction), else 0.0.

    Follows the GeneTuring paper: "A score of 0.5 was assigned if one of the
    GO terms partially matched the gold standard."
    """

    name: str = "containment"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.extracted_answer is None:
            return 0.0
        gold = instance.gold_answer.strip().lower()
        pred = str(output.extracted_answer).strip().lower()
        if gold == pred:
            return 1.0
        if gold in pred or pred in gold:
            return 0.5
        return 0.0


@dataclass(frozen=True, slots=True)
class SpeciesScorer(Scorer):
    """Species matching with partial credit for superset answers.

    Follows the GeneTuring paper: 1.0 for exact match, 0.5 if the model's
    response contains the gold species (a "correct superset"), 0.0 otherwise.
    """

    name: str = "species"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None:
            return 0.0
        gold = instance.gold_answer.strip().lower()
        # Check extracted answer first for exact match
        if output.extracted_answer is not None:
            pred = str(output.extracted_answer).strip().lower()
            if gold == pred:
                return 1.0
        # Check if the gold species appears anywhere in the raw output
        # (the model mentioned the right answer among other text)
        raw = _strip_thinking(output.text).lower()
        if gold in raw:
            return 0.5
        # Also check aliases in reverse: if gold is "worm", check for "c. elegans"
        for alias, common in _SPECIES_ALIASES.items():
            if common == gold and alias in raw:
                return 0.5
        return 0.0


# =============================================================================
# Custom Metrics (thin wrappers binding scorer to AccuracyMetric pattern)
# =============================================================================


@dataclass(frozen=True, slots=True)
class MeanScoreMetric(Metric):
    """Mean of per-response scores for a given scorer."""

    name: str = "mean_score"
    scorer: type[Scorer] = ExactMatchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


# =============================================================================
# Answer Extraction Helpers
# =============================================================================

_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_CHR_PATTERN = re.compile(r"chr(?:omosome\s*)?(\d+|[XYxy])", re.IGNORECASE)
_AMINO_ACID_PATTERN = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{5,}", re.IGNORECASE)
_DNA_PATTERN = re.compile(r"[ACGTacgt]{10,}")
_KNOWN_SPECIES = frozenset(
    {"human", "mouse", "rat", "zebrafish", "worm", "yeast", "fruit fly", "chicken"}
)

# Scientific / Latin name → common name mapping for species extraction.
_SPECIES_ALIASES: dict[str, str] = {
    "homo sapiens": "human",
    "mus musculus": "mouse",
    "rattus norvegicus": "rat",
    "rattus": "rat",
    "danio rerio": "zebrafish",
    "caenorhabditis elegans": "worm",
    "c. elegans": "worm",
    "c.elegans": "worm",
    "saccharomyces cerevisiae": "yeast",
    "s. cerevisiae": "yeast",
    "s.cerevisiae": "yeast",
    "drosophila melanogaster": "fruit fly",
    "drosophila": "fruit fly",
    "d. melanogaster": "fruit fly",
    "gallus gallus": "chicken",
}


def _strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from model output."""
    return _THINK_PATTERN.sub("", text).strip()


def _normalize_chromosome(text: str) -> str | None:
    """Extract and normalize a chromosome identifier to ``chrN`` format."""
    m = _CHR_PATTERN.search(text)
    if m:
        num = m.group(1).upper()
        return f"chr{num}"
    # Handle bare numbers like "18" or "Y"
    stripped = text.strip().upper()
    if re.fullmatch(r"\d{1,2}", stripped):
        return f"chr{stripped}"
    if stripped in ("X", "Y"):
        return f"chr{stripped}"
    return None


def _extract_chromosome(text: str) -> str | None:
    """Extract chromosome from model output."""
    return _normalize_chromosome(_strip_thinking(text))


def _extract_boolean(text: str) -> str | None:
    """Map yes/no/true/false to TRUE/NA (protein-coding convention)."""
    lower = _strip_thinking(text).strip().lower()
    # Check for explicit TRUE/NA/FALSE first
    if "true" in lower and "na" not in lower and "false" not in lower:
        return "TRUE"
    if re.search(r"\bfalse\b", lower):
        return "NA"
    if (
        lower.startswith("na")
        or lower.startswith("n/a")
        or "not a protein" in lower
        or "not protein" in lower
        or "non-coding" in lower
    ):
        return "NA"
    # Check yes/no
    first_line = lower.split("\n")[0]
    if re.search(r"\byes\b", first_line):
        return "TRUE"
    if re.search(r"\bno\b", first_line):
        return "NA"
    return None


def _extract_activation(text: str) -> str | None:
    """Extract Activation or Repression from model output."""
    lower = _strip_thinking(text).lower()
    has_activation = "activat" in lower
    has_repression = "repress" in lower
    if has_activation and not has_repression:
        return "Activation"
    if has_repression and not has_activation:
        return "Repression"
    # Both mentioned — check which comes first in the first sentence
    if has_activation and has_repression:
        first_line = lower.split("\n")[0].split(".")[0]
        act_pos = first_line.find("activat")
        rep_pos = first_line.find("repress")
        if act_pos >= 0 and (rep_pos < 0 or act_pos < rep_pos):
            return "Activation"
        return "Repression"
    return None


def _extract_sequence(text: str, pattern: re.Pattern[str]) -> str | None:
    """Extract the longest contiguous sequence matching the pattern."""
    matches: list[str] = pattern.findall(_strip_thinking(text))
    if not matches:
        return None
    best = matches[0]
    for m in matches[1:]:
        if len(m) > len(best):
            best = m
    return best.upper()


def _extract_first_line_answer(text: str) -> str | None:
    """Extract the first meaningful line, stripping thinking traces and boilerplate."""
    text = _strip_thinking(text)
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        # Strip markdown bold/italic
        line = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", line)
        # Strip common verbose prefixes (e.g. "The official gene symbol is ...")
        line = re.sub(
            r"^(the\s+(?:official\s+)?(?:gene\s+)?(?:symbol|name|answer|identifier)\s+"
            r"(?:is|for|of|corresponding to)\s+[^:]*?(?:is\s+|:\s*)"
            r"|the\s+(?:ensembl\s+)?(?:gene\s+)?identifier\s+\S+\s+"
            r"(?:corresponds?\s+to|maps?\s+to|is)\s+"
            r"(?:the\s+)?(?:official\s+)?(?:gene\s+)?(?:symbol\s+)?"
            r"|based on[^,]*,\s*"
            r"|according to[^,]*,\s*"
            r"|the\s+gene\s+symbol\s+for\s+.*?\s+is\s+"
            r"|the\s+official\s+gene\s+symbol\s+.*?\s+is\s+"
            r"|the\s+SNP\s+\S+\s+is\s+(?:most\s+closely\s+)?associated\s+with\s+"
            r"(?:the\s+)?(?:gene\s+)?)",
            "",
            line,
            flags=re.IGNORECASE,
        )
        line = line.rstrip(".")
        # Take first sentence only ("GENE. This gene encodes..." → "GENE")
        cleaned = line.split(". ")[0].strip()
        # Strip trailing parenthetical notes ("PTK7 (Protein Tyrosine Kinase 7)" → "PTK7")
        cleaned = re.sub(r"\s*\(.*?\)\s*$", "", cleaned).strip()
        if cleaned:
            return cleaned
    return None


def _extract_species(text: str) -> str | None:
    """Extract a species name from model output.

    Prefers the *last* mentioned species (the model's conclusion) over earlier
    incidental mentions.  Also resolves scientific / Latin names via
    ``_SPECIES_ALIASES``.
    """
    lower = _strip_thinking(text).lower()

    # Build list of (position, common_name) for every mention.
    mentions: list[tuple[int, str]] = []
    for species in _KNOWN_SPECIES:
        idx = lower.rfind(species)
        if idx >= 0:
            mentions.append((idx, species))
    for alias, common in _SPECIES_ALIASES.items():
        idx = lower.rfind(alias)
        if idx >= 0:
            mentions.append((idx, common))

    if not mentions:
        return None
    # Return the species whose *last* mention is latest in the text.
    mentions.sort(key=lambda t: t[0], reverse=True)
    return mentions[0][1]


def _parse_name_set(text: str) -> set[str]:
    """Parse a comma/semicolon-separated list of names into a normalized set."""
    # Split on comma, semicolon, " and ", or newline
    parts = re.split(r"[,;\n]|\band\b", text)
    result = set()
    for part in parts:
        cleaned = part.strip().strip(".")
        if cleaned and cleaned.lower() not in ("", "no gene", "none"):
            result.add(cleaned.lower())
    return result


def _extract_gene_names(text: str) -> str | None:
    """Extract gene/protein names from model output for gene_name_extraction."""
    lower = _strip_thinking(text).strip().lower()
    if "no gene" in lower or "no protein" in lower or lower.startswith("none"):
        return "No gene"
    return _extract_first_line_answer(text)


# Pattern for plausible human gene symbols: 1-2 uppercase letters followed by
# at least one alphanumeric char, optionally with hyphens (e.g. HLA-A).
_GENE_SYMBOL_PATTERN = re.compile(r"\b([A-Z][A-Z0-9][A-Z0-9/.-]{0,12})\b")

# Common abbreviations that look like gene symbols but aren't.
_NOT_GENE_SYMBOLS = frozenset(
    {
        "THE",
        "AND",
        "FOR",
        "NOT",
        "ARE",
        "BUT",
        "ALL",
        "CAN",
        "HAS",
        "WAS",
        "ONE",
        "OUR",
        "OUT",
        "YOU",
        "DNA",
        "RNA",
        "SNP",
        "GO",
        "NCBI",
        "OMIM",
        "PHP",
        "PFK",
        "MRI",
        "QT",
        "AD",
        "AR",
        "IV",
        "CKD",
        "MDS",
        "ASD",
        "ECM",
        "BAM",
        "HERE",
        "THIS",
        "THAT",
        "WITH",
        "HAVE",
        "FROM",
        "THEY",
        "BEEN",
        "SAID",
        "EACH",
        "WILL",
        "ALSO",
        "THAN",
        "NOW",
        "ITS",
        "WHO",
        "MAY",
        "USE",
        "DUE",
        "TYPE",
        "GENE",
        "HUGO",
        "HGNC",
        "HGMD",
        "KEY",
        "RARE",
        "NOTE",
        "HTTP",
        "HTTPS",
        "STEP",
        "VIA",
    }
)


def _extract_gene_symbols(text: str) -> str | None:
    """Extract gene symbols from the full model output.

    Scans all uppercase tokens that look like gene symbols, returning them as
    a comma-separated string for use with set-based scorers.
    """
    text = _strip_thinking(text)
    # Prefer bold-formatted symbols (**GENE**) as the model's primary answers
    bold_genes = re.findall(r"\*\*([A-Z][A-Z0-9/.-]{1,12})\*\*", text)
    bold_genes = [g for g in bold_genes if g not in _NOT_GENE_SYMBOLS]
    if bold_genes:
        return ", ".join(bold_genes)

    # Fall back to all uppercase gene-like tokens
    candidates = _GENE_SYMBOL_PATTERN.findall(text)
    candidates = [c for c in candidates if c not in _NOT_GENE_SYMBOLS]
    if candidates:
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return ", ".join(unique)
    return None


def _extract_go_term(text: str) -> str | None:
    """Extract a Gene Ontology term name from model output.

    Looks for quoted terms or terms following "is (likely)" patterns, which is
    how models typically present GO term predictions.
    """
    text = _strip_thinking(text)

    # Try to find quoted GO term (the most reliable signal)
    # Prefer the first quoted term after "is likely" or "is"
    quoted_after_is = re.search(
        r'is\s+(?:likely\s+)?["\u201c]([^"\u201d]+)["\u201d]',
        text,
        re.IGNORECASE,
    )
    if quoted_after_is:
        return quoted_after_is.group(1).strip()

    # Any quoted string that isn't a gene name or GO ID
    quoted = re.findall(r'["\u201c]([^"\u201d]{5,})["\u201d]', text)
    for q in quoted:
        if not re.fullmatch(r"GO:\d+", q) and not q.isupper():
            return q.strip()

    # Fall back to bold text after "is (likely)"
    bold_after_is = re.search(
        r"is\s+(?:likely\s+)?\*\*([^*]+)\*\*",
        text,
        re.IGNORECASE,
    )
    if bold_after_is:
        term = bold_after_is.group(1).strip()
        # Strip GO ID suffix like "(GO:0006629)"
        term = re.sub(r"\s*\(GO:\d+\)\s*$", "", term)
        return term

    return _extract_first_line_answer(text)


# =============================================================================
# System Prompt and Defaults
# =============================================================================

_SYSTEM_PROMPT = """\
You are a genomics expert. Answer the following question concisely and accurately.
Give only the answer with no explanation unless asked."""

_DEFAULT_SAMPLING = SamplingParams(temperature=0.0, max_tokens=512)
_DEFAULT_FORMATTER = ChatFormatter(system_prompt=_SYSTEM_PROMPT)

# Per-task formatters for modules where the gold answer format is non-obvious.
_PROTEIN_CODING_FORMATTER = ChatFormatter(
    system_prompt="""\
You are a genomics expert. You will be asked whether a gene is protein-coding.
Answer "TRUE" if the gene is protein-coding, or "NA" if it is not. \
Give only the answer with no explanation.""",
)

_TF_REGULATION_FORMATTER = ChatFormatter(
    system_prompt="""\
You are a genomics expert. You will be asked whether a transcription factor \
activates or represses a target gene.
Answer "Activation" or "Repression". Give only the answer with no explanation.""",
)

_GENE_DISEASE_FORMATTER = ChatFormatter(
    system_prompt="""\
You are a genomics expert. You will be asked which genes are associated with \
a disease. List the official gene symbols separated by commas. \
Give only the gene symbols with no explanation.""",
)

_GENE_ALIAS_FORMATTER = ChatFormatter(
    system_prompt="""\
You are a genomics expert. You will be asked for the official gene symbol \
corresponding to a gene alias. Give only the official symbol with no explanation.""",
)

_GENE_ONTOLOGY_FORMATTER = ChatFormatter(
    system_prompt="""\
You are a genomics expert. You will be given a set of genes and asked for the \
enriched Gene Ontology term they share. Give only the GO term name with no \
explanation.""",
)

_MULTI_SPECIES_FORMATTER = ChatFormatter(
    system_prompt="""\
You are a genomics expert. You will be given a DNA sequence and asked which \
organism it comes from. The answer is one of: human, mouse, rat, zebrafish, \
worm, yeast, fruit fly, chicken. Give only the organism name with no explanation.""",
)

# Lower max_tokens for tasks where the answer is short but inputs are long
# (DNA sequences), to reduce generation time and avoid vLLM timeouts.
_SHORT_ANSWER_SAMPLING = SamplingParams(temperature=0.0, max_tokens=128)


# =============================================================================
# Base Task
# =============================================================================


class GeneTuringTask(Task):
    """Base class for GeneTuring subtasks."""

    def __init_subclass__(
        cls,
        subset: str | None = None,
        scorer: type[Scorer] = ExactMatchScorer,
        metric_name: str = "accuracy",
        formatter: ChatFormatter | None = None,
        sampling_params: SamplingParams | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if subset is None:
            return

        name = f"geneturing_{subset}"

        cls.data_source = DataSource(path="allenai/geneturing", subset=subset)
        cls.formatter = formatter or _DEFAULT_FORMATTER
        cls.sampling_params = sampling_params or _DEFAULT_SAMPLING

        if metric_name == "accuracy":
            metric: Metric = AccuracyMetric(scorer=scorer)
        else:
            metric = MeanScoreMetric(name=metric_name, scorer=scorer)

        cls.metrics = (metric,)
        cls.primary_metric = metric
        register(name)(cls)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        question = doc.get("question", "")
        gold_answer = doc.get("gold_answer", "")
        if not question:
            return None
        return Instance(
            question=question,
            gold_answer=gold_answer,
            metadata={
                "id": doc.get("id", f"geneturing_{index}"),
                "index": index,
            },
        )

    @property
    def request_type(self) -> RequestType:
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.CHAT

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )


# =============================================================================
# Tier 1 — Exact Match Tasks (9 modules)
# =============================================================================


class GeneNameConversion(GeneTuringTask, subset="gene_name_conversion"):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_first_line_answer(output.text)


class GeneLocation(GeneTuringTask, subset="gene_location", scorer=ChromosomeScorer):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_chromosome(output.text)


class SnpLocation(GeneTuringTask, subset="snp_location", scorer=ChromosomeScorer):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_chromosome(output.text)


class GeneSnpAssociation(GeneTuringTask, subset="gene_snp_association"):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_first_line_answer(output.text)


class ProteinCodingGenes(
    GeneTuringTask, subset="protein_coding_genes", formatter=_PROTEIN_CODING_FORMATTER
):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_boolean(output.text)


class TfRegulation(GeneTuringTask, subset="tf_regulation", formatter=_TF_REGULATION_FORMATTER):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_activation(output.text)


class HumanGenomeDnaAlignment(
    GeneTuringTask, subset="human_genome_dna_alignment", scorer=ChromosomeScorer
):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_chromosome(output.text)


class AminoAcidTranslation(
    GeneTuringTask,
    subset="amino_acid_translation",
    sampling_params=_SHORT_ANSWER_SAMPLING,
):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_sequence(output.text, _AMINO_ACID_PATTERN)


class DnaSequenceExtraction(GeneTuringTask, subset="dna_sequence_extraction"):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_sequence(output.text, _DNA_PATTERN)


# =============================================================================
# Tier 2 — Set Overlap Tasks (3 modules)
# =============================================================================


class GeneNameExtraction(
    GeneTuringTask,
    subset="gene_name_extraction",
    scorer=JaccardScorer,
    metric_name="jaccard",
):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_gene_names(output.text)


class GeneAlias(
    GeneTuringTask,
    subset="gene_alias",
    scorer=JaccardScorer,
    metric_name="jaccard",
    formatter=_GENE_ALIAS_FORMATTER,
):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_first_line_answer(output.text)


class GeneDiseaseAssociation(
    GeneTuringTask,
    subset="gene_disease_association",
    scorer=GeneRecallScorer,
    metric_name="recall",
    formatter=_GENE_DISEASE_FORMATTER,
):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_gene_symbols(output.text)


# =============================================================================
# Tier 3 — Partial Credit Tasks (2 modules)
# =============================================================================


class GeneOntology(
    GeneTuringTask,
    subset="gene_ontology",
    scorer=ContainmentScorer,
    metric_name="containment",
    formatter=_GENE_ONTOLOGY_FORMATTER,
):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_go_term(output.text)


class MultiSpeciesDnaAlignment(
    GeneTuringTask,
    subset="multi_species_dna_alignment",
    scorer=SpeciesScorer,
    metric_name="species",
    formatter=_MULTI_SPECIES_FORMATTER,
    sampling_params=_SHORT_ANSWER_SAMPLING,
):
    def extract_answer(self, output: LMOutput) -> str | None:
        return _extract_species(output.text)
