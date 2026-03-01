"""Biology evaluation suite."""

from olmo_eval.evals.suites.registry import make_suite

# =============================================================================
# LAB-Bench Suite
# =============================================================================

_LAB_BENCH_TASKS = (
    "lab_bench_litqa2",
    "lab_bench_dbqa",
    "lab_bench_seqqa",
    "lab_bench_protocolqa",
    "lab_bench_suppqa",
    "lab_bench_cloning_scenarios",
)

LAB_BENCH = make_suite(
    "lab_bench",
    _LAB_BENCH_TASKS,
    description="LAB-Bench biology research benchmark (futurehouse/lab-bench)",
)

LAB_BENCH_MC = make_suite(
    "lab_bench:mc",
    tuple(f"{t}:mc" for t in _LAB_BENCH_TASKS),
    description="LAB-Bench with logprob-based MC scoring",
)

LAB_BENCH_BPB = make_suite(
    "lab_bench:bpb",
    tuple(f"{t}:bpb" for t in _LAB_BENCH_TASKS),
    description="LAB-Bench with bits-per-byte evaluation",
)

# =============================================================================
# GeneTuring Suite
# =============================================================================

_GENETURING_TASKS = (
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

GENETURING = make_suite(
    "geneturing",
    _GENETURING_TASKS,
    description="GeneTuring genomics Q&A benchmark (14 modules, 1,400 questions)",
)

# =============================================================================
# Combined Biology Suite
# =============================================================================

BIOLOGY = make_suite(
    "biology",
    (LAB_BENCH, GENETURING),
    description="Combined biology benchmarks (LAB-Bench + GeneTuring)",
)
