# Science & science-literature evals: running inventory

Owner: Rory (roryd). Last updated: 2026-06-22. Contributors welcome.

A running list of the science and science-literature evals we have or are building, with
what each measures, its size, and current status. Four sections:

1. Literature evals we have
2. Literature evals we are working on
3. General science evals we have
4. Uncovered eval gaps

Each eval entry uses the same fields (Summary / Items / Dataset / Scoring / Suites / Status)
so it maps cleanly onto spreadsheet columns. The new literature work lives on the
`roryd/science-lit` branch, with design notes under `plans/`. The reusable piece across the
literature evals is one citation-grounding scorer
(`src/olmo_eval/common/scorers/citation.py`); the benchmarks are delivery vehicles for it.

## 1. Literature evals we have

### ExpertQA

- Summary: attribution and on-topic precision of cited long-form answers (citation
  precision/recall plus answer precision). It does not measure factual correctness against a
  reference, so a high score means well-cited, on-topic prose – notably this does not check
  for hallucinated sources.
- Items: 2,177 questions across 32 fields.
- Dataset: cmalaviya/expertqa.
- Scoring: LLM-as-judge (citation scorer plus an irrelevant-paragraph judge).
- Suites: science:research, science:judge.
- Status: done.

### AstaBench ScholarQA

- Summary: cited scientific report synthesis (ScholarQA-CS2: long-form CS literature-review
  answers, scored for coverage and citation precision).
- Items: 100 questions.
- Dataset: allenai/asta-bench (astabench_scholarqa, tasks/sqa/rubrics_v1_recomputed.json).
- Scoring: LLM-as-judge (citation precision/recall plus answer precision).
- Suites: astabench, science:research, science:judge.
- Status: done; pre-existing on main. This branch only extracted the shared citation
  scorer out of it (the two evals built on this branch are ExpertQA and LitSearch).

### Citation scorer + judge (shared infrastructure, not an eval)

- Summary: the citation precision/recall grading shared by ExpertQA and AstaBench ScholarQA
  (entry point `score_citations_for_sections`).
- Judge: validated by an adversarial kill-test (`common/scorers/citation_validation.py`).
  gpt-4o-mini failed the score-inflating cases and was retired; gpt-5.5:medium is the
  current default, overridable via `$OLMO_EVAL_JUDGE`.
- Status: done. A human-agreement labeling study is scaffolded but not yet populated, and is
  the gate before adding more judge-based tasks.

## 2. Literature evals we are working on

### LitSearch

- Summary: literature retrieval / discernment. Given a research query, find the relevant
  prior papers from the literature, scored against gold papers by Recall@5 and Recall@20.
  The target version runs over a fixed, pinned corpus (no live API, so it is reproducible),
  needs no judge, and scores the model's own selection rather than whatever a tool surfaced.
- Items: 597 queries (reranked over a frozen 64,183-paper corpus).
- Dataset: princeton-nlp/LitSearch (queries with gold Semantic Scholar corpus IDs; corpus
  is corpus_clean).
- Scoring: deterministic Recall@5 (primary) / @20 over the model's selected set.
- Suites: science:research, science:nojudge, science:all.
- Status: partially built. A fixed-corpus reranking version is coded and tested, pending a
  one-time pool build and a beaker run for real numbers. An agentic stopgap exists (the
  model searches the live S2 API), but that is a retrieval smoke test, not the published
  Recall@k. The two should converge on the fixed-corpus version; a dense retriever baseline
  (encoder plus FAISS) is the follow-on.

### SAGE

- Summary: retrieval for deep-research agents over scientific literature. Reuses the
  LitSearch infrastructure.
- Items: 1,200 queries across four domains (computer science, healthcare, humanities,
  natural science), over a 200,000-paper corpus.
- Dataset: SAGE (arXiv 2602.05975).
- Scoring: retrieval metrics (Recall@k style), judge-free.
- Suites: TBD (planned for the science:research / retrieval path).
- Status: planned; Yilun.

### ResearchQA

- Summary: scholarly question answering. Long-form answers graded against query-specific
  rubric items (citing papers, giving explanations, describing limitations).
- Items: 21,414 questions across 75 fields / 7 domains (3,750 test, 703 validation); 160K
  rubric items.
- Dataset: ResearchQA (arXiv 2509.00496).
- Scoring: rubric-based automatic pairwise judge (74% agreement with expert annotators).
- Suites: TBD; likely needs a new rubric-judge eval type (science:rubric-judge).
- Status: planned; Yilun.

### DeepScholar-Bench

- Summary: generative research synthesis. Given a paper, generate its related-work section
  by retrieving, synthesizing, and citing prior work. No published system exceeds ~31%
  geomean.
- Items: 63 queries (each an arXiv paper with an expert-written exemplar; we pin a shipped
  snapshot rather than the live monthly refresh).
- Dataset: DeepScholar-Bench (arXiv 2508.20033).
- Scoring: geometric mean across knowledge synthesis, retrieval quality, and verifiability;
  gpt-4o judge (needs OPENAI + TAVILY keys). Track-not-hillclimb.
- Suites: not wired yet.
- Status: skeleton committed but unvalidated (never run end to end; LOTUS config and results
  schema unconfirmed). Needs a beaker validation run.

### Citation selection-bias probe

- Summary: fairness diagnostic for gender and majority bias when a model picks references
  from a pool of equally-relevant candidates. Track, don't hillclimb.
- Items: not yet built; target is a few hundred focal papers across six field groups, each
  with three pool compositions.
- Dataset: reconstructed from Semantic Scholar and pinned (He, arXiv 2508.02740).
- Scoring: deterministic SRR / NSD, no judge, with a gender-even pool as the null control.
- Suites: would be a standalone science:fairness suite, kept out of the science aggregates.
- Status: plan only; needs the offline pool build.

## 3. General science evals we have

The conventional MCQ and QA exams already wired into the science hierarchy in
`src/olmo_eval/evals/suites/science.py`. science:all contains each task exactly once;
science:nojudge and science:judge split by whether an external judge is needed. Item counts
below are per constituent task (the split each task uses).

### science:core

- Summary: broad STEM knowledge and school-science exam QA.
- Items: ~7,566 total. arc_easy 2,376 (test), arc_challenge 1,172 (test), sciq 1,000
  (validation), mmlu:stem 3,018 (test, 18 STEM subjects).

### science:biology

- Summary: biology, genomics, and wet-lab science, plus the GPQA biology slices.
- Items: LAB-Bench 1,542 (LitQA2 199, DbQA 520, SeqQA 600, ProtocolQA 108, SuppQA 82,
  CloningScenarios 33); GeneTuring 1,400 (14 implemented modules x 100 questions); GPQA biology ~38 / 85 / 104 for
  diamond / main / extended.

### science:medicine

- Summary: medical QA and clinical knowledge.
- Items: medmcqa 4,183 (validation), medqa_en 1,273 (test, USMLE-style 4-way MCQ), plus 8
  medicine-focused MMLU subjects, 1,640 total (test): anatomy 135, clinical_knowledge 265,
  college_medicine 173, human_aging 223, medical_genetics 100, nutrition 306,
  professional_medicine 272, virology 166.

### science:physical

- Summary: chemistry and physics, via the GPQA chemistry and physics slices.
- Items: GPQA chemistry + physics for each of diamond / main / extended. GPQA has 198 / 448
  / 546 questions per variant; chemistry and physics together are ~80% of each (physics
  ~41%, chemistry ~39%, biology ~19%), so the physical slice is roughly 160 / 360 / 440
  questions respectively.

### science:research

- Summary: scientific literature understanding and evidence use.
- Items: qasper_yesno 319 (train), sciriff_yesno 1,582 (train); plus the literature members
  detailed in sections 1 and 2 (expertqa 2,177, litsearch 597, astabench 100).
- Note: qasper_yesno (allenai/qasper-yesno) and sciriff_yesno (allenai/sciriff-yesno) are
  yes/no QA grounded in paper text.

### science:math

- Summary: mathematical reasoning.
- Items: gsm8k 1,319 (test), gsm_symbolic 5,000 (apple/GSM-Symbolic main), minerva_math
  5,000 (MATH test, 7 subsets), math500 500, aime_2024 30, aime_2025 30.

### GPQA (legacy convenience suites)

- Summary: graduate-level diamond / main / extended, sliced into biology, chemistry, and
  physics.
- Items: 198 (diamond) / 448 (main) / 546 (extended) questions. gpqa, gpqa:mc, gpqa:bpb are
  retained as entry points but kept out of science:all to avoid double-counting questions
  already allocated to the domain suites.

## 4. Uncovered eval gaps

### Paper discernment / research taste

- Gap: judging which research ideas or papers are high-impact or worth pursuing. No clean,
  high-quality dataset found to onboard directly.
- Candidates: "AI Can Learn Scientific Taste" (arXiv 2603.14473), "Predicting Empirical AI
  Research Outcomes with Language Models" (arXiv 2506.00794, includes a RAG setting), and
  paper-review / limitation-generation tasks.

### Hallucinated-source detection

- Gap: none of our citation evals directly catch fabricated references. ExpertQA and
  AstaBench ScholarQA reward well-attributed, on-topic prose but do not verify that a cited
  source exists or actually supports the claim's specifics.

### Factual correctness against a reference

- Gap: our literature evals measure attribution and retrieval, not whether the answer is
  factually true relative to a gold reference. ExpertQA's annotations grade its
  pre-generated answers, so they cannot grade a fresh model's answer.

### Non-MCQ physical science

- Gap: physical-science coverage is GPQA multiple choice. We have no open-ended or
  figure/table-reasoning physical-science eval.
