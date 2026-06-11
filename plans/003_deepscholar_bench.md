# DeepScholar-Bench integration plan

Status: SKELETON committed, UNVALIDATED. Tracks step 4 of
`plans/002_science_literature_evals.md`. Nothing here has been run end to end;
validation requires a real environment (Docker + GPU/vLLM + API keys), e.g. a
beaker job.

## What it is

DeepScholar-Bench (UC Berkeley Sky Computing Lab, arXiv 2508.20033) evaluates
generative research synthesis: given a paper, generate its related-work section
by retrieving, synthesizing, and citing prior work. Scored across knowledge
synthesis, retrieval quality, and verifiability, aggregated as a geometric mean.
No published system exceeds ~31% geomean.

Repo: https://github.com/guestrin-lab/deepscholar-bench (Python 3.10, LOTUS
framework). Reference pipeline: DeepScholar-ref (`deepscholar_base`).

## Skeleton (committed)

`src/olmo_eval/evals/external/benchmarks/deepscholar/`:
- `eval.py` — `DeepScholarExternalEval(SandboxedExternalEval)`, modeled on
  `tau2/eval.py`. Registered as `deepscholar_bench`.
- `args.py` — `DeepScholarArgs`.
- `result_parser.py` — `parse_results_csv` (best-effort).
- `__init__.py` — registers the eval.

Run shape (intended): `olmo-eval run-external -m <model> -e deepscholar_bench`.
Two phases inside the sandbox: generation (`python -m deepscholar_base.main
--queries-file dataset/queries.csv --config-yaml <model config>`) then eval
(`python -m eval.main --evals all --input-folder <generated> --model-name
gpt-4o`), which writes `results.csv`.

## Design decisions

- **Pin to the shipped snapshot.** Use the repo's `dataset/*.csv` rather than the
  live arXiv `data_pipeline`. This makes runs reproducible and sidesteps the
  "live = non-reproducible" problem flagged in plan 002. The live monthly-refresh
  pipeline is explicitly out of scope.
- **Model under test plugs in via the LOTUS config**, not a CLI flag. We write a
  config YAML pointing LOTUS's `LM` at the model; for local vLLM, a
  `hosted_vllm/<model>` string + `api_base` (litellm convention, as in tau2).
- **Judge stays gpt-4o** (the upstream eval default). Generation needs Tavily.
  Required secrets: `OPENAI_API_KEY`, `TAVILY_API_KEY`.

## Open / unverified items (must fix during validation)

1. **LOTUS config schema** (`_write_model_config`). The YAML structure
   (`lm: {model, api_base, ...}`) is a guess. Confirm against
   `configs/deepscholar_base.yaml` and how `deepscholar_base.main` loads it; verify
   a local vLLM endpoint is reachable through litellm.
2. **results.csv schema** (`result_parser.py`). Column names and which are metrics
   vs. metadata are unconfirmed; the geomean is computed locally over positive
   metric columns rather than read from upstream. Reconcile with a real file and,
   if upstream emits its own geomean, use that.
3. **`--evals` / `--modes` values.** Defaulted to `all` / `deepscholar_base`
   without enumerating the `EvaluationFunction` / `ParserType` enums. Confirm the
   metric set (organization, nugget_coverage, reference_coverage, cite_p, and the
   rest) and that `all` is accepted.
4. **Generation limit flag** (`--num-queries`). Guessed; confirm upstream supports
   limiting queries for smoke runs, and the exact flag name.
5. **Commit pin.** `DEEPSCHOLAR_REF = "main"`; pin to a specific commit once a run
   succeeds.
6. **Install path.** Assumes `uv venv && uv pip install -r requirements.txt` on a
   `uv:python3.10` image. Confirm requirements install cleanly (LOTUS, faiss,
   litellm, etc.) and the `.venv/bin/python` invocation works.
7. **Dataset/reference paths.** Generation reads `dataset/queries.csv`; eval
   defaults reference `dataset/papers_with_related_works.csv`,
   `dataset/important_citations.csv`, `dataset/gt_nuggets_outputs`,
   `test/baselines_results/openscholar`. Confirm these ship in the repo.

## Validation steps (beaker)

1. Build/pull the image; run setup (`git clone` + install) and confirm it
   completes.
2. Smoke run with `num_queries` small against a known model; inspect the written
   LOTUS config, generation outputs, and `results.csv`.
3. Fix items 1-4 against real output; pin the commit (item 5).
4. Full run; sanity-check the geomean against the paper's ~31% ceiling.

## Expectations

Track, not hillclimb. An early OLMo may fail to drive the LOTUS pipeline at all
(errors / empty output, not just low scores). Do not gate progress on gradient
from this benchmark until a model clears the floor. Not wired into any suite yet;
add to `science:research` only after a successful run.
