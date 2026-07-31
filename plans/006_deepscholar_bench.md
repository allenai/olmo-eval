# DeepScholar-Bench integration

Status: IMPLEMENTED, awaiting first beaker validation run. Ported from the
shelved skeleton (`../olmo-eval-shelved/003_deepscholar_bench.md`), this time
modeled on `tau2`/`asta` and built against the verified upstream interface.

## What it is

DeepScholar-Bench (UC Berkeley, arXiv 2508.20033) evaluates generative research
synthesis: given a paper's context, a system retrieves prior work and writes the
related-work section, scored on `organization`, `nugget_coverage`,
`reference_coverage`, `cite_p` and aggregated as a geometric mean. No published
system exceeds ~31% geomean. Upstream: https://github.com/guestrin-lab/deepscholar-bench
(Python 3.10, LOTUS framework).

## Implementation

`src/olmo_eval/evals/external/benchmarks/deepscholar/` — registered as
`deepscholar_bench`, a `SandboxedExternalEval`:
- `eval.py` — `DeepScholarExternalEval`. Stock image `ghcr.io/astral-sh/uv:python3.10-bookworm`;
  clones the repo and `uv pip install -r requirements.txt` in `setup_command`
  (no custom image/registry, like tau2). Two phases in one sandbox: generation
  then eval. Model under test wired in via a generated LOTUS config, not a flag.
- `args.py` — `DeepScholarArgs` + `PRIMARY_METRICS`.
- `result_parser.py` — best-effort JSON/CSV metric extraction + geomean.
- `__init__.py` — registers the eval (picked up by the lazy pkgutil walk).

Run: `olmo-eval run-external -m <model> -e deepscholar_bench [-a limit=N]`.

## Verified upstream interface (from source, 2026-06-30)

Generation (`deepscholar_base/main.py`):
- `--output-folder` (required), `--config-yaml` (default `configs/deepscholar_base.yaml`),
  `--queries-file` (default `dataset/queries.csv`), `--start-idx`/`--end-idx`,
  `--model` (overrides only the LM *name*, keeping other YAML lm settings).
- Writes per query `{out}/{idx}/{final_report.md,intro.md,paper.csv,stats.json}`
  and `{out}/summary.json`.
- The `lm:` YAML block becomes `LM(**lm)` (LOTUS over litellm). We write our own
  config (JSON is valid YAML) so we can inject `api_base` for local vLLM —
  `--model` alone can't carry a base URL.

Eval (`eval/main.py`):
- `--modes deepscholar_base`, `--evals {organization,nugget_coverage,reference_coverage,cite_p,all}`,
  `--input-folder <gen out>`, `--output-folder`, `--model-name` (judge, default gpt-4o).
- `--dataset-path`, `--important-citations-path`, `--nugget-groundtruth-dir-path`,
  `--reference-folder` all default to files that ship in `dataset/` — we leave them.

Secrets: `OPENAI_API_KEY` (judge gpt-4o) is the only hard requirement. Retrieval
defaults to the **keyless ARXIV corpus** (the dataset is arXiv CS papers), so no
Tavily account is needed. `TAVILY_API_KEY` (and `S2_API_KEY`/`SERPAPI_API_KEY`) are
forwarded into the sandbox only if set, for users who opt into another corpus via
`-a web_corpuses=TAVILY` (`SERPAPI_API_KEY` for GOOGLE/GOOGLE_SCHOLAR). Corpus
options: ARXIV, TAVILY, GOOGLE, GOOGLE_SCHOLAR, BING (only ARXIV is keyless). This
diverges from the paper's TAVILY default — acceptable since local runs already
diverge by using recursive search; track, don't compare to published numbers.

Caveat: `web_corpuses` is honored only by **recursive** search. Upstream **agentic**
search hardwires a `search_web` tool on `WebSearchCorpus.TAVILY` (when
`enable_web_search=True`) and ignores `web_corpuses` — so it still needs Tavily.
Local providers default to recursive (Tavily-free), but external API models keep
agentic by default; set `-a search_mode=recursive` to stay keyless there too.

Dataset: `papers_with_related_works.csv`, `important_citations.csv`,
`related_works_combined.csv`, `gt_nuggets_outputs/` ship. `queries.csv` does NOT —
generation auto-derives queries from `papers_with_related_works.csv` when absent.

## Local-model wiring

Generation lm block for local vLLM:
`{"model": "openai/<model>", "api_base": <sandbox_url>, "api_key": "EMPTY", ...}`.
`openai/` routes through litellm's OpenAI handler to the vLLM server; fallback
prefix `hosted_vllm` is exposed via `-a local_model_prefix=hosted_vllm`. The judge
(gpt-4o) hits real OpenAI — we never set `OPENAI_BASE_URL`, so the base URL stays
scoped to the generation lm config.

**Search mode (matters for the model string).** Upstream's *agentic* search builds
a raw OpenAI Agents SDK client (`_lotus_lm_to_openai_lm`): it reads `api_base` from
the lm kwargs (good) but sends `lm.model` *verbatim*, so the `openai/` prefix the
LOTUS sem-ops require would be rejected by vLLM — one `lm.model` field can't serve
both paths. *Recursive* search keeps every model call on the LOTUS/litellm path,
where the prefix is consistent. So for local (vLLM) providers we default
`search_mode=recursive`; external API models keep the upstream default (agentic).
Override with `-a search_mode=...`. Agentic-mode-on-vLLM would need upstream changes
(raw model name for the agent client + prefixed name for sem-ops).

**Eval set.** Default `evals` is the four headline metrics (`organization`,
`nugget_coverage`, `reference_coverage`, `cite_p`) — the geomean inputs and a
smaller failure surface for the first run. `-a evals=all` opts into the full
upstream set (adds `document_importance`, `claim_coverage`, `coverage_relevance_rate`).

## Run log

- **Run 1 (2026-07-01): FAILED at sandbox startup**, before the eval ran. The bare
  `uv:python3.10-bookworm` image has no swe-rex, and the runtime bootstrap invoked
  the wrong console script (`swe-rex` vs the installed `swerex-remote`) →
  "Container process terminated." Fix: override `_create_sandbox_config` to set
  `inject_swerex=True` (builds a derived image with swe-rex + git/curl preinstalled,
  as `scicode` does). Re-run after committing + pushing.
- **Run 2 (2026-07-01): swe-rex OK, stalled on `git clone`.** No submodules, but the
  repo is ~1.3GB (dataset CSVs + baseline outputs) so a full clone crawls. Fix:
  shallow `git init` + `git fetch --depth 1 origin <ref>` + `checkout FETCH_HEAD`
  (works for branch or SHA). Also: the swe-rex derived-image registry
  (`ai2-allennlp/olmo-eval`) returns 403 for pull AND push, so every fresh job
  rebuilds the swe-rex image (~3 min) locally. Infra/permissions, not our code; not
  blocking. If clone/install stay painful, bake the repo+deps into a custom image
  (asta-style) instead of cloning at setup.
- **Run 3 (2026-07-01): setup fully OK, generation crashed `ModuleNotFoundError:
  pandas`.** `uv pip install -r requirements.txt` installed into the swe-rex image's
  active `/root/venv` (3.12), not our repo `.venv` (3.10), so `.venv/bin/python` had
  no deps. Fix: `uv pip install --python {.venv}/bin/python`. Confirmed working this
  run: shallow-clone-less full clone (11 min), requirements resolve+install cleanly
  (lotus-ai + nuggetizer build), LOTUS config written (model=openai/allenai/Olmo-3-7B-
  Instruct), provider health check passes. Both Run 2 + Run 3 fixes still UNPUSHED.
- **Run 4 (2026-07-01): GENERATION fully works; eval crashed on missing NLTK data.**
  Generation ran end to end (recursive + arXiv retrieval, vLLM OLMo via litellm openai/
  prefix, structured output — all fine). Eval phase got deep into scoring (organization/
  reference matching done) then `cite_p` hit `LookupError: Resource punkt_tab not found`
  (`nltk.sent_tokenize`). Fix: setup step `{.venv}/python -m nltk.downloader punkt_tab`
  (only sent_tokenize is used across eval/; no other NLTK resources needed). Also set an
  explicit error message when the eval phase exits nonzero (was surfacing "Failed: None").
  Big milestone: the whole model-under-test path is validated; only the eval env was short
  a data file.
- **Run 5 (2026-07-01): full success; parser rewritten to match real schema.** Downloaded
  results via `beaker experiment results <id>`. Confirmed eval layout:
  `evaluation/<metric>/aggregated_results.csv` (`baseline_name,<metric>` with a
  `deepscholar_base` row) + `deepscholar_base.csv` (per-query rows). There is NO overall
  aggregate file. The old generic parser was noisy: all `aggregated_results.csv` share a
  stem (collision), and per-query rows leaked in as path-named "metrics". Rewrote
  `_extract_results` to read `<metric>/aggregated_results.csv` canonically → clean
  {organization, nugget_coverage, reference_coverage, cite_p} + geomean. Verified against
  real files: nugget_coverage 0.33, organization 0.0, reference_coverage 0.0.
  Interpretation: generation is real (2 queries, 15/11 papers retrieved, ~25K-char
  reports). Zeros are genuine scorer outputs, not artifacts. organization=0 (v1+v2 both 0
  on both queries) is SUSPECT — text is clearly organized; citations render malformed
  (`\[1]\]`), which plausibly zeroes reference_coverage/cite_p and may confuse the org
  judge. Upstream-scorer question, not our bug; revisit before trusting org/cite numbers.

- **Run 6 (2026-07-01, limit=10): arXiv 429 + hang.** Recursive search fired ~60 arXiv
  requests (10 papers x 3 steps x 2 queries); export.arxiv.org rate-limited (429) despite
  the arxiv lib's 3s spacing (shared cluster egress IP), then one request hung with no
  socket timeout -> sandbox watchdog aborted after 3x300s -> generation killed (-1), no
  output. The 2-query run worked because volume was ~12 requests. Added `search_steps` /
  `search_queries_per_step` args to throttle (retry: limit=10 search_steps=1
  search_queries_per_step=1 -> ~10 requests). Keyless arXiv is fragile at volume; Tavily
  (free tier, `-a web_corpuses=TAVILY` + secret) is the robust corpus for larger runs.

## Search backends (branch roryd/deepscholar-bench-s2patch)

Recursive search runs an **unconditional** arXiv search (`recursive_search.py`:
"Run search for arxiv" -> `web_search([ARXIV], ...)`), plus an optional web search
over `web_corpuses` gated by `enable_web_search`. So `web_corpuses=TAVILY` *adds*
Tavily but does not remove arXiv — which is why Tavily runs still hit arXiv 429s/hangs.

`-a search_backend=` selects the retrieval backend without editing the external
clone, via a runtime shim (`sandbox_search_shim.py`) that monkeypatches
`recursive_search._process_single_lotus_search_task` (module globals resolve at
call time, so overriding the attribute after import works):
- `arxiv` (default): upstream behavior, no shim.
- `s2`: routes every search to the Semantic Scholar API (keyed via S2_API_KEY,
  request timeouts + 429 backoff -> no arXiv-style hangs). Eval sets
  `enable_web_search=false` so only one S2 pass runs. Populates authors/id/date, so
  citations should be better-formed (may lift cite_p/reference_coverage).
- `tavily`: shim skips the ARXIV corpus, keeps the TAVILY web corpus (needs
  TAVILY_API_KEY).

Row schema the shim must emit (from upstream `_safe_lotus_async_search`):
`title, url, snippet, query, context, date` (+ optional authors, id). Standardized
output confirmed by reading recursive_search + summary_generation + final_generation.

Codex review fixes (2026-07-01): (1) CRITICAL — the eval parser only registers a
citation when the markdown URL matches `arxiv.org/abs/<id>`, so S2 semanticscholar
URLs would score 0 on reference_coverage/cite_p; the shim now emits `arxiv.org/abs/<id>`
(+ that id) when the S2 record has an ArXiv external id (non-arXiv hits keep their S2
URL as unscorable context; URL is never fetched, so no arXiv API calls). (2) non-arxiv
backends now force `search_mode=recursive` (the shim only intercepts the recursive path;
agentic uses separate LOTUS tools). (3) S2 date cutoff post-filters rows strictly before
end_date (year-only falls back to strict prior-year) instead of the coarse year= param.
(4) missing S2_API_KEY/TAVILY_API_KEY now fails fast in execute() with an actionable
message rather than deep in generation.

Stage-LM token budget (Fix A): upstream `initialize_lms` drops the configured
budget, so filter/search/taxonomize/generation LMs default to LOTUS's 512 and
truncate structured outputs (taxonomy `Categories` JSON -> parse failure -> query
error; cost 3/10 queries in run 7). The shim's `_patch_stage_max_tokens` wraps
`lotus.models.LM.__init__` to default a missing `max_tokens` to the configured
budget. `max_tokens` is a cap not a target, so short stage calls are unaffected;
the model-under-test LM and the judge pass it explicitly and are untouched.
Default `stage_max_tokens=4096` (constant `DEFAULT_STAGE_MAX_TOKENS`): LOTUS sends
this as `max_completion_tokens` and vLLM rejects `prompt_tokens + max_completion_tokens
> max_model_len`, so a large budget (e.g. 10000 at 16384 context) would starve prompt
room on stage prompts that aggregate ~30 reference abstracts. 4096 clears the 512-cap
truncation while leaving ~12k for the prompt; override with `-a stage_max_tokens`. The
shim now runs for **every** backend (arxiv too) so the fix applies universally;
s2/tavily additionally get search rerouting.

`DEEPSCHOLAR_REF` pinned to `c95413b` (the shim depends on upstream internals).
Pure mapping helpers (`map_s2_paper`) import stdlib only, so they're unit-testable
without deepscholar_base/pandas; deepscholar_base is resolved via importlib at runtime.

- **Run 7 (2026-07-01, S2 backend, limit=10): SUCCESS, S2 patch validated.**
  cite_p **0.05** (nonzero — up from 0; confirms the arXiv-URL citation fix works),
  nugget_coverage 0.26, organization 0.0, reference_coverage 0.0, geomean 0.
  Generation 7/10 succeeded — the 3 failures were the upstream max_tokens=512 cap
  truncating the taxonomize `Categories` JSON (invalid JSON -> query error); eval
  then skipped those folders (intro.md FileNotFoundError). organization=0 is a
  bidirectional pairwise win-rate vs the human related-work — likely a real loss for
  a 7B, not a bug. reference_coverage=0 (title-similarity vs gold important_citations)
  is the remaining real-or-matching-gap question. The 512 cap comes from upstream
  `initialize_lms` popping the token budget so every stage LM defaults to 512
  (independent of backend) -> Fix A (raise stage-LM max_tokens via the shim) is
  justified: it cost 30% of yield here. (Harmless: eval mis-iterates our top-level
  summary.json as a query folder and skips it.)

- **Run 8 (2026-07-01, S2 + stage-budget fix, exp 01KWFXM8): SUCCESS + due diligence.**
  Stage fix confirmed working: truncation 5x(4096) vs 104x(512), gen 8/10 (up from 7).
  Metrics: nugget 0.28, cite_p 0.05, organization 0.0, reference_coverage 0.0, geomean 0.
  Diligence verdict — the zeros are REAL, not bugs:
  - organization is a bidirectional pairwise judge, related_work_a=model vs
    related_work_b=human/published section; v1=v2=0 all queries = model lost every
    comparison in both orderings (position-bias-controlled). Real, not a bug.
  - reference_coverage matches gold cited papers by arxiv-id-in-text or title-sim>0.8
    against parser.docs; cite_p uses the same docs and is nonzero, so docs are
    populated with real titles/abstracts -> 0 = model cited none of the gold refs
    (harsh but real for keyword retrieval vs authors' hand-picked citations).
  - cite_p/nugget real with sensible per-query spread (nugget 0.0-0.5, cite_p 0.0-0.14).
  Consistent with the plan's "early model fails to clear the floor" expectation (no
  published system exceeds ~31% geomean, and those are strong agentic systems).
  Real shim BUG found (not the cause of the zeros): S2 mapping emitted year-only dates
  ("2006") that crash a strict %Y-%m-%d parse -> "Failed to search and filter after 3
  retries" -> 2 queries fail with 'intro_section'. FIXED: pad year-only to YYYY-01-01.
  To verify reference_coverage definitively next run, consider copying back per-query
  paper.csv/final_report.md (currently only summary.json is copied back).

- **Run 9 (2026-07-02, S2 FULL run, exp 01KWG0PC): FAILED ~22/63, hung.** Pace was
  ~1.8 min/query (full 63 ~= 2h, not 5h). Got through queries 0-21 (reports saved) then
  HUNG on query 22's LM call at 00:19; watchdog killed the run at 00:34 -> all 22
  completed queries lost (gen killed before summary.json, so allow_partial couldn't
  save them). Two causes: (1) context overflow (19x) — stage prompts aggregating ~30
  abstracts hit ~12289 tokens, +4096 stage budget = 16385 > max_model_len 16384 (exactly
  Codex's warning); vLLM 500s. (2) a vLLM LM call hung with no timeout -> gen produced no
  output for 900s -> watchdog abort. Fixes: raise `-K max_model_len=32768` (16384 too
  small for reference-heavy prompts); added `lm_timeout` arg (default 240s) ->
  `timeout` in the LOTUS lm config (propagates to stage LMs), so a stalled call fails +
  retries instead of wedging the run. Full run is only ~2h so the 8h timeout is fine.

- **Runs 10-11 (2026-07-02, FULL, max_model_len=32768): both HUNG ~40 min in.** 32768
  fixed the context overflow; got ~22 then ~24 queries before generation stalled and the
  watchdog killed the sandbox (all work lost, no partial eval). Diagnosis: not a memory
  leak (gen process flat ~1.0G), not overflow — a **vLLM stall** (server went silent mid
  "Aggregating" LM call), **time-correlated ~40 min** not query-specific. `lm_timeout=240`
  was applied but didn't fire (litellm streaming path likely ignores it, and/or the sandbox
  itself wedged). 10-query runs are reliable (~20 min, inside the safe window). Conclusion:
  a single generation process over all 63 is unreliable; don't chase the vLLM hang. Get the
  full number by batching (~6 jobs of start_idx=N limit=12, combine per-query CSVs) or
  accept the 10-query result as the baseline (model scores at the floor regardless).

## Metrics: report all seven + geomean (2026-07-07)

The paper (Table 1/Table 2, confirmed from the PDF) defines **seven** metrics
across three dimensions and its headline score is the **geometric mean over all
seven** ("Geo. Mean" column; abstract: "no system surpasses a geometric mean of
31% across all metrics"). Our old default ran only four
(`organization,nugget_coverage,reference_coverage,cite_p`) and geomeaned those
four — that split has **no basis in the paper**; it was copied from the README's
first `--evals` example. Verified: all-7 geomean of the paper's OpenAI
DeepResearch row = 0.309 (matches Table 2); the old 4-metric geomean would have
read ~0.42.

Also corrected: the paper's §5 setup restricts every benchmarked system to
**arXiv-API-only retrieval** with a publication-date cutoff, so our default
`arxiv` backend is *paper-consistent*, not a divergence (the `TAVILY_API_KEY` in
`.env.example` is for the DeepScholar-ref pipeline's own default, not the
benchmark runs). Remaining divergences from a paper-comparable run: model under
test, and dataset snapshot (we pin `c95413b`; paper is DeepScholar-June-2025, 63
queries — count matches).

Changes made (branch `roryd/deepscholar-bench-s2patch`):
- `PRIMARY_METRICS` and the default `evals` are now the seven upstream
  `EvaluationFunction` values: `organization, nugget_coverage,
  coverage_relevance_rate, document_importance, reference_coverage, cite_p,
  claim_coverage`.
- The geomean is computed over all seven and reported alongside the seven
  individual values. It is `None` for any partial subset (can't form the 7-mean)
  and `0.0` if any metric is exactly 0 — expected for a floor model, so **lead
  with the seven individual numbers**, not the geomean, when reporting OLMo.

## Reliability plan (runs crashing)

**Root cause, from the executor code (not speculation).** Generation is a single
~2h `execute_command`. `SandboxExecutor._execute_streaming` polls the swe-rex
runtime every 1s and, after **3 consecutive poll failures**, logs "Sandbox
unresponsive, aborting" and returns a failed result (`executor.py:641-654`). That
3-strikes abort — a container/runtime-level unresponsiveness — is what kills runs
~40 min in, **not** the 8h `timeout_seconds`. Two consequences:
1. The abort calls `kill_process_group()` on the command only; the
   `async with SandboxExecutor` context stays open and the completed per-query
   folders (`{idx}/final_report.md`, `paper.csv`, ...) are still on disk.
2. Our `execute()` bails on `gen_result.success == False` and returns before
   copy-back or eval, so completed queries are discarded by *our wrapper*, not
   destroyed by the sandbox.

### Step 0 — 1-hr interactive diagnostic (decides Step 2 vs 3)

Goal: reproduce the ~40-min stall and see *which* layer dies. Don't hand-launch
vLLM — let `run-external` launch it exactly as batch does (faithful config), and
just find its port to poke it. All values below are code-derived (job_assembler /
launch constants / vllm_server_utils); the `beaker session create` flags are the
only part to adapt to your setup.

**1. Session** (GPU node, the batch sandbox image, secrets as env). NB
`session create` takes `--secret-env VAR=secret-name` (opposite order from
`launch`'s `secret-name:VAR`), and `exec`/`attach` must run on the session host,
so from a laptop use `--remote`:
```
beaker session create \
  --remote --bare --cluster ai2/ceres \
  --budget ai2/oe-other -w ai2/olmo-eval-debug \
  --image ai2-tylerm/olmo-eval-cu1281-trc2100-amd64-sandbox \
  --gpus 1 --shared-memory 10GiB --timeout 3h \
  --secret-env OPENAI_API_KEY=<openai-beaker-secret> \
  --secret-env S2_API_KEY=roryd_S2_API_KEY \
  -- bash
```
Two shells: run `tmux` inside the session (run generation in one pane, the
monitor loop in the other), or background the run to a log and monitor in the
foreground.

**2. Inside: env + source + install** (mirrors BEAKER_INFRA_ENV_VARS; source
normally comes from gantry, so clone the branch by hand):
```
export OLMO_CONTAINER_RUNTIME=podman OLMO_PASTA_HOST_IP=169.254.1.2 \
       OLMO_RESULT_DIR=/results \
       BEAKER_ALLOW_SUBCONTAINERS=1 BEAKER_SKIP_DOCKER_SOCKET=1
git clone https://github.com/allenai/olmo-eval.git && cd olmo-eval
git checkout roryd/deepscholar-bench-s2patch
uv sync && uv pip install -e '.[s3,sandbox,vllm,clients]'
```

**3. Terminal 1 — run the failing configuration** (same args as runs 9-11: S2,
full 63; `allow_partial_generation` so the Step-1 salvage engages):
```
uv run olmo-eval run-external -m allenai/OLMo-3-7B-Instruct -e deepscholar_bench \
  --runtime podman -O /results \
  -K max_model_len=32768 \
  -a search_backend=s2 -a limit=63 -a allow_partial_generation=true
```
This launches vLLM (`--host 0.0.0.0`, a *random* free port) and logs it to
`/results/logs/vllm_server_<port>.log`; then it clones deepscholar, writes the
LOTUS config pointed at `169.254.1.2:<port>/v1`, and starts generation.

**4. Terminal 2 (`beaker session exec` / tmux) — monitor until it stalls:**
```
PORT=$(ls /results/logs/ | sed -n 's/vllm_server_\([0-9]*\)\.log/\1/p' | head -1)
while true; do date
  curl -s -m 5 -o /dev/null -w "vllm/models HTTP:%{http_code} t:%{time_total}s\n" \
    http://localhost:$PORT/v1/models
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  free -m | awk '/Mem:/{print "mem MB:",$3"/"$2}'; echo ---; sleep 30; done
```

**5. When generation goes silent (watch the gen log), take three readings:**
- **Real completion** (not just cheap `/models`):
  `curl -s -m 30 http://localhost:$PORT/v1/completions -H 'content-type: application/json' -d '{"model":"allenai/OLMo-3-7B-Instruct","prompt":"hello","max_tokens":16}'`
- **Sandbox container liveness:**
  `podman ps` → `podman exec <id> bash -c 'echo alive; curl -s -m5 -o/dev/null -w "%{http_code}\n" http://169.254.1.2:'$PORT'/v1/models'`
- **GPU state** from the terminal-2 loop at that timestamp.

**Decision:**
- `/models` fast **and** completion hangs, GPU pegged 100% → **vLLM inference
  wedged** = case (a) → Step 3 (fresh vLLM per chunk) or a vLLM-level fix.
- `/models` **and** completion both fine, but `podman exec` hangs → **container /
  swe-rex runtime wedged** (explains the 3-poll "Sandbox unresponsive" abort) =
  case (b) → Step 2 (in-sandbox chunking) is enough.
- Both host-side vLLM calls fine **and** `podman exec` fine, only the generation
  process is stuck → a single hung in-flight request (litellm ignoring
  `lm_timeout`) → Step 4 fixes it directly; Step 2 also sidesteps it.

### Step 1 — Salvage (cheap, do regardless of Step 0)

Make `execute()` never throw away completed work:
- On generation failure/timeout, still `_copy_back()` the completed folders.
- If `allow_partial_generation`, run eval over whatever completed and report the
  seven metrics + geomean with `queries_succeeded/total` in metadata (already
  plumbed). Without the flag, still copy back for inspection.
This alone turns "lose everything at 40 min" into "keep the ~24 completed."

### Step 2 — In-sandbox chunking (IMPLEMENTED 2026-07-09)

Replace the single generation command with a loop over disjoint index ranges
(chunk ~8-10 queries ≈ ~20 min, inside the proven-reliable window), each a
separate `execute_command`. Catch a per-chunk timeout, log it, advance to the
next range; completed folders accumulate on disk. Then run eval once over the
union and report. Keeps every command inside the window where 10-query runs are
reliable and never puts a 2h process behind one watchdog. No upstream resume
needed — ranges are disjoint (`--start-idx/--end-idx`); worst case we lose the
one chunk that stalls (optionally retry its remaining indices once).

Implementation (`eval.py`, `args.py`):
- New args: `chunk_size` (default 10; 0/none disables → old single-command path),
  `chunk_timeout` (default 1800s, kept under the ~40-min wedge threshold),
  `chunk_retries` (default 3, tolerated no-progress chunks before salvage-bail).
- `_run_generation` dispatches: single-command path when chunking is disabled, the
  bound is unknown, or the whole run fits one chunk (limit ≤ chunk_size — so the
  proven 10-query run is byte-identical to before); otherwise the chunked loop.
- The chunked loop is **frontier-driven**: each iteration runs a chunk starting at
  the lowest still-missing index. Completion is tracked by scanning `final_report.md`
  folders (`_completed_indices`), not summary.json (each chunk overwrites it). The
  `completed` set is unioned across chunks so a transient failed scan can't erase
  progress. A chunk whose *head* index produced no report is skipped (its head
  errored, or it stalled) so the loop always advances — provably ≤ n_total
  iterations. A query that merely errors (upstream skips it, chunk still finishes
  neighbors) is distinguished from a true stall (zero new folders): only the latter
  increments `no_progress_chunks`, so a floor model's ~20-30% error rate does NOT
  trigger the give-up path.
- Bail conditions (all salvage what completed, then eval if allow_partial): overall
  `timeout_seconds` budget exhausted; `no_progress_chunks > chunk_retries`; or the
  container fails an `echo alive` liveness probe after a no-progress chunk (the
  nested-podman wedge case — stop launching doomed 30-min commands).
- `_total_query_count` resolves the bound when `limit` is absent by counting dataset
  rows via pandas (queries.csv if present else papers_with_related_works.csv; pandas
  not `wc -l` because CSV cells contain embedded newlines). Confirmed against upstream
  `main.py@c95413b`: absent queries.csv → one query per papers-CSV row, sliced
  `iloc[start:end]`, folders named by original row index → disjoint chunks are safe.
- Verified locally with a fake-executor simulation of the loop (not committed):
  clean run, poisoned-middle query, high error rate (no false bail), mid-run wedge
  (clean break, no spin), limit=None counting, and stall-bail all behave correctly.
  Still needs a real beaker validation run.

### Step 3 — Chunk across jobs (REQUIRED: Step 2 proven insufficient, 2026-07-09)

Run 12 (2026-07-09, S2, full 63, in-sandbox chunking on): the chunked loop worked
mechanically — chunks `[0:10)` and `[10:20)` completed cleanly, the loop rolled over
to `[20:30)` — but generation **still wedged at query 24**, only ~5 queries into that
chunk, on a hung vLLM call in the Filtering step. `lm_timeout=240` did not fire
(litellm streaming path). The 3×300s poll-abort fired, the loop marked the chunk
skipped and tried to continue, but the container stayed unresponsive (a post-abort
`find` scan itself hung ~15 min). **Key finding: the wedge is correlated with
cumulative generation wall-clock (~45 min / ~query 24), not per-chunk time.** Early
chunks are healthy; a *late* chunk dies fast. In-sandbox chunks all share one vLLM +
one sandbox, so restarting the generation *command* does not reset the accumulating
resource pressure. Step 2 is a salvage layer, not a fix.

The fix is fresh isolation per shard: run N beaker jobs, each a disjoint index range
(`-A start_idx=N -A limit=M`) with a fresh sandbox + fresh vLLM, so the pressure that
builds to the ~query-24 wedge never accumulates. ~15 queries/job clears it with margin
(a 10-query chunk succeeded; ~20 reached the wedge). Submit script:
`scratchpad/submit_deepscholar_shards.sh` (4 shards: `[0:16) [16:32) [32:48) [48:63)`).
Do NOT combine the four shard-level `aggregated_results.csv` — see aggregation below.

### Metric aggregation across shards (and the cross-model denominator)

Verified from upstream `eval/evaluator/*.py@c95413b`: all seven metrics use the base
`Evaluator.aggregate` = `round(mean(per-query value), 2)` (no override), and every
`_calculate` scores each query independently (no corpus-level ratio-of-sums, no
cross-query normalization). So the exact single-run number is reproduced by **pooling
the per-query rows** (`<metric>/deepscholar_base.csv`, copied back under
`deepscholar_results/evaluation/`) across all shards and re-applying the mean — never
by averaging the four rounded shard aggregates (that double-rounds and mis-weights
unequal shard sizes). Combine script: `scratchpad/combine_deepscholar_shards.py`.

**Parse-failure semantics (matters for a multi-model sweep).** In `process_mode`, a
query is *dropped* (no row, excluded from the denominator — NOT scored 0) when its
folder fails to parse: (1) `intro.md` missing (generation errored before writing it —
the observed `'intro_section'` case, which still writes `final_report.md`); (2)
`paper.csv` missing/unreadable; (3) `docs` empty, i.e. the section has no citation
matching the `arxiv.org/abs/<id>` markdown-link pattern. Upstream then means over only
the parsed queries, so the **denominator is per-model** — a model that fails more
queries is scored on an easier self-selected subset. That is fine for reproducing the
paper/leaderboard (DeepResearch 0.309) but is NOT comparable across models.

For the 15-model internal sweep, report **two numbers** (the combine script emits both):
- `mean_fixed` = sum / 63, every missing/failed query counted as 0 — a fixed
  denominator, consistent across models → use this for cross-model ranking.
- `mean_parsed` = sum / n_scored (upstream) → paper/leaderboard-comparable secondary.
Lead with `mean_fixed` for internal comparisons; keep `mean_parsed` for the paper check.

### Step 4 — Hard LM-call timeout (IMPLEMENTED 2026-07-09)

Run 13 (2026-07-09, S2, 4 fresh shards): 3/4 completed; the `q17-q32` shard
wedged at query 20 — a *fresh* vLLM/sandbox, only ~4 queries / ~10 min in — on a
hung `sem_agg` ("Aggregating") LM call in the category-summary stage. Confirmed
this is NOT input-driven (abstract lengths for idx 20–24 are ~dataset mean; the
longest abstracts are elsewhere) and NOT cumulative (fresh shard died fast). It is
a per-request stall in the ~q20–24 band (5/5 historical runs die there: 22, 22,
24, 24, 20), where this model reliably provokes it. So fresh-job sharding (Step 3)
shrinks the blast radius but can't save a shard containing a trigger query.

Root cause of the non-firing timeout, from source: LOTUS calls
`litellm.batch_completion` **non-streaming** with a `timeout` that *is* propagated
to the stage LMs (`Configs.initialize_lms` pops only `max_completion_tokens`, keeps
`timeout`), but litellm does not enforce it on the vLLM/`openai/` path — so a stalled
request never aborts. (The earlier "litellm streaming ignores it" note was wrong;
it's non-streaming and litellm just doesn't honor it here.)

Fix (`sandbox_search_shim._patch_lm_hard_timeout`): wrap `lotus.models.LM.__call__`
so each call runs in a worker thread and is abandoned after `lm_timeout` seconds
(hard wall-clock guard, independent of litellm). On timeout it raises `TimeoutError`,
which upstream's per-query handler catches → the query is marked failed → the
pipeline advances. A stall now costs one query instead of the whole run/chunk — and
that is the *desired* scoring behavior: a weak model that provokes stalls is
penalized (failed queries → 0 under the fixed-63 denominator), not a harness hang.
Wired via `DEEPSCHOLAR_LM_HARD_TIMEOUT` env (= `lm_timeout`, default 240s, kept under
the 300s poll interval; worst-case legit call is a stage LM at ~4096 tokens ≪ 240s,
so genuine work is never cut). The abandoned worker isn't killed, but the litellm
`timeout` in kwargs closes that request so vLLM frees the slot; the pool is oversized
so lingering workers never block new calls.

Run 14 (2026-07-10, full 63, LM timeout deployed): still wedged at query 22, but the
LM timeout was NOT the miss — it worked (every LM call completed in ~28s). The hang
moved to a *different silent path*: the S2 search. Query 22's search-query-gen LM
repeatedly produced truncated/invalid JSON (the degenerate whitespace flood at the
4096 cap), so upstream re-searched (3 filter-attempts x 3 steps), bursting S2 calls
until S2 returned 429s. The shim's 429 handler slept **silently** (`time.sleep`, no
log), so ~900s passed with no output -> 3x300s poll-abort -> container wedged
(confirmed: last log line is `Searching [ARXIV] for queries` with no following
`Aggregating`, i.e. the hang is in the search, before any LM call). Generalized
lesson: the wedge is triggered by **any code path silent for >300s**, not specifically
LM calls. Fix: `s2_search_rows` now logs every retry (429 included) and is bounded by
`S2_SEARCH_BUDGET_S` (45s) total, so output keeps flowing (poll counter never
escalates) and a single search can't hang. Between the LM timeout and this, the two
known silent paths (LM calls, S2 search) are both bounded < 300s. Needs a beaker
validation run.

Run 15 (2026-07-10, full 63): wedged at **query 6** (much earlier; run was also
slower). CORRECTION to runs 13-14: the "silence -> poll abort" theory was WRONG.
Re-reading `_execute_streaming`, the poll runs `self._runtime.execute(poll_cmd,
timeout=10)` against the **swe-rex server inside the sandbox**; "Poll timed out
after 300s" means that HTTP call never returned (swerex retries to ~300s then
raises aiohttp TimeoutError -- see traceback), i.e. the **swe-rex runtime itself is
unreachable**. Generation stdout silence does NOT cause this -- a healthy-but-silent
container returns the poll instantly with no new output and resets the counter. So
the abort is the **sandbox container wedging** (nested-podman resource pressure) --
exactly the "container/runtime-level unresponsiveness" in the Root-cause note above,
the infra blocker from the original handoff. Non-deterministic (q22, q20, q6) and
load-correlated (slower run wedged earlier): resource pressure, not a query bug.

Implication: the per-call timeouts (LM, S2) target generation *silence*, which is
NOT the trigger, so they can't fix this (still correct hygiene; keep them). The one
lever our code has over container pressure is **concurrency**: LOTUS fires up to
`max_batch_size` (default **64**) concurrent vLLM requests per sem-op. New
`max_batch_size` arg (propagates to stage LMs) throttles this; try `-a
max_batch_size=8` (or 4). Unverified whether it prevents the wedge -- right
mechanism, but the wedge may be a pure nested-podman bug independent of our load.

**Reliable path that has produced results: short-lived sandboxes = small fresh-job
shards.** Run 13's 16-query shards got 3/4; 8-query shards (8 jobs,
`scratchpad/submit_deepscholar_shards.sh`, now with `max_batch_size=8`) should clear
most, and any shard that still wedges loses only ~8 queries (scored 0 under the
fixed-63 denominator). For the 15-model sweep this is many jobs; a durable fix needs
the nested-podman/swe-rex wedge addressed at the infra level (Tyler).

**Order:** Step 1 (salvage) + Step 2 (in-sandbox chunking) + Step 4 (hard LM
timeout) are all in place and compose: Step 4 stops a single stalled call from
hanging the run (the query just fails), so the chunk loop keeps advancing and a
full run should now complete on its own — a stall costs one query, not the run.
Step 3 (fresh-job sharding) stays useful as extra isolation / parallelism but may
no longer be required once Step 4 is validated. NB the shard submit + combine
scripts currently live in the session scratchpad and should be promoted into the
repo (e.g. `scripts/` or the deepscholar package) before the 15-model sweep.

## Open items to confirm on the first beaker run

1. **Recursive search on vLLM end to end.** Default for local is now `recursive`
   (see above). Confirm the LOTUS sem-ops + keyless ARXIV retrieval drive a vLLM
   OLMo without the agentic OpenAI-client path. If the litellm prefix is still
   wrong, switch with `-a local_model_prefix=hosted_vllm`.
2. **`use_structured_output: true` / `use_responses_model`.** Likely fails on a
   non-OpenAI model (no `response_format`/responses API). If generation errors
   here, pass `-a extra_gen_args=--no-structured-output`.
3. **Eval output schema.** `result_parser.py` is best-effort; once a real run
   writes its files (copied back to `output_dir/deepscholar_results/`), confirm
   metric file names/columns and tighten the parser + geomean key matching.
4. **Query auto-generation.** Confirm generation produces queries when
   `queries.csv` is absent (and whether that itself burns judge/LLM calls);
   otherwise add a query-gen step.
5. **requirements install** under uv/py3.10 (lotus-ai, faiss, litellm) completes.
6. **Pin `DEEPSCHOLAR_REF`** to a commit once a run succeeds (currently `main`;
   Codex verified flags/outputs against `c95413b`).

Note: generation catches per-query exceptions and still exits 0 (summary.json is
a list of `{idx, status: success|error, ...}`), and the eval scores only the
folders that succeeded. So the wrapper is strict by default: it fails unless every
query succeeded (`n_success == n_total`), which also covers the all-failed case.
This stops a partial generation from silently scoring a smaller subset than
requested. `-a allow_partial_generation=true` scores whatever generated (still
fails on zero successes). `queries_succeeded`/`queries_total` land in metadata.

## Validation steps (beaker)

1. Smoke: `... -e deepscholar_bench -a limit=2` against a known-good model; inspect
   the copied-back `generation/summary.json` and `evaluation/` files + raw_output.
2. Fix items 1-4 against real output; tighten the parser.
3. Full run; sanity-check geomean against the paper's ~31% ceiling.
4. Pin the commit; only then wire into `science:research`.

## Expectations

Track, not hillclimb. An early OLMo may fail to drive the LOTUS pipeline at all
(errors / empty output, not just low scores). Not wired into any suite yet.
