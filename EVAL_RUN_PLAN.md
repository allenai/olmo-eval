# Evaluation Run Plan

Last updated: 2026-07-31

## Objective

Run the following evaluations across the target model set while preserving the
upstream benchmark semantics and detecting model/provider failures before full
GPU and API spend:

- ExpertQA (`expertqa`)
- LitSearch open (`litsearch`)
- LitSearch rerank (`litsearch_rerank`)
- SAGE open (`sage_open_ended`)
- SAGE short (`sage_short_form`)
- DeepScholar-Bench (`deepscholar_bench`)
- IFEval OOD (`ifeval_ood`)
- MMLU (`mmlu`)
- MATH-500 (`math500`)

## Current full-versus-dev policy (2026-07-29)

Run the relatively quick non-agentic benchmarks in full and use fixed dev sets
for the expensive agentic benchmarks. Dev scores are for model iteration and
paired comparison only; label them with their sample size and do not mix them
with historical full-benchmark scores.

The runtime estimates below are medians from the completed Beaker jobs in the
results sheet. Agentic medians include only runs using the intended agentic
harness; default/non-agentic ExpertQA or LitSearch runs are excluded. Dev
runtime is inferred by scaling the median full runtime by the instance count,
so it is a planning estimate rather than a service-level guarantee.

| Evaluation | Median full runtime | Full instances | Standard scope | Inferred runtime |
| --- | ---: | ---: | ---: | ---: |
| MMLU | 24m (8 runs) | 14,042 | Full 14,042 | 24m |
| IFEval | 1h (8 runs) | 300 | Full 300 | 1h |
| MATH-500 | 1h (8 runs) | 500 | Full 500 | 1h |
| LitSearch-rerank | 1h (8 runs) | 597 | Full 597 | 1h |
| DeepScholar-Bench | 2h 59m (6 runs) | 63 | Dev 10 | 31m |
| ExpertQA agentic | 9h 03m (3 runs) | 2,176 | Dev 100 | 25m |
| LitSearch-open agentic | 8h 19m (3 runs) | 597 | Dev 50 | 42m |
| SAGE-open | 8h 19m (3 runs) | 600 | Dev 50 | 42m |
| SAGE-short | 8h 19m (3 runs) | 599 | Dev 50 | 42m |

### Chat-protocol corrections (2026-07-31)

Three completed scores were initially excluded from numerical analysis because
their protocols were invalid:

- GPT-OSS MMLU 0.4791 (`01KYMNWFQCJ7DQYVGHJY645T97`): raw completion
  loglikelihood bypassed the required Harmony chat format.
- Gemma4 26B-A4B MMLU 0.5566 (`01KY84PXVTJ56NXRFHGCS4K709`): raw completion
  loglikelihood bypassed Gemma's chat/reasoning format.
- Gemma4 26B-A4B MATH-500 0.2940 (`01KY7ZC75EVGGAAQ9X09CHD5B9`): no chat
  template, and 235/500 outputs hit the 1,024-token ceiling.

The three superseded values are retained here for provenance but were replaced
in `scripts/analysis/data/results.csv` on 2026-07-31 by their audited
protocol-correct full runs. The GPT-OSS replacement is explicitly documented
as conservative because some full-set responses exhausted its output budget.

Commit `b5f4126` adds an opt-in five-shot generative `mmlu:chat` suite without
changing the existing MMLU task or provider behavior. The launcher selects it
only for Gemma4 26B-A4B and GPT-OSS; Gemma MATH-500 is chat-wrapped and no
longer uses the legacy double-newline stop. These protocol canaries were
submitted from that commit:

| Model/eval | Scope | Experiment |
| --- | ---: | --- |
| Gemma4 26B-A4B / MATH-500 | fixed seed-42 dev40 | `01KYWHEHGQYRRV8XZ6NT58MWHF` |
| Gemma4 26B-A4B / MMLU chat | 5 per subject, 285 total | `01KYWHEHHC0CK7MHN0KD16EWX9` |
| GPT-OSS-20b / MMLU chat | 5 per subject, 285 total | `01KYWHEG42NRMRYRP8X3SKA1S8` |

The Gemma canaries passed their protocol audits. MATH-500 had 40/40 nonempty,
extractable, correct outputs with no output caps or server errors (the same
sample scored 15/40 under the superseded profile). MMLU produced 284/285 valid
A-D answers, scored 260/285, had no output caps or server errors, and its only
invalid item is a malformed source example with missing numbered statements.
Both used `/chat/completions` with chat messages and are promoted below.

The first GPT-OSS MMLU canary scored 247/285, but five responses exhausted the
4,096-token completion budget and were empty after reasoning extraction. Its
8,192-token medium-effort replacement (`01KYWKMF3G3SY0W8R6BKRDEGE3`) still
produced two capped empty outputs, so neither canary is eligible for the
analysis ledger. The paired low-effort canary
(`01KYWN9VCDT9F6GASDJ4VN4MVQ`) passed: 285/285 nonempty and extracted, zero
8,192-token caps, maximum output length 5,929, no harness/server errors, and
251/285 correct (suite aggregate 0.880464). Its only non-A-D extraction was a
substantive but wrong `None of the above` response, not a parser or transport
failure. The launcher now defaults GPT-OSS MMLU to native Harmony
`reasoning_effort=low`, 8,192 output tokens, and a 16,384-token context.

| Model/eval | Promoted/retry scope | Experiment | Status |
| --- | ---: | --- | --- |
| Gemma4 26B-A4B / MATH-500 | full 500 | `01KYWKMD3XD5PSC103N9PGN76V` | complete: 0.952; audited valid |
| Gemma4 26B-A4B / MMLU chat | full 14,042 | `01KYWKMEHGAFR1NAF7BN359377` | complete: 0.899077; audited valid |
| GPT-OSS-20b / MMLU chat, medium | retry: 5 per subject, 285 total | `01KYWKMF3G3SY0W8R6BKRDEGE3` | rejected: two capped empties |
| GPT-OSS-20b / MMLU chat, low | retry: 5 per subject, 285 total | `01KYWN9VCDT9F6GASDJ4VN4MVQ` | accepted: 0.880464, zero caps |
| GPT-OSS-20b / MMLU chat, low | full 14,042 | `01KYWP4WKTAJCHB011TCBB6ZJV` | complete: 0.850694; audited valid with truncation caveat |

All protocol-correction launches use commit `b5f4126`. Canary scores remain
validation-only; only the audited full-run metrics were imported. The GPT-OSS
full run completed 14,042/14,042 requests with no harness/server errors or raw
Harmony leakage. It produced 13,877 nonempty finals and 13,853 valid A-D
extractions. All 165 empty finals corresponded exactly to the 8,192-token
ceiling; 92 occurred in professional law. The remaining 24 non-A-D outputs
were non-capped: 13 bare-letter answers missed by the strict extractor, five
substantive `None of the above` answers, three refusals, and three other
nonconforming responses. The recorded 0.850694 macro is conservative: applying
the non-capped accuracy within each subject projects 0.856141, while treating
every capped response as correct gives an upper bound of 0.857541. The exact
official score is retained and marked valid with this limitation, consistent
with the existing treatment of the conservative GPT-OSS MATH-500 result.

SAGE-open and SAGE-short run in one Beaker job, so their combined dev50 wall
time should remain about 42 minutes rather than the sum of both rows. ExpertQA
dev100 is about 4.6% of the full dataset and should cost roughly $4-5 when a
full run costs about $100.

### Fixed sample contract

- Standard tasks use `limit=N, seed=42`. The async runner implements this as
  `random.Random(seed).sample(instances, limit)`, not as the first N rows.
  Keep the same code and dataset revision across models so the sampled document
  IDs remain identical.
- The seed-42 dev50 samples avoid the ordering bias in the raw prefixes. The
  LitSearch sample covers all four source groups; the SAGE samples cover
  computer science, healthcare, humanities, and natural science. By contrast,
  the first 50 raw SAGE rows are all computer science.
- DeepScholar's external interface only supports `start_idx` plus `limit`.
  Keep the simple fixed slice at indices 0-9 for dev10 across every model. If
  dev10 results appear unstable, add a committed question-ID manifest before
  changing the slice.
- Use the reporting names `ExpertQA-dev100`, `LitSearch-open-dev50`,
  `SAGE-open-dev50`, `SAGE-short-dev50`, and `DeepScholar-Bench-dev10`.
  The launcher also includes `devN` in Beaker experiment names.
- Dev50 is deliberately coarse for low-scoring SAGE models. Expand to a fixed
  dev100 only when two models are close enough that dev50 cannot separate them.

### Standard launch

The dev launcher submits six non-following jobs for one model: full Core, full
MMLU, and one job for each expensive harness group at the standard dev size.

```bash
# Inspect all commands first; this does not submit anything.
scripts/beaker/launch_dev_evals.sh \
  --model openai/gpt-oss-20b \
  --slug gpt-oss-20b \
  --print-only

# Full non-agentic jobs only.
scripts/beaker/launch_dev_evals.sh \
  --model openai/gpt-oss-20b \
  --slug gpt-oss-20b \
  --only small

# Expensive dev jobs only.
scripts/beaker/launch_dev_evals.sh \
  --model openai/gpt-oss-20b \
  --slug gpt-oss-20b \
  --only large
```

Canary a model's agent/tool-call profile before its first `--only large`
submission. The current next-round order is:

1. Add and canary the chat-formatted MATH-500 variant on Gemma4 26B-A4B and
   GPT-OSS, then run all 500 examples once validated.
2. Canary LitSearch-open, then run dev50 for GPT-OSS, Nemotron, GLM, OLMo
   Instruct, and OLMo Think.
3. Fill DeepScholar-Bench-dev10 for GLM and OLMo Think.
4. Fill the remaining ExpertQA-dev100 and SAGE dev50 cells. Do the costlier
   SAGE/ExpertQA work only after the cheaper harness checks pass.

### LitSearch-open dev50 rollout (2026-07-29)

The initial dev3 canary audit checked both aggregate metrics and raw trajectory
tool turns. Nemotron and OLMo Instruct passed. The following fixed seed-42
dev50 runs were launched, together with the previously validated Gemma and
Qwen profiles. The first Gemma launch exposed a launcher regression and was
replaced as noted below:

| Model | Dev50 experiment | Found rate | Gold recall |
| --- | --- | ---: | ---: |
| Gemma4 26B-A4B | `01KYR2GD0P7ZN4Y4VKQZNNDKMA` (parser-fixed replacement) | 0.28 | 0.28 |
| Qwen3.5 9B | `01KYR1Y6S6DWSW0VXVYQGAPMST` | 0.36 | 0.35 |
| Qwen3.5 35B-A3B | `01KYR1YE42G7WPDCD25PDM2TE9` | 0.46 | 0.45 |
| Nemotron 3 Nano 30B-A3B | `01KYR1X9M7JVC80R2S2TA71Q6V` | 0.36 | 0.35 |
| OLMo-3 7B Instruct | `01KYR1X9MM1G288B3ZNZTBVP8K` | 0.16 | 0.16 |

All five completed 50/50 with no task errors. The 250-prediction trace audit
found tool turns on every instance, no empty outputs, and no OOM, request,
parser, Semantic Scholar, or traceback failures. Nemotron instance 44 exhausted
the ten-turn agent budget and ended with an unexecuted eleventh tool call; this
is retained as a legitimate scored-zero model/horizon failure.

Do not log Gemma experiment `01KYR1X9KDNF57QVP96XSMSYJH`: although it
completed 50/50 with no task errors, all 50 native `<|tool_call>` outputs were
left as literal assistant text and no tool turns ran, producing an invalid
0.00. The launcher again defaults Gemma to `reasoning_parser=gemma4` and
`tool_call_parser=gemma4`, matching the prior successful full agentic run.

GPT-OSS initially failed 3/3 requests because automatic parser detection
  selected Hermes. The launcher now defaults GPT-OSS tool use to the `openai`
  parser. Recanary `01KYR1X9K33ZWW1M8MAW35T9FM` completed 3/3 and scored 2/3,
  but the remaining instance emitted the malformed tool name
  `semantic_scholar_snippet_search<|channel|>commentary`. The serving profile
  now also sets `reasoning_parser=openai_gptoss` so Harmony channel markers are
  parsed separately from tool names. Corrected dev3 canary experiment
  `01KYR3EM5GG9MSBE2SD03NTKYM` (job `01KYR3EM90MK44Z3971E5PEQAA`)
  completed 3/3 at 0.667. All three traces used tools, with no empty output,
  raw markup, or tool errors. LitSearch dev50 experiment
  `01KYR4TXZDWE1CEDD0EJDBYB65` (job `01KYR4TY356E13JAVR2281QP9R`)
  was subsequently submitted, but the larger run showed that the three-item
  canary was not sufficient: 8/50 outputs were explicit invalid-tool errors
  whose parsed tool names retained `<|channel|>` suffixes, and one additional
  output ended at the turn cap without a tool call. The apparent 0.32 found
  rate / 0.31 gold recall is therefore invalid and must not be logged. The
  contemporaneous SAGE canary also had the same invalid-tool signature on one
  of three open-ended examples. vLLM 0.19.1 logs additionally contain Harmony
  parsing exceptions and malformed-JSON tool-call errors. Keep GPT-OSS gated
  for every tool-dependent harness until a Harmony-capable parser fix passes a
  larger canary; the current `openai` plus `openai_gptoss` pair is necessary
  but not reliable enough for benchmark runs.

GPT-OSS Harmony recovery resumed on 2026-07-30. Commits `f3a3834`, `2e97d8c`,
`54e087d`, and `db667bf` route the agent loop through the Responses API,
disable hosted compaction, backport vLLM's channel-independent function-call
classification, make the forced-final pass compatible with vLLM 0.19.1, and
schema-bound normalization of leaked Harmony suffixes. LitSearch canary
`01KYT9C3QQ4HYKEACV01D842RV` (job `01KYT9C3WWRXJRCEHQ5TRHD9AH`, result
`01KYT9C3R1D9Y2NS12SHH16NBW`) completed 20/20 with no task errors and scored
0.20 found rate / 0.20 gold recall. Its 125 function calls all used the exact
`semantic_scholar_snippet_search` name, all had matched results, and none were
tool errors or MCP calls. One example made four successful searches but
returned an empty final answer; retain that as a model-output caveat rather
than a parser failure. The server transparently retried three Harmony header
500s. The first SAGE canary `01KYTA242QVHCXR8DZRGFJXQDE` completed 40/40,
but five examples invented unsupported tool aliases and two returned empty
final answers, so it did not pass promotion. The first ExpertQA canary
`01KYTA2GT35C655844GEVA1C7D` completed 20/20; all 247 calls used its two exact
tools and had matched results, but two answers were empty. Commit `0e30b43`
adds a bounded low-reasoning, no-tool final pass for GPT-OSS empty completions,
and the local SAGE launcher now explicitly enumerates its three exact tool
names. Corrected SAGE canary `01KYTANX9WHXYV14AN7YWT1BS8` (job
`01KYTANXD8KHA7T5CV5NT3MHRX`) completed 40/40 with every output nonempty, 585
matched calls/results, only the three configured tools, and no tool errors,
fallback strings, MCP calls, or orphaned IDs. Its dev20 scores were 0.40 short
exact match and 0.12283730158730159 open weighted recall. Seven Harmony header
500s were transparently retried and did not leak into scored trajectories.
Corrected ExpertQA canary `01KYTAP863JRTSM9BBKM5YYPN1` (job
`01KYTAP89HT3VRRMK1RG13GK5V`) completed 20/20 with every output nonempty, 268
matched calls/results, only its two configured tools, and no tool errors,
fallback strings, MCP calls, or orphaned IDs. Its dev20 citation recall was
0.08243986190967614 and global average was 0.26595775249671016. Two Harmony
header 500s were transparently retried without trajectory leakage. Both
canaries pass the structural promotion gate.

The first attempted promotions were submitted incorrectly with
`launch_safe_evals.sh` and no `--limit`: LitSearch
`01KYTB5ZSB7CNWNJVV8KN7WTHP` targets all 597 instances, and SAGE
`01KYTB6BG85R3MFXXTHEMDJWYW` targets all 1,199 combined instances. These are
full-dataset runs, not the fixed seed-42 dev50 contract, and must not be logged
as dev results. Both were manually canceled on 2026-07-30 and finalized with
exit 143. Correct fixed-sample replacements were submitted from CI-green commit
`b808dd7`: LitSearch-dev50 `01KYTCPQ8M1FNHYD59EBVB6DG6`, SAGE-dev50
`01KYTCQ6N53FY2K1WQMH7W7N3E`, and ExpertQA-dev100
`01KYTCQMJJFVF3C9ZRM136WFCX`. All use seed 42 and group
`safe-evals-gpt-oss-20b-gptoss-responses-valid-dev2-20260730`. Audit their raw
trajectories before logging metrics.

LitSearch-dev50 finalized successfully and passed trajectory audit: 50/50
nonempty outputs, 328 matched calls/results, only
`semantic_scholar_snippet_search`, and no tool errors, fallback strings, MCP
calls, or orphaned IDs. Three Harmony header 500s were transparently retried.
Its 0.34 found rate and 0.33 gold recall were logged to `results.csv`.

SAGE-dev50 also finalized successfully and passed trajectory audit: 100/100
nonempty outputs, 1,364 matched calls/results, only the three configured tools,
and no tool errors, fallback strings, MCP calls, or orphaned IDs. Eighteen
Harmony header 500s were transparently retried. Its 0.09463492063492064 open
weighted recall and 0.32 short exact match were logged to `results.csv`.
ExpertQA-dev100 finalized successfully and passed trajectory audit: 100/100
nonempty outputs, 1,351 matched calls/results, only
`serper_fetch_webpage_content` and `serper_google_webpage_search`, and no tool
errors, fallback strings, MCP calls, or orphaned IDs. Eleven Harmony header
500s were transparently retried. Its 0.14824615300158778 citation recall,
0.2975378392440893 citation precision, 0.5338730158730158 answer precision,
0.4833761353026059 snippet grounding, and 0.32655233603956435 global average
were logged to `results.csv`.

Analysis-ledger correction on 2026-07-30: OLMo-3 7B Think experiment
`01KXHFR26TBT2G928EJKSPFW0W` reported LitSearch-open `found_rate`, not
LitSearch-rerank `recall@5`; its 0.00 was removed from the rerank row. Gemma4
26B-A4B MMLU `01KY84PXVTJ56NXRFHGCS4K709` is a clean, fully validated
14,042/14,042 retry and should not be flagged merely because its 0.5566 score
is lower than OLMo Instruct. Remaining non-agentic gaps are Core
(LitSearch-rerank, IFEval, MATH-500) and the separately launched MMLU for OLMo
Think and GLM. Neither is intrinsically unsupported: canary the current
reasoning-aware generation profile for Core and the logprob-safe profile for
MMLU before promoting each to a full run.

The remaining sentinel/base gaps were launched on 2026-07-30 from CI-green
commit `b808dd7`. OLMo Think Core is `01KYTGM7XAW9TDS9X8S9A6XGFS`; GLM Core is
`01KYTGM8JWY3WFQWVBNB9BGQ1V`. Both cover full LitSearch-rerank, IFEval, and
MATH-500. The TP=2, 4,096-context, 2,048-prefill-token MMLU canaries completed
285/285 with no logprob failures: OLMo Think `01KYTGMAKMDXHTK202W221BKGM`
scored 0.6263 and GLM `01KYTGM643898B7WJDTHXQ43KY` scored 0.7447. Their full
14,042-instance promotions are `01KYTGVCXRD27HTR8122PR02AD` and
`01KYTGW0H17H3G443TGPNERQFF`, respectively. Audit output completeness and log
the four full experiments when they finalize.

All four gap-filling experiments finalized successfully and were integrated
into `scripts/analysis/data/results.csv` on 2026-07-30. Both Core runs covered
all 1,397 expected instances with `errors: []`: OLMo Think scored 0.4260748185
on LitSearch-rerank recall@5, 0.2933333333 on IFEval prompt-level loose
accuracy, and 0.558 on MATH-500; GLM scored 0.4749860413, 0.2466666667, and
0.58 respectively. Both MMLU runs covered all 57 subjects and 14,042
instances with `errors: []`: OLMo Think scored 0.6417625433 and GLM scored
0.7620444844. The ledger importer added 138 scalar metric rows, including
task-level and MMLU category submetrics.

FrontierScience integration on 2026-07-30 retains all three benchmark metrics:
100-question Olympiad `accuracy`, plus 60-question open-ended Research
`success_rate` and the more sensitive partial-credit `rubric_score`. Seven jobs
were verified at exit 0 with 100/100 and 60/60 instances and no harness errors:
OLMo Instruct `01KYR3HKPAG6H92ZTCS3NPJAKE`, OLMo Think
`01KYR3HQ3PV64NPH69F461MNWK`, GLM `01KYR3HTBFEFDF1VPCQETBVV6Q`, Gemma4 26B
`01KYR3JYFB74J6GRFZ1BVJ8CT6`, Nemotron `01KYR3K2HW1MD13SPFYYV8KJJ7`, Qwen3.5
35B-A3B `01KYT9S9SAG57CT25DKD0TCNHJ`, and GPT-OSS
`01KYT9SGKA6NE6ZBJHJEXSB47T`. GPT-OSS's three values remain usable reported
lower bounds: 10/160 responses had no visible output after reasoning
truncation. Qwen3.5 9B remains pending and should fill the three intentionally
blank FrontierScience cells when job `01KYT9RZW9XD7MNZX6ABYC2N0M` (experiment
`01KYT9RZRMVEPRV1MXY7G0XW2T`) finalizes and passes the same checks.

GLM remains gated and must not receive tool-dependent dev runs. Recanary
`01KYR1X9XYRAZDA0DSSXPJFTF4` completed 3/3 with zero score and zero tool turns,
and requiring tool choice did not change the behavior. Inspection of the
checkpoint's official chat template confirmed that it renders only system,
user, and assistant messages; it has no tools or tool-result branch. These
zeros are a harness incompatibility, not valid benchmark scores. DeepScholar
does not use this tool harness and remains in scope.

OLMo Think's corresponding zero-tool canary `01KYR1ZER8F5HN83GJG288T75M`
was traced to its stock legacy `functions` template, while the harness sends
OpenAI-style `tools`. The launcher now enables the repository's bundled OLMo3
tool template and parser patch. A first repaired LitSearch canary
`01KYRPK2T33XBZ2BYMWJWE16K1` proved that the template reached the model, but
the Think model's leading `<think>` block prevented the tool parser from seeing
the following `<function_calls>` block; it again had zero parsed tool turns.
The matching SAGE and ExpertQA jobs were canceled once this was confirmed.
The profile now also sets `reasoning_parser=olmo3`, as required to separate the
reasoning and content before tool parsing. Second fixed dev3 recanaries passed
raw review for LitSearch (`01KYRPZS9555D9JDP390PQ6R3N`, job
`01KYRPZSCJGZ7M24VEJN4B7YJ6`) and SAGE (`01KYRQ05A58W0GRXHQ1NNXHXDW`, job
`01KYRQ05DWBVEY9V081EQRCC4W`). LitSearch used tools on all three examples;
SAGE made valid parsed calls on three examples and exposed explicit model-side
invalid-tool behavior on the others, rather than parser corruption. They were
promoted to fixed dev50 as LitSearch `01KYRQJ94E955QK4MSJBC1VDRN` (job
`01KYRQJ988YCA0A5QMJVZ2C167`) and SAGE `01KYRQRT3J39NCT4Q3QDW059DQ` (job
`01KYRQRT704562FRFCRQDN42ZT`). The LitSearch promotion completed 50/50, but
raw-log review found the same escaped-apostrophe sanitizer defect later fixed
by commit `106ef62`; its metrics are invalid and must not be logged. Replacement
dev50 `01KYRSNZPQJ2D1ZK227PXH8RKE` completed 50/50 but is also invalid: 15
official wrappers leaked because of the multiline parser defect and one
trajectory exceeded the 32,768-token context. The `1f1a6a9` dev3 recanary
`01KYRV11H3MVN1GE1NF8ESCZQ6` used tools on 3/3 with nine matched calls/results
and no tool errors. Its sole raw final contains two separate wrapper blocks and
is retained as malformed model behavior. Replacement dev50
`01KYRVFCN5MKFSVWSXGQG67F5N` uses the corrected parser and the checkpoint's
supported 65,536-token context. It completed 50/50 with no empty outputs,
447 matched tool calls/results, no tool-result or context errors, and a 0.14
found rate / 0.14 gold recall; both metrics were logged. Forty-three examples
used parsed tools. Four of the seven zero-tool examples mixed official call
wrappers with fabricated final prose in the same response, which vLLM left as
model output. Fourteen finals contained wrappers, but all except one were
mixed-output or terminal-horizon model behavior. The sole parser exception was
an extreme malformed response containing hundreds of calls in one list; the
escaped-apostrophe and multiline parser failures did not recur. The first
ExpertQA recanary
(`01KYRQ0AXVM0S2XCJSCK8Z9VB9`, job `01KYRQ0B1XQVP6VG4ENFMP3VAQ`) still
skipped tools. A required-tool attempt was canceled before starting in favor
of the stronger explicit-search system prompt now being tested by
`01KYRQN75TNCR9NQPTT51FE4MW` (job `01KYRQN7BADHPFZCFY5EQTBC5A`).
That final canary also made zero tool calls on 3/3 and fabricated citations,
so OLMo Think ExpertQA is unsupported under the validated serving profile and
must not be logged or replicated.

### SAGE dev rollout (2026-07-29)

The canonical SAGE profile uses `dr_tulu_crawl4ai`, seed 42, a 65,536-token
context, and the S2 and Serper secrets. The three profiles with prior full SAGE
runs received combined open/short dev50 calibration jobs:

| Model | Experiment | Job |
| --- | --- | --- |
| Gemma4 26B-A4B | `01KYR4T30JWDSB6E33VW61G2NS` | `01KYR4T347BDN30CC1YXNHYERC` |
| Qwen3.5 9B | `01KYR4T8MRV9GSV7MH9GBX8B5T` | `01KYR4T8SDC0PVBP8AJW0KRWTW` |
| Qwen3.5 35B-A3B | `01KYR4TEACF1MJBPSC19ZEV13R` | `01KYR4TEDTB7DMS8B2AH2K3ADS` |

New SAGE profiles received combined open/short dev3 canaries:

| Model | Experiment | Job |
| --- | --- | --- |
| Nemotron 3 Nano 30B-A3B | `01KYR4TKPKXQMMWWH03Y9MF5N8` | `01KYR4TKT4D52NTNGTK0V1F13X` |
| OLMo-3 7B Instruct | `01KYR4TRW4X2GX888HV0H4VBFA` | `01KYR4TRZCJHZQMGCA1C8ZZR9Y` |
| GPT-OSS-20B | `01KYR4V2TXS3BP8EJ1384419P1` | `01KYR4V2YGR9GRAZF4WPHG0SPW` |

All three original dev50 jobs completed 100/100 with no evaluator or tool
errors, no empty outputs, and no raw tool markup. Their fixed seed-42 scores
were:

| Model | SAGE-open weighted recall | SAGE-short exact match |
| --- | ---: | ---: |
| Gemma4 26B-A4B | 0.049640415140415145 | 0.42 |
| Qwen3.5 9B | 0.08639682539682539 | 0.54 |
| Qwen3.5 35B-A3B | 0.09883760683760684 | 0.76 |

Nemotron's canary used tools on all six examples with no tool errors or parser
leakage. OLMo Instruct used tools on all three open examples and one of three
short examples; the two direct short answers are well-formed model behavior,
not a harness failure. Both were promoted to fixed dev50 on 2026-07-29:

| Model | Experiment | Job |
| --- | --- | --- |
| Nemotron 3 Nano 30B-A3B | `01KYRPBFDFWMJKYMZCY24W1SRV` | `01KYRPBFHJK3VNWV0GAEVSEMT8` |
| OLMo-3 7B Instruct | `01KYRPDY7HKZE87BQ3E1NG6V1R` | `01KYRPDYAWNZA83PM7XRA6E22X` |

Nemotron completed 100/100 with tools on every example, 1,080 matched
calls/results, no tool-result errors, and scores of 0.06628083028083027 open
weighted recall and 0.42 short exact match; both metrics were logged. Fourteen
raw final calls occurred only after the full 20-call agent horizon. One other
final call used a malformed `<parameter=query=...>` tag and produced the sole
parser exception; because the same run parsed the model's other 1,080 calls,
this is retained as a legitimate model-side scored-zero failure.

The launcher retains the validated non-GPT profiles: Gemma uses both `gemma4`
parsers; Qwen SAGE uses TP=2, Triton GDN prefill, raw thinking,
`qwen3_coder`, a 600-second request timeout, and 8,192 SAGE-short output
tokens. GPT-OSS passed the larger Harmony-recovery gates described above and
its fixed dev promotions are now in progress.

### Additional dev coverage wave (2026-07-29)

The following fixed ExpertQA dev100 jobs were launched only for profiles with
previously validated full ExpertQA runs:

| Model | Experiment | Job |
| --- | --- | --- |
| Gemma4 26B-A4B | `01KYRPBFD93A6PVZCZP8VC952W` | `01KYRPBFGXV0H322GS06N7ATKP` |
| Qwen3.5 9B | `01KYRPBGB8PKGG09AMAW7XVW1W` | `01KYRPBGFE80ZJV6DBR1WF3A4F` |
| Qwen3.5 35B-A3B | `01KYRPBFEMED6DDB0D0D06KFC1` | `01KYRPBFQ0687BHM69TJHEP7D7` |

Gemma completed 100/100 with no evaluation or tool errors, tool use on every
example, and no parser leakage. Four examples legitimately ended after their
last tool result at the agent horizon and scored zero. Its fixed-seed metrics
were citation recall 0.4836235146432515, citation precision
0.6338224275724276, answer precision 0.9183333333333333, snippet grounding
0.7751593684093685, and global average 0.6785930918496708; all five were logged
to `results.csv`.

Both Qwen jobs also completed 100/100 with no evaluation/tool errors, empty
answers, or parser leakage. Qwen3.5 9B used tools on 93 examples (1,693 matched
calls/results); its seven direct answers are valid model-side failures because
tool use works throughout the remainder of the same run. Qwen3.5 35B-A3B used
tools on all 100 examples (1,411 matched calls/results). Their fixed-seed
metrics were:

| Model | Citation recall | Citation precision | Answer precision | Grounding | Global average |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5 9B | 0.2476049519864783 | 0.4151819010755676 | 0.707074768368886 | 0.5853652926993057 | 0.4566205404769773 |
| Qwen3.5 35B-A3B | 0.36550173213715753 | 0.47306394094737697 | 0.94758658008658 | 0.8042919727394271 | 0.5953840843903715 |

All ten Qwen metric rows were logged to `results.csv`.

New ExpertQA profiles received fixed seed-42 dev3 canaries and must not be
promoted until their raw citation/tool trajectories pass review:

| Model | Experiment | Job |
| --- | --- | --- |
| Nemotron 3 Nano 30B-A3B | `01KYRPBGSS52FTBXXY0CQRT4D4` | `01KYRPBGXNC8J744F8BWAYKHGR` |
| OLMo-3 7B Instruct | `01KYRPBHBFE9VVDNZMVXW126VG` | `01KYRPBHFCXZND7PJGTN6HWS0X` |

The first OLMo Instruct ExpertQA canary completed, but all three examples made
zero tool calls and fabricated citations directly, so its apparent metrics are
invalid. A repository-supported OLMo3-template recanary
`01KYRPS7ZA658N8QV40CH3W183` (job `01KYRPS82ZNT9KNN4FSY17RRV3`) was
structurally clean but still skipped tools on 3/3. The launcher now sets
`tool_choice=required` at the ExpertQA harness level for this profile, but
canary `01KYRQ88H1PHN39RN81FGSDRGX` (job
`01KYRQ88TH9N6FEDDEP3ERSF18`) also skipped tools on 3/3. A final recanary adds
an explicit system instruction that the first response must search:
`01KYRQN1N6KCMJRC37FC1PJGWS` (job `01KYRQN1SE8QG6T0XWE5S9588V`). If it
also makes no tool calls, mark this model/ExpertQA cell unsupported rather than
recording fabricated-citation scores. It did make zero tool calls on 3/3 and
fabricated citations, so OLMo Instruct ExpertQA is unsupported under the
validated profile and must not be logged or replicated.

Nemotron's first ExpertQA canary used tools on every example and had no tool
errors, but leaked raw `<think>` traces into all three structured answers. The
official serving recipe requires a separate `nano_v3` reasoning-parser plugin;
because that plugin is absent from the pinned serving environment, the
launcher now uses the checkpoint's supported `enable_thinking=false` mode for
ExpertQA and explicitly retains `qwen3_coder` tool parsing. Recanary
`01KYRPWWHESSN2GQF6VAJEYA90` (job `01KYRPWWPEQVJ1N7B752KCNQE0`) used tools on
all three examples without tool errors, but still leaked a residual reasoning
block into every scored answer. A final recanary adds the compatible
string-space `olmo3` reasoning parser: `01KYRQETN2NZV2JMW9KW5GS1J7` (job
`01KYRQETRZY82FDP6E9BENJCDS`). That final canary used tools on all three
examples (18 calls total), with no tool errors, empty answers, raw reasoning,
or parser markup. Its dev3 global average was 0.46296296296296297. It was
promoted to fixed ExpertQA dev100 as `01KYRQXQWH0ZTSPFG6G2H9RTE9` (job
`01KYRQXR0GVBJ383KGAXAJF1K4`). It completed 100/100 with 95 tool-using
trajectories, 375 matched calls/results, no tool errors, malformed arguments,
empty finals, raw markup, or evaluation errors. The logged scores are citation
recall 0.31705372405372406, citation precision 0.5132947954822955, answer
precision 0.9183333333333333, snippet grounding 0.8167531055900621, and global
average 0.582893950956451.

The promoted OLMo Instruct SAGE dev50 run completed all 100 examples, but one
short-form trajectory ended in an unparsed legacy `<function_calls>` block.
Its scores (open 0.028534798534798535, short 0.00) are therefore not logged.
The OLMo3 template is now used for every tool-dependent OLMo Instruct harness;
fixed dev3 recanaries passed for LitSearch (`01KYRPX1X6722SB7EGWCC6X0KG`, job
`01KYRPX20KH0JNEC6M59ACFASZ`) and SAGE (`01KYRPX77RHQ21WE4895DE419G`, job
`01KYRPX7D2X5TCDHB2H9M14828`). Every LitSearch example and all six SAGE
examples used parsed tools without tool errors. Two SAGE final outputs were
unparsed malformed calls generated by the model itself (one omitted a closing
parenthesis; one mixed unavailable tools and invalid syntax), so they are
legitimate scored-zero behavior rather than serving corruption. Corrected
fixed dev50 promotions are LitSearch `01KYRQB65W8NSQRK9XH1J6JJZC` (job
`01KYRQB698AQKVQZHVXDSTXF7D`) and SAGE `01KYRQF4K68GKEMCB622GD06G5` (job
`01KYRQF4PNPTJS4WVETE3EPSD9`). The corrected LitSearch promotion completed
50/50 with tools on every example, 87 matched tool calls/results, no tool
errors, and a 0.20 found rate / 0.20 gold recall; both metrics were logged.
One scored-zero final contained a visibly malformed model-generated call with
an unmatched quote, which is benchmark behavior rather than a parser failure.

The corrected SAGE promotion exposed a separate serving defect during raw-log
review: two otherwise valid calls containing escaped apostrophes were broken
by the repository's vLLM sanitizer, which doubled the escape before `ast.parse`.
Commit `106ef62` preserves already escaped apostrophes while retaining the
sanitizer for raw backslashes; its focused regression assertions and replay of
both captured failures pass. Do not log the affected 0.027857142857142855
open / 0.00 short result. Parser-fixed dev3 recanaries are OLMo Instruct
`01KYRR7PVVJWW7QYH5QKAVK5CJ` (job `01KYRR7PZB610YXFMXHJ6NYRR5`) and OLMo
Think `01KYRR7YYB0EFYTJ7QZT0EXGBC` (job `01KYRR7Z5SZ6DE6SPZGQGBNWCP`). The
old OLMo Think SAGE dev50 job `01KYRQRT704562FRFCRQDN42ZT` was canceled after
startup because it was pinned to the superseded parser commit. The OLMo
Instruct recanary passed with 16 matched calls/results, no tool errors or raw
markup, and one valid direct answer; replacement dev50
`01KYRRFXN8SZVWECF3PWZ7HB9D` (job `01KYRRFXRZETJYHT236770ZHG7`) was then
submitted on the fixed commit. It completed 100/100 with 222 matched
calls/results, no tool-result errors or empty answers, and scores of
0.01641025641025641 open weighted recall and 0.00 short exact match; both were
logged. Six raw final calls were retained as model behavior: the parser log
shows missing closing quotes, an invalid positional argument, or final-horizon
calls rather than the escaped-apostrophe sanitizer defect.
The OLMo Think recanary was also parser-valid: five matched calls/results, no
tool-result errors, three direct answers, one explicit invalid-tool error, and
one malformed model call. Replacement dev50 `01KYRRJ8BZ5JAZJAGY53WQRFHZ`
(job `01KYRRJ8HD99SR9PWMW6AWHT8F`) completed 100/100, but raw review found 35
finals containing official `<function_calls>` wrappers that vLLM failed to
parse. The pinned parser separated every nonempty line with a comma, corrupting
multiline arguments inside one call. Do not log its 0.006666666666666666 open /
0.00 short scores. Commit `1f1a6a9` now separates only complete top-level calls
while preserving argument newlines; 18 focused assertions and an idempotent
patched-source compile pass. Replay of the 35 captured wrappers makes 30
parseable; the remaining five are genuinely malformed model calls. Fixed dev3
recanaries are SAGE
`01KYRV0FFH0QPMYS4T094TMN0P` and LitSearch `01KYRV11H3MVN1GE1NF8ESCZQ6`.
The SAGE recanary passed 6/6 with eight matched calls/results, no tool or parser
errors, and no empty finals. Its only raw wrapper used a nonexistent XML-style
tool and is retained as malformed model behavior. Replacement dev50
`01KYRV9CPS6QR0TQSY23YAAHQD` was then submitted on `1f1a6a9`. It completed
100/100 with no empty outputs, 203 matched tool calls/results, no tool-result or
context errors, and scores of 0.008076923076923077 open weighted recall and
0.00 short exact match; both metrics were logged. The open and short tasks used
parsed tools on 33/50 and 16/50 examples, respectively. The four parser
exceptions contained plainly malformed model syntax (unquoted `OR`, prose or
comments inside call lists, and an unterminated call); the escaped-apostrophe
and multiline failures did not recur. Remaining raw wrappers were direct,
mixed-output, malformed-tool, or terminal-horizon model behavior.

The two missing fixed DeepScholar dev10 cells were also launched with the
validated partial-generation and `_fixed`-metric policy:

| Model | Experiment | Job |
| --- | --- | --- |
| OLMo-3 7B Think | `01KYRPBX0QDQ78EXXXCQZZMSV8` | `01KYRPBX42NZXSS3QT1RSSG01B` |
| GLM-4.1V-9B-Thinking | `01KYRPC4ETRV5BPMMQMDMPT84V` | `01KYRPC4JDQKRXD17DMM0D6E0S` |

GLM completed the 10-instance slice successfully. Three generation attempts
produced empty reports under the allowed partial-generation policy, yielding a
valid fixed geomean of 0.00 rather than a job failure. The other fixed metrics
were citation precision 0.03267230756592458, claim coverage
0.04393785883147585, coverage relevance 0.24, document importance 0.00,
nugget coverage 0.1622103386809269, organization 0.15, and reference coverage
0.01; all eight `_fixed` rows were logged.

The first OLMo Think DeepScholar dev10 is invalid and must not be logged. Only
2/10 reports were nonempty; the other eight surfaced `KeyError:
'intro_section'` only after the underlying provider rejected prompts whose
16,384-token output reservation pushed the total beyond the model's 32,768
context window. The checkpoint's official config and validated SAGE profile
support 65,536 tokens, so the initial 8,192-token-stage recanary
`01KYRSMTTHEQY1MK5YF82ATH7T` was stopped before evaluation. Replacement dev3
`01KYRSV77X34WSG241TMEYZEY5` keeps the standard 16,384-token stage budget and
raises the provider context to 65,536. Promote it to dev10 only after confirming
that the context-window failures are gone, and retain both limits for its
replicates.

That 64K recanary generated 2/3 reports without context or API failures. Query 2
failed under the partial-generation policy because the model returned the
search instructions themselves as search queries; this is retained model
behavior. Fixed dev10 `01KYRVH85J39SXQMMD6EW4694C` was then submitted with the
same 65,536-token context and 16,384-token stage budget. It completed all ten
attempts without provider-context or API failures and produced eight nonempty
reports. Indices 3 and 7 exhausted search/filter retries and surfaced the known
secondary `intro_section` error, so they remain partial-generation zeros. The
eight logged `_fixed` metrics are citation precision 0.0386048566132582, claim
coverage 0.06762362790851814, coverage relevance 0.35666666666666663,
document importance 0.08057170542635658, geomean 0.09139352380170106, nugget
coverage 0.21116204057380528, organization 0.25, and reference coverage
0.013448275862068964. This closes every supported first-pass cell.

Once every in-scope first-pass agentic dev cell has a valid result, run two
additional independent copies of the same fixed sample for three total runs
per model/evaluation. Do not replicate invalid or still-gated profiles. Gemma4
12B dense remains excluded from this rollout.

The replicate matrix has 24 experiments per copy (48 new experiments total):

| Profile | LitSearch dev50 | SAGE open/short dev50 | ExpertQA dev100 | DeepScholar dev10 |
| --- | :---: | :---: | :---: | :---: |
| Gemma4 26B-A4B | yes | yes | yes | yes |
| Qwen3.5 9B | yes | yes | yes | yes |
| Qwen3.5 35B-A3B | yes | yes | yes | yes |
| Nemotron 3 Nano 30B-A3B | yes | yes | yes | yes |
| OLMo-3 7B Instruct | yes | yes | unsupported | yes |
| OLMo-3 7B Think | yes | yes | unsupported | yes, 65,536/16,384 limits |
| GPT-OSS-20B | yes | yes | yes | yes |
| GLM-4.1V-9B-Thinking | unsupported | unsupported | unsupported | yes |

Use run tags `agentic-dev-replicate2-20260730` and
`agentic-dev-replicate3-20260730`. Every standard task keeps sample seed 42 so
the same examples are compared; stochastic generation supplies run-to-run
variation. Submit tool-harness work in controlled batches and DeepScholar at
approximately two concurrent jobs so S2/Serper load does not recreate the old
retry-budget failures. Audit and log each finalized experiment independently.

Replica 2 tool batch 1 was submitted on commit `1f1a6a9`:

| Model / eval | Experiment | Job |
| --- | --- | --- |
| Gemma4 26B-A4B LitSearch | `01KYS2GJ927E92E3M888NCWT61` | `01KYS2GJCG9R2Z0NMTWHCJ0M06` |
| Gemma4 26B-A4B SAGE | `01KYS2G0DCSG2XYY6VD15V5XSX` | `01KYS2G0GSZKJ4C43SC2SSJXCG` |
| Gemma4 26B-A4B ExpertQA | `01KYS2G047ET9VB74F57MAPRHB` | `01KYS2G07T9CTF66QWG51BN3AW` |
| Qwen3.5 9B LitSearch | `01KYS2G0GJ7Y4P5ARHKC37VYP4` | `01KYS2G0KXGZMQA0P5VB64F101` |
| Qwen3.5 9B SAGE | `01KYS2GSPKRAYWPPPKMHZTM33S` | `01KYS2GST9Y9P6Y3KMS8AF0M5Y` |
| Qwen3.5 9B ExpertQA | `01KYS2G056WX51SCX42EQX1GED` | `01KYS2G08QD8YACJ1XCBHNA9AB` |

Replica 2 DeepScholar batch 1 keeps the two-job concurrency cap:

| Model | Experiment | Job |
| --- | --- | --- |
| GPT-OSS-20B | `01KYS2NY9SZCGJT1FAD2YAG126` | `01KYS2NYE4H64V58HCAG8QX46B` |
| GLM-4.1V-9B-Thinking | `01KYS2P7DBYVV8CSVH5K8690Q8` | `01KYS2P7HB9JJYRF3NZQNARHQC` |

Replica 2 OLMo tool batch:

| Model / eval | Experiment | Job |
| --- | --- | --- |
| OLMo-3 7B Instruct LitSearch | `01KYS398P2K4M71VCXZVQ15E6N` | `01KYS398SQ3EFVRYXQW1B3FRTT` |
| OLMo-3 7B Instruct SAGE | `01KYS39FPK9HTSQJEFW3W4FP92` | `01KYS39FT6GYWSWNJRNQSJQ9ZJ` |
| OLMo-3 7B Think LitSearch | `01KYS39PA4KHZSMN15NCYX9Y15` | `01KYS39PE7CNT36YR4M6K256RP` |
| OLMo-3 7B Think SAGE | `01KYS39Z1PQC558JYEWT7V44J8` | `01KYS39Z56WGXR45ECZ81FYN7D` |

Replica 2 tool batch 3 began with the two freed LitSearch slots:

| Model | Experiment | Job |
| --- | --- | --- |
| Qwen3.5 35B-A3B | `01KYS3EH3EK0JSTXX8WQ34ACBE` | `01KYS3EH6RVPT76TE51A1Z2T20` |
| Nemotron 3 Nano 30B-A3B | `01KYS3ERDVEVKGCBBWKFZ8ZST9` | `01KYS3ERH40G12GHXX5B99K6ZA` |

### Results logging

The canonical long-form local mirror is
`scripts/analysis/data/results.csv`. It uses the same six columns as the
Google Sheet Results tab:

```text
Model Name,Eval Name,Metric,Score,Beaker Run ID,Notes
```

Keep `scripts/analysis/data/eval_matrix.csv` as the primary-score-only input to
the existing slide analysis. The long-form `results.csv` is seeded from that
snapshot and additionally stores every scalar task sub-metric on its own row.

Import a completed group with:

```bash
scripts/analysis/log_beaker_results.py \
  --group roryd/deepscholar-dev10-compare-20260729-193621
```

The importer selects the latest finalized successful job for each experiment,
rejects runs exposing evaluation errors, validates DeepScholar's instance
count, preserves full numeric precision, and upserts on model, eval, metric,
and Beaker experiment ID. It can also take repeated `--experiment ID` flags and
supports `--dry-run`.

Log every scalar metric exposed under each task. For DeepScholar-Bench only,
discard metric names whose base name does not end in `_fixed`; the raw metrics
describe only the successfully generated subset. Dev eval names include their
scope, such as `DeepScholar-Bench-dev10`, so they cannot be confused with full
benchmark results.

## Canonical code state

- Branch: `roryd/deepscholar-bench-s2patch`
- Current GPT-OSS promotion commit:
  `b808dd74debf2fc297d0ce5c33c3a7119b2cbb34`. Replica-2 jobs launched before
  the GPT-OSS recovery use `1f1a6a91470d473030fffed78e7378928e5ddb98`. Earlier non-OLMo dev results on
  `b5530dc4ca00ce65854a29c690de02905045acb1` remain valid. OLMo parser repair
  `106ef62` fixed escaped apostrophes; `1f1a6a9` additionally fixes multiline
  calls and is required for all replicas.
- All runs from old commit `6c2385b2c23744491a52e9d417492b9e190fe2fb`
  are invalid and must not be entered into results tables.
- Keep tracked source changes committed before submission so Gantry launches the
  intended revision.
- The files under `scripts/beaker/` and `scripts/internal/` are local helpers.
  A submitted Beaker job does not depend on them after its spec is created, so
  they do not need to be committed unless they should be shared or reviewed.

## Secrets and external services

The Beaker workspace is `ai2/olmo-eval-debug`. Use these mappings without
printing secret values:

- `roryd_S2_API_KEY:S2_API_KEY`
- `roryd_SERPER_API_KEY:SERPER_API_KEY`
- `roryd_OPENALEX_API_KEY:OPENALEX_API_KEY`
- `roryd_OPENAI_API_KEY:OPENAI_API_KEY`

Current service state:

- Semantic Scholar quota was raised to approximately 10 requests/second.
  Occasional 429s may still occur and should be retried with backoff.
- Serper has a newly funded key. A full ExpertQA run completed successfully.
- OpenAlex has a newly funded key. DeepScholar produced nonzero
  `document_importance`, confirming authenticated citation lookups worked.
- `scripts/internal/check_s2_rate_limit.py` is available for a local S2 probe.

## DeepScholar implementation state

The validated branch contains the required fixes:

- Pinned LOTUS behavior and an organization-scoring compatibility patch.
- Organization pairwise outputs are parsed correctly; the validated score was
  nonzero.
- `coverage_relevance_rate` is normalized from the upstream 0-2 implementation
  to the published 0-1 scale exactly once.
- OpenAlex uses the configured email and API key and fails visibly after
  exhausted retries instead of silently converting missing lookups to zeros.
- S2 uses `S2_API_KEY` in the `x-api-key` header and hard-fails if it is absent.
- Full launches use `allow_partial_generation=true`, a default 32,768-token
  provider context, and a 16,384-token stage budget. OLMo Think uses its
  officially supported 65,536-token context because its longer reasoning
  prompts otherwise exceed the default when the standard stage budget is
  reserved. This model-specific context must remain identical across its three
  dev10 runs. A model that consumes its validated budget without emitting a
  valid answer is scored as a model failure rather than receiving a post-hoc
  allowance.
- Known upstream issue: the recurrent `KeyError: 'intro_section'` is a
  secondary, misleading error rather than the original failure. DeepScholar's
  search/filter stage exhausts all three retries, the pipeline catches and
  suppresses that exception, and then returns an empty report plus a `stats`
  dictionary without `intro_section`. The caller subsequently indexes
  `stats["intro_section"]`, masking the underlying search/filter error. Dev
  example index 7 (`2506.00085v1`) exhibited this pattern across every tested
  model, indicating a query/pipeline-level failure rather than a
  model-specific one.
- Decision as of 2026-07-29: do not patch or modify the upstream
  `deepscholar-bench` repository yet. In particular, do not paper over the
  failure with `stats.get("intro_section", "")`, because that could mark an
  empty report as successful. Continue using `allow_partial_generation=true`,
  retain failures in `_fixed` metric denominators, and mark affected aggregate
  results as partial. If revisited, first preserve and propagate the concrete
  exception from the final search/filter retry, then rerun only the failing
  index to diagnose the actual retrieval/filter failure.

For partial runs, report `geomean_fixed` and the other `_fixed` metrics. Raw
metrics only describe the successfully generated subset.

## Serper credit incident and Monday reruns

Serper credits were exhausted on 2026-07-24. The following runs contain
explicit HTTP 400 `Not enough credits` responses and are invalid regardless of
their process exit status:

| Evaluation | Model | Experiment | Matching error lines | Action |
| --- | --- | --- | ---: | --- |
| ExpertQA | Qwen3.5 9B | `01KYAE4B2VZ817HFN29W7C4RJV` | 9,820 | Rerun Monday 2026-07-27 |
| ExpertQA | Qwen3.5 35B-A3B | `01KYAN38XC0QJK4NT11ATVA5FM` | 9,828 | Canceled; rerun Monday |
| SAGE open + short, first 100 each | Qwen3.5 35B-A3B | `01KYAN39GPC1VZGK365AMP2QE1` | 1,114 | Rerun Monday |
| SAGE open + short, first 100 each | Qwen3.5 9B | `01KYAN398XA2FZTYT2M9ZDGVB4` | 1,898 | Rerun Monday |

The counts are matching log lines, not unique API calls; the client commonly
logs both an API error and an HTTP error for one failed call. Do not launch new
ExpertQA or canonical `dr_tulu_crawl4ai` SAGE jobs until the account top-up is
confirmed with a successful Serper probe. Exit code zero does not make these
four results usable.

## Serper-free evaluation queue

These evaluations can continue before the top-up, in priority order:

1. DeepScholar-Bench: uses Semantic Scholar, OpenAlex, and the OpenAI judge,
   but not Serper. Keep partial-generation scoring because of the known
   `intro_section` failure.
2. Core: LitSearch-rerank, IFEval OOD, and MATH-500 require no search API.
3. MMLU: requires no search API; retain the validated logprob-safe provider
   profile and its separate job.
4. LitSearch open: use `paper_search_agent`, which exposes Semantic Scholar
   search and does not use Serper.

Canonical SAGE is not on this list: `dr_tulu_crawl4ai` includes Serper web
search. Running SAGE with the older S2-only harness would change the evaluation
and would not be comparable to the colleague's results.

## Validated Gemma4 26B results

Model: `google/gemma-4-26B-A4B-it`

Group: `gemma4-26b-validated-full-20260723-013500`

### ExpertQA

- Experiment: <https://beaker.org/ex/01KY69SZMNDSKT049T6F61W9E2>
- Exit code: 0
- Instances: 2,176
- Citation recall: 0.508169
- Citation precision: 0.639788
- Global average: 0.689883
- Answer precision: 0.921692
- Snippet grounding: 0.722809
- Integrity note: 90/2,176 final answers were empty and were retained as
  zero-scored examples. There was no systematic parser or API failure.

### Paper-search agent

- Experiment: <https://beaker.org/ex/01KY69T9V1Z5QPW0E0Q6DXCTH0>
- Exit code: 0
- LitSearch: 597 instances, found rate 0.244556, gold recall 0.238191
- SAGE open: 600 instances, weighted recall 0.046346
- SAGE short: 599 instances, exact match 0.093489
- Integrity note: only two empty outputs across 1,796 examples. SAGE short
  often used the complete ten-turn budget, but retained substantive final
  answers. No systematic S2 or tool-parser failure was found.

### DeepScholar-Bench

- Experiment: <https://beaker.org/ex/01KY69TMTF1AV7ZVJ5G13X3M32>
- Exit code: 0
- Dataset size: 63
- Successful generations: 56/63
- Fixed geomean: 0.181747
- Raw successful-subset geomean: 0.223943
- Organization: 0.61 raw, 0.492063 fixed
- Coverage relevance: 0.625 raw, 0.506944 fixed
- Document importance: 0.09 raw, 0.075699 fixed
- Seven generations failed with `KeyError: 'intro_section'` at indices 7, 22,
  28, 29, 30, 31, and 57. This appears to be a model/pipeline-stage failure,
  not an S2 or OpenAlex outage. The fixed metrics penalize these failures.

These three runs are usable and do not need to be rerun for infrastructure
reasons.

### Core and MMLU (remaining non-agentic tasks)

The `core` harness group (`litsearch_rerank`, `ifeval_ood`, `math500`) and MMLU
were never completed for Gemma: the two prior `safe-core` full runs on
2026-07-22 were canceled (exit 143), and only the canaries succeeded. Relaunched
on the validated commit on 2026-07-23, group
`safe-evals-gemma4-26b-a4b-it-20260723-171131`.

Core: <https://beaker.org/ex/01KY7ZC75EVGGAAQ9X09CHD5B9> (exit 0, `errors: []`)

- `math500`: `accuracy:minerva_math_flex` = 0.294, 500 instances
- `litsearch_rerank`: `recall@5` = 0.591401, `recall@20` = 0.621050,
  597 instances
- `ifeval_ood`: `prompt_level_loose_acc` = 0.436667 (primary),
  `prompt_level_strict_acc` = 0.403333, 300 instances

`litsearch_rerank` is part of `core`. The completed paper-agent run
(`01KY69T9V1Z5QPW0E0Q6DXCTH0`) is task `litsearch` = LitSearch **open**, so its
0.244556 found rate belongs on the LitSearch-open row, not LitSearch-rerank.

The first MMLU attempt
(<https://beaker.org/ex/01KY7ZCCR9GVPEC2KZM13SRA8B>) is invalid. vLLM ran out
of memory in the prompt-logprob `log_softmax` path after 1,696/14,042 requests;
the remaining 12,346 requests produced empty outputs. The harness still exited
0 and reported ~0.069, so job success alone was not sufficient validation.

The config-only retry
(<https://beaker.org/ex/01KY84PXVTJ56NXRFHGCS4K709>) completed cleanly:

- Exit 0 with all 57 subjects and 14,042/14,042 instances scored
- No OOM or request failures
- `mmlu` `primary_score:average` = 0.5566
- Category scores: STEM 0.5348, humanities 0.6163, social sciences 0.5719,
  other 0.5033

The validated MMLU provider profile is TP equal to the two requested GPUs,
`max_model_len=4096`, `max_num_batched_tokens=2048`, and
`gpu_memory_utilization=0.85`. The batched-token cap directly bounds the
float32 token-by-vocabulary allocation that caused the original OOM.

## Model matrix

| Status | Model | Hugging Face reference |
| --- | --- | --- |
| Complete baseline | Gemma4 26B-A4B | `google/gemma-4-26B-A4B-it` |
| Pending | OLMo-3 7B Instruct | `allenai/Olmo-3-7B-Instruct` |
| Pending | OLMo-3 7B Think | `allenai/Olmo-3-7B-Think` |
| In progress; repair/comparison runs submitted | Qwen3.5 9B | `Qwen/Qwen3.5-9B` |
| Pending | GLM-4.1V-9B-Thinking | `zai-org/GLM-4.1V-9B-Thinking` |
| Pending | GPT-OSS-20B | `openai/gpt-oss-20b` |
| In progress; repair/comparison runs submitted | Qwen3.5-35B-A3B | `Qwen/Qwen3.5-35B-A3B` |
| Pending | Nemotron 3 Nano 30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Skip for now | Gemma4 12B | `google/gemma-4-12B-it` |

Gemma4 12B is skipped because it was not starting reliably with the current
vLLM stack.

## Rollout sequence

Do not launch the complete matrix at once. Process one new model through the
following gates:

1. Confirm vLLM starts with a two-example generation task.
2. Run two examples each of ExpertQA and LitSearch. Run SAGE separately with
   the richer `dr_tulu_crawl4ai` harness.
3. Run a three-query DeepScholar canary with all metrics enabled.
4. Inspect raw predictions for empty answers, raw tool-call markup, exhausted
   turn budgets, and tool/API errors.
5. Only after those checks pass, launch the model's fixed-size ExpertQA,
   LitSearch, SAGE, and DeepScholar dev jobs as separate harness groups.
6. Run the non-agentic evaluations in full and separately, especially MMLU.
7. Record exact experiment IDs, commit, instance counts, primary metrics, and
   failed-example counts before moving to the next model.

This staged approach prevents one model's parser or vLLM incompatibility from
creating a full set of unusable jobs.

## Local launch helpers

### Agentic and non-agentic tasks

`scripts/beaker/launch_safe_evals.sh` can select one model and one harness
group:

```bash
scripts/beaker/launch_safe_evals.sh \
  --model allenai/Olmo-3-7B-Instruct \
  --slug olmo3-7b-instruct \
  --only paper

# SAGE uses the 20-turn search-and-browse harness and a 65,536-token context.
# Use the standard fixed dev50 sample.
scripts/beaker/launch_safe_evals.sh \
  --model allenai/Olmo-3-7B-Instruct \
  --slug olmo3-7b-instruct \
  --only sage \
  --limit 50 \
  --sample-seed 42

scripts/beaker/launch_safe_evals.sh \
  --model allenai/Olmo-3-7B-Instruct \
  --slug olmo3-7b-instruct \
  --only expertqa \
  --limit 100 \
  --sample-seed 42

# MMLU automatically uses the validated logprob-safe provider profile.
scripts/beaker/launch_safe_evals.sh \
  --model allenai/Olmo-3-7B-Instruct \
  --slug olmo3-7b-instruct \
  --only mmlu
```

For `--only mmlu`, the helper defaults to two GPUs with TP=2, a 4,096-token
context, 2,048 maximum batched tokens, and 0.85 GPU memory utilization. Override
these only for a model-specific reason with `--gpus`,
`--mmlu-max-model-len`, `--mmlu-max-batch-tokens`, and
`--mmlu-gpu-memory-utilization`. These defaults completed the full Gemma4
26B-A4B MMLU run without truncating any current 5-shot prompt (maximum observed
length: 3,088 Gemma4 tokens).

`--only paper` now means LitSearch open under `paper_search_agent`, with the
Semantic Scholar tool. `--only sage` uses `dr_tulu_crawl4ai`: 20 turns,
Semantic Scholar search, Serper web search, webpage browsing, and a default
65,536-token context. Qwen3.5 SAGE keeps thinking/raw output, omits the
reasoning parser, uses the native `qwen3_coder` tool parser, and gives SAGE
short 8,192 output tokens. Keep these harnesses separate; results from the old
ten-turn S2-only SAGE setup are not directly comparable.

For both supported Qwen3.5 checkpoints, Core and ExpertQA use direct-response
mode and omit `reasoning_parser=qwen3`, preventing parser extraction from
turning a substantive or budget-exhausted response into an empty answer.
ExpertQA additionally uses an 8,192-token output budget and a strict JSON-only
system prompt naming the two valid Serper tools.

Use `--print-only` first. The helper is convenient, but it is not yet a
universally validated model-profile matrix. Add or verify each model's vLLM
tool-call and reasoning parsers before submitting agentic tasks. Gemma4 26B was
validated with:

```text
provider.kwargs.language_model_only=true
provider.kwargs.tool_call_parser=gemma4
provider.kwargs.reasoning_parser=gemma4
```

### DeepScholar

Launch one selected model with:

```bash
scripts/beaker/launch_deepscholar_full.sh \
  --only-model olmo3-7b-instruct
```

Run a three-query canary first:

```bash
scripts/beaker/launch_deepscholar_full.sh \
  --limit 3 \
  --only-model olmo3-7b-instruct
```

The helper already includes `--no-follow`, S2 secret mapping, partial-generation
scoring, a uniform 16,384-token stage limit, and known DeepScholar provider profiles. It
also accepts `--start-idx N`; `--limit N` is a count beginning at that index, and
`--max-model-len N` records validated model-specific context exceptions such as
OLMo Think's 65,536-token profile.
Use these together to shard a slow model below DeepScholar's fixed eight-hour
inner evaluation timeout.

### Validated-code DeepScholar sweep (2026-07-24)

Group: `deepscholar-validated-20260724`
(`01KYB10CPN9DE724X0BT6NQAF0`). All jobs use commit `b5530dc`, S2 search,
authenticated OpenAlex, the OpenAI judge, and no Serper calls.

| Model | Scope | Experiment |
| --- | --- | --- |
| OLMo-3 7B Instruct | Full 63 | `01KYB10JSJ131GRCB0YT50B7KA` |
| OLMo-3 7B Think | Indices 0-31 | `01KYB141MYZMV2KRW7CDBWA3TM` |
| OLMo-3 7B Think | Indices 32-62 | `01KYB149586P3H9EWXPPC9ZDJ0` |
| GLM-4.1V-9B-Thinking | Full 63 | `01KYB10SWAEEX8ZJ34YWJVJJ8S` |
| GPT-OSS-20B | Full 63 | `01KYB11177ECQ5ZE92QTFTRT68` |
| Nemotron 3 Nano 30B-A3B | Full 63 | `01KYB117RP02WT1D52DR4VEJDJ` |

The older exit-zero runs for these models are not comparable: their metrics
show pre-fix signatures such as `coverage_relevance_rate > 1`, organization
forced to zero, or document importance forced to zero. Qwen3.5 9B,
Qwen3.5-35B-A3B, and Gemma4 26B-A4B already have validated-code DeepScholar
runs. Gemma4 12B remains excluded because of its vLLM incompatibility.

### Serial 16K DeepScholar rerun (2026-07-24)

The six concurrent validated-code jobs above were canceled after Semantic
Scholar repeatedly exhausted its 45-second retry budget and returned no
results. Relaunch one job at a time with the uniform 32K context and 16K stage
budget. Current group: `deepscholar-serial16k-20260724`
(`01KYB335YYSVTBVYWHVKQY0CEB`).

- OLMo-3 7B Instruct: experiment `01KYB33B3SEBRW9R5382WJE9S6`, job
  `01KYB33B7GD3F93HNY79YDNAWF`. Early queries showed no S2 or generation
  errors; do not launch the next model until this job reaches a terminal state.

Startup logs showed retried Semantic Scholar 500s/timeouts (15 lines each in
the first OLMo Instruct and GLM intervals). This is not the Serper-credit
incident, and a prior usable Qwen shard contained 217 retried S2 failure lines,
so the sweep remains running. At acceptance, reject any run where S2 failures
are systematic enough to prevent representative retrieval rather than merely
intermittent and recovered by retry.

## Qwen3.5-35B-A3B rollout (repair runs submitted)

Model: `Qwen/Qwen3.5-35B-A3B`

- Full-run Beaker group:
  <https://beaker.org/gr/01KY8B2JRMAFZ1D7N362DEYB8C>
- Original Core full: <https://beaker.org/ex/01KY8B2QH2YJ0GM622F5MQNBHN>.
  MATH-500 is usable at 0.694, but IFEval and LitSearch-rerank are invalid:
  283/300 and 163/597 outputs, respectively, were empty under the reasoning
  parser. Corrected direct-response Core rerun:
  <https://beaker.org/ex/01KYAN2QH2FZ3VPVP5GPW3TMF7>.
- Original ExpertQA full (provisional, non-thinking but parser-enabled):
  <https://beaker.org/ex/01KY8DDDE9BCVH6P0CX25T3TJW>. The five-item canary
  produced four substantive finals (4.3K-7.3K characters) and citation recall
  0.1095. The superseded default-thinking full run
  (`01KY8BNNSCSRW2140GSPFN46H7`) was stopped.
- Corrected ExpertQA canary:
  <https://beaker.org/ex/01KY8CDZ3MWJJDDHKV9C5V51QY>. It uses non-thinking
  mode and five examples; replace the provisional full run if empty finals are
  materially lower.
- Strict-JSON/raw-output ExpertQA rerun:
  <https://beaker.org/ex/01KYAN38XC0QJK4NT11ATVA5FM>. **Invalid and canceled:**
  Serper ran out of credits. Rerun Monday with the same profile (no reasoning
  parser, thinking disabled, 8K output, strict JSON final instruction).
- MMLU: <https://beaker.org/ex/01KY8A85XVZHHK61P08GFJSC26>. Although named
  as a canary, the suite-level limit did not propagate to the 57 expanded
  subjects, so this is the full 14,042-instance run. It uses the validated
  logprob-safe profile and must not be duplicated. It completed 14,042/14,042
  without OOM; aggregate `primary_score:average` is 0.8493 (STEM 0.8109,
  humanities 0.8503, social sciences 0.9016, other 0.8342).
- Initial paper canary:
  <https://beaker.org/ex/01KY8A8ANBAD2DVAEQ5TMWYGSR>. Native tool parsing
  worked, but both SAGE-short samples spent their 2,048-token completion budget
  in Qwen's default thinking mode and returned no final title.
- Corrected paper canary:
  <https://beaker.org/ex/01KY8BXDZG4E1R4BKHQ1C7AG1G>. It uses the official
  `chat_template_kwargs.enable_thinking=false` switch and five instances per
  task. It completed quickly, but four of five SAGE-short finals were empty.
- Thinking/8K paper canary:
  <https://beaker.org/ex/01KY8CR931W2S8D0MVDXNVHQK2>. The merged SAGE
  validation fixtures show Qwen-style thinking outputs, but the reasoning
  parser still yielded four empty SAGE-short finals.
- Raw-thinking paper canary:
  <https://beaker.org/ex/01KY8DJPSY3QSFJ3GJJVZAJ4V0>. Omitting Qwen's
  reasoning parser preserved substantive final outputs while retaining native
  `qwen3_coder` tool calls. All five SAGE-short examples searched extensively
  (8-21 calls); their 0/5 score is therefore a retrieval result under the merged
  live-S2 agent semantics, not an empty-output failure. SAGE-open improved to
  0.175 and LitSearch found rate was 0.4 on the five-item sample.
- Paper-search full (raw-thinking profile):
  <https://beaker.org/ex/01KY8ER3MX15WMK80Q1AHYAJ9G>.
- Agentic SAGE comparison, first 100 examples per task:
  <https://beaker.org/ex/01KYAN39GPC1VZGK365AMP2QE1>. **Invalid:** Serper ran
  out of credits; rerun Monday. This uses the colleague-
  comparable `dr_tulu_crawl4ai` harness, 20 turns, S2 + Serper + browsing, and
  a 65,536-token context. The earlier paper-search SAGE numbers used a
  ten-turn S2-only harness and must not be compared directly to 0.76/0.094.
- Initial DeepScholar canary:
  <https://beaker.org/ex/01KY8A8YDE520154JHZZMBFW9H>. It demonstrated that an
  8,192-token stage can end with reasoning only.
- Corrected 16K DeepScholar canary:
  <https://beaker.org/ex/01KY8C0MQQZDRDHGTRJDFDSK94>. It generated and scored
  all three reports without the 8K reasoning-only truncation. The sample fixed
  metrics include organization 0.3333, claim coverage 0.5409, and citation
  precision 0.4758; reference coverage was zero, so the tiny-sample geomean was
  zero.
- DeepScholar full is split to stay below the fixed eight-hour inner timeout:
  shard A (indices 0-31) is
  <https://beaker.org/ex/01KY8E3NETSFMRZE0MXGYRNPVM>; shard B (32-62) is
  <https://beaker.org/ex/01KY8E3TW06X854HZ3KKMM3HZ0>. The initial unsharded
  submission (`01KY8DSR29FE6NRFV2927J9Z9N`) was stopped before it started.

The Qwen provider profile uses TP=2, text-only serving, Triton GDN prefill, and
`qwen3_coder` tool parsing. Core and ExpertQA use direct-response/raw-output
mode; MMLU retains its independently validated `qwen3` logprob profile.
LitSearch keeps the S2-only paper harness, while SAGE now uses the richer
20-turn search-and-browse harness. The launch helper's `--limit`
implementation attaches MMLU limits to every expanded subject rather than the
suite name.

## Qwen3.5-9B rollout (repair/comparison runs submitted)

Model: `Qwen/Qwen3.5-9B`

- Initial five-item canary group:
  <https://beaker.org/gr/01KY8H2BZG81THY5VNQ47ZAQN0>. MMLU completed 285/285
  examples at 0.7908; Paper search produced substantive outputs (LitSearch
  0.40, SAGE-open 0.05, SAGE-short 0.20); and MATH-500 scored 0.80. IFEval was
  invalid because all five generations spent exactly 2,048 tokens in reasoning
  and yielded empty parsed answers. ExpertQA was also invalid because four of
  five trajectories ended after a tool result without a final assistant answer.
- Initial three-query DeepScholar canary:
  <https://beaker.org/ex/01KY8H4HYDH54TRKJQX9FRYMC4>. It generated all three
  reports and scored a fixed geomean of 0.2376.
- MMLU full: <https://beaker.org/ex/01KY8MKJ7A495QT3WYF4C17G29>.
- Paper-search full: <https://beaker.org/ex/01KY8MKW3JTTPS065DAKRNG6B9>.
- DeepScholar full shard A (indices 0-31):
  <https://beaker.org/ex/01KY8MM7PRKT9AXD56ZC66JYCN>.
- DeepScholar full shard B (indices 32-62):
  <https://beaker.org/ex/01KY8MMT418J1VJB7QVGAPS89Y>.
- Corrected direct-response Core canary:
  <https://beaker.org/ex/01KY8MNBN6CE15EPC11DJE3NW0>. All five IFEval
  outputs were substantive, with loose prompt accuracy 0.40; MATH-500 accuracy
  was 0.60 and LitSearch-rerank recall@5 was 0.60.
- Core full: <https://beaker.org/ex/01KY8ND3RHEQ3EBKNBJPGC8DT3>
  (exit 0). IFEval loose prompt accuracy was 0.3767, MATH-500 accuracy
  was 0.612, and LitSearch-rerank recall@5 was 0.5797.
- Intermediate direct-response ExpertQA canary:
  <https://beaker.org/ex/01KY8MNXNVWFKRY57S3V3V8003>. Only one of five
  finals parsed and two trajectories used invalid tool aliases, so this profile
  was not promoted.
- Intermediate reasoning-parser/8K ExpertQA canary:
  <https://beaker.org/ex/01KY8NGDNF5KERN1M264ZH8X3Y>. Parsing improved to
  two of five, but three trajectories still ended without a final answer.
- Accepted strict-JSON ExpertQA canary:
  <https://beaker.org/ex/01KY8P2MC2KD6MS351MZAKGWXK>. Four of five finals
  parsed, no invalid tool aliases or `<think>` leakage remained, citation recall
  was 0.2153, and the global average was 0.4216.
- ExpertQA full: <https://beaker.org/ex/01KYAE4B2VZ817HFN29W7C4RJV>.
  **Invalid:** Serper ran out of credits; rerun Monday.
- Agentic SAGE comparison, first 100 examples per task:
  <https://beaker.org/ex/01KYAN398XA2FZTYT2M9ZDGVB4>. **Invalid:** Serper ran
  out of credits; rerun Monday. It uses the same
  `dr_tulu_crawl4ai`, 20-turn, 65,536-context profile as the 35B comparison.
- Both DeepScholar shards are affected by the recurrent
  `KeyError: 'intro_section'`; do not launch replacement shards until that
  separate issue is taken back up.

For both Qwen3.5 checkpoints, Core and ExpertQA disable thinking and omit the
`qwen3` reasoning parser so a reasoning-only or parser-misaligned response
cannot be silently converted into an empty answer. They retain TP=2, text-only
serving, Triton GDN prefill, and the native `qwen3_coder` tool parser. ExpertQA
also uses an 8,192-token output budget and a strict system instruction naming
the two valid Serper tools and requiring a JSON-only final response. Paper
search retains the validated raw-thinking profile. SAGE is a separate richer
agentic harness. DeepScholar is split 32/31 to remain below its fixed
eight-hour inner timeout, but its current `intro_section` failures are deferred.

## Non-agentic work still requiring care

- Small IFEval, LitSearch-rerank, and MATH-500 generation canaries worked.
- MMLU uses prompt log probabilities and must retain the launcher's validated
  MMLU-specific memory profile. Keep it in its own Beaker job so logprob memory
  behavior and failures remain isolated.
- Do not use a successful plain-generation canary as proof that MMLU will run.
- The full profile is validated for Gemma4 26B-A4B; canary it for materially
  different model architectures before launching their full MMLU jobs.

## Acceptance checklist for every full run

- Job ran the intended commit and branch.
- Exit code is zero and harness-level `errors` is empty.
- Instance count matches the benchmark's expected count.
- Prediction files parse completely.
- No raw `<|tool_call>` or equivalent markup leaked into final answers.
- Empty outputs and failed samples are counted, not silently omitted.
- Search/API errors are bounded rather than repeated across most examples.
- DeepScholar reports all seven metrics and uses `_fixed` metrics when partial.
- `coverage_relevance_rate` is within the intended 0-1 scale.
- `document_importance` is not being suppressed by failed OpenAlex lookups.
- Organization is nonzero unless raw judge outputs genuinely support zero.
- Save the experiment and result-dataset IDs with the result record.

## Immediate next action

Monitor all six jobs in DeepScholar group `01KYB10CPN9DE724X0BT6NQAF0` and
record fixed metrics, successful-generation counts, and `intro_section`
failures. Combine OLMo Think's 32/31 shards with query-count weighting. Continue
Serper-free Core, MMLU, and LitSearch-open work as capacity allows. On Monday
2026-07-27, verify the Serper top-up and rerun the two invalid Qwen ExpertQA
jobs and two invalid Qwen SAGE-100 jobs listed above. Do not promote any metric
from those four exhausted-credit runs. Gemma4 12B remains excluded.
