# PR 251 review — Rewrite ExpertQA as agentic web-grounded attributed QA

**Branch:** `yilun/expertqa-grounded` → `main`
**Files:** `src/olmo_eval/common/scorers/citation.py` (+), `src/olmo_eval/evals/suites/science.py`, `src/olmo_eval/evals/tasks/expertqa.py` (rewrite), `src/olmo_eval/harness/presets.py`, tests (new + modified)
**Verdict:** Largest PR of the set and the biggest *semantics* change. The code is solid — I verified `_aggregate_output_scores` exists (`base.py:827`, handles empty dict → 0.0) and that expertqa stays registered in `science:research` (`science.py:148`). The main thing to be deliberate about is that this redefines what "expertqa" measures.

## Must fix before merging

- **`science.py` + task semantics — this changes what ExpertQA reports.** ExpertQA goes from closed-book (answer-from-memory, judged against self-cited claims) to agentic web-grounded, and is *removed from* `science:judge` (`science.py:209`), living only in `science:research` (`science.py:148`). Consequences to confirm are intended and communicated:
  1. Historical `expertqa` numbers are **not comparable** to numbers produced after this lands. Any tracked dashboard/leaderboard using expertqa needs a reset/annotation.
  2. `expertqa` now only produces meaningful citation scores when run through a tool-providing harness (`web_search_agent` / `*_crawl4ai`). This is intended, but it is a footgun: run it under a plain non-agentic config and citation metrics collapse toward 0. The task logs a warning (`expertqa.py` `score_responses`, "scored N/N responses with no trajectory") rather than erroring, which is the right call — just make sure the suite/launcher wiring for `science:research` always attaches an agentic harness.

  Neither of these is a code bug; they are release-communication items I'm flagging as must-address so the numbers aren't silently misread.

## Could fix if time

- **`citation.py` `ground_citations_in_sources` line ~122 (`len(normalized_snippet) >= 20`).** Snippets whose normalized (alphanumeric-only) form is under 20 chars can never ground, so a legitimately short verbatim quote counts as ungrounded and gets dropped before judging. Reasonable anti-noise threshold, but it will slightly undercount citation recall for terse quotes. Worth a comment stating the 20 is deliberate.

- **`expertqa.py` `_score_precision` (inherited, lines ~168–177).** Returns `1.0` when the judge output is empty/unparseable ("no irrelevant paragraphs = perfect precision"). A model that emits zero tool calls and fabricates citations still scores ~high `answer_precision` on fluent prose (the documented Olmo-3-7B case: citations ground to 0 but answer_precision ~0.84). Since `global_avg` averages the three, a non-searching model gets a non-trivial floor from `answer_precision` alone. Documented behavior, but consider whether `global_avg` should weight grounded metrics more, or report the tiers only.

- **`citation.py` `_CITE_TAG_RE` lines ~17–22.** Dense regex with lookbehinds for the `url=` attribute and quote handling. It is covered by tests, but it is the kind of expression that is hard to modify safely later. A short comment with an example match, or a note that cite-tag parsing is intentionally lenient, would help future maintainers.

## Small nit but probably fine

- **`expertqa.py` `_iter_trajectory_tool_results` lines ~482–520.** Pairs tool calls to results by `tool_call_id` with an order-based fallback for saved traces that carry empty ids. Careful and correct, but it duplicates pairing logic that arguably belongs on `AgentTrajectory` itself; if a third task needs this, promote it.

- **`expertqa.py` cite variant (`ExpertQACite`, lines ~668–691).** Cleanly subclasses and overrides only `format_request`/`extract_answer`/`_ground`. Good. Just note (as the docstring already does) that cite-mode and JSON-mode `snippet_grounding_rate` are not comparable — the rate denominators differ (URLs vs quoted snippets).

## Dependency / merge-order note

- Independent of #250 at the code level: `EXPERTQA_CRAWL4AI_FETCH_TOOL = "browse_webpage"` is just a string constant, no import from crawl4ai — so #251 grounds correctly under either serper or crawl4ai presets without #250 installed.
- The PR body's *reproduction* numbers need #239 + #249 + #250, but the task itself runs standalone (numbers just differ).
- Conflicts with #250 on `presets.py` (both add presets; #251 adds `web_search_agent` + `WEB_SEARCH_SYSTEM_PROMPT`).
