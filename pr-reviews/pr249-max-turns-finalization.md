# PR 249 review — Preserve agent work on max_turns

**Branch:** `yilun/max-turns-finalization` → `main`
**Files:** `src/olmo_eval/harness/scaffolds/openai_agents.py` (+), `tests/core/harness/test_openai_agents_scaffold.py` (new)
**Verdict:** Sound. One cross-PR merge hazard, a couple of small improvements. Safe to merge once the test-file collision with #252 is resolved.

The core idea is good: on `MaxTurnsExceeded`, preserve the partial trajectory and run one tool-free generation to force a final answer, with a strict fallback to the old empty result if anything throws. I traced the downstream consumers of `HarnessResult`:
- `result.success` is `error is None and not max_turns_reached` (`harness/result.py:41`), so recovered runs still report `success=False`. Good — scoring/accounting is unaffected by flipping `error` to `None`.
- `base.py:747` `result.error` is the *scorer's* error, not the harness error, so nothing there breaks.

## Must fix before merging

- **`tests/core/harness/test_openai_agents_scaffold.py` (new file, whole file).** PR #252 creates a file at the *same path* with different contents. This is an add/add conflict — git will not auto-merge it, and "rebase" is not enough; whoever lands second must hand-merge the two test modules into one. The author flagged this in both PR bodies. Decide a merge order and combine the two test files deliberately.

## Could fix if time

- **`openai_agents.py` ~lines 42–47 (finalization-failure fallback).** When `_force_final_answer` raises, the handler returns `AgentTrajectory(turns=())` even though `trajectory = self._convert_trajectory(partial_result)` was already computed a few lines up. Scorers that read tool results (litsearch) would still benefit from the preserved trajectory even when the *forced answer* fails. Consider returning `trajectory=trajectory` (the partial one) in the fallback branch instead of an empty one — still "never worse," strictly better for trajectory-reading scorers.

- **`openai_agents.py` line 56 (`error=None` on recovered path).** Correct for `success` semantics, but any log analysis or dashboard that greps the `error` string to detect capped runs will no longer see it. `max_turns_reached=True` is still set, so anything keying on that is fine. Worth a one-line mention in the PR so downstream consumers know to switch from `error` to `max_turns_reached`.

## Small nit but probably fine

- **`openai_agents.py` `_build_forced_final_input` lines ~104–116.** The second branch (manual reconstruction from `partial_result.input` + `new_items`) is effectively dead whenever `to_input_list()` works, which is the normal SDK case. It's harmless defensive code; fine to keep, but it's the kind of fallback that rots silently.

- **`openai_agents.py` `_force_final_answer` (whole method).** The forced generation is an *extra* model call per capped instance and is not itself recorded in the trajectory (only its text lands in `final_output`). Both are acceptable and documented, just noting the added cost/latency on capped instances.

- **`_convert_trajectory` lines ~141–149 (dict `raw_item` handling).** Populating `tool_call_id` from dict-shaped `raw_item`s is a genuine correctness fix, but it also changes the *normal* (non-capped) path: `tool_call_id` will now be populated where it was previously `""`. More correct, low risk, but it is a behavior change for every agentic run, not only capped ones — worth being aware of if any scorer keyed on the old empty value.

## Merge-order note

Overlaps `openai_agents.py` with #252 and #253. See the cross-PR section in the #252/#253 reviews. #249 itself is based on `main` and has no upstream dependency.
