# PR 252 review — Self-correct unknown tool calls via a hidden fallback tool

**Branch:** `yilun/unknown-tool-correction` → **`yilun/harness-robustness` (#239)**, *not* `main`
**Files:** `src/olmo_eval/harness/scaffolds/openai_agents.py` (+), `tests/core/harness/test_openai_agents_scaffold.py` (new), `tests/core/harness/test_scaffold.py`
**Verdict:** The mechanism is clean and well-tested (SDK-signature drift guard, reserved-name guard, hidden-tool wiring, kwargs forwarding). But it has the hardest merge story of the set: it is stacked on an unmerged PR and collides with #249's new test file.

## Must fix before merging

- **Dependency on #239, which is OPEN and not in the 249–254 set.** This PR's base is `yilun/harness-robustness` (#239), and the diff's context (the `ModelBehaviorError` handler at `openai_agents.py` lines ~144–149, and the "instance stays in the denominator" accounting) comes from #239. I confirmed the current `main`-based scaffold has **only** a `MaxTurnsExceeded` handler and no `ModelBehaviorError` handler (`grep` on the branch shows just the max-turns catch). So #252 cannot be merged to `main` until #239 lands (or #252 is retargeted and rebased onto `main`, which would pull in #239's changes anyway). This is the top blocker for this PR — resolve the #239 question first.

- **`tests/core/harness/test_openai_agents_scaffold.py` (new file).** Same-path add/add conflict with #249. Must hand-merge the two test modules, not just rebase. (See #249 review.)

## Could fix if time

- **`openai_agents.py` `_ToolCallCorrectingModel.get_response` (lines ~36–84).** The override only covers the non-streaming path; `stream_response` is intentionally inherited, per the docstring. Today the scaffold uses non-streaming `Runner.run`, so this is fine — but it is a silent gap if any future preset switches to `run_streamed`: unknown-tool rewriting would just stop happening and instances would die on `ModelBehaviorError` again. Worth a guard or an assertion somewhere that the scaffold path stays non-streaming, so the assumption fails loudly if it changes.

- **`openai_agents.py` lines ~68–82 (in-place mutation of `response.output` items).** The rewrite mutates the SDK's `ResponseFunctionToolCall` Pydantic objects in place (`item.arguments = ...`, `item.name = ...`). This relies on those objects being mutable and on the SDK not re-validating them downstream. It works with the pinned SDK and is tested, but it couples the scaffold tightly to SDK-internal object shapes. If the SDK ever freezes these models, this breaks. Low probability, but note the coupling.

## Small nit but probably fine

- **`openai_agents.py` `_get_tool_call_correcting_model_class` (module-global cache, lines ~21–24).** Caching the dynamically-built subclass in a module global is fine and avoids importing `agents` at module import. Just the usual caveat that a module-global memo is process-wide state.

- **The forced-answer path from #249 interacts benignly.** If both land, `_force_final_answer` builds a `tools=[]` agent with `tool_choice="none"`, so the correcting model has no visible tools and no handoffs; a stray tool call in that single generation is extremely unlikely. No action needed, just verified.
