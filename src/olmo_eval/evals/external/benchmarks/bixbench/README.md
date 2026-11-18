# BixBench-Verified-50

`bixbench_verified` runs [BixBench-Verified-50](https://huggingface.co/datasets/phylobio/BixBench-Verified-50),
Phylo's expert-revised 50-question subset of FutureHouse's
[BixBench](https://github.com/Future-House/BixBench) ([paper](https://arxiv.org/abs/2503.00096)).
About a third of the 50 questions were rewritten to state the covariates, thresholds or
methods they depend on, or had their answer key fixed
([audit](https://arxiv.org/abs/2604.24955)). Each question comes with a capsule of
input data. An agent answers it by analysing that data in a Jupyter notebook, and the
answer is graded by string match or an LLM judge, as upstream grades open answers.

```bash
# All 50 questions, three episodes each, on Beaker
uv run olmo-eval beaker launch -E bixbench_verified -m <model> --gpus 1 --timeout 12h

# Two questions, one episode each
uv run olmo-eval beaker launch -E bixbench_verified -m <model> --gpus 1 \
    -A question_ids=bix-18-q1,bix-30-q3 -A n_attempts=1
```

The dataset is gated: accept its terms on the Hub with the account whose token the run
uses. The judge needs `OPENAI_API_KEY`. `olmo-eval external-evals` lists every argument
with its default.

The data is for evaluation only. Every row carries BixBench's canary string; keep it out
of training corpora.

## How an episode runs

The episode follows upstream's v1.5 open-answer runs
(`bixbench/run_configuration/*_no_image.yaml`), whose agent and environment come from
[data-analysis-crow](https://github.com/Future-House/data-analysis-crow) v1.5.0, ldp
0.26 and fhaviary 0.19.

- **Workspace.** The capsule's input data is copied into an empty directory, mounted at
  `/workspace` in a container of the BixBench-env image. The capsule's reference
  notebook is removed first, as upstream does.
- **Prompt.** The system prompt and the question prompt are upstream's, vendored in
  `prompts.py`: the open-answer template with its step-by-step notebook instructions, R
  guidelines and the instruction to avoid plots. Notebooks are Python; R runs through
  rpy2 and shell commands through `%%bash`.
- **Tools.** `edit_cell(contents, idx=None)` appends a code cell or replaces one, saves
  the notebook and reruns all of it with `jupyter nbconvert --execute --allow-errors`.
  `list_workdir()` returns the directory tree as JSON. `submit_answer(answer)` ends the
  episode. The schemas are what fhaviary builds from upstream's functions.
  Only the first tool call of a reply runs, as upstream's `parallel_tool_calls: false`
  makes the model send one; later calls are left out of the conversation and counted as
  `ignored_tool_calls`.
  Every request sets `tool_choice="required"`, as upstream's does. vLLM enforces it by
  constraining decoding to a JSON array of calls, which skips the model's own
  tool-call format and the server's `--tool-call-parser`, and drops a reply cut off
  by `max_tokens` as one without a call. `-A tool_choice=auto` sends `auto` instead.
- **Message order.** The question comes first, then the first notebook render, then the
  system prompt, as upstream's environment sends them. Olmo 3, Qwen3 and gpt-oss chat
  templates accept this. Templates that require the system message first, such as
  Qwen3.5's, need `-A system_first=true`, which sends it first; results from such runs
  are not comparable with upstream's order. With gpt-5.5 on ten questions, system-first
  scored 0.867 against 0.833, with the same outcome on nine of the ten questions, and
  made episodes about 1.6 steps longer.
- **Observations.** After each step the agent sees its tool results and the whole
  notebook rendered as markdown, each output cut to 3000 characters. Only the newest
  render stays in the conversation; older ones are replaced by a placeholder.
- **Limits.** 20 steps (model calls) per episode, and 1200 seconds per run of the
  notebook. A run that times out ends the episode without an answer.
- **Sampling.** Temperature 1.0, as upstream, and up to 16384 tokens per reply.

Each question runs `n_attempts` episodes (default 3), because one pass over 50
questions is noisy.

## Grading

`grader.py` ports upstream's `OpenEndedGrader` as its grading script calls it:

- `llm_verifier`: the judge compares the submitted answer with the target.
- `str_verifier`: an answer equal to the target after keeping only lowercased letters and
  digits is correct, and so is one contained in the target (upstream's partial match);
  otherwise the judge decides.
- `range_verifier`: the judge decides, given the target range as `(lower,upper)`.

The answer is graded as submitted, `<answer>` tags included, as upstream grades it; in
practice the judge decides most `str_verifier` questions too. The judge prompt is
upstream's and asks for `correct`, `incorrect` or `refused`; only `correct` counts.
The default judge is `gpt-6-luna` at temperature 1.0 (`judge_model` changes it).
Upstream's grading script defaults to gpt-4o. `gpt-6-luna` is an alias with no dated
snapshot, so the model behind it can change.

## Results

The primary metric is `accuracy` over all episodes. An episode that never submits, runs
out of steps or context, or times out in the notebook counts as incorrect. The run also
reports:

- `accuracy_attempt_<i>` per attempt, with `accuracy_std` and `accuracy_stderr` across
  attempts;
- accuracy for each verifier type;
- how episodes ended: `submitted_rate`, `max_steps_rate`, `exec_timeout_rate`,
  `context_overflow_rate`, and `mean_steps`;
- how steps were spent: `length_truncated_reply_rate`, `no_tool_call_step_rate` and
  `ignored_tool_call_rate`, each over all steps;
- `substring_match_rate`, the share of episodes graded correct by the partial match.

A run fails rather than reporting a score when an episode's sandbox or model server
fails, or an answer cannot be graded; the partial metrics are still saved. Each
prediction holds the answer, its grade and the judge's reply, step counts, token usage
and the trajectory. The final notebook of each episode, and its markdown render, are
kept under `bixbench_verified/<question>-<attempt>/` in the output directory.

## Differences from upstream

- **No images.** Images in cell outputs are not sent; the markdown keeps upstream's
  numbered placeholders. The prompt already tells the agent to avoid plots.
- **Ungradable judge replies.** A judge reply without a `<grade>` tag is asked again and,
  if none ever has one, the answer is left ungraded. Upstream counts it incorrect.
- **Directory listings are sorted**, where upstream takes the file system's order, and
  list a symlink as a file instead of following it.
- **The notebook file is not followed through a symlink.** If the agent's code replaces
  `notebook.ipynb` with a link, reading it fails as a tool exception and the next save
  replaces the link. Upstream follows it.
- **Image architecture.** BixBench's harness defaults to `futurehouse/bixbench:aviary-notebook-env`,
  published for linux/arm64 only. The image here is built for linux/amd64 from the recipe
  whose pinned packages match that image's layer history: `packages/notebook/docker/` of
  Future-House/aviary at `dc6fe209` (`sandbox/`). data-analysis-crow's own
  `Dockerfile.pinned` differs from the published image (other Python requirements,
  numpy 1.26 instead of 2.0, no BUSCO), so it is not used. The conda environment is the
  arm64 image's package list solved for linux-64 (`sandbox/conda-pins.txt`). The
  unversioned lines there are low-level packages (certificates, compiler headers,
  compression and FFI libraries, conda's own dependencies) whose arm64 version cannot be
  installed on linux-64 with the rest or has no conda-forge build; they take other
  versions. Analysis results can still differ in the last digits between architectures:
  replaying notebooks in both images gave the same answers, with DESeq2 fits differing by
  about 0.3% in standard errors. BUSCO, which the image runs through udocker, works here;
  the arm64 image cannot run its amd64-only container.
- **No Anaconda packages.** The image takes nothing from Anaconda's repositories: its base
  is Debian 12.7 instead of `continuumio/miniconda3`, the packages the arm64 image takes
  from Anaconda's channel come from conda-forge at the same versions where they exist, its
  `anaconda-anon-usage` telemetry plugin is left out, and conda installs from conda-forge
  and bioconda.

## Pins

| What | Pin |
|---|---|
| Dataset | `phylobio/BixBench-Verified-50` @ `c77fc8ea4611b546947b8a4710ed36ab72ee233c` |
| Benchmark harness and grader | `Future-House/BixBench` @ `49311180bdacb324c596f2e07596c126f2004008` |
| Agent environment and prompts | `Future-House/data-analysis-crow` v1.5.0 @ `953a6b13c5f8e15354525bca08636f620994f954` |
| Image recipe | `Future-House/aviary` @ `dc6fe209cddb7b99343ff26bae3d4ac607ea0b19`, `packages/notebook/docker/` |
| Sandbox image | `ghcr.io/allenai/olmo-eval-bixbench-env:2026.11.03` @ `sha256:624403103a4cfbcb9a33091cf56d8f164901f91e5fabe9a5c5fe3b4864421227` (`SANDBOX_IMAGE` in `eval.py`) |
