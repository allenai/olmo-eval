# AutomationBench-AA

`automationbench_aa` runs the public tasks of Zapier's
[AutomationBench](https://github.com/zapier/AutomationBench) and scores them the way
Artificial Analysis scores
[AutomationBench-AA](https://artificialanalysis.ai/evaluations/automationbench-aa).

Each task gives an agent a business workflow to carry out, such as routing a closed
deal or reconciling expenses. The agent works in simulated SaaS apps (Gmail, Salesforce,
Google Sheets, Slack and about 40 others) through REST-style tools. Programmatic
assertions then check the apps' final state. No model judge is involved.

```bash
# All 600 public tasks on Beaker
uv run olmo-eval beaker launch -E automationbench_aa -m <model> --gpus 1 --timeout 24h

# One domain, ten tasks, locally against a running server
uv run olmo-eval run-external -e automationbench_aa -m <model> --base-url http://localhost:8000/v1 \
    -a domains=finance -a limit=10
```

`olmo-eval external-evals` lists every argument with its default.

## Scoring

Artificial Analysis sorts each assertion into one of two classes:

- an **objective** fails in the task's initial state, and the agent must make it true;
- a **guardrail** already passes in the initial state, and the agent must not break it.

A task scores 0 if the agent broke any guardrail. Otherwise it scores the share of its
objectives the agent met. The primary metric, `aa_score`, is the mean task score.

The run also reports:

| Metric | Meaning |
|---|---|
| `objectives_completed` | Objectives met, pooled over tasks, whether or not a guardrail broke |
| `tasks_completed` | Share of tasks with every objective met and no guardrail broken |
| `violations_per_task`, `objectives_per_violation` | Guardrail violations per task, and objectives met per violation (left out when nothing was violated) |
| `zapier_pass_rate` | AutomationBench's own score: share of tasks where every scored assertion passes |
| `zapier_partial_credit` | AutomationBench's per-task reward, averaged |
| `aa_score_<domain>` | `aa_score` for each of the six domains |
| `max_turns_rate`, `context_overflow_rate`, `model_error_rate` | How tasks ended |
| `mean_turns`, `mean_tool_calls` | Model replies and tool calls per task |

The objective and guardrail classes come from AutomationBench's own `partial_credit`.
That function leaves out an assertion that passes at the start and still passes at
the end, and it fails one that the agent broke. Two cases follow it exactly:

- Assertions a task marks `"scored": false` or `"excluded": true` are left out.
- An assertion marked `"excluded": false` is an objective even when it passes at the
  start. Tasks where the right action is to do nothing are built from these. Read
  literally, AA's definition would make them guardrails, which leaves such a task with
  no objectives. This eval follows `partial_credit` instead, so an agent that rightly
  does nothing scores 1.

For every task, the driver checks that its counts reproduce AutomationBench's
`partial_credit`. A task where they don't is recorded as a scoring error.

## How a task runs

The eval uses AutomationBench's own environment rather than a port of it:

- **Code.** It checks out the AutomationBench 1.0.6 release (commit `6d210543`, the
  version Artificial Analysis names) and installs it with `uv sync --frozen`. The
  benchmark needs Python 3.13, so it gets its own environment, kept under
  `~/.cache/olmo-eval/automationbench/<commit>` and reused. The host needs `git` and `uv`.
- **Driver.** `driver/run_tasks.py` runs in that environment. It builds
  AutomationBench's verifiers environment and its retrying OpenAI-compatible client,
  as the `auto-bench` CLI does, and points them at olmo-eval's model server. It adds
  one rubric function after AutomationBench's that records the objective and guardrail
  counts. It writes one JSON line per task as tasks finish.
- **Settings.** It uses the `api` toolset (`api_search`, `api_fetch`, `base64_encode`)
  and a cap of 50 model replies per task, as Artificial Analysis does. Each task runs
  once. The tasks' own system prompt is the one Artificial Analysis shows. A task ends
  when the model replies without a tool call, at the turn cap, when the history no
  longer fits the context window, or on an error (see Failures). In the first three
  cases it is scored on the state it reached. A tool that raises does not end the task:
  its error goes back to the model as the tool's reply.
- **Sampling.** AutomationBench sends no sampling parameters, and by default this eval
  sends none either. vLLM then applies the model's generation config, and a reply may
  use the rest of the context window. Set `temperature`, `top_p` or `max_tokens` to
  override these. The provider's `chat_template_kwargs` are passed through.

The output directory keeps `automationbench_aa/tasks.jsonl`. It has one line per task,
with the stop condition, every assertion's class and outcome, token usage, and the full
message history. The same fields are in each prediction.

## Failures

AutomationBench's client retries a request that fails with anything but a client error
(4xx) up to 40 times, with backoff of up to a minute. A request that still fails, or
gets a 4xx other than a context overflow, ends the task with an error. So does a
failure to check the task's assertions. Such a task is left unscored: it is left out of
every metric and listed under `failed_tasks` in the metadata.

A reply that the server returns but that cannot be used, such as an empty one, is the
model's failure. The client asks again up to 40 times first. A task that ends this way
scores 0, as Artificial Analysis scores errored tasks, and none of its objectives count
as met.

The run fails if more than `max_failure_rate` (default 5%) of its tasks are unscored, if
the benchmark process exits with an error, or if it outlasts `run_timeout` (default 20
hours), at which point it is stopped. Its metrics over the finished tasks are still
saved. An unexpected exception inside AutomationBench, such as an authentication error,
stops the whole benchmark process, as it does under AutomationBench's own CLI. A failed
run starts again from the first task: there is no resume.

The eval sends `OPENAI_API_KEY` to a hosted provider and no key to a local server.

## Differences from the leaderboard

- **Public tasks.** Artificial Analysis runs a private, held-out split of 657 tasks
  from the same release. Zapier says its private tasks are harder. Scores here are not
  comparable to the leaderboard's numbers, only between models run with this eval.
- **Infrastructure errors.** Artificial Analysis scores every errored task 0. Here an
  error of the server or the harness leaves the task unscored, and the failure budget
  above bounds how many can be left out.
- **Do-nothing tasks.** Assertions marked `"excluded": false` count as objectives (see
  Scoring).
- **Sampling.** Artificial Analysis uses temperature 0.6 for reasoning models and 0 for
  others, unless the model's lab recommends otherwise. It caps output at 16,384 tokens
  for non-reasoning models. This eval uses the server's defaults unless told otherwise
  (see Settings above).
