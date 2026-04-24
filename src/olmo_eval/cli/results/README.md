# `olmo-eval results` — CLI guide

Subcommands for inspecting evaluations in the results DB, discovering what
you have, launching the local results viewer, and exporting pairwise stats.

| Command                    | Purpose                                                                                 |
| -------------------------- | --------------------------------------------------------------------------------------- |
| `query`                    | List experiments / task results / instance predictions with flexible filters.           |
| `groups`                   | List experiment groups in the DB with summary counts.                                   |
| `group`                    | Drill into one experiment group — its models, tasks, and covered suites.                |
| `suites`                   | List registered suites and how many of their tasks have results in the filtered scope.  |
| `viewer` **(experimental)** | Launch the local DB-backed results viewer, or dump pairwise data as JSON / CSV. |

Each command accepts `--db-host`, `--db-port`, `--db-name`, `--db-user`,
`--db-password` (or the matching `OLMO_EVAL_DB_*` env vars). Auth falls back
to AWS Secrets Manager via `OLMO_EVAL_DB_SECRET_ARN` when set.

## `query`

The general-purpose lookup — lists experiments, task-level results, or
instance-level predictions with arbitrary filter combinations in table /
JSON / CSV form. Use this for anything that isn't pairwise comparison or
discovery.

## `groups`, `group`, `suites`

Discovery helpers that complement the viewer workflow. See `--help` on each
command for the exact flags.

- `results groups` — one row per experiment_group with counts (experiments,
  models, tasks) and most-recent timestamp.
- `results group <GROUP_NAME>` — drill-down view: three tables (models,
  tasks, covered suites) plus a header with totals and date range.
- `results suites [-G group]` — for each registered suite, show `covered /
  total` tasks against the filtered scope. Sorted by coverage.

### Discovery workflow

Work top-down when you don't know which group, suite, or task to compare:

```
# 1. What experiment groups exist?
olmo-eval results groups

# 2. For a group, what models / tasks / covered suites does it contain?
olmo-eval results group my-benchmark

# 3. (Alternative) Which suites have coverage in this group?
olmo-eval results suites -G my-benchmark
```

## `viewer` *(experimental)*

> **Experimental.** Interface, output format, and summary statistics may
> change. Numbers are statistically sound but have not been thoroughly
> validated on every task family — treat the matrix as directional unless
> MDE80 is small and the per-cell saturation / SE support your
> specific claim.

`results viewer` is the single entrypoint for pairwise analysis.

- Default mode launches the local DB-backed web UI so you can discover groups,
  choose a suite or task, inspect the paired-test heatmap, and switch into the
  per-task results table.
- `-f json` or `-f csv` dumps the same underlying pairwise
  comparison data from the CLI.

Statistical interpretation of the matrix lives in
[`../../analysis/README.md`](../../analysis/README.md).

### Browser mode

Browser mode is the default. Use these optional flags to open the viewer on a
specific group or scope:

| Flag                       | Use |
| -------------------------- | --- |
| `-G, --experiment-group GROUP` | Experiment group prefix to open initially. |
| `-t, --task TASK_NAME`     | Exact task name to open initially. |
| `-S, --suite SUITE_NAME`   | Suite name to open initially. |
| `--host HOST`              | Bind address for the viewer (default `127.0.0.1`). |
| `--port PORT`              | Listen port for the viewer (default `8765`). |
| `--margin FLOAT`           | Tie threshold for continuous metrics in the paired test. |
| `--repeated-runs/--latest-only` | Seed whether the viewer keeps repeated runs as separate rows or collapses to the latest run per model hash. |
| `--require-full-coverage/--no-require-full-coverage` | Control suite-mode full-coverage filtering. |

Browser-mode constraints:

- The viewer starts from experiment-group discovery rather than direct
  model/experiment filters.
- Seed filters are limited to one `--experiment-group` plus at most one of
  `--task` or `--suite`.
- `--task-hash`, the exclude flags, and `--output` are not supported in browser mode.

Once the page is open, use the `runs` selector in the viewer header to switch
between the latest run per model hash and repeated runs without restarting the
server.

Start from discovery:

```
olmo-eval results viewer
olmo-eval results viewer -G my-benchmark
olmo-eval results viewer -G my-benchmark -S multipl_e:pass_at_1
```

Open a specific task:

```
olmo-eval results viewer -G my-benchmark -t humaneval:3shot:pass_at_1
```

### Dump mode

Use `-f json` or `-f csv` when you want the pairwise data without
the browser.

#### Filters (who to compare)

Combine one or more of the following to pick the experiments that go into the
matrix. All filters are ANDed; within a flag, multiple values are ORed.

| Flag                            | Use                                             |
| ------------------------------- | ----------------------------------------------- |
| `-e, --experiment EXP_ID`       | Specific experiment IDs.                        |
| `-m, --model PREFIX`            | Model-name prefix match (repeatable).           |
| `-M, --model-hash PREFIX`       | Model-hash prefix match (repeatable).           |
| `-G, --experiment-group PREFIX` | Experiment-group prefix match (repeatable).     |

At least one filter is required. Within the matched experiments, the tool
keeps one row per `(model_name, model_hash)` — the most recent by timestamp.

#### Scope (what to compare them on)

Exactly one of these is required:

| Flag                       | Use                                                                             |
| -------------------------- | ------------------------------------------------------------------------------- |
| `-t, --task TASK_NAME`     | **Exact** task name (full variant/regime, e.g. `humaneval:3shot:pass_at_1`). Unlike `results query -t` this does not prefix-match — use `results query` to browse matching names, or `--suite` to pool. |
| `-T, --task-hash PREFIX`   | Single task by hash prefix. Must resolve to one `task_name`.                    |
| `-S, --suite SUITE_NAME`   | Registered suite — pools instances across every task the suite resolves to.    |

Suite mode keys instances by `(task_name, native_id)` so identical native IDs
across different tasks don't collide.

#### Other dump options

| Flag                  | Default | Use                                                                                      |
| --------------------- | ------- | ---------------------------------------------------------------------------------------- |
| `--exclude-model PFX` | none    | Drop model rows whose names start with any supplied prefix.                              |
| `--exclude-model-hash PFX` | none | Drop model rows whose hashes start with any supplied prefix.                           |
| `--exclude-task TASK` | none    | Drop exact task names from the scoped comparison (useful with `--suite`).                |
| `--exclude-task-hash PFX` | none | Drop task rows whose hashes start with any supplied prefix.                            |
| `--metric METRIC`     | none    | Metric in `metric:scorer` format. Defaults to each task's `primary_metric`.              |
| `--margin FLOAT`      | `0.0`   | Tie threshold for continuous scores. Scores within `margin` of each other count as tied. |
| `-o, --output PATH`   | stdout  | Save JSON / CSV to a file.                                                               |
| `-f, --format FMT`    | none    | One of `json`, `csv`. Omit it to launch the browser viewer.                               |
| `--repeated-runs/--latest-only` | `latest-only` | Keep repeated runs as distinct rows instead of collapsing to the latest run per model hash. |

By default matched experiments are deduped to one row per
`(model_name, model_hash)`, keeping the most recent by timestamp. The CLI
prints a line summarizing how many were kept vs. dropped. Use
`--repeated-runs` to keep every re-run as a distinct row in the matrix
(labels include the timestamp for disambiguation) — useful for comparing
historical re-runs of the same model.

#### Dump workflows

Export a JSON win-rate matrix for downstream analysis:

```
olmo-eval results viewer -G my-benchmark -S multipl_e:pass_at_1 -f json -o matrix.json
```

Export a CSV for two specific models on a task hash:

```
olmo-eval results viewer -m llama3.1-8b -m qwen2.5-7b -T abc12345 -f csv
```

Exclude one noisy suite member and a stale model family from the dump:

```
olmo-eval results viewer \
  -G my-benchmark \
  -S multipl_e:pass_at_1 \
  --exclude-task mbpp_plus:pass_at_1 \
  --exclude-model old-baseline- \
  -f json
```

### Errors you might see

- **"Only 1 experiment(s) matched the filters — need at least 2."** — the
  filter combination is too narrow. The error echoes the filter values and,
  when a group was specified, suggests `results group <name>` to inspect
  coverage.
- **"No task results found for `<scope>` in the matched experiments."** —
  the models matched, but none of them ran that task/suite. Check via
  `results group <group>` or `results suites -G <group>`.
- **"No primary_metric set for task `<task>` — specify --metric explicitly."**
  — the task's `primary_metric` column is null. Pass `--metric`
  (`metric:scorer` format) yourself.
- **"Suite `<name>` not found."** — the suite isn't registered. The error
  lists close matches from the suite registry.
