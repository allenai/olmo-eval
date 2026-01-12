# olmo-eval

Evaluation toolkit for OLMo and other language models.

## Quick Start

```bash
# Install
pip install -e .

# List available commands
olmo-eval --help

# List model presets
olmo-eval models

# List task suites
olmo-eval suites

# List tasks and their regimes
olmo-eval tasks

# Run evaluation (dry run)
olmo-eval run -m llama3.1-8b -t arc_challenge::olmes --dry-run

# Run evaluation
olmo-eval run -m olmo-2-7b -t olmes_core --limit 100
```

## Key Concepts

### Tasks and Regimes

Tasks live in `olmo_eval/tasks/` and are registered with the `@register` decorator. Regimes are named configuration variants:

```python
from olmo_eval.tasks import Task, TaskConfig, register

@register("arc_challenge", lambda: TaskConfig(...))
class ARCChallenge(Task): ...

# Regime: task_name::regime_name
olmo-eval run -m model -t arc_challenge::olmes
```

### Task Suites

Suites live in `olmo_eval/evals/suites/` and group multiple tasks for batch evaluation:

```python
from olmo_eval.evals.suites import Suite, register_suite

register_suite(Suite(
    name="olmes_core",
    tasks=["arc_easy::olmes", "arc_challenge::olmes", "hellaswag::olmes"],
))
```

### Model Presets

Pre-configured model settings in `olmo_eval/core/constants/models.py`:

```python
from olmo_eval.core import get_model_presets

# Returns dict of preset name -> ModelConfig
presets = get_model_presets()
# {
#     "llama3.1-8b": ModelConfig(model="meta-llama/Meta-Llama-3.1-8B"),
#     "olmo-2-7b": ModelConfig(model="allenai/OLMo-2-1124-7B", trust_remote_code=True),
#     ...
# }
```

## Launching on Beaker

olmo-eval includes built-in support for launching evaluation jobs on [Beaker](https://beaker.org).

### Installation

Install with the Beaker optional dependency:

```bash
pip install 'olmo-eval-internal[beaker]'
```

### CLI Usage

Launch an evaluation job:

```bash
# Basic evaluation
olmo-eval launch -n "eval-llama3-mmlu" -m llama3.1-8b -t mmlu

# Multiple tasks
olmo-eval launch -n "eval-llama3-suite" \
    -m llama3.1-8b \
    -t mmlu -t gsm8k -t hellaswag

# Large model with multiple GPUs
olmo-eval launch \
    --name "eval-70b-full" \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --task mmlu --task gsm8k --task arc \
    --cluster h100 \
    --gpus 4 \
    --priority high \
    --timeout 48h

# Preview the Beaker spec without launching
olmo-eval launch -n "test" -m llama3.1-8b -t arc_easy --dry-run
```

### Multiple Models

Run the same suite across multiple models by specifying `-m` multiple times.
Each model will be launched as a separate experiment:

```bash
# Compare two models on the same tasks
olmo-eval launch -n "eval-compare" \
    -m llama3.1-8b \
    -m olmo-2-7b \
    -t mmlu -t gsm8k -t hellaswag

# Creates 2 experiments:
#   eval-compare-llama3.1-8b: runs all tasks on llama3.1-8b
#   eval-compare-olmo-2-7b:   runs all tasks on olmo-2-7b

# Combine with per-task priorities (creates model x priority experiments)
olmo-eval launch -n "eval-full" \
    -m llama3.1-8b -m olmo-2-7b \
    -t "mmlu@high" -t "gsm8k@normal"

# Creates 4 experiments:
#   eval-full-llama3.1-8b-high, eval-full-llama3.1-8b-normal
#   eval-full-olmo-2-7b-high, eval-full-olmo-2-7b-normal
```

### Per-Task Priorities

Tasks can include an optional `@priority` suffix to set different priorities per task.
Tasks with different priorities will be launched as separate Beaker experiments:

```bash
# Mixed priorities - creates separate experiments per priority level
olmo-eval launch -n "eval-suite" -m llama3.1-8b \
    -t "mmlu@high" \
    -t "gsm8k@normal" \
    -t "arc@low"

# Creates 3 experiments:
#   eval-suite-high:   runs mmlu at high priority
#   eval-suite-normal: runs gsm8k at normal priority
#   eval-suite-low:    runs arc at low priority

# With task regimes (@ comes after ::)
olmo-eval launch -n "eval" -m llama3.1-8b -t "mmlu::olmes@high"

# Tasks without @priority use the --priority flag (default: normal)
olmo-eval launch -n "eval" -m llama3.1-8b -t mmlu -t gsm8k --priority high
```

### CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-f` | none | YAML config file (CLI args override config values) |
| `--name` | `-n` | required | Experiment name |
| `--model` | `-m` | required | Model name or HuggingFace path (can specify multiple) |
| `--task` | `-t` | required | Task name with optional `@priority` suffix (can specify multiple) |
| `--cluster` | `-c` | `h100` | Cluster alias (`h100`, `a100`, `aus`) or full name |
| `--gpus` | | `1` | Number of GPUs |
| `--priority` | | `normal` | Job priority (`low`, `normal`, `high`, `urgent`) |
| `--preemptible` | | `true` | Allow preemption |
| `--timeout` | | `24h` | Job timeout (e.g., `24h`, `30m`) |
| `--retries` | | none | Number of retries on failure |
| `--workspace` | | `ai2/oe-data` | Beaker workspace |
| `--budget` | | `ai2/oe-base` | Beaker budget |
| `--dry-run` | | `false` | Print spec without launching |

### YAML Configuration

For complex or reusable configurations, use YAML config files with the `--config/-f` option.
CLI arguments override values from the config file.

**Basic config file** (`eval_config.yaml`):

```yaml
name: eval-llama3-core
models:
  - llama3.1-8b
tasks:
  - mmlu
  - gsm8k
  - hellaswag
  - arc_challenge

cluster: h100
gpus: 1
priority: normal
timeout: 24h
```

**Usage**:

```bash
# Run from config file
olmo-eval launch -f eval_config.yaml --dry-run

# Override specific values
olmo-eval launch -f eval_config.yaml --gpus 4 --priority high

# Add additional models via CLI
olmo-eval launch -f eval_config.yaml -m olmo-2-7b
```

**Multi-model comparison config**:

```yaml
name: eval-model-comparison
models:
  - llama3.1-8b
  - olmo-2-7b
  - mistral-7b
tasks:
  - mmlu
  - gsm8k
  - hellaswag
cluster: h100
gpus: 1
```

**Per-task priorities in config** (`examples/configs/prioritized_tasks.yaml`):

Use `@priority` suffix on tasks to run different tasks at different priority levels.
Tasks with different priorities create separate Beaker experiments:

```yaml
name: eval-prioritized
models:
  - llama3.1-8b
  - olmo-2-7b
tasks:
  # High priority - run first
  - mmlu@high
  - gsm8k@high
  # Normal priority
  - hellaswag@normal
  - arc_challenge@normal
  # Low priority - run when resources available
  - winogrande@low
  - truthfulqa@low
cluster: h100
gpus: 1
timeout: 24h
```

This creates **6 experiments** (2 models × 3 priority levels):

```
eval-prioritized-llama3.1-8b-high:   tasks=[mmlu, gsm8k]
eval-prioritized-llama3.1-8b-normal: tasks=[hellaswag, arc_challenge]
eval-prioritized-llama3.1-8b-low:    tasks=[winogrande, truthfulqa]
eval-prioritized-olmo-2-7b-high:     tasks=[mmlu, gsm8k]
eval-prioritized-olmo-2-7b-normal:   tasks=[hellaswag, arc_challenge]
eval-prioritized-olmo-2-7b-low:      tasks=[winogrande, truthfulqa]
```

**Large model config**:

```yaml
name: eval-70b-full
models:
  - meta-llama/Llama-3.1-70B-Instruct
tasks:
  - mmlu
  - gsm8k
  - hellaswag
cluster: h100
gpus: 4
priority: high
preemptible: false
timeout: 48h
retries: 2
description: "Full evaluation suite for Llama 70B"
```

**Config file fields**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Experiment name |
| `models` | list | yes | List of model names or HuggingFace paths |
| `tasks` | list | yes | List of task specs (with optional `@priority`) |
| `cluster` | string | no | Cluster alias or full name (default: `h100`) |
| `gpus` | int | no | Number of GPUs (default: `1`) |
| `priority` | string | no | Default priority (default: `normal`) |
| `preemptible` | bool | no | Allow preemption (default: `true`) |
| `timeout` | string | no | Job timeout (default: `24h`) |
| `retries` | int | no | Retry count on failure |
| `workspace` | string | no | Beaker workspace |
| `budget` | string | no | Beaker budget |
| `description` | string | no | Experiment description |

See `examples/configs/` for more configuration examples.

### Cluster Aliases

| Alias | Clusters |
|-------|----------|
| `h100` | ai2/augusta, ai2/jupiter, ai2/ceres |
| `a100` | ai2/saturn |
| `aus` | ai2/jupiter, ai2/neptune, ai2/saturn, ai2/ceres |
| `80g` | ai2/augusta, ai2/jupiter, ai2/saturn, ai2/ceres |

### Programmatic API

```python
from olmo_eval.launch import BeakerJobConfig, BeakerLauncher

config = BeakerJobConfig(
    name="eval-llama3-mmlu",
    command=["olmo-eval", "run", "-m", "llama3.1-8b", "-t", "mmlu"],
    cluster="h100",
    num_gpus=1,
)

launcher = BeakerLauncher()
experiment = launcher.launch(config)
print(f"Launched: {launcher.beaker.experiment.url(experiment)}")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check src/

# Run tests
pytest
```
