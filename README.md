# olmo-eval

Evaluation toolkit for OLMo and other language models.

## Quick Start

```bash
# Install
uv pip install -e .

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

# Run tasks in parallel across GPUs (faster)
olmo-eval run --async -m olmo-2-7b -t mmlu -t gsm8k -t arc

# Specify number of workers and GPUs per worker
olmo-eval run --async --num-workers 4 --gpus-per-worker 2 -m llama3.1-70b -t mmlu
```

## Parallel Execution

By default, tasks run sequentially. Use `--async` to run tasks in parallel across multiple GPUs:

```bash
# Sequential (default) - runs one task at a time
olmo-eval run -m llama3.1-8b -t mmlu -t gsm8k -t arc

# Parallel - runs tasks in parallel across available GPUs
olmo-eval run --async -m llama3.1-8b -t mmlu -t gsm8k -t arc

# Control number of workers (default: auto-detect from GPUs)
olmo-eval run --async --num-workers 4 -m llama3.1-8b -t mmlu -t gsm8k

# For multi-GPU models, specify GPUs per worker
olmo-eval run --async --num-workers 2 --gpus-per-worker 4 -m llama3.1-70b -t mmlu
```

## Key Concepts

### Tasks and Regimes

Tasks live in `olmo_eval/evals/tasks/` and are registered with the `@register` decorator. Regimes are named configuration presets that override task settings:

```python
from olmo_eval.evals.tasks import Task, TaskConfig, register, register_regime

# Register the base task
@register("arc_challenge", lambda: TaskConfig(
    name="arc_challenge",
    hf_dataset="allenai/ai2_arc",
    num_fewshot=0,
))
class ARCChallenge(Task): ...

# Register a regime with configuration overrides
register_regime(
    "arc_challenge",
    "olmes",
    num_fewshot=5,
    fewshot_seed=42,
)

# Usage: task_name::regime_name
olmo-eval run -m model -t arc_challenge::olmes
```

Regimes allow you to define reusable evaluation configurations (e.g., few-shot settings, prompts) that can be applied to any task.

### Task Suites

Suites live in `olmo_eval/evals/suites/` and group multiple tasks for batch evaluation:

```python
from olmo_eval.evals.suites import Suite, register

register(Suite(
    name="olmes_core",
    tasks=("arc_easy::olmes", "arc_challenge::olmes", "hellaswag::olmes"),
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
uv pip install 'olmo-eval-internal[beaker]'
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

### Experiment Groups

Organize multiple experiments into a Beaker group for result aggregation:

```bash
# Launch with grouping
olmo-eval launch -n "benchmark-v1" --group "benchmark-2024" \
    -m llama3.1-8b -m olmo-2-7b \
    -t mmlu -t gsm8k -t hellaswag

# Creates experiments and adds them to "benchmark-2024" group
# Output:
#   Launched: benchmark-v1-llama3.1-8b -> https://beaker.org/ex/...
#   Launched: benchmark-v1-olmo-2-7b -> https://beaker.org/ex/...
#   Group: Added 2 experiment(s) to 'benchmark-2024'

# Check results
olmo-eval results --group "benchmark-2024"

# Wait for completion and export as CSV
olmo-eval results --group "benchmark-2024" --wait --format csv > results.csv

# Export as JSON
olmo-eval results --group "benchmark-2024" --format json
```

### Runtime Backend Installation

Docker images do NOT include inference backends (vllm, transformers, litellm) by default. Install them at runtime when launching jobs using optional dependency group names:

```bash
# Install vLLM backend
olmo-eval launch -n "eval-vllm" -m llama3.1-8b -t mmlu --backends vllm

# Install HuggingFace transformers backend
olmo-eval launch -n "eval-hf" -m llama3.1-8b -t mmlu --backends hf

# Install multiple backends
olmo-eval launch -n "eval-multi" -m llama3.1-8b -t mmlu \
    --backends vllm \
    --backends hf

# Short flag
olmo-eval launch -n "eval-vllm" -m llama3.1-8b -t mmlu -b vllm
```

Available backend groups (defined in `pyproject.toml`):
- `vllm` - vLLM inference engine
- `hf` - HuggingFace transformers
- `litellm` - LiteLLM for API-based models

Backends are installed via `uv pip install -e '.[backend]'` at job startup.

### CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-f` | none | YAML config file (CLI args override config values) |
| `--name` | `-n` | required | Experiment name |
| `--model` | `-m` | required | Model name or HuggingFace path (can specify multiple) |
| `--task` | `-t` | required | Task name with optional `@priority` suffix (can specify multiple) |
| `--cluster` | `-c` | required | Cluster alias (`h100`, `a100`, `aus`) or full name |
| `--gpus` | `-G` | `1` | Number of GPUs per model instance |
| `--parallelism` | `-P` | `1` | Number of model instances to run in parallel |
| `--max-gpus-per-node` | | `8` | Maximum GPUs per node (tasks split if exceeded) |
| `--priority` | `-p` | `normal` | Job priority (`low`, `normal`, `high`, `urgent`) |
| `--preemptible` | | `true` | Allow preemption |
| `--timeout` | `-T` | `24h` | Job timeout (e.g., `24h`, `30m`) |
| `--retries` | `-r` | none | Number of retries on failure |
| `--workspace` | `-w` | required | Beaker workspace |
| `--budget` | `-B` | required | Beaker budget |
| `--group` | `-g` | none | Add experiments to Beaker group(s) (can specify multiple) |
| `--backends` | `-b` | none | Backends to install at runtime (can specify multiple) |
| `--async` | `-a` | `false` | Enable parallel task execution |
| `--async-stream` | | `false` | Use vLLM's AsyncLLMEngine for continuous batching |
| `--num-workers` | `-W` | auto | Number of workers for async mode |
| `--gpus-per-worker` | | `1` | GPUs per worker for async mode |
| `--fa3` | | `false` | Use Flash Attention 3 (for Hopper GPUs) |
| `--no-flash-attn` | | `false` | Disable Flash Attention |
| `--dry-run` | `-d` | `false` | Print spec without launching |
| `--follow/--no-follow` | | `true` | Follow logs after launch |

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

**Config with runtime backends**:

```yaml
name: eval-vllm
models:
  - llama3.1-8b
tasks:
  - mmlu
  - gsm8k
backends:
  - vllm
cluster: h100
gpus: 1
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
| `models` | list | yes | List of model names/paths or ModelConfig objects |
| `tasks` | list | yes | List of task specs (with optional `@priority`) |
| `cluster` | string | yes | Cluster alias or full name |
| `gpus` | int | no | GPUs per model instance (default: `1`) |
| `parallelism` | int | no | Model instances to run in parallel (default: `1`) |
| `max_gpus_per_node` | int | no | Max GPUs per node, splits tasks if exceeded (default: `8`) |
| `priority` | string | no | Default priority (default: `normal`) |
| `preemptible` | bool | no | Allow preemption (default: `true`) |
| `timeout` | string | no | Job timeout (default: `24h`) |
| `retries` | int | no | Retry count on failure |
| `workspace` | string | yes | Beaker workspace |
| `budget` | string | yes | Beaker budget |
| `groups` | list | no | Beaker groups to add experiments to |
| `backends` | list | no | Backends to install at runtime (e.g., `["vllm"]`) |
| `use_async` | bool | no | Enable parallel task execution (default: `false`) |
| `num_workers` | int | no | Number of workers for async mode |
| `gpus_per_worker` | int | no | GPUs per worker for async mode (default: `1`) |
| `flash_attn` | int | no | Set to `3` to use Flash Attention 3 |
| `no_flash_attn` | bool | no | Disable Flash Attention (default: `false`) |
| `description` | string | no | Experiment description |

See `examples/configs/` for more configuration examples.

### Cluster Aliases

| Alias | Clusters |
|-------|----------|
| `h100` | ai2/jupiter, ai2/ceres |
| `a100` | ai2/saturn |
| `l40` | ai2/neptune |
| `aus` | ai2/jupiter, ai2/neptune, ai2/saturn, ai2/ceres |
| `aus80g` | ai2/jupiter, ai2/saturn, ai2/ceres |
| `80g` | ai2/jupiter, ai2/saturn, ai2/ceres |

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

## Docker Image Management

Docker images provide the runtime environment (Python, PyTorch, CUDA) but do NOT include:
- **Source code** - Gantry mounts your git repository at runtime
- **Backends** - Install at job startup using `--backends` flag

This approach allows you to:
- Use any git commit without rebuilding images
- Mix and match backend versions per job
- Keep images small and cacheable

### Building Images

Images are tagged with CUDA and PyTorch versions: `cuda{version}-torch{version}-{arch}`

```bash
# Build with defaults (CUDA 12.8.1 + PyTorch 2.9.0)
./scripts/build_image.sh

# Specific CUDA + PyTorch version
./scripts/build_image.sh --cuda-version 12.8.1 --torch-version 2.9.0

# Production build (amd64)
./scripts/build_image.sh --platform linux/amd64

# See supported CUDA+PyTorch pairs
./scripts/build_image.sh --help
```

**Supported CUDA versions**: 12.6.1, 12.8.0, 12.8.1, 12.9.1
**PyTorch version**: Configurable via `--torch-version` (default: 2.9.0)
**Configuration**: See `scripts/build_config.sh`

### What's in the Image

The image contains:
- Python 3.12 (via uv)
- PyTorch with CUDA support
- Flash Attention 2 (pre-installed)
- Flash Attention 3 (pre-built wheel, installed on-demand)
- System dependencies (git, uv, ca-certificates)

The image does NOT contain:
- olmo-eval source code (provided by gantry at runtime)
- olmo-eval dependencies like click, datasets, rich, etc. (installed at job startup)
- Storage backends like boto3, psycopg (installed at job startup if needed)
- Inference backends like vllm, transformers, litellm (installed at job startup)

### Installing Backends at Runtime

Inference backends are NOT baked into images. Install them when launching jobs using optional dependency group names:

```bash
# Install vLLM backend
olmo-eval launch -n "eval" -m llama3.1-8b -t mmlu --backends vllm

# Install multiple backends
olmo-eval launch -n "eval" -m llama3.1-8b -t mmlu \
  --backends vllm \
  --backends hf

# Or manually inside container
uv pip install -e '.[vllm]'
```

### Pushing to Beaker

```bash
# Push most recent build
./scripts/beaker/push_beaker_image.sh

# Preview without pushing
./scripts/beaker/push_beaker_image.sh --dry-run
```

The script auto-detects the image name from the tag (e.g., `olmo-eval-cuda128-amd64`)

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run linter
ruff check src/

# Run tests
pytest
```
