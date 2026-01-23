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

By default, tasks run sequentially. Two parallel execution modes are available:

| Mode | Flag | Backend | Best For |
|------|------|---------|----------|
| Sequential | (default) | Any | Simple runs, debugging |
| Async | `--async` | Any | Multi-GPU batch processing |
| Streaming | `--async-stream` | vLLM only | Maximum throughput |

### Sequential Mode (Default)

Runs one task at a time on a single model instance:

```bash
olmo-eval run -m llama3.1-8b -t mmlu -t gsm8k -t arc
```

### Async Mode (`--async`)

Spawns worker processes that each load the model and process batches. All instances are queued upfront, then processed in parallel:

```bash
# Auto-detect workers from available GPUs
olmo-eval run --async -m llama3.1-8b -t mmlu -t gsm8k -t arc

# Specify number of workers
olmo-eval run --async --num-workers 4 -m llama3.1-8b -t mmlu -t gsm8k

# Multi-GPU models (e.g., 70B on 4 GPUs per worker)
olmo-eval run --async --num-workers 2 --gpus-per-worker 4 -m llama3.1-70b -t mmlu
```

### Streaming Mode (`--async-stream`)

Uses vLLM's AsyncLLMEngine for true continuous batching. Requests are added continuously and results stream back as they complete:

```bash
# Streaming with auto-detected workers
olmo-eval run --async-stream -m llama3.1-8b -t mmlu -t gsm8k -t arc

# Streaming with specific worker config
olmo-eval run --async-stream --num-workers 2 --gpus-per-worker 4 -m llama3.1-70b -t mmlu
```

### Architecture

**Async Mode (`--async`)**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Process                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Prepare Task │───▶│ Prepare Task │───▶│ Prepare Task │      │
│  │   (mmlu)     │    │   (gsm8k)    │    │    (arc)     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                             │                                   │
│                             ▼                                   │
│                  ┌─────────────────────┐                        │
│                  │   Instance Queue    │ (all instances mixed)  │
│                  │ [inst1, inst2, ...] │                        │
│                  └─────────────────────┘                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  Worker 0   │     │  Worker 1   │     │  Worker 2   │
   │  (GPU 0)    │     │  (GPU 1)    │     │  (GPU 2)    │
   │             │     │             │     │             │
   │ Load Model  │     │ Load Model  │     │ Load Model  │
   │ Collect All │     │ Collect All │     │ Collect All │
   │ Batch Infer │     │ Batch Infer │     │ Batch Infer │
   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
          │                   │                   │
          └───────────────────┴───────────────────┘
                              │
                              ▼
                  ┌─────────────────────┐
                  │    Result Queue     │
                  └─────────────────────┘
                              │
                              ▼
                  ┌─────────────────────┐
                  │  Score & Aggregate  │
                  └─────────────────────┘
```

**Streaming Mode (`--async-stream`)**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Process                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Prepare Task │───▶│ Prepare Task │───▶│ Prepare Task │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                             │                                   │
│                             ▼                                   │
│                  ┌─────────────────────┐                        │
│                  │   Instance Queue    │ (streams continuously) │
│                  └─────────────────────┘                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Streaming Worker     │
                 │   (AsyncLLMEngine)     │
                 │                        │
                 │  ┌──────────────────┐  │
                 │  │ Add requests     │◀─┼─── Continuous input
                 │  │ continuously     │  │
                 │  └────────┬─────────┘  │
                 │           │            │
                 │  ┌────────▼─────────┐  │
                 │  │ vLLM Continuous  │  │
                 │  │ Batching Engine  │  │
                 │  └────────┬─────────┘  │
                 │           │            │
                 │  ┌────────▼─────────┐  │
                 │  │ Stream results   │──┼─── Results as they complete
                 │  │ as completed     │  │
                 │  └──────────────────┘  │
                 └────────────────────────┘
                              │
                              ▼
                  ┌─────────────────────┐
                  │  Score immediately  │
                  │  as tasks complete  │
                  └─────────────────────┘
```

**Key Differences:**
- **Async**: Workers collect all instances first, then batch process. Works with any backend.
- **Streaming**: Requests flow continuously through vLLM's async engine. Maximum throughput, earliest completion. vLLM only.

## Key Concepts

### Tasks and Regimes

Tasks live in `olmo_eval/evals/tasks/` and are registered with the `@register` decorator. Regimes are named configuration presets that override task settings:

```python
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks import Task, TaskConfig, register, register_regime

# Register the base task
@register("arc_challenge", lambda: TaskConfig(
    name="arc_challenge",
    data_source=DataSource(path="allenai/ai2_arc", subset="ARC-Challenge"),
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

# Usage: task_name:regime_name
olmo-eval run -m model -t arc_challenge:olmes
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

## Adding New Tasks

This section explains how to create new evaluation tasks.

### Quick Start: Minimal Task Example

Here's a complete, minimal task implementation:

```python
"""Example: Minimal task implementation."""
from collections.abc import Iterator
from typing import Any

from olmo_eval.core import (
    AccuracyMetric,
    Instance,
    LMOutput,
    LMRequest,
    MultipleChoiceFormatter,
    MultipleChoiceScorer,
    RequestType,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.core import Task, TaskConfig, register


class MyTask(Task):
    """Base class for my task."""

    default_source: str = "my-org/my-dataset"

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Load and yield instances from the dataset."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        """Get data source for a specific split."""
        try:
            return self.config.get_data_source(split=split)
        except ValueError:
            return DataSource(path=self.default_source, split=split)

    def process_doc(self, doc: dict[str, Any]) -> Instance:
        """Convert a dataset document to an Instance."""
        return Instance(
            question=doc["question"],
            gold_answer=doc["answer"],
            choices=tuple(doc["choices"]),  # For MC tasks
            metadata={"id": doc["id"]},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Format instance for the language model."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        # Fallback formatting
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract the answer from model output."""
        return output.text.strip()


def _my_task_config() -> TaskConfig:
    return TaskConfig(
        name="my_task",
        data_source=DataSource(path="my-org/my-dataset"),
        formatter=MultipleChoiceFormatter(template="Q: {question}\n\nA:"),
        scorers=(MultipleChoiceScorer(),),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
    )


@register("my_task", _my_task_config)
class MyTaskImpl(MyTask):
    """Registered task implementation."""
    pass
```

### Task Class Overview

| Method | Required | Purpose |
|--------|----------|---------|
| `instances` | Yes | Property that yields `Instance` objects from the dataset |
| `process_doc(doc)` | Yes | Converts a raw document dict into an `Instance` |
| `format_request(instance)` | Yes | Converts an `Instance` into an `LMRequest` for the model |
| `extract_answer(output)` | Yes | Extracts the answer string from `LMOutput` |
| `_build_fewshot()` | No | Override to customize few-shot example loading |
| `score_responses(...)` | No | Override to customize scoring logic |
| `compute_metrics(...)` | No | Override to customize metric computation |

### TaskConfig Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | Required | Task identifier used in CLI |
| `data_source` | `DataSource \| str` | `None` | Dataset source (HuggingFace, S3, GCS, or local path) |
| `fewshot_source` | `DataSource \| str` | `None` | Optional separate source for few-shot examples |
| `formatter` | `Formatter` | `None` | Request formatter |
| `scorers` | `tuple[Scorer, ...]` | `()` | Answer scorers |
| `metrics` | `tuple[Metric, ...]` | `()` | Evaluation metrics |
| `num_fewshot` | `int` | `0` | Number of few-shot examples |
| `fewshot_seed` | `int` | `42` | Random seed for few-shot |
| `limit` | `int \| None` | `None` | Max instances to evaluate |
| `split` | `Split` | `Split.TEST` | Dataset split to use |

### Data Sources

Tasks can load data from multiple sources using `DataSource`:

```python
from olmo_eval.data import DataSource

# HuggingFace datasets
DataSource(path="cais/mmlu", subset="abstract_algebra")

# Local JSONL files
DataSource(path="/path/to/dataset.jsonl")

# S3
DataSource(path="s3://my-bucket/datasets/data.jsonl")

# GCS
DataSource(path="gs://my-bucket/datasets/data.parquet")

# URI strings are also supported in TaskConfig
TaskConfig(
    name="my_task",
    data_source="hf://cais/mmlu?subset=abstract_algebra",
)
```

### Common Patterns

**Multiple Choice Tasks:**
```python
formatter=MultipleChoiceFormatter(template="Question: {question}\n\nAnswer:")
scorers=(MultipleChoiceScorer(),)
metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),)
```

**Generation Tasks (exact match):**
```python
formatter=CompletionFormatter(template="{question}")
scorers=(ExactMatchScorer(),)
metrics=(AccuracyMetric(scorer=ExactMatchScorer),)
```

**Tasks with Multiple Subsets** (like MMLU with 57 subjects):
```python
class MMLUTask(Task):
    def __init__(self, config: TaskConfig, subset: str) -> None:
        super().__init__(config)
        self.subset = subset

# Register each subset
@register("mmlu_anatomy", _mmlu_anatomy_config)
class MMLUAnatomy(MMLUTask):
    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config, subset="anatomy")
```

### Adding Variants and Regimes

**Variants** modify how a task is formatted/scored (e.g., `:mc`, `:bpb`):
```python
from olmo_eval.evals.tasks import register_variant

# Register after task is defined
register_variant("my_task", "3shot", num_fewshot=3)
```

**Regimes** are configuration presets (e.g., `:olmes`, `:zero`):
```python
from olmo_eval.evals.tasks import register_regime

register_regime("my_task", "olmes", num_fewshot=5, fewshot_seed=1234)
register_regime("my_task", "zero", num_fewshot=0)
```

Usage: `olmo-eval run -t my_task:3shot:olmes`

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
olmo-eval beaker launch -n "eval-llama3-mmlu" -m llama3.1-8b -t mmlu

# Multiple tasks
olmo-eval beaker launch -n "eval-llama3-suite" \
    -m llama3.1-8b \
    -t mmlu -t gsm8k -t hellaswag

# Large model with multiple GPUs
olmo-eval beaker launch \
    --name "eval-70b-full" \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --task mmlu --task gsm8k --task arc \
    --cluster h100 \
    --gpus 4 \
    --priority high \
    --timeout 48h

# Preview the Beaker spec without launching
olmo-eval beaker launch -n "test" -m llama3.1-8b -t arc_easy --dry-run
```

### Multiple Models

Run the same suite across multiple models by specifying `-m` multiple times.
Each model will be launched as a separate experiment:

```bash
# Compare two models on the same tasks
olmo-eval beaker launch -n "eval-compare" \
    -m llama3.1-8b \
    -m olmo-2-7b \
    -t mmlu -t gsm8k -t hellaswag

# Creates 2 experiments:
#   eval-compare-llama3.1-8b: runs all tasks on llama3.1-8b
#   eval-compare-olmo-2-7b:   runs all tasks on olmo-2-7b

# Combine with per-task priorities (creates model x priority experiments)
olmo-eval beaker launch -n "eval-full" \
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
olmo-eval beaker launch -n "eval-suite" -m llama3.1-8b \
    -t "mmlu@high" \
    -t "gsm8k@normal" \
    -t "arc@low"

# Creates 3 experiments:
#   eval-suite-high:   runs mmlu at high priority
#   eval-suite-normal: runs gsm8k at normal priority
#   eval-suite-low:    runs arc at low priority

# With task regimes (@ comes after ::)
olmo-eval beaker launch -n "eval" -m llama3.1-8b -t "mmlu::olmes@high"

# Tasks without @priority use the --priority flag (default: normal)
olmo-eval beaker launch -n "eval" -m llama3.1-8b -t mmlu -t gsm8k --priority high
```

### Experiment Groups

Organize multiple experiments into a Beaker group for result aggregation:

```bash
# Launch with grouping
olmo-eval beaker launch -n "benchmark-v1" --group "benchmark-2024" \
    -m llama3.1-8b -m olmo-2-7b \
    -t mmlu -t gsm8k -t hellaswag

# Creates experiments and adds them to "benchmark-2024" group
# Output:
#   Launched: benchmark-v1-llama3.1-8b -> https://beaker.org/ex/...
#   Launched: benchmark-v1-olmo-2-7b -> https://beaker.org/ex/...
#   Group: Added 2 experiment(s) to 'benchmark-2024'

# Check group status and results
olmo-eval beaker group info benchmark-2024

# Show detailed task info
olmo-eval beaker group info benchmark-2024 --verbose

# Wait for completion and export as CSV
olmo-eval beaker group info benchmark-2024 --wait --format csv > results.csv

# Export as JSON
olmo-eval beaker group info benchmark-2024 --format json
```

### Runtime Backend Installation

Docker images do NOT include inference backends (vllm, transformers, litellm) by default. Install them at runtime when launching jobs using optional dependency group names:

```bash
# Install vLLM backend
olmo-eval beaker launch -n "eval-vllm" -m llama3.1-8b -t mmlu --backends vllm

# Install HuggingFace transformers backend
olmo-eval beaker launch -n "eval-hf" -m llama3.1-8b -t mmlu --backends hf

# Install multiple backends
olmo-eval beaker launch -n "eval-multi" -m llama3.1-8b -t mmlu \
    --backends vllm \
    --backends hf

# Short flag
olmo-eval beaker launch -n "eval-vllm" -m llama3.1-8b -t mmlu -b vllm
```

Available backend groups (defined in `pyproject.toml`):
- `vllm` - vLLM inference engine (includes `vllm[runai]` for S3 model loading)
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
olmo-eval beaker launch -f eval_config.yaml --dry-run

# Override specific values
olmo-eval beaker launch -f eval_config.yaml --gpus 4 --priority high

# Add additional models via CLI
olmo-eval beaker launch -f eval_config.yaml -m olmo-2-7b
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
| `beaker_image` | string | no | Container image to use (config-only) |
| `groups` | list | no | Beaker groups to add experiments to |
| `backends` | list | no | Backends to install at runtime (e.g., `["vllm"]`) |
| `use_async` | bool | no | Enable parallel task execution (default: `false`) |
| `use_async_stream` | bool | no | Enable streaming async with vLLM (default: `false`) |
| `num_workers` | int | no | Number of workers for async modes |
| `gpus_per_worker` | int | no | GPUs per worker for async modes (default: `1`) |
| `description` | string | no | Experiment description (config-only) |

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

Images are tagged with CUDA and PyTorch versions: `cu{version}-trc{version}-{arch}`

```bash
# Build with defaults
./scripts/build_image.sh

# Specific CUDA + PyTorch version
./scripts/build_image.sh --cuda-version 12.8.1 --torch-version 2.9.0

# Production build
./scripts/build_image.sh --platform linux/amd64

# See supported CUDA+PyTorch pairs
./scripts/build_image.sh --help
```

**Supported CUDA versions**: 12.6.1, 12.8.0, 12.8.1, 12.9.1
**PyTorch version**: Configurable via `--torch-version`
**Configuration**: See `scripts/build_config.sh`

### What's in the Image

The image contains:
- Python 3.12 (via uv)
- PyTorch with CUDA support
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
olmo-eval beaker launch -n "eval" -m llama3.1-8b -t mmlu --backends vllm

# Install multiple backends
olmo-eval beaker launch -n "eval" -m llama3.1-8b -t mmlu \
  --backends vllm \
  --backends hf

# Or manually inside container (runai extras enable S3 model loading)
uv pip install -e '.[vllm]'  # includes vllm[runai]
```

### Pushing to Beaker

```bash
# Push most recent build
./scripts/beaker/push_beaker_image.sh

# Preview without pushing
./scripts/beaker/push_beaker_image.sh --dry-run
```

The script auto-detects the image name from the tag (e.g., `olmo-eval-cu128-trc291-amd64`)

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run linter
ruff check src/

# Run tests
pytest
```
