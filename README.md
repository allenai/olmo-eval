# olmo-eval

[![CI](https://github.com/allenai/olmo-eval-internal/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/allenai/olmo-eval-internal/actions/workflows/ci.yml)
![Alpha](https://img.shields.io/badge/status-alpha-orange)

Evaluation toolkit for OLMo and other language models.

> **Warning**
> This project is in alpha. APIs may change without warning.

## Quick Start

```bash
# Development setup (includes pre-commit hooks)
make setup

# Full setup with beaker + storage (for launching jobs and fetching results)
make setup-all

# Or with specific extras
make setup EXTRAS=beaker

# List available commands
olmo-eval --help

# List model presets
olmo-eval models

# List task suites
olmo-eval suites

# List tasks and their regimes
olmo-eval tasks

# Run evaluation (dry run)
olmo-eval run -m llama3.1-8b -t arc_challenge:olmes --dry-run

# Run evaluation
olmo-eval run -m olmo-2-7b -t olmes_core --limit 100

```

## Key Concepts

The evaluation framework is built around these core abstractions:

| Abstraction | Description |
|-------------|-------------|
| **Task** | Defines a single evaluation (data loading, formatting, scoring) |
| **Suite** | Groups tasks and/or nested suites with aggregation |
| **Formatter** | Converts instances into LM requests |
| **Scorer** | Scores individual instance/output pairs |
| **Metric** | Aggregates scores into final metrics |

### Tasks

Tasks define how to load data, format prompts, and score outputs. Register with `@register`:

```python
from olmo_eval.evals.tasks import Task, TaskConfig, register

@register("my_task", lambda: TaskConfig(
    name="my_task",
    data_source="hf://dataset/path",
    formatter=MultipleChoiceFormatter(),
    scorers=(MultipleChoiceScorer(),),
    metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
))
class MyTask(Task): ...
```

**Regimes** are named presets that override task settings (e.g., few-shot count):

```python
from olmo_eval.evals.tasks import register_regime

register_regime("my_task", "olmes", num_fewshot=5, fewshot_seed=42)
# Usage: olmo-eval run -m model -t my_task:olmes
```

### Suites

Suites group multiple tasks for batch evaluation:

```python
from olmo_eval.evals.suites import Suite, register

register(Suite(
    name="my_suite",
    tasks=("task_a::olmes", "task_b::olmes", "task_c::olmes"),
))
```

#### Aggregation

Suites support different strategies for combining task results:

| Strategy | Description |
|----------|-------------|
| `AVERAGE` | Simple average of all task scores (default) |
| `AVERAGE_OF_AVERAGES` | Average over child suite averages (equal weight per child) |
| `DISPLAY_ONLY` | Display child results without computing suite average |
| `NONE` | No aggregation - just collect individual task results |

**Average of Averages Example:**

```python
from olmo_eval.evals.suites import Suite, AggregationStrategy, register

# Nested suite with 3 tasks
multilingual_code = Suite(
    name="multilingual_code",
    tasks=("mbpp_python", "mbpp_java", "mbpp_rust"),
    aggregation=AggregationStrategy.AVERAGE,
)

# Parent suite using average of averages
register(Suite(
    name="code_eval",
    tasks=(
        "humaneval",        # Single task (score: 0.80)
        multilingual_code,  # Nested suite with 3 tasks (scores: 0.40, 0.50, 0.60)
    ),
    aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
))

# Results:
# - humaneval: 0.80
# - multilingual_code average: (0.40 + 0.50 + 0.60) / 3 = 0.50
#
# AVERAGE_OF_AVERAGES: (0.80 + 0.50) / 2 = 0.65
# vs AVERAGE:          (0.80 + 0.40 + 0.50 + 0.60) / 4 = 0.575
```

Note: Currently `AVERAGE_OF_AVERAGES` gives each child equal weight regardless of how many tasks it contains. Custom weighting may be supported in the future.

### Formatters

Formatters convert instances into LM requests. Available formatters:

| Formatter | Description |
|-----------|-------------|
| `CompletionFormatter` | Text completion with template |
| `ChatFormatter` | Chat messages (system/user/assistant) |
| `MultipleChoiceFormatter` | MC with continuations for logprob scoring |
| `PPLFormatter` | Perplexity/BPB evaluation |

### Scorers

Scorers compute a score for each instance/output pair. Available scorers:

| Scorer | Description |
|--------|-------------|
| `ExactMatchScorer` | Exact string match (1.0 or 0.0) |
| `MultipleChoiceScorer` | Compare selected choice index/letter |
| `F1Scorer` | Token-level F1 score |
| `BitsPerByteScorer` | Bits per byte from logprobs |
| `CodeExecutionScorer` | Execute code against test cases |

### Metrics

Metrics aggregate scores across responses. Available metrics:

| Metric | Description |
|--------|-------------|
| `AccuracyMetric` | Mean accuracy for a scorer |
| `F1Metric` | Mean F1 score |
| `BPBMetric` | Byte-weighted bits per byte |
| `PassAtKMetric` | Pass@k for code generation |

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

Install with the Beaker extra:

```bash
make setup EXTRAS=beaker
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
Models with compatible runtimes (same GPUs, parallelism, cluster, inference provider) are
grouped into a single experiment:

```bash
# Compare two models on the same tasks
olmo-eval beaker launch -n "eval-compare" \
    -m llama3.1-8b \
    -m olmo-2-7b \
    -t mmlu -t gsm8k -t hellaswag

# Creates 1 experiment running both models

# Models with different resource requirements get separate experiments
olmo-eval beaker launch -n "eval-mixed" \
    -m llama3.1-8b --gpus 1 \
    -m llama3.1-70b --gpus 4 \
    -t mmlu -t gsm8k

# Creates 2 experiments (different GPU requirements)
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

Groups logically organize experiments for management and result retrieval:

```bash
# Launch with grouping
olmo-eval beaker launch -n "benchmark-v1" --group "benchmark-2024" \
    -m llama3.1-8b -m olmo-2-7b \
    -t mmlu -t gsm8k -t hellaswag

# Creates experiment and adds it to "benchmark-2024" group

# Check group status and results
olmo-eval beaker group info benchmark-2024

# Show detailed task info
olmo-eval beaker group info benchmark-2024 --verbose

# Wait for completion and export as CSV
olmo-eval beaker group info benchmark-2024 --wait --format csv > results.csv

# Export as JSON
olmo-eval beaker group info benchmark-2024 --format json
```

### Inference Provider Configuration

Docker images do NOT include inference providers (vllm, transformers, litellm) by default.
Each model must specify its provider, which is installed at job startup.

**Via config file (recommended):**

```yaml
name: eval-mixed-providers
models:
  - name_or_path: llama3.1-8b
    provider: vllm
  - name_or_path: gpt-4o
    provider: litellm
tasks:
  - mmlu
cluster: h100
```

**Via CLI inline override:**

```bash
olmo-eval beaker launch -n "eval" -m "llama3.1-8b::provider=vllm" -t mmlu
```

Models with the same provider (and other compatible settings) are grouped into the same experiment.
Models with different providers run in separate experiments.

Available inference providers:
- `vllm` - vLLM inference engine
- `hf` - HuggingFace transformers
- `litellm` - LiteLLM for API-based models (OpenAI, Anthropic, etc.)

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
  - name_or_path: llama3.1-8b
    provider: vllm
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

**Multi-model comparison config**:

```yaml
name: eval-model-comparison
models:
  - name_or_path: llama3.1-8b
    provider: vllm
  - name_or_path: olmo-2-7b
    provider: vllm
  - name_or_path: mistral-7b
    provider: vllm
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
  - name_or_path: llama3.1-8b
    provider: vllm
  - name_or_path: olmo-2-7b
    provider: vllm
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

This creates **3 experiments** (one per priority level, with both models in each):

```
eval-prioritized-high:   models=[llama3.1-8b, olmo-2-7b], tasks=[mmlu, gsm8k]
eval-prioritized-normal: models=[llama3.1-8b, olmo-2-7b], tasks=[hellaswag, arc_challenge]
eval-prioritized-low:    models=[llama3.1-8b, olmo-2-7b], tasks=[winogrande, truthfulqa]
```

**Large model config**:

```yaml
name: eval-70b-full
models:
  - name_or_path: meta-llama/Llama-3.1-70B-Instruct
    provider: vllm
    gpus: 4
tasks:
  - mmlu
  - gsm8k
  - hellaswag
cluster: h100
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
| `models` | list | yes | List of ModelConfig objects (each must have `name_or_path` and `provider`) |
| `tasks` | list | yes | List of task specs (with optional `@priority`) |
| `cluster` | string | yes | Cluster alias or full name |
| `gpus` | int | no | Default GPUs per model instance (default: `1`) |
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
- **Inference providers** - Installed at job startup based on each model's `provider` config

This approach allows you to:
- Use any git commit without rebuilding images
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
- Inference providers like vllm, transformers, litellm (installed at job startup)

### Installing Inference Providers at Runtime

Inference providers are NOT baked into images. They are installed at job startup based on each model's `provider` configuration:

```yaml
# In config file
models:
  - name_or_path: llama3.1-8b
    provider: vllm  # Installs vllm at job startup
  - name_or_path: gpt-4o
    provider: litellm  # Installs litellm at job startup
```

```bash
# Or via CLI inline override
olmo-eval beaker launch -n "eval" -m "llama3.1-8b::provider=vllm" -t mmlu

# Manual installation inside container
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

## Querying Results

Evaluation results are stored in PostgreSQL and can be queried via the CLI.

### Basic Queries

```bash
# Query by experiment ID
olmo-eval results query --experiment exp_001

# Query by model
olmo-eval results query --model llama3.1-8b

# Query by task (shows comparison matrix)
olmo-eval results query --task mmlu --task gsm8k

# Combine filters
olmo-eval results query --model llama3.1-8b --task mmlu --format json
```

### Instance-Level Predictions

Include `--instances` to retrieve instance-level predictions:

```bash
# Get instances for an experiment
olmo-eval results query --experiment exp_001 --task mmlu --instances --format json

# Paginate through large result sets using keyset pagination
olmo-eval results query --task mmlu --instances --limit 1000 --format json

# Get next page using last_id from previous response
olmo-eval results query --task mmlu --instances --limit 1000 --after-id 1000 --format json
```

JSON output includes pagination metadata:
```json
{
  "experiments": [...],
  "pagination": {
    "last_id": 12345,
    "has_more": true
  }
}
```

### Output Formats

| Format | Flag | Description |
|--------|------|-------------|
| Table | `--format table` | Rich terminal tables (default) |
| JSON | `--format json` | Structured JSON with pagination metadata |
| CSV | `--format csv` | CSV output to stdout |

### Database Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLMO_EVAL_DB_HOST` | `localhost` | Database host |
| `OLMO_EVAL_DB_PORT` | `5432` | Database port |
| `OLMO_EVAL_DB_NAME` | `olmo_eval` | Database name |
| `OLMO_EVAL_DB_USER` | `postgres` | Database user |
| `OLMO_EVAL_DB_PASSWORD` | `postgres` | Database password |

## Advanced Usage

### Parallel Execution

By default, tasks run sequentially. Two parallel execution modes are available for faster evaluation:

| Mode | Flag | Backend | Best For |
|------|------|---------|----------|
| Sequential | (default) | Any | Simple runs, debugging |
| Async | `--async` | Any | Multi-GPU batch processing |
| Streaming | `--async-stream` | vLLM only | Generative tasks only |

**Sequential Mode (Default)** - Runs one task at a time:

```bash
olmo-eval run -m llama3.1-8b -t mmlu -t gsm8k -t arc
```

**Async Mode (`--async`)** - Spawns worker processes that each load the model and process batches in parallel:

```bash
# Auto-detect workers from available GPUs
olmo-eval run --async -m llama3.1-8b -t mmlu -t gsm8k -t arc

# Specify number of workers
olmo-eval run --async --num-workers 4 -m llama3.1-8b -t mmlu -t gsm8k

# Multi-GPU models (e.g., 70B on 4 GPUs per worker)
olmo-eval run --async --num-workers 2 --gpus-per-worker 4 -m llama3.1-70b -t mmlu
```

**Streaming Mode (`--async-stream`)** - Uses vLLM's AsyncLLMEngine for true continuous batching:

```bash
olmo-eval run --async-stream -m llama3.1-8b -t mmlu -t gsm8k -t arc
```

## Development

```bash
# Setup (installs dev dependencies and pre-commit hooks)
make setup

# Run linter
make lint

# Run tests
make test
```
