# olmo-eval

[![CI](https://github.com/allenai/olmo-eval-internal/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/allenai/olmo-eval-internal/actions/workflows/ci.yml)
![Alpha](https://img.shields.io/badge/status-alpha-orange)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/allenai/olmo-eval-internal/blob/main/LICENSE)

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

# For agent tasks
make setup EXTRAS=agents

# List available commands
olmo-eval --help

# List model presets
olmo-eval models

# List task suites
olmo-eval suites

# Show tasks in a specific suite
olmo-eval suite-info core

# List tasks and their regimes
olmo-eval tasks

# List harness presets
olmo-eval harnesses

# List external evaluations
olmo-eval external-evals

# Run evaluation (dry run)
olmo-eval run -m llama3.1-8b -t humaneval:3shot --dry-run

# Run evaluation with limit
olmo-eval run -m olmo-2-7b -t humaneval:bpb -o limit=100

```

## Key Concepts

The evaluation framework is built around these core abstractions:

| Abstraction | Description |
|-------------|-------------|
| **Task** | Defines a single evaluation (data loading, formatting, scoring) |
| **Suite** | Groups tasks and/or nested suites with aggregation |
| **Harness** | A model provider configured with specific capabilities |
| **Formatter** | Converts instances into LM requests |
| **Scorer** | Scores individual instance/output pairs |
| **Metric** | Aggregates scores into final metrics |

### Tasks

Tasks define how to load data, format prompts, and score outputs. Register with `@register`:

```python
from olmo_eval.evals.tasks.common import Task, register
from olmo_eval.data import DataSource

@register("my_task")
class MyTask(Task):
    # DataSource specifies path, subset (optional), and split
    data_source = DataSource(path="cais/mmlu", subset="abstract_algebra", split="test")
    ...
```

**Regimes** are named presets that override task settings (e.g., few-shot count):

```python
from olmo_eval.evals.tasks.common import register_regime

register_regime("my_task", "olmes", num_fewshot=5, fewshot_seed=42)
# Usage: olmo-eval run -m model -t my_task:olmes
```

**Runtime Dependencies** allow tasks to specify packages installed at job startup:

```python
@register("code_eval")
class CodeEvalTask(Task):
    data_source = DataSource(path="my-org/code-dataset", split="test")
    dependencies = ["code-sandbox==1.0", "git+https://github.com/user/repo@v2.0"]
    ...
```

### Suites

Suites group multiple tasks for batch evaluation:

```python
from olmo_eval.evals.suites import Suite, register

register(Suite(
    name="my_suite",
    tasks=("task_a:olmes", "task_b:olmes", "task_c:olmes"),
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

Formatters convert instances into LM requests. See `olmo_eval.common.formatters` for available options.

```python
from olmo_eval.common.formatters import MultipleChoiceFormatter, ChatFormatter

# Multiple choice with logprob scoring
formatter = MultipleChoiceFormatter(template="Q: {question}\n\nA:")

# Chat-based formatting
formatter = ChatFormatter(system_prompt="You are a helpful assistant.")
```

### Scorers

Scorers compute a score for each instance/output pair. See `olmo_eval.common.scorers` for available options.

```python
from olmo_eval.common.scorers import ExactMatchScorer, MultipleChoiceScorer

# Exact string match
scorer = ExactMatchScorer()

# Multiple choice comparison
scorer = MultipleChoiceScorer()
```

### Metrics

Metrics aggregate scores across responses. See `olmo_eval.common.metrics` for available options.

```python
from olmo_eval.common.metrics import AccuracyMetric, F1Metric
from olmo_eval.common.scorers import ExactMatchScorer, F1Scorer

# Mean accuracy
metric = AccuracyMetric(scorer=ExactMatchScorer)

# Mean F1 score
metric = F1Metric(scorer=F1Scorer)
```

### Model Presets

Pre-configured model settings in `olmo_eval/common/constants/models.py`:

```python
from olmo_eval.common.constants import get_model_presets

# Returns dict of preset name -> ModelConfig
presets = get_model_presets()
# {
#     "llama3.1-8b": ModelConfig(model="meta-llama/Meta-Llama-3.1-8B"),
#     "olmo-2-7b": ModelConfig(model="allenai/OLMo-2-1124-7B"),
#     ...
# }
```

### Harness

A **Harness** configures a model provider with specific capabilities like tools, system prompts, and backends. It wraps an inference provider and injects configuration into requests, enabling tool-augmented evaluation or multi-turn execution.

**Key concept**: Any task can be run with or without tools—that's determined by the Harness configuration, not the task definition. This allows comparing baseline vs tool-augmented performance on the same task.

#### Using Harness via CLI

```bash
# Run task without tools or backend (baseline)
olmo-eval run -m llama3.1-8b -t simpleqa

# Run task with search tools via harness preset
olmo-eval run -m llama3.1-8b -t simpleqa --harness dr_tulu

# Use a custom harness config file
olmo-eval run -m llama3.1-8b -t simpleqa --harness-config ./my_harness.yaml
```

#### HarnessConfig

Configuration for a harness:

```python
from olmo_eval.harness import HarnessConfig, ProviderConfig, get_harness_preset
from olmo_eval.harness.tools.search import (
    semantic_scholar_search,
    serper_web_search,
    serper_fetch_page,
)

# Get a preset
config = get_harness_preset("dr_tulu")

# Or create custom config with tools
config = HarnessConfig(
    name="my_harness",
    provider=ProviderConfig(model="gpt-4o", kind="litellm"),
    tools=(semantic_scholar_search, serper_web_search, serper_fetch_page),
    system_prompt="You are a helpful assistant with search tools.",
    max_turns=10,
    max_concurrency=8,
    backend="openai_agents",
    required_secrets=("S2_API_KEY", "SERPER_API_KEY"),
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | Required | Harness identifier |
| `provider` | `ProviderConfig` | `ProviderConfig()` | Model provider configuration |
| `tools` | `tuple[Tool \| str, ...]` | `()` | Tool instances or registered tool names |
| `system_prompt` | `str \| None` | `None` | System prompt to inject |
| `tool_choice` | `str` | `"auto"` | Tool selection mode (`auto`, `none`, `required`) |
| `backend` | `str \| None` | `None` | Execution backend (e.g., `openai_agents`) |
| `max_turns` | `int \| None` | `None` | Max turns for multi-turn execution |
| `max_concurrency` | `int \| None` | `None` | Concurrent executions |
| `scoring_concurrency` | `int \| None` | `None` | Max concurrent scoring operations |
| `sandboxes` | `tuple[SandboxConfig, ...]` | `()` | Sandbox configurations for isolated tool execution |
| `backend_kwargs` | `dict[str, Any]` | `{}` | Backend-specific options (e.g., `enable_compaction`) |
| `metrics` | `MetricsConfig \| None` | `None` | Inference metrics collection config |
| `batching` | `BatchConfig \| None` | `None` | Batching strategy configuration |
| `required_secrets` | `tuple[str, ...]` | `()` | Required environment variables |

#### Backends

Backends define how the Harness executes multi-turn requests with tool calling. The backend handles the agentic loop: calling the model, executing tools, and feeding results back.

```bash
# List available backends
olmo-eval backends
```

**When to use a backend:**
- For multi-turn execution with `harness.run()`, you must specify a backend
- For single-turn generation with `harness.generate()`, no backend is needed

```python
# Multi-turn execution requires a backend
config = HarnessConfig(
    name="my_agent",
    provider=ProviderConfig(model="gpt-4o", kind="litellm"),
    tools=(semantic_scholar_search, serper_web_search),
    backend="openai_agents",  # Required for run()
)
harness = Harness(config)
result = await harness.run(request)  # Uses the backend

# Single-turn generation works without a backend
config = HarnessConfig(
    name="simple",
    provider=ProviderConfig(model="gpt-4o", kind="litellm"),
)
harness = Harness(config)
outputs = harness.generate(requests)  # No backend needed
```

#### Inference Metrics

Harness configurations can include `MetricsConfig` to collect inference performance metrics during evaluation:

```python
from olmo_eval.harness import HarnessConfig, ProviderConfig
from olmo_eval.inference.metrics import MetricsConfig

config = HarnessConfig(
    name="with_metrics",
    provider=ProviderConfig(model="llama3.1-8b", kind="vllm_server"),
    metrics=MetricsConfig(
        enabled=True,
        reporters=("file", "db"),  # Save to file and database
        collect_vllm_server=True,  # Poll vLLM server /metrics endpoint
    ),
)
```

**Visualizing Metrics:**

```bash
# Plot metrics from database (requires at least one filter)
olmo-eval metrics plot -G my-benchmark-group
olmo-eval metrics plot -m OLMo-3 --metric throughput

# Show statistics table without interactive plots
olmo-eval metrics plot -e experiment_123 --stats-only
```

When using the `db` reporter, metrics are stored in a PostgreSQL database (default name: `olmo_eval_metrics`). You must configure your own database connection using the `OLMO_EVAL_DB_*` environment variables (see [Database Configuration](#database-configuration)).

#### Defining Tools

Tools combine schema (for the LLM) and implementation (for execution) in a single definition:

```python
from olmo_eval.harness import tool, registered_tool

# Option 1: @tool decorator (local use)
@tool(description="Search the web for information")
async def web_search(query: str) -> str:
    """Search implementation."""
    return await search_api(query)

# Option 2: @registered_tool decorator (global registry, for cross-process use)
@registered_tool(description="Fetch a webpage")
async def fetch_page(url: str) -> str:
    """Fetch implementation."""
    return await fetch_url(url)
```

Tools are automatically registered when using `@registered_tool`, making them available by name in HarnessConfig.

#### Custom Harness Config File

Create a YAML file for custom harness configurations:

```yaml
# my_harness.yaml
name: custom_search
tool_names:
  - semantic_scholar_snippet_search
  - serper_google_webpage_search
system_prompt: |
  You are a research assistant with web search capabilities.
  Use search tools to find accurate information before answering.
max_turns: 15
max_concurrency: 4
required_secrets:
  - S2_API_KEY
  - SERPER_API_KEY
```

```bash
olmo-eval run -m llama3.1-8b -t simpleqa --harness-config my_harness.yaml
```

#### Programmatic Usage

```python
from olmo_eval.harness import Harness, HarnessConfig, ProviderConfig, get_harness_preset
from olmo_eval.harness.tools.search import (
    semantic_scholar_search,
    serper_web_search,
)

# Create harness with preset and provider override
config = get_harness_preset("dr_tulu").with_provider(
    ProviderConfig(model="meta-llama/Llama-3.1-8B-Instruct", kind="vllm")
)
harness = Harness(config)

# Or create from scratch
config = HarnessConfig(
    name="my_harness",
    provider=ProviderConfig(model="gpt-4o", kind="litellm"),
    tools=(semantic_scholar_search, serper_web_search),
    system_prompt="You are a helpful assistant.",
    backend="openai_agents",
)
harness = Harness(config)

# Multi-turn execution with tool calling
result = await harness.run(request, sampling_params)
print(result.trajectory)  # Shows all turns including tool calls
print(result.final_output)  # Final model response
```

## Adding New Tasks

This section explains how to create new evaluation tasks.

### Quick Start: Minimal Task Example

Here's a complete, minimal task implementation:

```python
"""Example: Minimal task implementation."""
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register


@register("my_task")
class MyTask(Task):
    """My task implementation."""

    # DataSource arguments:
    #   path: HuggingFace dataset path (e.g., "cais/mmlu")
    #   subset: Dataset subset/config (e.g., "abstract_algebra")
    #   split: Dataset split (e.g., "test", "validation")
    data_source = DataSource(path="cais/mmlu", subset="abstract_algebra", split="test")

    @property
    def instances(self) -> Iterator[Instance]:
        """Load and yield instances from the dataset."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self.config.get_data_source()
            for doc in loader.load(source):
                self._instances_cache.append(self.process_doc(doc))
        yield from self._instances_cache

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
| `data_source` | `DataSource \| str` | `None` | Dataset source (HuggingFace, S3, GCS, local, or URI string) |
| `fewshot_source` | `DataSource \| str` | `None` | Optional separate source for few-shot examples |
| `formatter` | `Formatter` | `None` | Request formatter |
| `metrics` | `tuple[Metric, ...]` | `()` | Evaluation metrics (scorers are inferred from metrics) |
| `num_fewshot` | `int` | `0` | Number of few-shot examples |
| `fewshot_seed` | `int` | `42` | Random seed for few-shot selection |
| `seed` | `int` | `42` | General random seed for task |
| `limit` | `int \| None` | `None` | Max instances to evaluate |
| `split` | `Split` | `Split.TEST` | Dataset split to use |
| `primary_metric` | `MetricName \| Metric \| None` | `None` | Primary metric for ranking (defaults to single metric if only one) |
| `sampling_params` | `SamplingParams \| None` | `None` | Default sampling parameters for this task |
| `dependencies` | `list[str] \| None` | `None` | Runtime packages to install (e.g., `["pkg==1.0"]`) |

### Data Sources

Tasks can load data from multiple sources using `DataSource`:

```python
from olmo_eval.data import DataSource

# HuggingFace datasets - specify path, subset, and split
DataSource(path="cais/mmlu", subset="abstract_algebra", split="test")

# Using URI string (alternative syntax)
DataSource.from_uri("hf://cais/mmlu?subset=abstract_algebra&split=test")

# Without subset (for datasets that don't have subsets)
DataSource(path="openai_humaneval", split="test")

# With specific data files and revision
DataSource(path="my-org/dataset", data_files="data/test.jsonl", revision="v1.0")

# Local JSONL files
DataSource(path="/path/to/dataset.jsonl")

# S3
DataSource(path="s3://my-bucket/datasets/data.jsonl")

# GCS
DataSource(path="gs://my-bucket/datasets/data.parquet")
```

**DataSource Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `str` | Required | Dataset path (HuggingFace repo, S3/GCS URI, or local path) |
| `subset` | `str \| None` | `None` | Dataset subset/config name |
| `split` | `str` | `"test"` | Dataset split |
| `data_files` | `str \| None` | `None` | Specific data files to load |
| `revision` | `str \| None` | `None` | Dataset revision/version |

### Common Patterns

**Multiple Choice Tasks:**
```python
formatter=MultipleChoiceFormatter(template="Question: {question}\n\nAnswer:")
metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),)
```

**Generation Tasks (exact match):**
```python
formatter=CompletionFormatter(template="{question}")
metrics=(AccuracyMetric(scorer=ExactMatchScorer),)
```

**Tasks with Multiple Subsets** (like MMLU with 57 subjects):
```python
# Base class with shared logic
class MMLUTask(Task):
    ...

# Register each subset - the subset is specified in DataSource
@register("mmlu_anatomy")
class MMLUAnatomy(MMLUTask):
    data_source = DataSource(path="cais/mmlu", subset="anatomy", split="test")

@register("mmlu_physics")
class MMLUPhysics(MMLUTask):
    data_source = DataSource(path="cais/mmlu", subset="high_school_physics", split="test")
```

### Adding Variants and Regimes

**Variants** modify how a task is formatted/scored (e.g., `:mc`, `:bpb`):
```python
from olmo_eval.evals.tasks.common import register_variant

# Register after task is defined
register_variant("my_task", "bpb", formatter=PPLFormatter(), metrics=(BPBMetric(scorer=BitsPerByteScorer),))
```

**Regimes** are configuration presets (e.g., `:olmes`, `:zero`):
```python
from olmo_eval.evals.tasks.common import register_regime

register_regime("my_task", "olmes", num_fewshot=5, fewshot_seed=1234)
register_regime("my_task", "3shot", num_fewshot=3)
```

Usage: `olmo-eval run -t my_task:bpb:3shot`

## Tool-Augmented Evaluation

olmo-eval supports evaluating models with tool use through the **Harness** abstraction. This enables comparing baseline model performance against tool-augmented performance on the same tasks.

### Recommended Approach: Harness

The **Harness** is the preferred way to add tools to evaluations. It separates tool configuration from task definition, allowing any task to be run with or without tools:

```bash
# Baseline evaluation (no tools)
olmo-eval run -m llama3.1-8b -t simpleqa

# Same task with search tools
olmo-eval run -m llama3.1-8b -t simpleqa --harness dr_tulu
```

See the [Harness](#harness) section above for full documentation on:
- Creating custom harness configurations
- Defining tools with the `@tool` decorator
- Programmatic usage

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

# Query by experiment group
olmo-eval results query -G my-benchmark-group --format json

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

#### AI2 Users (Recommended)

Set these two environment variables to connect to the shared database:

```bash
export OLMO_EVAL_DB_HOST="<database-host>"
export OLMO_EVAL_DB_SECRET_ARN="arn:aws:secretsmanager:us-west-2:..."
```

The password is automatically fetched from AWS Secrets Manager on first connection.
This requires AWS credentials configured (via `~/.aws/credentials` or environment variables).

#### All Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLMO_EVAL_DB_HOST` | `localhost` | Database host |
| `OLMO_EVAL_DB_PORT` | `5432` | Database port |
| `OLMO_EVAL_DB_NAME` | `olmo_eval` | Database name |
| `OLMO_EVAL_DB_USER` | `postgres` | Database user |
| `OLMO_EVAL_DB_PASSWORD` | - | Database password (use this OR `OLMO_EVAL_DB_SECRET_ARN`) |
| `OLMO_EVAL_DB_SECRET_ARN` | - | AWS Secrets Manager ARN for password (fetched on auth failure) |

## Advanced Usage

### Multi-GPU and Tool-Augmented Evaluation

```bash
# Basic evaluation
olmo-eval run -m llama3.1-8b -t mmlu -t gsm8k -t arc

# Large models with multi-GPU tensor parallelism
olmo-eval run -m llama3.1-70b -t mmlu --num-gpus 4

# Tool-augmented evaluation with harness
olmo-eval run -m llama3.1-8b -t simpleqa --harness dr_tulu
```

## Debugging and Inspection

olmo-eval provides tools for inspecting tasks, requests, and responses at various stages of evaluation.

### Task Inspection (`olmo-eval task inspect`)

Inspect task instances without running evaluation:

```bash
# View raw instance data
olmo-eval task inspect arc_easy

# View multiple instances
olmo-eval task inspect arc_easy -n 5 --skip 10

# View the LM request that will be sent to the model
olmo-eval task inspect mmlu:olmes --request

# View formatted prompt with chat template applied
olmo-eval task inspect humaneval -T meta-llama/Llama-3.1-8B-Instruct --formatted

# View tokenized representation
olmo-eval task inspect humaneval -T meta-llama/Llama-3.1-8B-Instruct --tokens

# Export as JSON for programmatic use
olmo-eval task inspect arc_easy --json
```

| Option | Description |
|--------|-------------|
| `-n, --count` | Number of instances to display |
| `-s, --skip` | Number of instances to skip |
| `--instance` | Show instance details (default if no other flags) |
| `--request` | Show the LM request |
| `-T, --tokenizer` | Tokenizer for formatting/tokenization |
| `--formatted` | Show prompt after template applied (requires `-T`) |
| `--tokens` | Show token array (requires `-T`) |
| `--max-tokens` | Max tokens to display (0 for no limit) |
| `--max-chars` | Max chars for formatted prompt (0 for no limit) |
| `--max-string-length` | Max chars for instance field values (0 for no limit) |
| `--json` | Output as JSON |

### Runtime Inspection Flags

Inspect data during evaluation runs with `olmo-eval run`:

```bash
# Enable all inspection flags at once
olmo-eval run -m llama3.1-8b -t mmlu --inspect

# Or use individual flags for specific inspection
olmo-eval run -m llama3.1-8b -t mmlu --inspect-instance --inspect-request

# Inspect the response after model generation
olmo-eval run -m llama3.1-8b -t mmlu --inspect-response

# Combine multiple inspection flags
olmo-eval run -m llama3.1-8b -t mmlu \
    --inspect-instance \
    --inspect-request \
    --inspect-response
```

| Flag | Description |
|------|-------------|
| `--inspect` | Enable all inspection flags below |
| `--inspect-instance` | Print the first instance of each task before running |
| `--inspect-request` | Print the first LM request before model generation |
| `--inspect-formatted` | Show formatted prompt (after chat template applied) |
| `--inspect-tokens` | Show token array before evaluation |
| `--inspect-response` | Print the first response after model generation |

### Mock Provider for Testing

Use the `mock` provider to test inspection tools without loading a real model:

```bash
# Quick inspection without vLLM or PyTorch
olmo-eval run -m mock -t humaneval:3shot:bpb --inspect-request

# Dry run with mock to preview configuration
olmo-eval run -m mock -t mmlu --dry-run
```

## External Evals

External evals are standalone evaluations that run outside the normal task pipeline.

### Defining an External Eval

```python
from typing import Any

from olmo_eval.evals.external import SandboxedExternalEval, ExternalEvalResult, register_external_eval

class MyBenchmarkExternalEval(SandboxedExternalEval):
    """My benchmark evaluation."""

    name = "my_benchmark"
    description = "Evaluates model on my benchmark"
    timeout_seconds = 3600
    required_secrets = ("MY_API_KEY",)

    @property
    def sandbox_image(self) -> str:
        return "my-benchmark:latest"

    @property
    def working_dir(self) -> str:
        return "/workspace"

    @property
    def setup_command(self) -> tuple[str, ...]:
        return ("pip install -r requirements.txt",)

    @property
    def arguments(self) -> dict[str, tuple[str, Any | None]]:
        # Returns dict of arg_name -> (description, default_value)
        return {"subset": ("Which subset to evaluate", "default")}

    async def execute(self, provider, args, output_dir, container_runtime):
        # Run benchmark in sandbox
        result = await self.run_in_sandbox(provider, args, output_dir)
        return ExternalEvalResult(
            name=self.name,
            metrics={"accuracy": result.score},
            success=True,
        )

# Register the eval
register_external_eval(MyBenchmarkExternalEval())
```

### Running External Evals

```bash
# List available external evals
olmo-eval external-evals

# Run an external eval
olmo-eval run-external -e my_benchmark --model llama3.1-8b -A subset=test
```

### ExternalEvalResult

External evals return structured results:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Eval identifier |
| `metrics` | `dict[str, float]` | Evaluation metrics |
| `metadata` | `dict` | Additional metadata |
| `success` | `bool` | Whether the eval completed successfully |
| `error` | `str \| None` | Error message if failed |
| `duration_seconds` | `float` | Execution time |
| `raw_output` | `str \| None` | Raw stdout/stderr from the evaluation |
| `predictions` | `list` | Instance-level predictions |

## Sandboxes

Sandboxes provide isolated execution environments for code execution, tool use, and external evals.

### Configuration

```python
from olmo_eval.harness.sandbox import SandboxConfig, SandboxMode, Capability

config = SandboxConfig(
    image="python:3.12-slim",
    mode=SandboxMode.DOCKER,
    command_timeout=30.0,
    startup_timeout=60.0,
    instances=4,  # Run 4 parallel executors
    working_dir="/workspace",
    environment=(("MY_VAR", "value"),),
    volumes=(("/host/path", "/container/path"),),
    capabilities=Capability.BASH | Capability.PYTHON,  # Union of frozensets
)
```

### SandboxConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | `str` | Required | Container image |
| `mode` | `SandboxMode` | `DOCKER` | `LOCAL`, `DOCKER`, or `MODAL` |
| `container_runtime` | `str` | `"podman"` | `"docker"` or `"podman"` |
| `command_timeout` | `float` | `30.0` | Timeout per command (seconds) |
| `startup_timeout` | `float` | `60.0` | Container startup timeout |
| `instances` | `int` | `1` | Number of parallel executors |
| `working_dir` | `str` | `"/workspace"` | Working directory in container |
| `environment` | `tuple` | `()` | Environment variables |
| `volumes` | `tuple` | `()` | Volume mounts (host, container) |
| `capabilities` | `frozenset[str]` | `Capability.DEFAULT` | Capabilities like `Capability.BASH`, `Capability.PYTHON` |
| `remove_container` | `bool` | `True` | Remove container after use |
| `docker_args` | `tuple[str, ...]` | `()` | Additional Docker/Podman arguments |
| `log_dir` | `str \| None` | `None` | Directory for container logs |
| `exec_shell` | `tuple[str, ...] \| None` | `None` | Custom shell for command execution |
| `enable_diagnostics` | `bool` | `True` | Run background diagnostics monitor |

### Using SandboxManager

The `SandboxManager` manages multiple executors with capability-based routing:

```python
from olmo_eval.harness.sandbox import SandboxConfig, SandboxManager, SandboxMode, Capability

configs = [
    SandboxConfig(image="python:3.12", mode=SandboxMode.DOCKER, capabilities=Capability.PYTHON, instances=2),
    SandboxConfig(image="ubuntu:22.04", mode=SandboxMode.DOCKER, capabilities=Capability.BASH),
]

manager = SandboxManager(configs, owner="my-scorer")
await manager.start()

# Execute with specific capability - routes to matching executor
result = await manager.execute_with_capabilities(
    "print('hello')",
    Capability.PYTHON
)

# Round-robin across matching executors
results = await asyncio.gather(*[
    manager.execute_with_capabilities(cmd, Capability.PYTHON)
    for cmd in commands
])

await manager.stop()
```

### Sandbox Modes

| Mode | Description |
|------|-------------|
| `LOCAL` | Run commands locally (development only) |
| `DOCKER` | Run in Docker/Podman containers |
| `MODAL` | Run on Modal cloud platform |

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
    --timeout 48h

# Preview the Beaker spec without launching
olmo-eval beaker launch -n "test" -m llama3.1-8b -t arc_easy --dry-run

# With a harness preset for tool-augmented evaluation
olmo-eval beaker launch -n "eval-with-tools" \
    -m llama3.1-8b \
    -t simpleqa \
    --harness dr_tulu \
    --cluster h100

# With inspection flags for debugging
olmo-eval beaker launch -n "debug-eval" \
    -m llama3.1-8b \
    -t mmlu -o limit=10 \
    --inspect-request \
    --inspect-response \
    --cluster h100

# Run external evaluations
olmo-eval beaker launch -n "external-eval" \
    -E my_benchmark \
    -m llama3.1-8b \
    -A subset=test \
    --cluster h100
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

# With task regimes (@ comes after regime)
olmo-eval beaker launch -n "eval" -m llama3.1-8b -t "mmlu:olmes@high"

# Tasks without @priority use the config file priority (default: normal)
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

# Watch experiment logs
olmo-eval beaker watch -e <experiment-id>

# Cancel all experiments in a group
olmo-eval beaker group cancel benchmark-2024

# List groups in a workspace
olmo-eval beaker group list -w <workspace>
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

### CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-f` | none | YAML config file (CLI args override config values) |
| `--name` | `-n` | auto | Experiment name (auto-generated from model/tasks if not provided) |
| `--model` | `-m` | required | Model name or HuggingFace path (can specify multiple) |
| `--task` | `-t` | required | Task name with optional `@priority` suffix (can specify multiple) |
| `--harness` | `-H` | `default` | Harness preset name |
| `--override` | `-o` | none | Override for preceding `-m` or `-t` or `-H` (can specify multiple) |
| `--cluster` | `-c` | required | Cluster alias (`h100`, `a100`, `aus`) or full name |
| `--gpus` | `-G` | auto | Number of GPUs (defaults to 1 for GPU providers, 0 otherwise) |
| `--max-gpus-per-node` | | `8` | Maximum GPUs per node (tasks split if exceeded) |
| `--priority` | `-p` | `normal` | Job priority (`low`, `normal`, `high`, `urgent`) |
| `--preemptible` | | `true` | Allow preemption |
| `--timeout` | `-T` | `24h` | Job timeout (e.g., `24h`, `30m`) |
| `--retries` | `-r` | none | Number of retries on failure |
| `--workspace` | `-w` | required | Beaker workspace |
| `--budget` | `-B` | required | Beaker budget |
| `--image` | `-I` | default | Custom Beaker image |
| `--group` | `-g` | auto | Add experiments to Beaker group(s) (auto-generated if not specified) |
| `--harness` | `-H` | none | Harness preset name |
| `--external-eval` | `-E` | none | External evaluation name(s) to run instead of tasks |
| `--eval-arg` | `-A` | none | Arguments for external evals (`key=value`) |
| `--provider-kwarg` | `-K` | none | Provider kwargs for external evals (`key=value`) |
| `--uv-cache-dir` | | default | UV cache directory for package downloads |
| `--dry-run` | `-d` | `false` | Print spec without launching |
| `--yes` | `-y` | `false` | Skip confirmation prompt |
| `--follow/--no-follow` | | `true` | Follow logs after launch |
| `--secret-env` | | none | Map Beaker secret to env var (`SECRET:VAR`) |
| `--aws-credentials` | | auto | Inject AWS credentials (auto-detected from s3:// paths) |
| `--gcp-credentials` | | auto | Inject GCP credentials (auto-detected from gs:// model paths) |
| `--store` | | `false` | Persist results to configured database |

### Per-Task Overrides

Use the `-o/--override` flag to apply configuration overrides to the preceding `-t`:

```bash
# Task overrides (apply to the preceding -t)
olmo-eval beaker launch -n "eval" \
    -m llama3.1-8b \
    -t mmlu -o limit=100 -o num_fewshot=5 \
    -t gsm8k -o limit=50
```

The `-o` flag uses OmegaConf dotlist syntax, supporting:

| Type | Syntax | Example |
|------|--------|---------|
| String | `key=value` | `-o formatter.template="Q: {q}"` |
| Number | `key=123` | `-o limit=100` |
| Boolean | `key=true` | `-o preemptible=false` |
| Nested | `a.b.c=val` | `-o scorer.normalize=true` |
| List | `key=[a,b]` | `-o 'dependencies=[pkg1, pkg2]'` |

**Note:** Quote complex values to prevent shell interpretation:
```bash
# Good - single quotes protect the value
-o 'extra_config={key: value, nested: {a: 1}}'
```

### Secret Environment Overrides

By default, Beaker secrets are mapped using the pattern `{username}_{ENV_VAR}` (e.g., `ai2-tylerm_OPENAI_API_KEY`).
Use `--secret-env` to override this with a custom Beaker secret name:

```bash
# Use a team-shared secret instead of your personal secret
olmo-eval beaker launch -n "eval" -m gpt-4o -t mmlu \
    --secret-env team-openai-key:OPENAI_API_KEY

# Multiple secret overrides
olmo-eval beaker launch -n "eval" -m gpt-4o -t simpleqa \
    --secret-env team-openai-key:OPENAI_API_KEY \
    --secret-env shared-serper-key:SERPER_API_KEY
```

Format: `BEAKER_SECRET_NAME:ENV_VAR_NAME`

This is useful for:
- Using team-shared API keys instead of personal secrets
- Testing with different credential sets
- Sharing jobs that use organization-level secrets

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
olmo-eval beaker launch -f eval_config.yaml --gpus 4

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
| `name` | string | no | Experiment name (auto-generated if not provided) |
| `models` | list | yes | List of ModelConfig objects (each must have `name_or_path` and `provider`) |
| `tasks` | list | yes | List of task specs (with optional `@priority`) |
| `cluster` | string | yes | Cluster alias or full name |
| `gpus` | int | no | Default GPUs per model instance (auto-detected based on provider) |
| `max_gpus_per_node` | int | no | Max GPUs per node, splits tasks if exceeded (default: `8`) |
| `priority` | string | no | Default priority (default: `normal`) |
| `preemptible` | bool | no | Allow preemption (default: `true`) |
| `timeout` | string | no | Job timeout (default: `24h`) |
| `retries` | int | no | Retry count on failure |
| `workspace` | string | yes | Beaker workspace |
| `budget` | string | yes | Beaker budget |
| `beaker_image` | string | no | Container image to use (config-only) |
| `groups` | list | no | Beaker groups to add experiments to |
| `harness` | string | no | Harness preset name |
| `description` | string | no | Experiment description (config-only) |

See `examples/beaker/configs/` for more configuration examples.

### Cluster Aliases

```bash
# List available cluster aliases
olmo-eval beaker clusters
```

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

### Pushing Images

```bash
# Push most recent build
./scripts/beaker/push_beaker_image.sh

# Preview without pushing
./scripts/beaker/push_beaker_image.sh --dry-run
```

The script auto-detects the image name from the tag (e.g., `olmo-eval-cu128-trc291-amd64`)

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
# Or via CLI override flag
olmo-eval beaker launch -n "eval" -m llama3.1-8b -o provider.kind=vllm -t mmlu

# Manual installation inside container
uv pip install -e '.[vllm]'  # includes vllm[runai]
```

### Task-Specific Dependencies

Tasks can declare runtime dependencies that are installed at job startup (see [Tasks](#tasks)). Dependencies are automatically merged, deduplicated, and installed after the inference provider.

You can also add or override dependencies via the CLI:

```bash
# Add dependencies to a task via -o flag
olmo-eval beaker launch -n "eval" -m llama3.1-8b \
    -t code_eval -o 'dependencies=["code-sandbox==1.0", "git+https://github.com/user/repo@v2.0"]'

# Dependencies from multiple tasks are merged
olmo-eval beaker launch -n "eval" -m llama3.1-8b \
    -t task_a -o 'dependencies=["pkg1"]' \
    -t task_b -o 'dependencies=["pkg2"]'
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
