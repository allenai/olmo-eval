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

# For agent tasks
make setup EXTRAS=agents

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

**Runtime Dependencies** allow tasks to specify packages installed at job startup:

```python
@register("code_eval", lambda: TaskConfig(
    name="code_eval",
    data_source="my-org/code-dataset",
    dependencies=["code-sandbox==1.0", "git+https://github.com/user/repo@v2.0"],
))
class CodeEvalTask(Task): ...
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

Formatters convert instances into LM requests. See `olmo_eval.core.formatters` for available options.

```python
from olmo_eval.core import MultipleChoiceFormatter, ChatFormatter

# Multiple choice with logprob scoring
formatter = MultipleChoiceFormatter(template="Q: {question}\n\nA:")

# Chat-based formatting
formatter = ChatFormatter(system="You are a helpful assistant.")
```

### Scorers

Scorers compute a score for each instance/output pair. See `olmo_eval.core.scorers` for available options.

```python
from olmo_eval.core import ExactMatchScorer, MultipleChoiceScorer

# Exact string match
scorer = ExactMatchScorer()

# Multiple choice comparison
scorer = MultipleChoiceScorer()
```

### Metrics

Metrics aggregate scores across responses. See `olmo_eval.core.metrics` for available options.

```python
from olmo_eval.core import AccuracyMetric, F1Metric

# Mean accuracy
metric = AccuracyMetric(scorer=ExactMatchScorer)

# Mean F1 score
metric = F1Metric(scorer=F1Scorer)
```

### Model Presets

Pre-configured model settings in `olmo_eval/core/constants/models.py`:

```python
from olmo_eval.core import get_model_presets

# Returns dict of preset name -> ModelConfig
presets = get_model_presets()
# {
#     "llama3.1-8b": ModelConfig(model="meta-llama/Meta-Llama-3.1-8B"),
#     "olmo-2-7b": ModelConfig(model="allenai/OLMo-2-1124-7B"),
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
| `dependencies` | `list[str] \| None` | `None` | Runtime packages to install (e.g., `["pkg==1.0"]`) |

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
metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),)
```

**Generation Tasks (exact match):**
```python
formatter=CompletionFormatter(template="{question}")
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
register_variant("my_task", "bpb", formatter=PPLFormatter(), metrics=(BPBMetric(scorer=BitsPerByteScorer),))
```

**Regimes** are configuration presets (e.g., `:olmes`, `:zero`):
```python
from olmo_eval.evals.tasks import register_regime

register_regime("my_task", "olmes", num_fewshot=5, fewshot_seed=1234)
register_regime("my_task", "3shot", num_fewshot=3)
```

Usage: `olmo-eval run -t my_task:bpb:3shot`

## Agent Tasks

Agent tasks support multi-turn evaluations with tool use, enabling evaluation of models' ability to use tools to complete tasks. They use the OpenAI Agents SDK for orchestration and support vLLM as the inference backend.

### Creating an Agent Task

Agent tasks extend `AgentTask` instead of `Task` and use `AgentTaskConfig` instead of `TaskConfig`:

```python
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from typing import Any

from olmo_eval.core import AccuracyMetric, Instance
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.core import AgentTask, AgentTaskConfig, register


async def my_tool(query: str) -> str:
    """A tool the agent can use.

    Args:
        query: The search query.

    Returns:
        Tool result as a string.
    """
    # Tool implementation
    return f"Result for: {query}"


class MyAgentTask(AgentTask):
    """Agent task with custom tools."""

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self.config.get_data_source()
            for idx, doc in enumerate(loader.load(source)):
                self._instances_cache.append(
                    Instance(
                        question=doc["question"],
                        gold_answer=doc["answer"],
                        metadata={"id": idx},
                    )
                )
        yield from self._instances_cache

    @asynccontextmanager
    async def _get_agent(
        self,
        model: str,
        model_url: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Create agent with tools."""
        from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, function_tool
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=model_url, api_key="EMPTY")
        llm = OpenAIChatCompletionsModel(openai_client=client, model=model)

        # Create tools using function_tool decorator
        tools = [function_tool(strict_mode=False)(my_tool)]

        agent = Agent(
            name="MyAgent",
            instructions=system_prompt or "You are a helpful assistant.",
            model=llm,
            model_settings=ModelSettings(temperature=temperature),
            tools=tools,
        )
        yield agent


def _my_agent_config() -> AgentTaskConfig:
    return AgentTaskConfig(
        name="my_agent_task",
        data_source=DataSource(path="my-org/my-dataset", split="test"),
        metrics=(AccuracyMetric(scorer=MultipleChoiceScorer),),
        system_prompt="You are a helpful assistant with tools.",
        max_turns=10,
        max_concurrency=1,
        required_secrets=("MY_API_KEY",),
    )


@register("my_agent_task", _my_agent_config)
class MyAgent(MyAgentTask):
    pass
```

### AgentTaskConfig Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | `str` | `""` | System prompt for the agent |
| `max_turns` | `int` | `10` | Maximum agent turns |
| `max_concurrency` | `int` | `1` | Concurrent agent executions |
| `required_secrets` | `tuple[str, ...]` | `()` | Required environment variables |
| `tools` | `tuple[ToolSchema, ...]` | `()` | Tool schemas for display |

### Running Agent Tasks

```bash
# Local run
olmo-eval run --task simpleqa_agent --model Qwen/Qwen3-8B -o limit=10

# Beaker launch
olmo-eval beaker launch -n "agent-eval" -m Qwen/Qwen3-8B -t simpleqa_agent -o limit=10
```

### vLLM Configuration

For agent tasks using vLLM, the tool call parser is auto-detected based on model name. You can override it via model overrides:

```bash
olmo-eval run --task simpleqa_agent -m my-model -o tool_call_parser=hermes
```

Supported parsers: `hermes`, `llama3_json`, `mistral`

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
```

### Multiple Models

Run the same suite across multiple models by specifying `-m` multiple times.
Models with compatible runtimes (same cluster, inference provider) are
grouped into a single experiment:

```bash
# Compare two models on the same tasks
olmo-eval beaker launch -n "eval-compare" \
    -m llama3.1-8b \
    -m olmo-2-7b \
    -t mmlu -t gsm8k -t hellaswag

# Creates 1 experiment running both models

# Models with different providers get separate experiments
olmo-eval beaker launch -n "eval-mixed" \
    -m llama3.1-8b -o provider.name=vllm \
    -m gpt-4o -o provider.name=litellm \
    -t mmlu -t gsm8k

# Creates 2 experiments (different inference providers)
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

**Via CLI override flag:**

```bash
olmo-eval beaker launch -n "eval" -m llama3.1-8b -o provider.name=vllm -t mmlu
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
| `--override` | `-o` | none | Override for preceding `-m` or `-t` (can specify multiple) |
| `--cluster` | `-c` | required | Cluster alias (`h100`, `a100`, `aus`) or full name |
| `--gpus` | `-G` | `1` | Number of GPUs per model instance |
| `--parallelism` | `-P` | `1` | Number of model instances to run in parallel |
| `--max-gpus-per-node` | | `8` | Maximum GPUs per node (tasks split if exceeded) |
| `--preemptible` | | `true` | Allow preemption |
| `--timeout` | `-T` | `24h` | Job timeout (e.g., `24h`, `30m`) |
| `--retries` | `-r` | none | Number of retries on failure |
| `--workspace` | `-w` | required | Beaker workspace |
| `--budget` | `-B` | required | Beaker budget |
| `--group` | `-g` | none | Add experiments to Beaker group(s) (can specify multiple) |
| `--runner-type` | `-R` | `sync` | Runner type: `sync`, `async`, `async-stream`, `agent` |
| `--num-workers` | `-W` | auto | Number of workers for async/async-stream modes |
| `--gpus-per-worker` | | `1` | GPUs per worker for async/async-stream modes |
| `--dry-run` | `-d` | `false` | Print spec without launching |
| `--follow/--no-follow` | | `true` | Follow logs after launch |

### Per-Model and Per-Task Overrides

Use the `-o/--override` flag to apply configuration overrides to the preceding `-m` or `-t`:

```bash
# Model overrides (apply to the preceding -m)
olmo-eval beaker launch -n "eval" \
    -m llama3.1-8b -o provider.name=vllm -o provider.package=vllm==0.14.0 \
    -m gpt-4o -o provider.name=litellm \
    -t mmlu -t gsm8k

# Task overrides (apply to the preceding -t)
olmo-eval beaker launch -n "eval" \
    -m llama3.1-8b \
    -t mmlu -o limit=100 -o num_fewshot=5 \
    -t gsm8k -o limit=50

# Mixed model and task overrides
olmo-eval beaker launch -n "eval" \
    -m llama3.1-8b -o provider.name=vllm \
    -m gpt-4o -o provider.name=litellm \
    -t mmlu -o limit=100 \
    -t gsm8k
```

The `-o` flag uses OmegaConf dotlist syntax, supporting:

| Type | Syntax | Example |
|------|--------|---------|
| String | `key=value` | `-o provider.name=vllm` |
| Number | `key=123` | `-o limit=100` |
| Boolean | `key=true` | `-o preemptible=false` |
| Nested | `a.b.c=val` | `-o provider.package=vllm==0.14.0` |
| List | `key=[a,b]` | `-o 'args=[--flag1, --flag2]'` |
| Dict | `key={a: 1}` | `-o 'config={distributed: true}'` |

**Note:** Quote complex values to prevent shell interpretation:
```bash
# Good - single quotes protect the value
-o 'extra_config={key: value, nested: {a: 1}}'
```

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

See `examples/beaker/configs/` for more configuration examples.

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
# Or via CLI override flag
olmo-eval beaker launch -n "eval" -m llama3.1-8b -o provider.name=vllm -t mmlu

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

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLMO_EVAL_DB_HOST` | `localhost` | Database host |
| `OLMO_EVAL_DB_PORT` | `5432` | Database port |
| `OLMO_EVAL_DB_NAME` | `olmo_eval` | Database name |
| `OLMO_EVAL_DB_USER` | `postgres` | Database user |
| `OLMO_EVAL_DB_PASSWORD` | `postgres` | Database password |

## Advanced Usage

### Runner Types

By default, tasks run sequentially. Different runner types are available for various use cases:

| Runner | Flag | Backend | Best For |
|--------|------|---------|----------|
| Sync | (default) | Any | Simple runs, debugging |
| Async | `--runner-type async` | Any | Multi-GPU batch processing |
| Async-Stream | `--runner-type async-stream` | vLLM only | Generative tasks only |
| Agent | `--runner-type agent` | vLLM | Multi-turn agent tasks |

**Sync Mode (Default)** - Runs one task at a time:

```bash
olmo-eval run -m llama3.1-8b -t mmlu -t gsm8k -t arc
```

**Async Mode** - Spawns worker processes that each load the model and process batches in parallel:

```bash
# Auto-detect workers from available GPUs
olmo-eval run --runner-type async -m llama3.1-8b -t mmlu -t gsm8k -t arc

# Specify number of workers
olmo-eval run --runner-type async --num-workers 4 -m llama3.1-8b -t mmlu -t gsm8k

# Multi-GPU models (e.g., 70B on 4 GPUs per worker)
olmo-eval run --runner-type async --num-workers 2 --gpus-per-worker 4 -m llama3.1-70b -t mmlu
```

**Async-Stream Mode** - Uses vLLM's AsyncLLMEngine for true continuous batching:

```bash
olmo-eval run --runner-type async-stream -m llama3.1-8b -t mmlu -t gsm8k -t arc
```

**Agent Mode** - For multi-turn agent tasks with tool use:

```bash
olmo-eval run --runner-type agent -m llama3.1-8b -t simpleqa_agent
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
| `--json` | Output as JSON |

### Runtime Inspection Flags

Inspect data during evaluation runs with `olmo-eval run`:

```bash
# Inspect the first instance and request before running
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

### Beaker Job Inspection

The same inspection flags work with Beaker jobs:

```bash
olmo-eval beaker launch \
    -n "debug-eval" \
    -m Qwen/Qwen3-8B \
    -t mmlu -o limit=10 \
    --inspect-request \
    --inspect-response \
    --cluster h100
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
