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

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check src/

# Run tests
pytest
```
