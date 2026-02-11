# Add RULER benchmark tasks to olmo-eval

## Summary

This PR migrates the RULER (What's the Real Context Size of Your Long-Context Language Models?) benchmark from the old `oe-eval` framework to the new `olmo-eval` framework.

RULER is a comprehensive benchmark for evaluating long-context language models across 4 task categories (NIAH, multi-hop tracing, aggregation, and QA) with 13 task types and 6 context lengths (4K-131K tokens), totaling **78 evaluation tasks**.

## What's Added

### Core Implementation
- **[src/olmo_eval/evals/tasks/ruler.py](src/olmo_eval/evals/tasks/ruler.py)** - RULER task implementation using new framework interface
  - Dynamically registers all 78 RULER task variants
  - Supports both generation-based (recall) and perplexity-based (BPB) evaluation
  - Compatible with new `Task`, `TaskConfig`, and `Instance` types

### Data & Configuration
- **[src/olmo_eval/data/ruler_tasks.py](src/olmo_eval/data/ruler_tasks.py)** - Programmatic task configuration (137 lines, down from 878 in old framework)
  - 13 base task types with configurable context sizes
  - Clean, maintainable structure with defaults

- **[src/olmo_eval/data/ruler_loader.py](src/olmo_eval/data/ruler_loader.py)** - Data loading utilities ported from HELMET
  - Downloads RULER dataset from HuggingFace (`allenai/ruler_data`)
  - Provides task-specific prompt templates
  - Handles data preprocessing

### Metrics & Scoring
- **[src/olmo_eval/core/metrics/base.py](src/olmo_eval/core/metrics/base.py)** - Added `RecallMetric` for recall-based evaluation (general-purpose)
- **[src/olmo_eval/core/scorers/substring.py](src/olmo_eval/core/scorers/substring.py)** - `SubstringRecallScorer` for substring matching (general-purpose)

### Task Suites
- **[src/olmo_eval/evals/suites/ruler.py](src/olmo_eval/evals/suites/ruler.py)** - 30 task suites for organized evaluation
  - Per-category suites: `ruler_niah__4096`, `ruler_aggregation__8192`, etc. (24 suites)
  - Combined suites: `ruler_all__4096`, `ruler_all__8192`, etc. (6 suites)

## Key Features

### Task Categories (78 tasks total)
1. **NIAH (Needle in a Haystack)** - 54 tasks
   - Single-needle: `niah_s_1`, `niah_s_2`, `niah_s_3`
   - Multi-key: `niah_mk_1`, `niah_mk_2`, `niah_mk_3`
   - Multi-value: `niah_mv`
   - Multi-query: `niah_mq`

2. **Multi-hop Tracing** - 6 tasks
   - Variable tracking: `vt`

3. **Aggregation** - 12 tasks
   - Common word extraction: `cwe`
   - Frequency word extraction: `fwe`

4. **Question Answering** - 12 tasks
   - `qa_1`, `qa_2`

### Context Sizes
Each task type is available at 6 context lengths: **4K, 8K, 16K, 32K, 65K, 131K tokens**

### Evaluation Modes
- **Base tasks**: Generation + recall/exact-match scoring (e.g., `ruler_niah_s_1__4096`)
- **BPB variant**: Perplexity-based evaluation (e.g., `ruler_niah_s_1__4096:bpb`)

## Framework Migration Details

### Old Framework → New Framework
| Component | Old | New |
|-----------|-----|-----|
| Base class | `oe_eval.tasks.base_task.Task` | `olmo_eval.evals.tasks.core.Task` |
| Data iteration | `validation_docs()` | `@property instances` |
| Formatting | `doc_to_text()`, `doc_to_target()` | `format_request()`, `extract_answer()` |
| Metrics | Manual `make_metrics()` | `TaskConfig.metrics` |
| Registration | None | `@register()` decorator |
| Data loading | Custom `download()` | Integrated in task `_load_data()` |

### Architecture Improvements
- **Programmatic configuration**: Base task definitions generate all variants, reducing config from 878 to 137 lines
- **Type safety**: Uses `Instance`, `LMRequest`, `LMOutput`, `SamplingParams` types
- **Modular scorers/metrics**: Separate `RulerRecallScorer` and `RulerRecallMetric`
- **Suite organization**: Hierarchical task grouping with configurable aggregation

## Usage Examples

```bash
# Run a single RULER task
olmo-eval run --model your-model --tasks ruler_niah_s_1__4096

# Run with BPB variant
olmo-eval run --model your-model --tasks ruler_niah_s_1__4096:bpb

# Run all NIAH tasks at 4K context
olmo-eval run --model your-model --tasks ruler_niah__4096

# Run all tasks at 8K context
olmo-eval run --model your-model --tasks ruler_all__8192

# List RULER tasks
olmo-eval tasks | grep ruler_
```

## Testing

All tasks verified to:
- ✅ Register correctly (78 tasks + 30 suites)
- ✅ Instantiate without errors
- ✅ Have proper configuration (metrics, context size, task type)
- ✅ Support BPB variant (78 additional variants)
- ✅ Generate correct task structures from programmatic config

## References

- **Paper**: [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654)
- **Original implementation**: https://github.com/hsiehjackson/RULER
- **Data source**: https://github.com/princeton-nlp/HELMET (ported)
- **Dataset**: `allenai/ruler_data` on HuggingFace
