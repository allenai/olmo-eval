# Distributed Evaluation for olmo-eval

## Overview

This document proposes approaches for running distributed evaluation matrices (N models × M tasks) using Beaker as the compute backend. Three options are presented, ranging from minimal changes to a full orchestration layer.

## Goals

1. **Matrix Evaluation**: Launch N models × M task suites as coordinated experiments
2. **Result Aggregation**: Collect and compare results across models/tasks
3. **Resource Optimization**: GPU-aware scheduling based on model requirements
4. **Flexibility**: Support both simple workflows and complex orchestration needs

## Beaker Background

Beaker is AI2's collaborative platform for reproducible research. Key features relevant to evaluation:

| Feature | Description |
|---------|-------------|
| **Experiments** | Container-based jobs with tasks, resources, and result datasets |
| **Groups** | Collections of related experiments with metric aggregation |
| **Queues** | Coordinator-worker pattern for distributed batch processing |
| **Priorities** | low, normal, high, urgent, immediate |
| **Preemption** | Jobs can be preempted and retried automatically |
| **Metrics Export** | `beaker.group.export_metrics()` for CSV export across experiments |

### Beaker Experiment Structure

```yaml
version: v2
budget: "ai2/oe-base"
tasks:
  - name: "eval"
    image:
      beaker: "ai2/olmo-eval-latest"
    command: ["olmo-eval", "run", "-m", "llama3.1-8b", "-t", "mmlu"]
    resources:
      gpuCount: 1
      memory: "32G"
    context:
      cluster: ["ai2/jupiter", "ai2/saturn"]
      priority: "normal"
      preemptible: true
    result:
      path: "/results"
```

### Beaker Groups for Result Aggregation

Groups organize related experiments and provide native metric aggregation:

```python
# Create a group for an evaluation run
group = beaker.group.create(
    name="eval-benchmark-2024-01",
    workspace="ai2/oe-data",
    experiment_ids=[exp1.id, exp2.id, exp3.id]
)

# Export aggregated metrics
csv_data = beaker.group.export_metrics(group)
```

---

## Approach Options

### Option A: Lightweight CLI Enhancement (Recommended)

**Complexity**: Low | **New Code**: ~200 lines | **No daemon required**

Enhance the existing `launch` command to create Beaker Groups automatically, then add a `results` command for aggregation. Leverages Beaker's native group features.

```
┌──────────────────────────────────────────────────────────────────┐
│                     EXISTING CLI (Enhanced)                       │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │ launch cmd   │──▶│ BeakerGroup  │──▶│ results cmd          │  │
│  │ (existing)   │   │ (new)        │   │ (new)                │  │
│  └──────────────┘   └──────────────┘   └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Beaker Groups   │
                     │  (native feature)│
                     └──────────────────┘
```

**Changes:**

1. **Add `--group` flag to `launch`**: Auto-create/update Beaker group
2. **Add `results` command**: Query group metrics, generate reports

**Example Usage:**

```bash
# Launch with automatic grouping
olmo-eval launch -n "benchmark-v1" --group "benchmark-2024" \
    -m llama3.1-8b -m olmo-2-7b \
    -t mmlu -t gsm8k -t hellaswag

# Creates experiments and adds them to "benchmark-2024" group
# Output:
#   Created group: benchmark-2024
#   Launched: benchmark-v1-llama3.1-8b -> https://beaker.org/ex/...
#   Launched: benchmark-v1-olmo-2-7b -> https://beaker.org/ex/...
#   Added 2 experiments to group

# Later: check results
olmo-eval results --group "benchmark-2024"
# Output: table of model × task scores

# Export to CSV
olmo-eval results --group "benchmark-2024" --format csv > results.csv
```

**Pros:**
- Minimal new code
- Uses Beaker's native grouping (battle-tested)
- No background process needed
- Works with existing `launch` workflow

**Cons:**
- No streaming progress (poll manually)
- Limited to Beaker-specific features
- Less portable to other backends

---

### Option B: Matrix Command with Polling

**Complexity**: Medium | **New Code**: ~500 lines | **No daemon required**

Add a dedicated `matrix` command that launches experiments, waits for completion, and collects results. Simple polling loop, no async complexity.

```
┌──────────────────────────────────────────────────────────────────┐
│                        matrix command                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Matrix       │  │ Job          │  │ Result               │   │
│  │ Expander     │  │ Monitor      │  │ Collector            │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  BeakerLauncher  │
                     │  (existing)      │
                     └──────────────────┘
```

**Example Usage:**

```bash
# Run matrix and wait for results
olmo-eval matrix -f benchmark.yaml --wait

# Or fire-and-forget with group tracking
olmo-eval matrix -f benchmark.yaml --group "benchmark-2024"
```

**Matrix Config (benchmark.yaml):**

```yaml
name: model-comparison
models:
  - llama3.1-8b
  - llama3.1-70b
  - olmo-2-7b
tasks:
  - mmlu
  - gsm8k
  - hellaswag
  - arc_challenge

cluster: h100
group_by: model  # or "task" for per-task priorities
```

**Pros:**
- Single command for full workflow
- Optional wait-for-completion
- YAML config for complex matrices
- Can add simple progress display

**Cons:**
- Blocking if using `--wait`
- Still Beaker-specific
- More code than Option A

---

### Option C: Full Orchestrator with Backend Abstraction

**Complexity**: High | **New Code**: ~1500 lines | **Future cloud portability**

Abstract compute backend interface designed for Beaker now, generalizable to AWS Batch/GCP Vertex AI later. Async streaming results, progress tracking, cancellation support.

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION ORCHESTRATOR                       │
│  ┌───────────┐  ┌────────────┐  ┌───────────┐  ┌─────────────┐  │
│  │  Matrix   │  │  Resource  │  │  Result   │  │  Progress   │  │
│  │  Expander │  │  Planner   │  │  Collector│  │  Tracker    │  │
│  └───────────┘  └────────────┘  └───────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ABSTRACT COMPUTE BACKEND                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  ComputeBackend Interface                  │  │
│  │  - submit_jobs(jobs) -> JobHandles                         │  │
│  │  - get_status(handle) -> JobStatus                         │  │
│  │  - stream_results(handles) -> AsyncIterator[Result]        │  │
│  │  - cancel_job(handle)                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │   Beaker    │   │  (Future)   │   │  (Future)   │
   │   Backend   │   │  AWS Batch  │   │  GCP Vertex │
   └─────────────┘   └─────────────┘   └─────────────┘
```

**Pros:**
- Cloud-portable architecture
- Async streaming results
- Rich progress tracking
- Cancellation support
- Reusable for other projects

**Cons:**
- Significant implementation effort
- Async complexity
- May be over-engineered for current needs

---

## Recommendation

**Start with Option A** (Lightweight CLI Enhancement) because:

1. **Minimal effort**: ~200 lines of code
2. **Leverages Beaker**: Uses native groups and metrics export
3. **Non-breaking**: Extends existing `launch` command
4. **Sufficient for current needs**: Group-based aggregation covers most use cases

If requirements grow, evolve to Option B or C incrementally. The group-based approach in Option A provides a foundation that Options B and C can build on.

---

## Implementation Details (Option A)

### Files to Modify

| File | Changes |
|------|---------|
| `src/olmo_eval/cli.py` | Add `--group` to launch, add `results` command |
| `src/olmo_eval/launch/beaker.py` | Add group management methods to BeakerLauncher |

### CLI Changes

**Enhanced `launch` command:**

```python
@main.command()
@click.option("--group", "-g", help="Add experiments to this Beaker group")
# ... existing options ...
def launch(..., group: str | None):
    # ... existing logic ...

    # After launching experiments, add to group
    if group:
        experiment_ids = [exp.id for exp in launched_experiments]
        add_to_group(launcher, group, experiment_ids)
```

**New `results` command:**

```python
@main.command()
@click.option("--group", "-g", required=True, help="Beaker group name")
@click.option("--format", type=click.Choice(["table", "csv", "json"]), default="table")
@click.option("--wait", is_flag=True, help="Wait for all experiments to complete")
def results(group: str, format: str, wait: bool):
    """Show results from a Beaker group."""
    from olmo_eval.launch import BeakerLauncher

    launcher = BeakerLauncher()

    if wait:
        wait_for_group(launcher, group)

    if format == "csv":
        csv_data = launcher.beaker.group.export_metrics(group)
        click.echo(csv_data)
    else:
        display_group_results(launcher, group, format)
```

### BeakerLauncher Extensions

```python
class BeakerLauncher:
    # ... existing methods ...

    def get_or_create_group(self, name: str, workspace: str | None = None) -> BeakerGroup:
        """Get existing group or create new one."""
        try:
            return self.beaker.group.get(name)
        except GroupNotFound:
            return self.beaker.group.create(name=name, workspace=workspace)

    def add_experiments_to_group(
        self,
        group: str | BeakerGroup,
        experiment_ids: list[str]
    ) -> BeakerGroup:
        """Add experiments to a group."""
        return self.beaker.group.update(
            group,
            add_experiment_ids=experiment_ids
        )

    def get_group_status(self, group: str | BeakerGroup) -> dict:
        """Get status summary for all experiments in group."""
        experiments = list(self.beaker.group.experiments(group))
        status_counts = {"succeeded": 0, "failed": 0, "running": 0, "pending": 0}
        for exp in experiments:
            status = exp.status.lower()
            if status in status_counts:
                status_counts[status] += 1
            elif status in ("queued", "initializing"):
                status_counts["pending"] += 1
        return status_counts
```

### Example Workflow

```bash
# 1. Launch evaluation suite with grouping
olmo-eval launch -n "benchmark-jan" --group "benchmark-2024-01" \
    -m llama3.1-8b -m llama3.1-70b -m olmo-2-7b \
    -t mmlu -t gsm8k -t hellaswag -t arc_challenge

# Output:
# Created group: benchmark-2024-01
# Launched: benchmark-jan-llama3.1-8b -> https://beaker.org/ex/abc123
# Launched: benchmark-jan-llama3.1-70b -> https://beaker.org/ex/def456
# Launched: benchmark-jan-olmo-2-7b -> https://beaker.org/ex/ghi789
# Added 3 experiments to group: benchmark-2024-01

# 2. Check progress
olmo-eval results --group "benchmark-2024-01"
# Output:
# Group: benchmark-2024-01
# Status: 1 succeeded, 1 running, 1 pending
#
# | Model          | mmlu  | gsm8k | hellaswag | arc_challenge |
# |----------------|-------|-------|-----------|---------------|
# | llama3.1-8b    | 0.65  | 0.58  | 0.79      | 0.52          |
# | llama3.1-70b   | -     | -     | -         | -             |
# | olmo-2-7b      | -     | -     | -         | -             |

# 3. Wait for completion and export
olmo-eval results --group "benchmark-2024-01" --wait --format csv > results.csv
```

---

## Implementation Details (Option C - Full Orchestrator)

For reference, the full orchestrator approach would include:

### File Structure

```
src/olmo_eval/orchestrate/
├── __init__.py
├── models.py           # EvalMatrix, EvalJob, ModelSpec, TaskSpec, EvalResult
├── orchestrator.py     # EvalOrchestrator
├── collector.py        # ResultCollector
└── backends/
    ├── __init__.py
    ├── base.py         # ComputeBackend ABC
    └── beaker.py       # BeakerBackend implementation
```

### Abstract Backend Interface

```python
class ComputeBackend(ABC):
    """Abstract interface for compute backends."""

    @abstractmethod
    async def submit_job(self, job: EvalJob) -> JobHandle: ...

    @abstractmethod
    async def get_status(self, handle: JobHandle) -> JobStatus: ...

    @abstractmethod
    async def stream_results(self, handles: list[JobHandle]) -> AsyncIterator[EvalResult]: ...

    @abstractmethod
    async def cancel_job(self, handle: JobHandle) -> bool: ...
```

This interface would enable future implementations for:
- **AWS Batch**: Submit to Batch queues, S3 results, CloudWatch polling
- **GCP Vertex AI**: Custom jobs, GCS results, Vertex API status

---

## Verification

### Option A Testing

```bash
# Test group creation
olmo-eval launch -n "test" --group "test-group" -m llama3.1-8b -t arc_easy --dry-run

# Test results command
olmo-eval results --group "test-group"

# Test CSV export
olmo-eval results --group "test-group" --format csv
```

### Integration Test

```bash
# Small real evaluation
olmo-eval launch -n "integration-test" --group "ci-test" \
    -m llama3.1-8b -t arc_easy --limit 10

# Wait and verify
olmo-eval results --group "ci-test" --wait
```
