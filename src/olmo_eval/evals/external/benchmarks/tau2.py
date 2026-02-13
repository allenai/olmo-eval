"""tau2_bench external evaluation configuration.

tau2_bench is a benchmark for evaluating language model agents on realistic
customer service tasks. It measures both task completion and constraint
satisfaction.

Repository: https://github.com/sierra-research/tau2-bench
"""

from olmo_eval.evals.external.config import ExternalEvalConfig
from olmo_eval.evals.external.registry import register_external_config

# tau2_bench configuration
TAU2_BENCH_CONFIG = ExternalEvalConfig(
    name="tau2_bench",
    sandbox_image="python:3.11",
    setup_commands=(
        "git clone --depth 1 https://github.com/sierra-research/tau2-bench.git "
        "/workspace/tau2-bench",
        "cd /workspace/tau2-bench && pip install -e .",
    ),
    run_command=(
        "cd /workspace/tau2-bench && "
        "python -m tau2_bench.run "
        "--model $OPENAI_MODEL "
        "--api-base $OPENAI_API_BASE "
        "--output /workspace/results.json"
    ),
    timeout=7200.0,  # 2 hours
)

# Register the configuration
register_external_config("tau2_bench", TAU2_BENCH_CONFIG)
