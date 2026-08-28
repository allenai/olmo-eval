"""The vision benchmarks; importing this package registers them.

One module per benchmark. Imports are explicit (unlike ``evals/tasks``\'
pkgutil scan) so a missing module is an ImportError here rather than a silently
absent task.
"""

from olmo_eval.evals.vision.benchmarks import countbench_qa as _countbench_qa  # noqa: F401
from olmo_eval.evals.vision.benchmarks import dense_caption as _dense_caption  # noqa: F401
from olmo_eval.evals.vision.benchmarks import pixmo_count as _pixmo_count  # noqa: F401
from olmo_eval.evals.vision.benchmarks import pixmo_points_eval as _pixmo_points_eval  # noqa: F401
from olmo_eval.evals.vision.benchmarks import sa_co_gold as _sa_co_gold  # noqa: F401

__all__: list[str] = []
