"""The vision benchmarks; importing this package registers them.

One module per benchmark. Imports are explicit (unlike ``evals/tasks``\'
pkgutil scan) so a missing module is an ImportError here rather than a silently
absent task.
"""

from olmo_eval.evals.vision.benchmarks import dense_caption as _dense_caption  # noqa: F401

__all__: list[str] = []
