"""DeepScholar-Bench external evaluation (EXPERIMENTAL / UNVALIDATED).

DeepScholar-Bench evaluates generative research synthesis: generating a paper's
related-work section by retrieving, synthesizing, and citing prior work
(Sky Computing Lab, arXiv 2508.20033). Scored on synthesis, retrieval, and
verifiability, aggregated as a geometric mean.

This registration exposes a SKELETON that has never been run end to end. It
needs validation on a real (Docker + GPU + API-key) environment and will need
fixes; see plans/003_deepscholar_bench.md and eval.py for the open items.

Repository: https://github.com/guestrin-lab/deepscholar-bench
"""

from olmo_eval.evals.external.benchmarks.deepscholar.eval import DeepScholarExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(DeepScholarExternalEval())
