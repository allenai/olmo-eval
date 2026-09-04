"""Berkeley Function Calling Leaderboard (BFCL) v4 external evaluations.

Repository: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
"""

from olmo_eval.evals.external.benchmarks.bfcl.eval import BFCLV4NonWebExternalEval
from olmo_eval.evals.external.registry import register_external_eval

register_external_eval(BFCLV4NonWebExternalEval())
