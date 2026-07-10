"""Standard IFEval instruction-following benchmark."""

from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.ifeval_ood import IFEvalOOD


@register("ifeval")
class IFEval(IFEvalOOD):
    """The original 541-prompt IFEval benchmark with chat formatting."""

    data_source = DataSource(path="wis-k/instruction-following-eval", split="train")
