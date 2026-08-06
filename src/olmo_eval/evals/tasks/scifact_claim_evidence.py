"""SciFact claim-evidence contrastive probe.

A likelihood-only probe of whether a model knows which text supports which claim, scorable on a
base checkpoint. Nothing is generated and no instruction is followed, so an annealed model that
cannot yet produce a formatted answer is still measured on what it knows.

Each pair scores ONE claim under two contexts: the gold rationale that supports it, and a hard
negative. The scored string is identical across the pair, so continuation length, fluency and
register cancel exactly and the difference isolates the conditioning. This is the design behind
MiST's canonical-vs-corrupted SMILES score (arXiv 2512.21231), which reports rho = 0.60-0.64
between a pre-RL likelihood contrast and post-RL task accuracy.

Two conditions, built by build_scifact_pairs.py:
    within  a non-rationale sentence from the SAME abstract -- which sentences support the claim
    cross   the nearest rationale from a different paper supporting a different assertion --
            topical adjacency without support, which is the failure mode LLM-judged citation
            metrics score most leniently

Metrics are the paired effect size (Cohen's d over the per-pair logprob difference, the primary)
and the win rate (share of pairs where the gold context makes the claim more likely). Chance is
d = 0 and win rate = 0.5.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import PPLFormatter
from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers import LogprobScorer
from olmo_eval.common.types import Instance, LMRequest, RequestType, Response, Split
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register

log = logging.getLogger(__name__)

PAIRS_URI = "s3://ai2-llm/sftlab/davidg/eval-data/scifact_claim_evidence_pairs.jsonl"


def _paired_diffs(responses: Sequence[Response]) -> list[float]:
    """Per-pair logprob(claim | gold context) - logprob(claim | negative context).

    The claim is the same string on both sides, so the raw sum is comparable without
    length normalization.
    """
    scorer = LogprobScorer()
    sides: dict[str, dict[str, float]] = {}
    for r in responses:
        pid = r.instance.metadata.get("pair_id")
        kind = r.instance.metadata.get("kind")
        if pid is None or kind not in ("gold", "neg") or not r.outputs:
            continue
        s = scorer.score(r.instance, r.outputs[0])
        if s == float("-inf"):
            continue
        sides.setdefault(pid, {})[kind] = s
    return [v["gold"] - v["neg"] for v in sides.values() if len(v) == 2]


@dataclass(frozen=True, slots=True)
class ContrastiveEffectSize(Metric):
    """Cohen's d of the paired logprob difference: mean(diff) / sd(diff).

    The paired form, since gold and negative share a claim. d = 0 is chance; MiST treats
    d ~ 1.5 as the point where a chemistry model becomes trainable by RL.
    """

    name: str = "effect_size"
    scorer: type[LogprobScorer] = LogprobScorer

    def compute(self, responses: Sequence[Response]) -> float:
        d = _paired_diffs(responses)
        if len(d) < 2:
            return 0.0
        sd = statistics.stdev(d)
        return statistics.mean(d) / sd if sd else 0.0

    def compute_instance(self, response: Response) -> float | None:
        return None  # defined only over a pair

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class ContrastiveWinRate(Metric):
    """Share of pairs where the gold context makes the claim more likely. Chance is 0.5."""

    name: str = "win_rate"
    scorer: type[LogprobScorer] = LogprobScorer

    def compute(self, responses: Sequence[Response]) -> float:
        d = _paired_diffs(responses)
        return sum(1 for x in d if x > 0) / len(d) if d else 0.0

    def compute_instance(self, response: Response) -> float | None:
        return None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


_EFFECT = ContrastiveEffectSize()
_METRICS = (_EFFECT, ContrastiveWinRate())


class ScifactClaimEvidenceTask(Task):
    """Base class; `condition` selects which negative type to score."""

    split = Split.TEST
    condition: str | None = None
    data_source = DataSource(path=PAIRS_URI)
    formatter = PPLFormatter()
    metrics = _METRICS
    primary_metric = _EFFECT

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            for doc in DataLoader().load(self.config.get_data_source()):
                inst = self.process_doc(doc)
                if inst is not None:
                    self._instances_cache.append(inst)
        yield from self._instances_cache

    @property
    def request_type(self) -> RequestType:
        return RequestType.LOGLIKELIHOOD

    def format_request(self, instance: Instance) -> LMRequest:
        return (self.config.formatter or PPLFormatter()).format(instance, self.get_fewshot())

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        if self.condition is not None and doc.get("cond") != self.condition:
            return None
        claim, context = doc.get("claim"), doc.get("context")
        if not claim or not context:
            return None
        # PPLFormatter scores gold_answer as a continuation of question.
        return Instance(
            question=f"{context.strip()}\n\nClaim:",
            gold_answer=claim.strip(),
            metadata={"pair_id": doc["pair_id"], "kind": doc["kind"], "cond": doc["cond"]},
        )


for _cond in ("within", "cross"):
    _name = f"scifact_claim_evidence_{_cond}"
    _cls = type(
        f"ScifactClaimEvidence{_cond.title()}",
        (ScifactClaimEvidenceTask,),
        {"__module__": __name__, "__qualname__": f"ScifactClaimEvidence{_cond.title()}",
         "condition": _cond},
    )
    globals()[_cls.__name__] = _cls
    register(_name)(_cls)
