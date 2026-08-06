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

Pairs are built here from SciFact's public release tarball rather than read from a hosted copy, so
the construction is versioned with the task and the eval job needs no storage credentials. The RNG
is seeded, so the pair set is identical across runs and across models.
"""

from __future__ import annotations

import collections
import functools
import io
import json
import logging
import math
import os
import random
import re
import statistics
import tarfile
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.formatters import PPLFormatter
from olmo_eval.common.metrics import Metric
from olmo_eval.common.scorers import LogprobScorer
from olmo_eval.common.types import Instance, LMRequest, RequestType, Response, Split
from olmo_eval.evals.tasks.common import Task, register

log = logging.getLogger(__name__)

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SEED = 17
N_WITHIN = 2              # within-abstract negatives per rationale
CLAIM_OVERLAP_MAX = 0.20  # a cross negative must support a sufficiently different claim


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


def _rec(pair_id: str, cond: str, kind: str, claim: str, context: str) -> dict[str, Any]:
    return {"pair_id": pair_id, "cond": cond, "kind": kind, "claim": claim, "context": context}


_STOP = frozenset(
    ["the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with", "by",
     "from", "at", "is", "are", "was", "were", "be", "as", "that", "this"]
)


def _toks(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower())} - _STOP


@functools.lru_cache(maxsize=1)
def _build_pairs() -> tuple[dict[str, Any], ...]:
    """Download SciFact and construct the minimal pairs. Cached per process."""
    with urllib.request.urlopen(SCIFACT_URL, timeout=300) as resp:  # noqa: S310 - pinned https
        blob = resp.read()
    corpus: dict[int, dict] = {}
    claims: list[dict] = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            base = os.path.basename(member.name)
            if base not in ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            lines = fh.read().decode("utf-8").splitlines()
            rows = [json.loads(ln) for ln in lines if ln.strip()]
            if base == "corpus.jsonl":
                corpus = {d["doc_id"]: d for d in rows}
            else:
                claims.extend(rows)

    # sort rather than trusting tar member order: the seeded sampling below draws in this order,
    # so the pair set must not depend on how the archive happens to be laid out
    claims.sort(key=lambda c: c["id"])
    gold: list[tuple[Any, str, int, str, set[int]]] = []
    for c in claims:
        for did, evs in (c.get("evidence") or {}).items():
            for e in evs:
                if e.get("label") != "SUPPORT":
                    continue
                doc = corpus.get(int(did))
                if not doc:
                    continue
                sents = doc["abstract"]
                idx = [i for i in e.get("sentences", []) if i < len(sents)]
                if idx:
                    rat = " ".join(sents[i] for i in idx)
                    gold.append((c["id"], c["claim"], int(did), rat, set(idx)))

    # TF-IDF over rationale text, for mining the cross negatives
    gtoks = [_toks(g[3]) for g in gold]
    df: collections.Counter = collections.Counter()
    for t in gtoks:
        df.update(t)
    n_gold = len(gold)

    def vec(t: set[str]) -> dict[str, float]:
        v = {w: math.log(n_gold / (1 + df[w])) for w in t}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    gvec = [vec(t) for t in gtoks]
    inv: dict[str, list[int]] = collections.defaultdict(list)
    for i, t in enumerate(gtoks):
        for w in t:
            inv[w].append(i)
    gidx = {(g[0], g[2]): i for i, g in enumerate(gold)}

    rng = random.Random(SEED)
    recs: list[dict[str, Any]] = []
    for cid, claim, did, rat, idx in gold:
        sents = corpus[did]["abstract"]
        pool = [i for i in range(len(sents)) if i not in idx and len(sents[i].split()) >= 6]
        for j, ni in enumerate(rng.sample(pool, min(N_WITHIN, len(pool)))):
            pid = f"{cid}_{did}_within{j}"
            recs.append(_rec(pid, "within", "gold", claim, rat))
            recs.append(_rec(pid, "within", "neg", claim, sents[ni]))

        gi = gidx[(cid, did)]
        ctoks = _toks(claim)
        cand: collections.Counter = collections.Counter()
        for w in gtoks[gi]:
            for j in inv[w]:
                if gold[j][2] == did:
                    continue
                oc = _toks(gold[j][1])
                # the negative must be evidence for a different assertion; without this the
                # nearest rationale is often a second source supporting the same claim
                if len(ctoks & oc) / max(1, len(ctoks | oc)) > CLAIM_OVERLAP_MAX:
                    continue
                cand[j] += gvec[gi].get(w, 0.0) * gvec[j].get(w, 0.0)
        if cand:
            j, _ = cand.most_common(1)[0]
            pid = f"{cid}_{did}_cross"
            recs.append(_rec(pid, "cross", "gold", claim, rat))
            recs.append(_rec(pid, "cross", "neg", claim, gold[j][3]))
    log.info("SciFact probe: %d rationales -> %d pairs", len(gold), len(recs) // 2)
    return tuple(recs)


class ScifactClaimEvidenceTask(Task):
    """Base class; `condition` selects which negative type to score."""

    split = Split.TEST
    condition: str | None = None
    formatter = PPLFormatter()
    metrics = _METRICS
    primary_metric = _EFFECT

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            for doc in _build_pairs():
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
