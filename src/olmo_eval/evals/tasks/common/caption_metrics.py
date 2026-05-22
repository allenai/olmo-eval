"""
Caption-quality metrics for evaluating VLM generation.

Implements three metrics from scratch (no external NLP deps) plus optional
CIDEr via ``pycocoevalcap`` when installed:

- **Token-F1**: F1 of the set of tokens in the prediction vs the reference.
  Order-insensitive. Closest analogue to QA-style F1.
- **BLEU-4**: standard corpus-level BLEU-4 with a brevity penalty.
- **ROUGE-L**: longest common subsequence over the unigram sequence,
  reported as the F1 of LCS-based precision/recall.

Tokenization is the simplest defensible thing: lowercase, strip punctuation,
split on whitespace. That matches how COCO-style caption metrics handle
free-form text and is what we want for PixMo-Cap.

The metrics are computed in pure Python on already-tokenized predictions
and references. They're cheap enough for a few thousand examples without
any optimization.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = [
    "tokenize",
    "token_f1",
    "bleu_4",
    "rouge_l",
    "CaptionMetricsReport",
    "compute_caption_metrics",
]


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


_PUNCT_RE = re.compile(r"[^\w\s]")


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return _PUNCT_RE.sub(" ", text.lower()).split()


# ---------------------------------------------------------------------------
# Token-F1 (order-insensitive)
# ---------------------------------------------------------------------------


def token_f1(prediction_tokens: Sequence[str], reference_tokens: Sequence[str]) -> float:
    """F1 over token multisets — micro-precision/recall counting duplicates.

    Standard formulation used by SQuAD-style evaluators. With multisets, a
    repeated token in the prediction is only credited up to its count in the
    reference.
    """
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    pred_counts = Counter(prediction_tokens)
    ref_counts = Counter(reference_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(ref_counts.values())
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# BLEU-4 (corpus-level)
# ---------------------------------------------------------------------------


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_4(
    predictions: Sequence[Sequence[str]],
    references: Sequence[Sequence[Sequence[str]]],
    max_n: int = 4,
) -> float:
    """Corpus-level BLEU-N with brevity penalty.

    Matches the standard formula used by SacreBLEU / NLTK in spirit (multi
    references supported by taking max-count-per-ngram across them).

    :param predictions: List of tokenized predictions, one per example.
    :param references: List of *lists* of tokenized references, one set per
        example. (PixMo-Cap has multiple transcripts per image; pass them
        all here.)
    """
    assert len(predictions) == len(references)
    if not predictions:
        return 0.0

    # Clipped n-gram count + total prediction n-gram count, per n.
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    pred_len_total = 0
    ref_len_total = 0  # sum of closest reference lengths per example

    for pred, refs in zip(predictions, references):
        pred_len_total += len(pred)
        # Pick the reference length closest to the prediction length (ties
        # broken in favour of shorter reference).
        ref_lens = [len(r) for r in refs]
        diffs = [(abs(rl - len(pred)), rl) for rl in ref_lens]
        diffs.sort()
        ref_len_total += diffs[0][1] if diffs else 0

        for n in range(1, max_n + 1):
            pred_ngrams = _ngrams(pred, n)
            total_counts[n - 1] += sum(pred_ngrams.values())
            if not pred_ngrams:
                continue
            # Max count for each n-gram across all references.
            max_ref: Counter = Counter()
            for r in refs:
                rc = _ngrams(r, n)
                for k, v in rc.items():
                    if v > max_ref.get(k, 0):
                        max_ref[k] = v
            for ngram, c in pred_ngrams.items():
                clipped_counts[n - 1] += min(c, max_ref.get(ngram, 0))

    # Geometric mean of clipped precisions, with smoothing (epsilon = 1e-10
    # to avoid log(0)).
    log_precisions = 0.0
    for c, t in zip(clipped_counts, total_counts):
        if t == 0:
            return 0.0
        p = c / t if c > 0 else 1e-10
        log_precisions += math.log(p)
    geo_mean = math.exp(log_precisions / max_n)

    # Brevity penalty.
    if pred_len_total == 0:
        return 0.0
    if pred_len_total > ref_len_total:
        bp = 1.0
    else:
        bp = math.exp(1.0 - ref_len_total / pred_len_total)
    return bp * geo_mean


# ---------------------------------------------------------------------------
# ROUGE-L (LCS-based F1)
# ---------------------------------------------------------------------------


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence."""
    if not a or not b:
        return 0
    # O(|a| * |b|) DP, rolling-array.
    m, n = len(a), len(b)
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
        for j in range(n + 1):
            curr[j] = 0
    return prev[n]


def rouge_l(
    prediction_tokens: Sequence[str],
    reference_tokens: Sequence[str],
    beta: float = 1.2,
) -> float:
    """ROUGE-L F-measure (LCS-based). ``beta`` weights recall over precision
    (standard ROUGE convention)."""
    if not prediction_tokens or not reference_tokens:
        return 0.0
    lcs = _lcs_length(prediction_tokens, reference_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return (1 + beta**2) * precision * recall / (recall + beta**2 * precision)


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


@dataclass
class CaptionMetricsReport:
    """Aggregated caption metrics across a dataset."""

    n_examples: int
    token_f1: float
    bleu_4: float
    rouge_l: float
    cider: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "token_f1": self.token_f1,
            "bleu_4": self.bleu_4,
            "rouge_l": self.rouge_l,
            "cider": self.cider,
        }


def compute_caption_metrics(
    predictions: Sequence[str],
    references: Sequence[Sequence[str]],
    *,
    include_cider: bool = False,
) -> CaptionMetricsReport:
    """Compute caption metrics over a parallel ``predictions`` / ``references`` set.

    :param predictions: One predicted caption per example.
    :param references: List of *lists* of reference captions per example. For
        single-reference datasets pass ``[[ref]]`` lists. For PixMo-Cap pass
        the full list of human transcripts plus the synthetic caption.
    :param include_cider: If ``True`` and ``pycocoevalcap`` is installed,
        compute CIDEr too. Silently skipped when unavailable.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"len(predictions)={len(predictions)} != len(references)={len(references)}"
        )
    if not predictions:
        return CaptionMetricsReport(0, 0.0, 0.0, 0.0, cider=None)

    pred_tokens = [tokenize(p) for p in predictions]
    ref_tokens = [[tokenize(r) for r in refs] for refs in references]

    # Token-F1 against the first reference (typical convention for VQA-style F1).
    f1_scores = [token_f1(pt, rt[0] if rt else []) for pt, rt in zip(pred_tokens, ref_tokens)]
    f1 = sum(f1_scores) / len(f1_scores)

    bleu = bleu_4(pred_tokens, ref_tokens)

    rouge_scores = [
        max((rouge_l(pt, rt) for rt in refs), default=0.0)
        for pt, refs in zip(pred_tokens, ref_tokens)
    ]
    rouge = sum(rouge_scores) / len(rouge_scores)

    cider: Optional[float] = None
    if include_cider:
        try:
            from pycocoevalcap.cider.cider import Cider

            gts = {i: [" ".join(r) for r in refs] for i, refs in enumerate(ref_tokens)}
            res = {i: [" ".join(pt)] for i, pt in enumerate(pred_tokens)}
            cider_score, _ = Cider().compute_score(gts, res)
            cider = float(cider_score)
        except ImportError:
            cider = None

    return CaptionMetricsReport(
        n_examples=len(predictions),
        token_f1=f1,
        bleu_4=bleu,
        rouge_l=rouge,
        cider=cider,
    )
