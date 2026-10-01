"""Scoring for n-gram copying: bits per byte restricted to repeated-span positions."""

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

from olmo_eval.common.types import Instance, LMOutput

from .base import Scorer


def compute_repeated_ngram_mask(tokens: Sequence[Hashable], k: int) -> list[bool]:
    """Flag positions whose preceding k-gram already occurred earlier in the sequence.

    Position i is flagged when tokens[i-k+1:i+1] is equal to some earlier k-gram
    tokens[j-k+1:j+1] with j < i. Match feasibility is monotonic in k (any earlier
    match of length k implies a match of every shorter length against the same
    earlier occurrence), so thresholding a document's longest-repeat score at k is
    equivalent to checking length-k matches directly.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    mask = [False] * len(tokens)
    seen: set[tuple[Hashable, ...]] = set()
    for i in range(len(tokens)):
        if i >= k - 1:
            ngram = tuple(tokens[i - k + 1 : i + 1])
            mask[i] = ngram in seen
            seen.add(ngram)
    return mask


@dataclass(frozen=True, slots=True)
class NGramCopyingBPBScorer(Scorer):
    """Bits per byte restricted to positions with a length-k+ repeated n-gram.

    Requires ``output.logprobs`` to reflect teacher-forced scoring of the actual
    document tokens (e.g. a loglikelihood request over the whole document), since
    repeat detection depends on the true token sequence rather than sampled output.
    """

    name: str = field(init=False)
    k: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", f"ngram_copying_bpb_k{self.k}")

    def masked_totals(self, output: LMOutput) -> tuple[float, int] | None:
        """Sum logprobs and byte counts over positions with a length-k+ repeat.

        Returns None if the output has no logprobs or no qualifying positions.
        """
        if not output.logprobs:
            return None

        # Prefer token IDs as the repeat identity: distinct tokens whose decoded
        # strings collide (e.g. multi-byte UTF-8 fragments that all render as the
        # replacement character) must not count as repeats of each other.
        tokens: list[Hashable] = [
            entry.get("token_id", entry["token"]) for entry in output.logprobs
        ]
        mask = compute_repeated_ngram_mask(tokens, self.k)
        if not any(mask):
            return None

        total_logprob = 0.0
        total_bytes = 0
        for entry, matched in zip(output.logprobs, mask, strict=True):
            if not matched:
                continue
            total_logprob += entry.get("logprob", 0.0)
            # Byte counts derive from the decoded token string (providers fill
            # "bytes" the same way), so a token that decodes to the replacement
            # character contributes its 3-byte encoding rather than the raw
            # token's true byte length. Exact counts would need tokenizer-level
            # raw bytes; the error is limited to repeats containing such tokens.
            token_bytes = entry.get("bytes")
            total_bytes += (
                len(token_bytes) if token_bytes is not None else len(entry["token"].encode("utf-8"))
            )
        return total_logprob, total_bytes

    def score(self, instance: Instance, output: LMOutput) -> float:
        totals = self.masked_totals(output)
        if totals is None:
            return 0.0
        total_logprob, total_bytes = totals
        if total_bytes == 0:
            return 0.0
        return -total_logprob / (total_bytes * math.log(2))
