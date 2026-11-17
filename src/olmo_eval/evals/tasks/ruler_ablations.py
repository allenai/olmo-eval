"""Prompt ablations for RULER FWE and VT (scaling-ladders #106 and #107).

FWE (``fwe_ablation`` task setting): the coded words of an instance are reassigned in place,
so the prompt keeps its length, its ``...`` separators and its wording.

- ``tail<K>``: in the last K words, each gold word becomes a decoy word. The tail's top 3 are
  then the decoys, while the whole text's top 3 stay the gold.
- ``span<m>``: the text is split into 10 segments. Each segment has 3 words of its own,
  ``c_local`` times each, and every segment has the 3 gold words ``c_gold`` times each, so
  a window must cover at least m segments for the gold to win.
- ``depth<d>``: the gold words are dense in one block (10% of the words) at depth d%, and
  3 decoy words are spread over the rest, below the gold in total.

Every rewrite is deterministic in the instance index and checked: the gold must be the strict
top 3 of the whole text, or the instance raises.

VT (``vt_ablation`` task setting) varies what precedes the question; see ``vt_prompt``.
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from collections.abc import Sequence
from typing import Any

FWE_WORD = re.compile(r"\b[a-z]{6}\b")
#: FWE ablations: tail-swap sizes (words), required spans (segments of 10), gold-block depths (%).
FWE_TAIL_K = (60, 200, 1000)
FWE_SPAN_M = (1, 2, 5)
FWE_DEPTH = (0, 25, 50, 75, 100)
FWE_SEGMENTS = 10
FWE_BLOCK = 0.10
#: Fewest coded words per FWE length in allenai/RULER. A tail swap needs K <= a sixth of it:
#: the top word is ~40% of the text, so K swapped words give a decoy ~0.4 K, which must stay
#: below the 3rd gold word (~10% of the rest).
FWE_MIN_WORDS = {4096: 624, 8192: 1172, 16384: 2584, 32768: 4824, 65536: 9721, 131072: 19553}


def fwe_variants(length: int) -> list[str]:
    """The FWE ablation variants valid at a nominal length."""
    tails = [f"tail{k}" for k in FWE_TAIL_K if 6 * k <= FWE_MIN_WORDS[length]]
    return [*tails, *(f"span{m}" for m in FWE_SPAN_M), *(f"depth{d}" for d in FWE_DEPTH)]


def _fresh_words(rng: random.Random, n: int, taken: set[str]) -> list[str]:
    out = []
    while len(out) < n:
        w = "".join(rng.choices(string.ascii_lowercase, k=6))
        if w not in taken:
            taken.add(w)
            out.append(w)
    return out


def _top3_strict(words: Sequence[str]) -> set[str] | None:
    """The 3 most frequent words, or None when the 3rd ties the 4th."""
    top = Counter(words).most_common(4)
    if len(top) < 3 or (len(top) == 4 and top[2][1] == top[3][1]):
        return None
    return {w for w, _ in top[:3]}


def _filler(rng: random.Random, n: int, cap: int, taken: set[str]) -> list[str]:
    """n filler words, each used at most ``cap`` times (cap >= 1)."""
    pool = _fresh_words(rng, -(-n // cap), taken)
    seq = [pool[i % len(pool)] for i in range(n)]
    rng.shuffle(seq)
    return seq


def fwe_rewrite(context: str, gold: Sequence[str], variant: str, seed: Any) -> tuple[str, list[str], list[str]]:
    """Rewrite an FWE context; return (context, gold, decoys)."""
    spans = [m.span() for m in FWE_WORD.finditer(context)]
    old = [context[a:b] for a, b in spans]
    n = len(old)
    rng = random.Random(f"ruler-fwe-{variant}-{seed}")
    taken = set(old)
    kind, param = re.fullmatch(r"(tail|span|depth)(\d+)", variant).groups()
    param = int(param)
    if kind == "tail":
        decoys = _fresh_words(rng, 3, taken)
        swap = dict(zip(gold, decoys))
        new = old[: n - param] + [swap.get(w, w) for w in old[n - param :]]
        new_gold = list(gold)
        local = set(decoys)
    elif kind == "span":
        new_gold = _fresh_words(rng, 3, taken)
        bounds = [round(i * n / FWE_SEGMENTS) for i in range(FWE_SEGMENTS + 1)]
        seg = min(b - a for a, b in zip(bounds, bounds[1:]))
        c_gold = max(2, round(0.03 * seg))
        # local words win inside m-1 segments and lose inside m: (m-1) c_gold < c_local < m c_gold
        c_local = max(1, (param - 1) * c_gold + c_gold // 2) if param > 1 else max(1, c_gold // 2)
        assert 3 * (c_gold + c_local) <= seg, f"{variant}: segments of {seg} words are too short"
        new, decoys = [], []
        for a, b in zip(bounds, bounds[1:]):
            loc = _fresh_words(rng, 3, taken)
            decoys = loc  # the last segment's words: what a reader of only the tail answers
            body = [w for w in new_gold for _ in range(c_gold)] + [w for w in loc for _ in range(c_local)]
            cap = max(1, min(c_gold, c_local) - 1) if min(c_gold, c_local) > 1 else 1
            body += _filler(rng, (b - a) - len(body), cap, taken)
            rng.shuffle(body)
            new += body
        local = set(decoys)
    else:  # depth
        new_gold = _fresh_words(rng, 3, taken)
        decoys = _fresh_words(rng, 3, taken)
        blk = max(30, round(FWE_BLOCK * n))
        start = round((n - blk) * param / 100)
        g = blk // 4  # each gold word fills a quarter of the block
        d = max(1, (4 * g) // 5)  # each decoy: 80% of a gold word's count, outside the block
        block = [w for w in new_gold for _ in range(g)]
        block += _filler(rng, blk - len(block), 1, taken)
        rng.shuffle(block)
        rest = [w for w in decoys for _ in range(d)]
        rest += _filler(rng, (n - blk) - len(rest), max(1, d // 4), taken)
        rng.shuffle(rest)
        new = rest[:start] + block + rest[start:]
        local = set(decoys)
    assert len(new) == n
    if _top3_strict(new) != set(new_gold):
        raise ValueError(f"FWE {variant} (seed {seed}): the gold is not the strict top 3")
    # span1 and depth100 are controls: there the last segment holds the gold by design
    if variant not in ("span1", "depth100") and kind != "tail" and _top3_strict(new[-60:]) == set(new_gold):
        raise ValueError(f"FWE {variant} (seed {seed}): the last 60 words still give the gold")
    out, pos = [], 0
    for (a, b), w in zip(spans, new):
        out += [context[pos:a], w]
        pos = b
    out.append(context[pos:])
    return "".join(out), list(new_gold), sorted(local)


def fwe_ablate(doc: dict[str, Any], variant: str, index: int) -> dict[str, Any]:
    """An FWE data row with its context rewritten by ``variant``."""
    seed = doc.get("index", index)
    gold = list(doc.get("outputs") or doc.get("answer") or [])
    ctx, new_gold, decoys = fwe_rewrite(doc["context"], gold, variant, seed)
    return {
        **doc,
        "input": doc["input"].replace(doc["context"], ctx, 1),
        "context": ctx,
        "outputs": new_gold,
        "answer": new_gold,
        "decoys": decoys,
    }


#: VT ablations (#107). All use the data row's example re-drawn per instance, as `:v3` does.
VT_VARIANTS = ("vt_fmt", "vt_late", "vt_name5", "vt_instr", "vt_3shot")
VT_EXAMPLE_ANSWER = re.compile(r"Answer: According to the chain\(s\).*?they are:\s*(.+?)\s*$", re.S)
VT_FORMAT_INSTRUCTION = "Answer with the variable names, separated by spaces."


def vt_prompt(question: str, example: str, variant: str, redraw) -> str:
    """The VT prompt for an ablation; ``redraw(example, seed_suffix, name_len)`` re-draws it.

    - ``vt_fmt``: one example whose answer has another format ("Answer: VAR QPE, VAR SEJ, ..."),
      so it never contains the real answer prefix "... they are:".
    - ``vt_late``: one example placed right before the question instead of at the start.
    - ``vt_name5``: one example with 5-letter names, like the real variables.
    - ``vt_instr``: no example; a format instruction before the question.
    - ``vt_3shot``: three examples, each re-drawn.
    """
    if variant == "vt_instr":
        q = question.rfind("Question:")
        return question[:q] + VT_FORMAT_INSTRUCTION + " " + question[q:]
    if variant == "vt_3shot":
        shots = [redraw(example, j, None) for j in range(3)]
        return "\n\n".join(shots) + "\n\n" + question
    ex = redraw(example, 0, 5 if variant == "vt_name5" else None)
    if variant == "vt_fmt":
        m = VT_EXAMPLE_ANSWER.search(ex)
        if m is None:
            raise ValueError("VT example in an unexpected shape")
        names = m.group(1).split()
        ex = ex[: m.start()] + "Answer: " + ", ".join(f"VAR {n}" for n in names)
    if variant == "vt_late":
        q = question.rfind("Question:")
        return question[:q] + ex + "\n\n" + question[q:]
    return ex + "\n\n" + question
