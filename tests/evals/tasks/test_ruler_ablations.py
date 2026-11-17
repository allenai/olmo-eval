"""RULER FWE/VT prompt ablations (scaling-ladders #106, #107)."""

import random
import string
from collections import Counter

import pytest

from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.evals.tasks.ruler_ablations import FWE_WORD, fwe_ablate, fwe_variants, vt_prompt


def _fwe_doc(n_words: int = 3000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    vocab = ["".join(rng.choices(string.ascii_lowercase, k=6)) for _ in range(60)]
    weights = [1 / (r + 1) ** 1.6 for r in range(len(vocab))]
    words = rng.choices(vocab, weights=weights, k=n_words)
    context = " ... ".join(words)
    gold = [w for w, _ in Counter(words).most_common(3)]
    return {"input": f"Read this.\n{context}\nQuestion: top 3? Answer:", "context": context,
            "outputs": gold, "index": seed}


def _top3(words):
    return {w for w, _ in Counter(words).most_common(3)}


@pytest.mark.parametrize("variant", ["tail60", "tail200", "span1", "span2", "span5", "depth0", "depth50", "depth100"])
def test_fwe_rewrite_keeps_length_and_makes_the_gold_the_top_3(variant):
    doc = _fwe_doc()
    out = fwe_ablate(doc, variant, 0)

    assert len(out["context"]) == len(doc["context"])
    assert out["input"] == doc["input"].replace(doc["context"], out["context"])
    words = FWE_WORD.findall(out["context"])
    assert _top3(words) == set(out["outputs"])
    assert fwe_ablate(doc, variant, 0) == out  # deterministic


def test_fwe_tail_swap_puts_the_decoys_on_top_of_the_tail_only():
    doc = _fwe_doc()
    out = fwe_ablate(doc, "tail200", 0)
    words = FWE_WORD.findall(out["context"])

    assert set(out["outputs"]) == set(doc["outputs"])
    assert not set(words[-200:]) & set(out["outputs"])
    assert set(out["decoys"]) <= set(words[-200:])
    assert FWE_WORD.findall(out["context"])[:-200] == FWE_WORD.findall(doc["context"])[:-200]


def test_fwe_span5_hides_the_gold_from_short_windows():
    words = FWE_WORD.findall(fwe_ablate(_fwe_doc(), "span5", 0)["context"])
    gold = _top3(words)

    assert not any(_top3(words[a:a + 300]) == gold for a in range(0, len(words) - 300, 50))


def test_fwe_variants_respect_the_tail_limit():
    assert fwe_variants(4096)[0] == "tail60" and "tail200" not in fwe_variants(4096)
    assert "tail1000" in fwe_variants(65536) and "tail1000" not in fwe_variants(32768)
    assert get_task("ruler_fwe__65536:tail1000").config.task_settings["fwe_ablation"] == "tail1000"


def test_vt_prompt_variants():
    q = "Memorize.\nFILLER\nQuestion: Find all variables? Answer: ... they are: "
    ex = "Memorize. VAR QPE = 64886 Question: Find Answer: According to the chain(s) x, they are:  QPE SEJ"
    redraw = lambda e, shot, n: e.replace("QPE", "AAA" if n is None else "AAAAA").replace("SEJ", f"B{shot}B")

    assert vt_prompt(q, ex, "vt_instr", redraw).endswith(
        "Answer with the variable names, separated by spaces. Question: Find all variables? Answer: ... they are: ")
    fmt = vt_prompt(q, ex, "vt_fmt", redraw)
    assert "Answer: VAR AAA, VAR B0B\n\n" in fmt and fmt.count("they are:") == 1
    late = vt_prompt(q, ex, "vt_late", redraw)
    assert late.index("FILLER") < late.index("VAR AAA") < late.rindex("Question:")
    assert "AAAAA" in vt_prompt(q, ex, "vt_name5", redraw)
    assert vt_prompt(q, ex, "vt_3shot", redraw).count("they are:") == 4
