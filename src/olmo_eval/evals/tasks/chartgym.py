"""ChartGym -- a synthetic chart-understanding instrument with exact ground truth.

Why this exists as a task rather than as a number on CharXiv: CharXiv can only resolve two
of the four capabilities ChartGym trains. Template 16 (general trend) is n=36 with a 16.4pp
2-sigma band, and the dense-OCR templates are already at 78-94%. So CharXiv is the
*transfer* target, and this is the instrument that can actually see whether the capability
was learned at all.

Two properties make it cheap and trustworthy:

* **No GPT judge.** Every answer is derived from a `FigureSpec` and scored by exact match,
  numeric tolerance, or multiset comparison. CharXiv's `n_invalid` failure mode -- a
  rate-limited judge still exits 0 and still reports a plausible score -- cannot happen here.
* **A held-out primitive.** Nothing about panel layout is ever trained (families `pnl.*`),
  while panel-layout questions still appear here. `chartgym_heldout_primitive` is therefore
  the headline: it measures capability transfer to a question shape the corpus never taught.
  Overall accuracy is a fit statistic and must not be read as evidence of transfer.

The `:text_only` variant is the leak floor. If a meaningful share is answerable without the
image, the corpus has a template leak and the training signal is contaminated.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response, SamplingParams, Split
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.tasks.common.image_qa_base import ImageQATask, lazy_hf_image

_CAPABILITIES = ("counting", "geometry", "trend", "ocr")
_DIFFICULTIES = ("easy", "medium", "hard")

#: Natural ways a model can express "this question does not apply here". CharXiv's own
#: prompt supplies its literal token at eval time, so what the corpus has to teach -- and
#: what this must therefore accept -- is the *detection*, expressed however the model likes.
_NA_MARKERS = (
    "not applicable", "n/a", "does not apply", "no legend", "no key", "no title",
    "no heading", "cannot be determined", "none", "no such", "there is no",
    "does not have", "doesn't have", "no curves", "not evenly spaced", "no name",
    "carries no", "has no",
)


def _dataset_dir() -> Path:
    explicit = os.environ.get("CHARTGYM_EVAL_DIR")
    if explicit:
        return Path(explicit)
    root = os.environ.get(
        "MOLMO_EXPERIMENT_DATA_DIR", "/weka/oe-training-default/donovanc/molmo-experimental-data"
    )
    return Path(root) / "chartgym" / "eval-v1"


def _norm(text: str) -> str:
    out = " ".join(str(text).strip().lower().split())
    # Matplotlib renders negatives with U+2212 unless axes.unicode_minus is off; normalize
    # both sides so a gold of "-5" can never be unmatchable by a model emitting "-5".
    return out.replace("−", "-").strip(" .\"'")


def _first_number(text: str) -> float | None:
    import re

    m = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(text).replace(",", ""))
    return float(m[0]) if m else None


def _expresses_na(text: str) -> bool:
    low = _norm(text)
    return any(marker in low for marker in _NA_MARKERS)


@dataclass(frozen=True)
class ChartGymScorer(Scorer):
    name: str = "chartgym"

    def score(self, instance: Instance, output: LMOutput) -> float:
        return float(self._score(getattr(output, "text", "") or "", instance.metadata))

    def _score(self, pred: str, meta: dict) -> bool:
        gold, atype = meta["answer"], meta["answer_type"]
        if atype == "na":
            return _expresses_na(pred)
        # A model that answers "not applicable" to an answerable question is wrong, and
        # saying so explicitly keeps the NA-bias regression visible rather than silent.
        if _expresses_na(pred) and not _expresses_na(gold):
            return False
        if atype == "int":
            n = _first_number(pred)
            return n is not None and abs(n - float(gold)) < 0.5
        if atype == "float":
            n = _first_number(pred)
            if n is None:
                return False
            tol = meta.get("tol", -1.0)
            g = float(gold)
            tol = abs(g) * 0.02 if tol is None or tol < 0 else tol
            return abs(n - g) <= max(tol, 1e-9)
        if atype == "yesno":
            p = _norm(pred)
            g = _norm(gold)
            said_yes = p.startswith("yes") or " yes" in p[:40]
            said_no = p.startswith("no") or " no " in p[:40]
            if said_yes == said_no:
                return False
            return said_yes == (g == "yes")
        if atype == "list":
            split = lambda s: sorted(  # noqa: E731
                filter(None, (_norm(p) for p in str(s).replace(";", ",").split(",")))
            )
            return split(pred) == split(gold)
        # verbatim / phrase / cat: the gold must appear, and for short golds the prediction
        # must not be padded with extra content that changes its meaning.
        g, p = _norm(gold), _norm(pred)
        return g == p or (len(g) > 3 and g in p)


_SCORER = ChartGymScorer()


@dataclass(frozen=True)
class ChartGymSubsetMetric(Metric):
    """Mean score over instances matching one metadata predicate.

    Same shape as ``ChartQaSubsetMetric``; the subset key is whichever metadata field the
    breakdown needs (capability, family, difficulty, held_out, is_na).
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    key: str | None = None
    value: object = None

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [
            r.scores.get(self.scorer.name, 0.0)
            for r in responses
            if self.key is None or r.instance.metadata.get(self.key) == self.value
        ]
        return sum(vals) / len(vals) if vals else 0.0


def _metrics() -> tuple[Metric, ...]:
    out: list[Metric] = [ChartGymSubsetMetric(name="chartgym_overall", scorer=_SCORER)]
    # The headline for the overfitting question -- see the module docstring.
    out.append(ChartGymSubsetMetric(
        name="chartgym_heldout_primitive", scorer=_SCORER, key="held_out", value=True))
    out.append(ChartGymSubsetMetric(
        name="chartgym_trained_families", scorer=_SCORER, key="held_out", value=False))
    out.append(ChartGymSubsetMetric(
        name="chartgym_not_applicable", scorer=_SCORER, key="is_na", value=True))
    out.append(ChartGymSubsetMetric(
        name="chartgym_answerable", scorer=_SCORER, key="is_na", value=False))
    for cap in _CAPABILITIES:
        out.append(ChartGymSubsetMetric(
            name=f"chartgym_{cap}", scorer=_SCORER, key="capability", value=cap))
    for d in _DIFFICULTIES:
        out.append(ChartGymSubsetMetric(
            name=f"chartgym_{d}", scorer=_SCORER, key="difficulty", value=d))
    return tuple(out)


@register("chartgym")
class ChartGymTask(ImageQATask):
    dependencies = ["pillow"]
    # 256, not 64. A stage-2 Molmo2 checkpoint answers these in prose ("The legend box in
    # the top right corner of the graph ...") and a 64-token cap truncates before the answer
    # appears, scoring 0 for verbosity rather than for error. That is the same confound that
    # made CharXiv template 11 look like a geometry failure when it was an applicability
    # failure -- measure the model, not the cap.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    metrics = _metrics()
    primary_metric = metrics[0]
    split = Split.VALIDATION

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        path = _dataset_dir()
        if not path.exists():
            raise FileNotFoundError(
                f"ChartGym eval corpus not found at {path}. Build it with "
                "chartgym/scripts/generate.py then chartgym/scripts/stage_eval.py, "
                "or point CHARTGYM_EVAL_DIR at an existing one."
            )
        ds = datasets.load_from_disk(str(path))
        if isinstance(ds, datasets.DatasetDict):
            ds = ds[next(iter(ds))]
        # `stage_eval.py` embeds the PNG bytes, so the dataset is self-contained and the
        # `path` cell is only a basename -- resolving it as a filesystem path would break
        # inside a Beaker job. Decode lazily from the cell, exactly as charxiv.py does.
        ds = ds.cast_column("image", datasets.Image(decode=False))
        for idx in range(len(ds)):
            ex = ds[idx]
            yield Instance(
                question=ex["question"],
                gold_answer=ex["answer"],
                metadata={
                    "answer": ex["answer"],
                    "answer_type": ex["answer_type"],
                    "tol": ex["tol"],
                    "family": ex["family"],
                    "capability": ex["capability"],
                    "held_out": ex["held_out"],
                    "is_na": ex["is_na"],
                    "difficulty": ex["difficulty"],
                    "example_id": ex["question_id"],
                    "image": lazy_hf_image(ds, idx, "image"),
                },
            )


register_variant("chartgym", "text_only", image_mode="none")
