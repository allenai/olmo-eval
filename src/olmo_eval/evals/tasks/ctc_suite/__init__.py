"""
The CTC (corpus-tracking-capacity) long-context suite: 22 tasks, context ladders from 2k to 1M
tokens.

Each task gives the model a corpus of N documents in-prompt and asks a question whose difficulty
scales with how much of the corpus must be *simultaneously* tracked -- from O(N) retrieval (find
the answer-bearing passage) to O(N^2) relational tasks (find every contradicting pair) and O(NM)
structural ones (cluster everything). Every task has a ladder of context rungs; a rung label is the
measured median prompt length through the reference prompt path, not a document count.

**Provenance.** Prompt templates, answer parsers, per-task metrics, gold-index conventions and stop
rules are vendored verbatim from the ``ctc`` package (AI2 OLMo-core branch ``prasann/ctc``) under
``_vendor/``, where they are golden-fixture-tested against the pre-migration implementation that
produced the suite's published numbers. Do not edit the vendored files here; fix upstream and
re-vendor. One spec (plain ``grouping``) is registered locally below from the vendored factory.

**Data.** Private HF dataset ``PrasannSinghal/ctc-suite-eval``: one config per task, one split per rung
(``r2k`` ... ``r1m``). Set ``CTC_SUITE_DATA_ROOT=/path/to/ladders`` to read local
``<task>/rung_<tokens>.jsonl`` files instead (same files the HF dataset is built from).

**Comparability notes.**

* Rungs at 256k and above carry ``eval_size=125`` (seeded subsample of the same question set;
  binomial SE ~ +/-0.041 at f1~0.7) -- quote the size next to any number from them. ``scifact``
  is 300 examples and ``obliq_twitter`` 126 at every rung; same rule.
* Prompts here are the plain-text prompt path (``spec.build_prompt``). The original suite's
  vLLM/native runs additionally wrapped each document in Qwen-reserved marker tokens at
  tokenization time; scores are therefore comparable *within* this harness, and near but not
  bit-identical to the historical grid.
* ``query_position`` is pinned to ``"both"`` -- the setting every published rung was graded with.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

from ._vendor.ctc.eval.stopping import STOP_PRESETS
from ._vendor.ctc.eval.stopping import apply as _apply_stop
from ._vendor.ctc.format import registry as _ctc_registry
from ._vendor.ctc.format.prompts import GROUPING_INSTRUCTION
from ._vendor.ctc.tasks import load_all as _load_all_specs
from ._vendor.ctc.tasks._grouping import make_grouping_spec

__all__ = ["HF_DATASET", "ROSTER", "CTCScorer", "CTCMeanMetric", "CTCSuiteTask"]

#: Private HF dataset holding every rung file. One config per ROSTER row, one split per rung.
HF_DATASET = "PrasannSinghal/ctc-suite-eval"

#: Env var pointing at a local ladder tree (``<subset>/rung_<tokens>.jsonl``) for offline runs.
DATA_ROOT_ENV = "CTC_SUITE_DATA_ROOT"

#: The setting every published rung was graded with. Tasks that hardcode their own position
#: (qdmatch, grouping, outlier) ignore this, and declare so via ``honors_query_position``.
QUERY_POSITION = "both"

# Import the vendored spec registrations (side-effect: populates the ctc registry).
_load_all_specs()

# Plain ``grouping`` (OpenAlex, unlabeled clusters) is in the suite roster but not in the vendored
# canonical set -- register it from the vendored factory, mirroring grouping_labeled minus labels.
if "grouping" not in _ctc_registry.names():
    _ctc_registry.register(
        make_grouping_spec(
            name="grouping",
            description="Partition abstracts into their (unnamed) field clusters.",
            instruction=GROUPING_INSTRUCTION,
            rungs=("2k", "4k", "8k", "16k", "32k"),
            query_builder=lambda ex: (
                f"{GROUPING_INSTRUCTION}\n\n{ex['queries'][0]}"
                if ex.get("queries")
                else GROUPING_INSTRUCTION
            ),
            sources=("openalex",),
        )
    )


#: Rung label -> token budget in the local filenames. ``contradiction_iid`` aliases r2k to 2560:
#: its 2048-doc-count file sat below the training minimum, so the IID ladder starts one step up
#: (the remap is explicit in the source repo's ``build_suite_table.py`` as well).
RUNG_TOKENS: dict[str, int] = {
    "r2k": 2048,
    "r4k": 4096,
    "r8k": 8192,
    "r16k": 16384,
    "r32k": 32768,
    "r64k": 65536,
    "r128k": 131072,
    "r256k": 262144,
    "r512k": 524288,
    "r1m": 1048576,
}

_LADDER_2K_32K = ("r2k", "r4k", "r8k", "r16k", "r32k")
_LADDER_FULL = tuple(RUNG_TOKENS)  # 2k .. 1m
_LADDER_TO_512K = _LADDER_FULL[:-1]


@dataclass(frozen=True)
class RosterRow:
    """One row of the 22-task suite.

    :param subset: HF config name == local ladder directory name.
    :param spec: The vendored :class:`TaskSpec` that formats and grades this row.
    :param rungs: Rung labels with data, ascending.
    :param eval_size: ``{rung_label: rows}`` for any rung below the 500-example floor. A number
        quoted from one of these must carry the size inline -- a small eval inflates noise into
        apparent findings.
    :param rung_alias: Rung label -> token budget override for the local filename.
    :param note: Anything a reader of the numbers has to know.
    """

    subset: str
    spec: str
    rungs: tuple[str, ...] = _LADDER_2K_32K
    eval_size: dict[str, int] = field(default_factory=dict)
    rung_alias: dict[str, int] = field(default_factory=dict)
    note: str = ""


_SUB500_XLONG = {"r256k": 125, "r512k": 125, "r1m": 125}

#: The frozen 22-row roster (records/ctc-final-suite.md, 2026-08-12). Row order is figure order.
ROSTER: dict[str, RosterRow] = {
    "ctc_fiqa": RosterRow(
        subset="fiqa", spec="retrieval", rungs=_LADDER_FULL, eval_size=dict(_SUB500_XLONG)
    ),
    "ctc_nq": RosterRow(
        subset="nq", spec="retrieval", rungs=_LADDER_FULL, eval_size=dict(_SUB500_XLONG)
    ),
    "ctc_hpqa": RosterRow(
        subset="hotpotqa", spec="retrieval", rungs=_LADDER_FULL, eval_size=dict(_SUB500_XLONG)
    ),
    "ctc_qdmatch_fiqa": RosterRow(
        subset="qdmatch_fiqa",
        spec="qdmatch",
        rungs=_LADDER_TO_512K,
        eval_size=dict(_SUB500_XLONG),
        note="caps at 512k: the 6,148 recoverable BEIR-FiQA units are the whole labeled universe, "
        "and a 1M example needs ~8k distinct units",
    ),
    "ctc_qdmatch_nq": RosterRow(
        subset="qdmatch_nq", spec="qdmatch", rungs=_LADDER_FULL, eval_size=dict(_SUB500_XLONG)
    ),
    "ctc_qdmatch_hpqa": RosterRow(
        subset="qdmatch_hpqa",
        spec="qdmatch",
        rungs=_LADDER_FULL[:-2],
        eval_size=dict(_SUB500_XLONG),
        note="caps at 256k: 4,000 recoverable HotpotQA units, and a 512k example needs ~7k",
    ),
    "ctc_outlier_amzn": RosterRow(
        subset="outlier_amzn", spec="outlier", rungs=_LADDER_FULL, eval_size=dict(_SUB500_XLONG)
    ),
    "ctc_outlier": RosterRow(
        subset="outlier",
        spec="outlier",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
        note="64k+ rungs are the true scale-K construction (K ~ n/9); the retired fixed-K xlong "
        "files are not used",
    ),
    "ctc_outlier_fixedm": RosterRow(
        subset="outlier_fixedM",
        spec="outlier",
        rungs=_LADDER_TO_512K,
        eval_size=dict(_SUB500_XLONG),
        note="K pinned at 3 -- the control for the scale-K row. Caps at 512k: three majority "
        "topics need ~2,400 same-topic chunks each at 1M and the wiki pool cannot supply that",
    ),
    "ctc_oolong": RosterRow(
        subset="oolong", spec="oolong", rungs=_LADDER_FULL, eval_size=dict(_SUB500_XLONG)
    ),
    "ctc_grouping": RosterRow(subset="grouping", spec="grouping"),
    "ctc_absence": RosterRow(
        subset="absence_gutenberg",
        spec="absence",
        rungs=("r2k", "r4k", "r8k", "r16k"),
        note="corpus is a contiguous Gutenberg passage; rung ceiling is bounded by book length",
    ),
    "ctc_xabsence": RosterRow(
        subset="xabsence",
        spec="xabsence",
        note="the one-sided EXACT-COPY Gutenberg build (2026-08-14): B-side twins are "
        "byte-identical and orphans sit A-side only. Supersedes the PubMed paraphrase build, "
        "kept in the dataset as the xabsence_paraphrase config",
    ),
    "ctc_rerank": RosterRow(
        subset="rerank", spec="rerank", rungs=_LADDER_TO_512K, eval_size=dict(_SUB500_XLONG)
    ),
    "ctc_msmarco": RosterRow(
        subset="msmarco", spec="retrieval", rungs=_LADDER_TO_512K, eval_size=dict(_SUB500_XLONG)
    ),
    "ctc_reorder": RosterRow(
        subset="reorder",
        spec="reorder",
        rungs=("r2k", "r4k", "r8k", "r16k"),
        note="chunks are contiguous in one book; rung ceiling is bounded by book length",
    ),
    "ctc_obliq": RosterRow(
        subset="obliq_twitter",
        spec="retrieval",
        rungs=_LADDER_FULL,
        eval_size={**{r: 126 for r in _LADDER_FULL[:7]}, "r256k": 125, "r512k": 125, "r1m": 125},
        note="126 examples at every rung -- flag the size inline",
    ),
    "ctc_niah": RosterRow(
        subset="niah", spec="retrieval", rungs=_LADDER_FULL, eval_size=dict(_SUB500_XLONG)
    ),
    "ctc_contradiction": RosterRow(
        subset="contradiction_iid",
        spec="contradiction",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
        rung_alias={"r2k": 2560},
        note="the IID realistic-mode ladder; never mix with the retired both-mode ladder",
    ),
    "ctc_strmatch": RosterRow(subset="strmatch", spec="strmatch"),
    "ctc_textgroups": RosterRow(subset="textgroups", spec="textgroups"),
    "ctc_scifact": RosterRow(
        subset="scifact",
        spec="retrieval",
        eval_size={r: 300 for r in _LADDER_2K_32K},
        note="300 examples at every rung -- flag the size inline",
    ),
}

#: Which rung a bare task name (no variant) evaluates: the top of the 2k-32k figure ladder.
DEFAULT_RUNG = "r32k"


def _resolve_spec(spec_name: str):
    return _ctc_registry.get(spec_name)


def _data_source(row: RosterRow, rung: str) -> DataSource:
    """The canonical HF source. The split IS the rung label, which is also how
    :meth:`CTCSuiteTask._load_instances` knows which local file to substitute when
    ``CTC_SUITE_DATA_ROOT`` is set (checked at load time, not import time)."""
    return DataSource(path=HF_DATASET, subset=row.subset, split=rung)


@dataclass(frozen=True)
class CTCScorer(Scorer):
    """Parse a generation with the task's own parser and score it with the task's own metric.

    The three calls mirror the reference runner exactly: stop-rule cleanup, then
    ``spec.parse(text, n_docs)``, then ``spec.score(parsed, gold)`` where the gold field is the
    spec's declared one (``gold_doc_indices`` for most tasks, ``gold_pairs`` for qdmatch). A
    ``None`` parse scores 0 on every metric, which is the reference behaviour -- but parse *rate*
    should be watched separately: a parse-rate collapse is a decoding/stopping regression wearing
    an accuracy drop's clothes.
    """

    name: str = "ctc"
    spec_name: str = ""

    def score(self, instance: Instance, output: LMOutput) -> float:
        spec = _resolve_spec(self.spec_name)
        example = instance.metadata["example"]
        cleaned = _apply_stop(output.text or "", STOP_PRESETS[spec.stop])
        parsed = spec.parse(cleaned, len(example["documents"]))
        gold_field = spec.extra.get("gold_field", "gold_doc_indices")
        # When the declared field is absent or empty (oolong's gold lives in _meta.gold_list /
        # answers), hand score() the whole example -- the specs that need that path document
        # accepting "the example carrying them".
        gold = example.get(gold_field) or example
        scored = spec.score(parsed, gold)
        value = scored.get(spec.primary_metric, 0.0)
        if output.metadata is None:
            output.metadata = {}
        output.metadata["ctc_parse_ok"] = parsed is not None
        output.metadata["ctc_all_metrics"] = {k: float(v) for k, v in scored.items()}
        return float(value)


@dataclass(frozen=True, slots=True)
class CTCMeanMetric(Metric):
    """Mean of the task's primary metric across responses (named for that metric, e.g. mrr@10)."""

    name: str = "ctc"
    scorer: type[Scorer] | Scorer = CTCScorer()

    def compute(self, responses) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


class CTCSuiteTask(Task):
    """Base class for every suite row. Subclasses set ``row_name`` and inherit everything else."""

    row_name: str = ""

    @property
    def row(self) -> RosterRow:
        return ROSTER[self.row_name]

    @property
    def spec(self):
        return _resolve_spec(self.row.spec)

    @property
    def instances(self) -> Iterator[Instance]:
        yield from self._load_instances_cached()

    def _load_instances(self, split: str | None = None) -> Iterator[Instance]:
        """Same as the base loader, except ``CTC_SUITE_DATA_ROOT`` (read here, at load time)
        substitutes the local ``<subset>/rung_<tokens>.jsonl`` file for the HF split -- for
        offline runs and for validating a ladder before it is uploaded."""
        root = os.environ.get(DATA_ROOT_ENV)
        if not root:
            yield from super()._load_instances(split=split)
            return
        from olmo_eval.data import DataLoader

        rung = self.config.data_source.split  # the split label IS the rung label
        tokens = self.row.rung_alias.get(rung, RUNG_TOKENS[rung])
        path = os.path.join(root, self.row.subset, f"rung_{tokens}.jsonl")
        source = DataSource(path=path, split="train")  # a bare JSONL is a single unnamed split
        for index, doc in enumerate(DataLoader().load(source)):
            instance = self.process_doc(doc, index)
            if instance is not None:
                yield instance

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        # The raw unified example travels whole: the prompt builder, parser and scorer all read
        # different parts of it, and slicing it here is how field conventions historically drifted.
        return Instance(
            question=doc["queries"][0] if doc.get("queries") else "",
            gold_answer=None,
            metadata={"id": index, "example": doc},
        )

    def format_request(self, instance: Instance) -> LMRequest:
        prompt = self.spec.build_prompt(instance.metadata["example"], query_position=QUERY_POSITION)
        return LMRequest(request_type=RequestType.COMPLETION, prompt=prompt)

    def get_sampling_params(self, instance: Instance) -> SamplingParams | None:
        # No decode-time text stops on purpose: the reference stop rules suppress text stops
        # inside unclosed <think> blocks and before first content, which a decode-time stop string
        # cannot honour. The same rules run post-hoc in CTCScorer instead; the only cost is decode
        # tokens on models that never emit EOS, bounded by max_tokens.
        stop = STOP_PRESETS[self.spec.stop]
        return SamplingParams(
            max_tokens=max(stop.max_new_tokens, self.spec.max_new_tokens),
            temperature=0,
        )


def _make_row_task(task_name: str, row: RosterRow) -> None:
    spec = _resolve_spec(row.spec)
    metric = CTCMeanMetric(name=spec.primary_metric, scorer=CTCScorer(spec_name=row.spec))

    cls = type(
        f"CTC_{row.subset}",
        (CTCSuiteTask,),
        {
            "__doc__": f"CTC suite row {row.subset!r} ({row.spec} spec). {row.note}".strip(),
            "row_name": task_name,
            "data_source": _data_source(
                row, DEFAULT_RUNG if DEFAULT_RUNG in row.rungs else row.rungs[-1]
            ),
            "metrics": (metric,),
            "primary_metric": metric,
        },
    )
    register(task_name)(cls)
    for rung in row.rungs:
        register_variant(task_name, rung, data_source=_data_source(row, rung))


for _name, _row in ROSTER.items():
    _make_row_task(_name, _row)
