"""Paloma perplexity tasks, one per Paloma source.

Paloma (Magnusson et al., 2023, https://arxiv.org/abs/2312.10523,
https://huggingface.co/datasets/allenai/paloma) is AI2's benchmark of 16
held-out corpora. Each task here scores one source's val text and reports its
byte-weighted bits per byte, which unlike loss per token does not depend on the
tokenizer. The ``paloma`` suite averages that over the 16 sources with equal
weight, so no one source dominates.

Following Paloma's inference format, documents are scored individually rather
than packed together, and a document longer than the model's context is split
into disjoint inputs of the context's length. Each document is one rolling
request: the provider scores it from a BOS token in windows of its context, so
every token is scored once and none is dropped, and the window follows each
model's context length.
"""

import sys
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import BPBMetricByteAvg
from olmo_eval.common.types import Instance, LMRequest, RequestType, Split
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.common import Task, register

PALOMA_PATH = "allenai/paloma"
PALOMA_REVISION = "65cd6fc59dba021b21db414fa5e8d7765ffbe5e6"

#: Paloma's 16 sources, as named by the dataset's configs.
PALOMA_SUBSETS: tuple[str, ...] = (
    "4chan_meta_sep",
    "c4_100_domains",
    "c4_en",
    "dolma-v1_5",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "falcon-refinedweb",
    "gab",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep",
    "mc4",
    "ptb",
    "redpajama",
    "twitterAAE_HELM_fixed",
    "wikitext_103",
)

#: Paloma names its splits ``val`` and ``test``.
_PALOMA_SPLITS = {Split.VALIDATION: "val", Split.TEST: "test"}

#: Files the datasets library cannot load because they hold no documents. At the
#: pinned revision dolma-v1_5's test split has an empty books file (42 bytes of
#: gzip), so that split names its other five files instead of a glob.
_SPLIT_FILES: dict[tuple[str, str], tuple[str, ...]] = {
    ("dolma-v1_5", "test"): tuple(
        f"dolma-v1_5/test/test_{name}.jsonl.gz"
        for name in ("common-crawl", "pes2o", "reddit_uniform", "stack_uniform", "wiki")
    ),
}


class PalomaBase(Task):
    """One Paloma source, each document scored as raw text in full."""

    split = Split.VALIDATION
    metrics = (BPBMetricByteAvg(),)
    primary_metric = BPBMetricByteAvg()

    @property
    def request_type(self) -> RequestType:
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.LOGLIKELIHOOD

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield one instance per document in the split."""
        if self._instances_cache is None:
            source = _split_source(self.config.data_source, self.config.split)
            self._instances_cache = [
                self.process_doc(doc, index) for index, doc in enumerate(DataLoader().load(source))
            ]
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance:
        """Convert one document to an Instance scored as a whole text."""
        text = doc["text"]
        return Instance(
            question="",
            gold_answer=text,
            metadata={
                "id": index,
                "doc_id": doc.get("id", index),
                "num_bytes": len(text.encode("utf-8")),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Score the document from BOS in windows of the model's context."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        gold = instance.gold_answer
        if gold is None:
            raise ValueError(f"Paloma instance {instance.metadata.get('id')} has no text")
        return LMRequest(
            request_type=self.request_type,
            continuations=(gold,),
            rolling=True,
        )


def _split_source(data_source: Any, split: Split) -> DataSource:
    """The files of one Paloma split, loaded on their own.

    ``load_dataset`` on a Paloma config builds every split, so one unloadable
    file breaks them all. Naming the split's files loads just those; the
    datasets library files them under ``train``.
    """
    if split not in _PALOMA_SPLITS:
        raise ValueError(f"Paloma has no {split} split; use validation or test")
    if not isinstance(data_source, DataSource) or data_source.subset is None:
        raise ValueError(f"Paloma tasks need a DataSource naming the subset, got {data_source!r}")
    paloma_split = _PALOMA_SPLITS[split]
    data_files = _SPLIT_FILES.get(
        (data_source.subset, paloma_split), f"{data_source.subset}/{paloma_split}/*"
    )
    return DataSource(
        path=data_source.path,
        subset=data_source.subset,
        split="train",
        data_files=data_files,
        revision=data_source.revision,
    )


def _register_paloma_tasks() -> None:
    """Register ``paloma_<subset>`` for every Paloma source, pinned to one revision."""
    for subset in PALOMA_SUBSETS:
        class_name = f"Paloma_{subset.replace('-', '_')}"
        cls = type(
            class_name,
            (PalomaBase,),
            {
                "data_source": _split_source(
                    DataSource(path=PALOMA_PATH, subset=subset, revision=PALOMA_REVISION),
                    Split.VALIDATION,
                ),
                "__module__": __name__,
                "__qualname__": class_name,
            },
        )
        # Generated classes must be importable by name to pickle.
        setattr(sys.modules[__name__], class_name, cls)
        register(f"paloma_{subset}")(cls)


_register_paloma_tasks()
