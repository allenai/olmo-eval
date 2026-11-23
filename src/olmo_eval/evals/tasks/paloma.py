"""Paloma (https://arxiv.org/abs/2312.10523) bits per byte, one task per source.

Each document is scored on its own as one rolling request, so a document longer
than the context is scored in full in windows of the context. A source is
byte-weighted over all its documents; the paper instead averages over a source's
domains, which the saved ``subdomain`` lets you recompute.
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

#: Paloma's sources, as named by the dataset's configs.
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
        return Instance(
            question="",
            gold_answer=doc["text"],
            metadata={
                "id": index,
                "doc_id": doc.get("id", index),
                "subdomain": doc.get("subdomain"),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Score the document from BOS in windows of the context or ``max_length``."""
        if self.config.formatter is not None:
            # A formatter's request would not roll, so long documents would be truncated.
            raise ValueError("Paloma tasks score raw text and do not take a formatter")
        if self.config.num_fewshot:
            raise ValueError("Paloma tasks score raw text and do not take few-shot examples")
        gold = instance.gold_answer
        if gold is None:
            raise ValueError(f"Paloma instance {instance.metadata.get('id')} has no text")
        return LMRequest(
            request_type=self.request_type,
            continuations=(gold,),
            max_length=self.config.max_length,
            rolling=True,
        )


def _split_source(data_source: DataSource, split: Split) -> DataSource:
    """The files of one Paloma split, loaded on their own.

    ``load_dataset`` on a Paloma config builds every split, so one unloadable
    file breaks them all. Naming the split's files loads just those; the
    datasets library files them under ``train``.
    """
    if split not in _PALOMA_SPLITS:
        raise ValueError(f"Paloma has no {split} split; use validation or test")
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
                # The split's files are chosen at load time, so the stored config
                # does not claim the val files for a test-split run.
                "data_source": DataSource(
                    path=PALOMA_PATH, subset=subset, revision=PALOMA_REVISION
                ),
                "__module__": __name__,
                "__qualname__": class_name,
            },
        )
        # Generated classes must be importable by name to pickle.
        setattr(sys.modules[__name__], class_name, cls)
        register(f"paloma_{subset}")(cls)


_register_paloma_tasks()
