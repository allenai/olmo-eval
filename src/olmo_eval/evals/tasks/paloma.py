"""Paloma perplexity tasks, one per Paloma source.

Paloma (Magnusson et al., 2023, https://huggingface.co/datasets/allenai/paloma)
is 16 held-out corpora. Each task here scores one source's text and reports its
token-weighted loss (nats per token) and byte-weighted bits per byte. The
``paloma`` suite averages those over the 16 sources with equal weight, the
"Paloma macro loss" that Levanter logs as ``eval/paloma/macro_loss`` and Marin
uses as its scaling-law target.

Long documents are split into chunks of at most ``MAX_CHUNK_CHARS`` characters,
broken at whitespace, and each chunk is scored from a BOS token. This plays the
role of Levanter packing eval text into fixed-length windows: every token is
scored and none is dropped to fit the context. The chunks join back to the
original text exactly, so bytes are counted once.
"""

import sys
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.metrics import BPBMetricByteAvg, CorpusLossMetric
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

#: Longest chunk scored as one request. At most one token per character for
#: almost all text, so a chunk fits an 8k-token context with room to spare.
MAX_CHUNK_CHARS = 8000

#: Paloma names its splits ``val`` and ``test``.
_PALOMA_SPLITS = {Split.VALIDATION: "val", Split.TEST: "test"}


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters.

    Each cut falls just after the last newline in the window, else the last
    space or tab, provided that leaves the chunk at least half full; otherwise
    the window is cut at ``max_chars``. The whitespace stays at the end of the
    earlier chunk, and ``"".join(split_text(text)) == text``.
    """
    if max_chars < 2:
        raise ValueError(f"max_chars must be at least 2, got {max_chars}")
    chunks: list[str] = []
    start = 0
    while len(text) - start > max_chars:
        window = text[start : start + max_chars]
        cut = window.rfind("\n")
        if cut < max_chars // 2:
            cut = max(window.rfind(" "), window.rfind("\t"))
        end = start + cut + 1 if cut >= max_chars // 2 else start + max_chars
        chunks.append(text[start:end])
        start = end
    if start < len(text):
        chunks.append(text[start:])
    return chunks


class PalomaBase(Task):
    """One Paloma source, scored as raw text in whitespace-bounded chunks."""

    split = Split.VALIDATION
    metrics = (CorpusLossMetric(), BPBMetricByteAvg())
    primary_metric = CorpusLossMetric()

    @property
    def request_type(self) -> RequestType:
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.LOGLIKELIHOOD

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield one instance per chunk of every document in the split."""
        if self._instances_cache is None:
            source = _split_source(self.config.data_source, self.config.split)
            self._instances_cache = []
            for doc_index, doc in enumerate(DataLoader().load(source)):
                chunks = split_text(doc["text"])
                for chunk_index, chunk in enumerate(chunks):
                    self._instances_cache.append(
                        self.process_chunk(
                            chunk,
                            index=len(self._instances_cache),
                            doc_id=doc.get("id", doc_index),
                            chunk_index=chunk_index,
                            num_chunks=len(chunks),
                        )
                    )
        yield from self._instances_cache

    def process_chunk(
        self, text: str, *, index: int, doc_id: Any, chunk_index: int, num_chunks: int
    ) -> Instance:
        """Convert one chunk of a document to an Instance scored as a continuation."""
        return Instance(
            question="",
            gold_answer=text,
            metadata={
                "id": index,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "num_chunks": num_chunks,
                "num_chars": len(text),
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        """Score the chunk as the continuation of an empty (BOS-only) prompt."""
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        gold = instance.gold_answer
        return LMRequest(
            request_type=self.request_type,
            prompt=instance.question,
            continuations=(gold,) if gold is not None else None,
        )


def _split_source(data_source: Any, split: Split) -> DataSource:
    """The files of one Paloma split, loaded on their own.

    ``load_dataset`` on a Paloma config builds every split, and the dolma-v1_5
    test split fails to build. Naming the split's files loads just those; the
    datasets library files them under ``train``.
    """
    if split not in _PALOMA_SPLITS:
        raise ValueError(f"Paloma has no {split} split; use validation or test")
    if not isinstance(data_source, DataSource) or data_source.subset is None:
        raise ValueError(f"Paloma tasks need a DataSource naming the subset, got {data_source!r}")
    return DataSource(
        path=data_source.path,
        subset=data_source.subset,
        split="train",
        data_files=f"{data_source.subset}/{_PALOMA_SPLITS[split]}/*",
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
