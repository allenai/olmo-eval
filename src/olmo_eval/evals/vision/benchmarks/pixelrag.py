"""PixelRAG reader benchmark: Wikipedia QA answered from retrieved screenshots.

Paper: https://arxiv.org/abs/2606.28344; data: https://huggingface.co/datasets/StarTrail-org/pixelrag-bench.

Each question comes with the Wikipedia screenshot tiles the PixelRAG retriever returned for it
(slices of rendered pages, 875 px wide and at most 1024 px tall), so the model is evaluated as
the reader of a visual RAG pipeline: it has to find and read the answer in the tiles, which
usually include distractors and sometimes miss the answer. One task per source benchmark:

* ``pixelrag_simpleqa`` — SimpleQA questions whose sources include a Wikipedia page;
* ``pixelrag_nq`` — Natural Questions validation;
* ``pixelrag_nq_tables`` — NQ-Tables dev, answered from tables;
* ``pixelrag_mmsearch`` — MMSearch end2end, most questions with a query image;
* ``pixelrag_evqa`` — Encyclopedic-VQA's automatic Google Landmarks questions, each with a query
  photo (the card says test; the ids rebuild from the val split).

The reader protocol is the paper's (``build_messages`` in the official ``eval/lib/llm.py``): the
top 3 tiles, the paper's system prompt, and a user turn of the question, the tiles, then the
subset's answer-format instruction. A question with a query image gets the query-image system
prompt and one text block (question, a note on the images, the instruction) followed by the query
image and the tiles. Images are sent as stored; a tile missing from the dataset is left out.
Decoding is greedy with the budgets of the official ``reproduce.sh``: 200 new tokens for
SimpleQA, NQ and NQ-Tables, 16,384 for MMSearch and EVQA (where the paper also turns on Qwen3.5's
thinking for MMSearch; models without a thinking mode answer directly). Grading is the official
GPT-4.1 judge's prompts (:mod:`olmo_eval.evals.vision.scoring.pixelrag`), called through the
shared judge: ``accuracy`` is the fraction graded correct, the paper's number, and NQ and
NQ-Tables also report exact match, which needs no judge. The judge model, reply budget and
temperature are task configuration (``-o judge_model=...``, ``OLMO_EVAL_JUDGE``), so they reach
the task hash, and a judge call that fails leaves the response unscored rather than wrong. The
official grader also sends ``seed=42``; the shared judge sends no seed.

How the turn reaches the model depends on its chat template: one without a system role (Molmo2)
gets no system prompt, and Molmo2's places every image before the text.

The questions are instruction-following QA over several images, so the tasks target
instruction-tuned checkpoints and have no stage-1 prompt form. The dataset revision, the number of
tiles and the reader prompts are ``task_settings``, so they are part of the task hash too. Data is
fetched from the Hub at the pinned revision; ``PIXELRAG_BENCH_DIR`` reads a local copy of the
dataset repository instead, which must match the pinned revision file for file (SHA-256).
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.llm_judge import Judge
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.common.base import _store_output_score
from olmo_eval.evals.tasks.common.judge import (
    JudgedScoreMetric,
    JudgedTask,
    JudgeScorer,
    TruncationRateMetric,
)
from olmo_eval.evals.vision.scoring.pixelrag import (
    EXACT_MATCH_SCORER_NAME,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_SYSTEM_MESSAGE,
    SCORER_NAME,
    Grader,
    PixelRagExactMatchScorer,
    judge_prompt,
    parse_verdict,
)
from olmo_eval.evals.vision.tasks.base import VisionTask

HF_REPO = "StarTrail-org/pixelrag-bench"
HF_REVISION = "43581cfcdd39f7e1e3734fa3e5e7e13b1a41a032"

#: SHA-256 of every parquet file at ``HF_REVISION``, by dataset config and file stem: a
#: local copy must match it.
DATA_MANIFEST: dict[str, dict[str, str]] = {
    "encyclopedic_vqa_landmarks": {
        "test-00000-of-00007": "e5ef3915f13dab2a4fe4127b9cf7988383d38381bdc95e7a7a6d6044c1722239",
        "test-00001-of-00007": "a7eb2304d2e23ce33da8d7e01dfc5b310868d958b7d4390782ff21c7c1fcece2",
        "test-00002-of-00007": "e1e6dc51f57656a428e33e121abe880d5cd091dc02957602c241af61b65c9e62",
        "test-00003-of-00007": "a199ef6a194378e6e0d97c62835db04a6158419735b70d9beb6eccd7d89f6aa3",
        "test-00004-of-00007": "464c2d9ee8392da517d71044d178501fa628a614f2b3f240f3c936fd514311e6",
        "test-00005-of-00007": "1d77e499f0e1c99cfceb94583c8ef4a3cde43db18fba566590d313b29db47d9e",
        "test-00006-of-00007": "6e356c6af4a5256d1d2050ea7294605dcc38287f5db3e3307a765ff49cda5db6",
    },
    "mmsearch": {
        "test-00000-of-00001": "a0dbdc07bb2b59be114f558b1a82b16e1e1245151b5b103de473af58dba75a8b",
    },
    "nq": {
        "test-00000-of-00003": "46c1db275a82809b5252120124c4d35daffaba54a3bd56dc7ca20cb59419a482",
        "test-00001-of-00003": "9c365f6a1b7d3b0551301745aa9949750ae390c26b5f95d30be0f0321c6d325a",
        "test-00002-of-00003": "f9682229891fc645cd72f3044bd676457dca3603c7e1026e98ad6243e6dd6889",
    },
    "nq_tables": {
        "test-00000-of-00003": "20003d6fe42ca67ca5d415d6d995f91435b0a2d563726bf70a52ffb14d163745",
        "test-00001-of-00003": "6153119976b69f62a0aa26d5fa4825a485083ac75d8992d0bbf1163172119194",
        "test-00002-of-00003": "8817b5ca5cf2fdd59be730a67415c6e1ddc12ce57159441614902a6416561873",
    },
    "simpleqa": {
        "test-00000-of-00003": "6b8befe2a0f2383b1c4b1e6502ce8b394df0c7aa3447ef8e97a4d4963fa7cba1",
        "test-00001-of-00003": "a20234779180581922334f84bb4d674f6800be1154585dcd633c63e20d9fa4cd",
        "test-00002-of-00003": "5f33ad5a42621795bc49569f6af62f3f93b1c7709ef572c8a29dbdd4438b5548",
    },
}

#: Tiles the paper's reader sees, best first.
READER_TOP_K = 3

#: ``SYSTEM_PROMPT_EVIDENCE_QA`` of ``eval/lib/llm.py``, verbatim.
SYSTEM_PROMPT = (
    "You are a research assistant who answers questions based on provided evidence.\n"
    "Use <think></think> tags to show your reasoning if needed.\n"
    "Answer the question directly and concisely based ONLY on the provided evidence.\n"
)
#: ``SYSTEM_PROMPT_MULTIMODAL_QUERY`` of ``eval/lib/llm.py``, verbatim.
SYSTEM_PROMPT_QUERY_IMAGE = (
    "You are a research assistant who answers questions based on retrieved visual evidence.\n"
    "You will receive: (1) a text question, (2) a query image, and (3) retrieved Wikipedia "
    "evidence images.\n"
    "Use the query image and evidence images to answer the question.\n"
    "Use <think></think> tags to show your reasoning if needed.\n"
    "Answer the question directly and concisely.\n"
)
#: The note ``build_messages`` adds when a query image comes with retrieved tiles.
QUERY_IMAGE_NOTE = (
    "The first image is the query image. The following images are retrieved Wikipedia "
    "evidence. Answer the question based on the evidence."
)

_IMAGE_PART = {"type": "image"}
_SCORER = JudgeScorer(name=SCORER_NAME)
_EXACT_MATCH_SCORER = PixelRagExactMatchScorer()

# The runner can prepare several of these tasks at once, and datasets' loader is not safe to
# run concurrently on one cache; serialize the load.
_LOAD_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.cache
def verify_local_copy(root: str, config: str, revision: str) -> None:
    """Fail unless ``root`` holds the pinned files of ``config`` with their SHA-256.

    Only the pinned revision has a manifest, so a local copy of any other is refused.
    """
    if revision != HF_REVISION:
        raise ValueError(
            f"PIXELRAG_BENCH_DIR can only stand in for revision {HF_REVISION}, not {revision}"
        )
    expected = DATA_MANIFEST.get(config)
    if expected is None:
        raise ValueError(f"{HF_REPO} has no config {config!r}")
    folder = Path(root) / config
    problems = []
    for stem, sha256 in sorted(expected.items()):
        path = folder / f"{stem}.parquet"
        if not path.is_file():
            problems.append(f"{config}/{stem}.parquet: missing")
        elif (actual := _sha256(path)) != sha256:
            problems.append(f"{config}/{stem}.parquet: sha256 {actual[:12]} != {sha256[:12]}")
    problems.extend(
        f"{config}/{path.name}: not in the pinned revision"
        for path in sorted(folder.glob("*.parquet"))
        if path.stem not in expected
    )
    if problems:
        raise ValueError(f"{root} does not match {HF_REPO}@{revision[:12]}: " + "; ".join(problems))


def _load_subset(config: str, revision: str):
    import datasets

    local = os.environ.get("PIXELRAG_BENCH_DIR")
    with _LOAD_LOCK:
        if local:
            verify_local_copy(local, config, revision)
            ds = datasets.load_dataset(local, config, split="test")
        else:
            ds = datasets.load_dataset(HF_REPO, config, split="test", revision=revision)
    # No cast to undecoded images: casting a multi-GB image column concatenates it into one
    # Arrow array, past its 2 GB offset limit. Images are read from Arrow as raw bytes instead.
    return ds


def _has_bytes(images) -> list[bool]:
    """Per entry of an Arrow image-struct array, whether it holds image bytes."""
    valid = images.is_valid().to_pylist()
    with_bytes = images.field("bytes").is_valid().to_pylist()
    return [a and b for a, b in zip(valid, with_bytes, strict=True)]


def _present_tile_ranks(dataset, top_k: int) -> list[tuple[int, ...]]:
    """Per row, the ranks among the top ``top_k`` tiles that have image bytes.

    Read through Arrow so that no image bytes are copied.
    """
    ranks: list[tuple[int, ...]] = []
    for chunk in dataset.data.column("retrieved_images").chunks:
        present = _has_bytes(chunk.values)
        offsets = chunk.offsets.to_pylist()
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            ranks.append(
                tuple(rank for rank in range(min(end - start, top_k)) if present[start + rank])
            )
    return ranks


def _present_query_images(dataset) -> list[bool]:
    """Per row, whether it has a query image (read through Arrow, like the tiles)."""
    present: list[bool] = []
    for chunk in dataset.data.column("query_image").chunks:
        present.extend(_has_bytes(chunk))
    return present


def _decode(record: dict[str, Any]):
    from PIL import Image

    return Image.open(io.BytesIO(record["bytes"]))


def _load_images(dataset, index: int, tile_ranks: tuple[int, ...], with_query: bool) -> list:
    """The query image (if any) then the row's tiles, decoded (module-level so it is picklable).

    The row is read from the Arrow table, so only the images sent are decoded.
    """
    row = dataset.data.slice(index, 1).select(["query_image", "retrieved_images"]).to_pylist()[0]
    images = [_decode(row["query_image"])] if with_query else []
    return images + [_decode(row["retrieved_images"][rank]) for rank in tile_ranks]


def _text(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def reader_messages(
    question: str,
    instructions: str,
    num_tiles: int,
    *,
    query_image: bool,
    settings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """The official reader conversation, with ``{"type": "image"}`` parts where images go.

    ``settings`` supplies the prompts (``system_prompt``, ``query_image_system_prompt``,
    ``query_image_note``); by default the official ones.
    """
    settings = settings or reader_settings("", "worldvqa")
    if query_image:
        text = f"Question: {question}\n\n{settings['query_image_note']}"
        if instructions:
            text += f"\n\n{instructions}"
        system = settings["query_image_system_prompt"]
        content = [_text(text), _IMAGE_PART, *([_IMAGE_PART] * num_tiles)]
    else:
        system = settings["system_prompt"]
        content = [_text(question), *([_IMAGE_PART] * num_tiles)]
        if instructions:
            content.append(_text(instructions))
    return ({"role": "system", "content": system}, {"role": "user", "content": content})


def reader_settings(subset: str, grader: Grader) -> dict[str, Any]:
    """A subset's data and reader protocol, as ``task_settings`` (part of the task hash)."""
    return {
        "dataset": HF_REPO,
        "revision": HF_REVISION,
        "subset": subset,
        "reader_top_k": READER_TOP_K,
        "system_prompt": SYSTEM_PROMPT,
        "query_image_system_prompt": SYSTEM_PROMPT_QUERY_IMAGE,
        "query_image_note": QUERY_IMAGE_NOTE,
        "grader": grader,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LowerIsBetterScoreMetric(JudgedScoreMetric):
    """A judged rate where lower is better (wrong or unattempted answers)."""

    def higher_is_better(self) -> bool:
        return False


_ACCURACY = JudgedScoreMetric(name="accuracy", scorer=_SCORER, score_key="accuracy")
_JUDGE_METRICS: tuple[Metric, ...] = (
    _ACCURACY,
    LowerIsBetterScoreMetric(name="incorrect", scorer=_SCORER, score_key="incorrect"),
    LowerIsBetterScoreMetric(name="not_attempted", scorer=_SCORER, score_key="not_attempted"),
    TruncationRateMetric(scorer=_SCORER),
)
#: Its own scorer, so a failed judge call does not drop a response from it.
_EM_METRIC = JudgedScoreMetric(
    name="exact_match", scorer=_EXACT_MATCH_SCORER, score_key=EXACT_MATCH_SCORER_NAME
)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class PixelRagTask(VisionTask, JudgedTask):
    """One subset of the PixelRAG reader benchmark."""

    dependencies = ["pillow", "openai"]
    required_secrets = ("OPENAI_API_KEY",)
    #: The official budget for the short-answer subsets; MMSearch and EVQA allow long answers.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    metrics = _JUDGE_METRICS
    primary_metric = _ACCURACY
    split = Split.TEST
    judge_model = JUDGE_MODEL
    judge_max_tokens = JUDGE_MAX_TOKENS
    judge_scorer_name = SCORER_NAME
    judge_score_key = "accuracy"
    judge_extra_score_keys = ("incorrect", "not_attempted")
    task_settings = reader_settings("", "worldvqa")

    @property
    def _settings(self) -> dict[str, Any]:
        settings = self.config.task_settings
        if not settings or not settings.get("subset"):
            raise ValueError(f"{self.config.name} has no PixelRAG task_settings")
        return settings

    def _build_instances(self) -> Iterator[Instance]:
        settings = self._settings
        subset = settings["subset"]
        ds = _load_subset(subset, settings["revision"])
        rows = ds.select_columns(["id", "question", "instructions", "answer", "original_data"])
        has_query = _present_query_images(ds)
        tile_ranks = _present_tile_ranks(ds, int(settings["reader_top_k"]))
        for index, row in enumerate(rows):
            source = json.loads(row["original_data"])["original_data"]
            with_query = bool(has_query[index])
            yield Instance(
                question=source["problem"],
                gold_answer=row["answer"],
                metadata={
                    "id": row["id"],
                    "example_id": row["id"],
                    "subset": subset,
                    "problem": source["problem"],
                    # The ground truth exactly as the official judge receives it.
                    "answer": row["answer"],
                    "gold_answers": source.get("gold_answers") or [],
                    "instructions": row["instructions"] or "",
                    "has_query_image": with_query,
                    "tile_ranks": tile_ranks[index],
                    "images": functools.partial(
                        _load_images, ds, index, tile_ranks[index], with_query
                    ),
                },
            )

    def format_request(self, instance: Instance) -> LMRequest:
        meta = instance.metadata
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=reader_messages(
                meta["problem"],
                meta["instructions"],
                len(meta["tile_ranks"]),
                query_image=meta["has_query_image"],
                settings=self._settings,
            ),
            images=(meta["images"],),
        )

    async def _score_output(self, response: Response, output: LMOutput, judge: Judge) -> float:
        meta = response.instance.metadata
        grader: Grader = self._settings["grader"]
        judge_result: dict[str, Any] = {"scorer": SCORER_NAME, "grader": grader}
        verdict = await self._judge_output(
            output,
            judge,
            judge_prompt(
                grader, question=meta["problem"], ground_truth=meta["answer"], response=output.text
            ),
            lambda reply: parse_verdict(grader, reply),
            judge_result,
            system_prompt=JUDGE_SYSTEM_MESSAGE,
            raw_key="judge_reply",
        )
        scores = {
            "accuracy": float(verdict == "correct"),
            "incorrect": float(verdict == "incorrect"),
            "not_attempted": float(verdict == "unattempted"),
        }
        judge_result["verdict"] = verdict
        return self._record_judgement(output, judge_result, scores)

    async def score_responses(self, responses: Sequence[Response], context: Any = None):
        responses = await super().score_responses(responses, context)
        if any(metric.scorer() is _EXACT_MATCH_SCORER for metric in self.config.metrics):
            # Exact match needs no judge, so every output gets it, judged or not.
            for response in responses:
                for output in response.outputs:
                    _store_output_score(
                        output,
                        scorer_name=EXACT_MATCH_SCORER_NAME,
                        score=_EXACT_MATCH_SCORER.score(response.instance, output),
                    )
                if response.outputs:
                    response.scores[EXACT_MATCH_SCORER_NAME] = response.outputs[0].metadata[
                        f"score:{EXACT_MATCH_SCORER_NAME}"
                    ]
        return responses


@register("pixelrag_simpleqa")
class PixelRagSimpleQaTask(PixelRagTask):
    """SimpleQA questions whose sources include a Wikipedia page, graded by the SimpleQA judge."""

    task_settings = reader_settings("simpleqa", "simpleqa")


@register("pixelrag_nq")
class PixelRagNqTask(PixelRagTask):
    """Natural Questions validation, graded by the judge as in the paper, plus exact match."""

    metrics = (*_JUDGE_METRICS, _EM_METRIC)
    task_settings = reader_settings("nq", "worldvqa")


@register("pixelrag_nq_tables")
class PixelRagNqTablesTask(PixelRagTask):
    """NQ-Tables dev, graded by the judge as in the paper, plus exact match."""

    metrics = (*_JUDGE_METRICS, _EM_METRIC)
    task_settings = reader_settings("nq_tables", "worldvqa")


@register("pixelrag_mmsearch")
class PixelRagMmSearchTask(PixelRagTask):
    """MMSearch end2end; questions with a query image send it before the tiles."""

    sampling_params = SamplingParams(temperature=0.0, max_tokens=16384)
    task_settings = reader_settings("mmsearch", "worldvqa")


@register("pixelrag_evqa")
class PixelRagEvqaTask(PixelRagTask):
    """Encyclopedic-VQA's Google Landmarks questions, each with its query photo."""

    sampling_params = SamplingParams(temperature=0.0, max_tokens=16384)
    task_settings = reader_settings("encyclopedic_vqa_landmarks", "worldvqa")
