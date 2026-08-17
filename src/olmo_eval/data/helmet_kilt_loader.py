"""HELMET RAG (KILT) data loading.

Ports HELMET's `load_qa` (https://github.com/princeton-nlp/HELMET, data.py):
an open-domain question plus a stack of retrieved Wikipedia passages, where the
model must answer from the passages.

Context length is set by how many passages were retrieved, not by truncation,
so each length tier is a separate pre-retrieved file. That retrieval was run
once by the HELMET authors and is re-hosted unpacked on `allenai/helmet-plus`;
see `kilt/manifest.json` there for the tier -> file mapping. These files are
large (the 128k tiers are 1-3 GB each) but are cached by huggingface_hub after
the first download.

Because retrieval depth is fixed in the data, these tasks cannot be extended
past standard HELMET's 128k the way the synthetic recall task can.

**`max_samples` caps questions, not instances.** Each question appears several
times in these files with the gold passage planted at different relative
depths -- 6 for nq/triviaqa/popqa (the `dep6` in the filenames: 0.0, 0.2, 0.4,
0.6, 0.8, 0.95) and 3 for hotpotqa -- so the score averages over where in the
context the answer sits, which is the "lost in the middle" effect these tasks
exist to measure. HELMET samples by question and keeps every depth for the
questions it picks, and that is reproduced here: a limit of 100 yields ~600
instances for nq, not 100. Budget accordingly at the longer tiers.
"""

import hashlib
import json
import logging
import random
from typing import Any

from olmo_eval.data.helmet_loader import download_helmet_plus_file

logger = logging.getLogger(__name__)

KILT_MANIFEST = "kilt/manifest.json"

_USER_TEMPLATE = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}{context}\n\nQuestion: {question}"
)
_SYSTEM_TEMPLATE = "Answer:"
_PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
_DEMO_TEMPLATE = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"


def load_kilt_manifest() -> dict[str, dict[str, Any]]:
    """Download and parse the KILT manifest: task -> length tier -> file paths."""
    with open(download_helmet_plus_file(KILT_MANIFEST), encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _sampled_rows(
    path: str,
    max_samples: int | None,
    seed: int,
    keep=None,
) -> list[dict[str, Any]]:
    """Load rows, sampling by question without holding the whole file parsed.

    The 128k tiers are 1-3GB of JSONL that expand severalfold when parsed
    (upstream HELMET avoids this by memory-mapping arrow), and each question
    repeats once per gold-passage depth. So: one pass parses rows only long
    enough to record each line's key (and apply `keep`, the PopQA popularity
    filter), the kept questions are sampled, and a second pass parses only the
    lines that survive. Selection is identical to sampling after a full load --
    same sorted unique key set, same RNG draw -- just without the resident
    memory. With no cap there is nothing to skip, so the file loads directly.
    """
    if max_samples is None:
        rows = _load_jsonl(path)
        return [r for r in rows if keep(r)] if keep is not None else rows

    line_keys: list[Any] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                line_keys.append(None)
                continue
            row = json.loads(line)
            if keep is not None and not keep(row):
                line_keys.append(None)
                continue
            line_keys.append(row["id"] if "id" in row else row["question"])

    unique = sorted({k for k in line_keys if k is not None})
    kept = set(random.Random(seed).sample(unique, min(max_samples, len(unique))))

    rows = []
    with open(path, encoding="utf-8") as f:
        for line, line_key in zip(f, line_keys, strict=True):
            if line_key in kept:
                rows.append(json.loads(line))
    return rows


def _render_passages(contexts: list[dict[str, Any]]) -> str:
    return "\n\n".join(_PASSAGE_TEMPLATE.format(**ctx) for ctx in contexts)


def _instance_seed(key_value: Any) -> int:
    """Per-instance seed for demo selection.

    hashlib rather than hash() because the latter is salted per process; this
    matches HELMET, so an instance draws the same demos on every run.
    """
    return int(hashlib.sha256(str(key_value).encode("utf-8")).hexdigest(), 16) % 2**31


def _drop_duplicates(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen, out = set(), []
    for record in records:
        if record[key] in seen:
            continue
        seen.add(record[key])
        out.append(record)
    return out


def load_kilt_dataset(
    task: str,
    length_name: str,
    shots: int = 2,
    max_samples: int | None = None,
    seed: int = 42,
    popularity_threshold: float | None = None,
) -> dict[str, Any]:
    """Load a HELMET RAG dataset for one task at one length tier.

    Args:
        task: Manifest key, e.g. "kilt_nq" or "kilt_popqa".
        length_name: Tier key, e.g. "4k" or "128k".
        shots: Number of worked examples to prepend. Demos carry their own
            (much shorter) passage sets, drawn from the k3 demo file.
        max_samples: Cap on the number of instances.
        seed: Seed for instance sampling. Demo selection is additionally keyed
            per instance, so it is stable regardless of this cap.
        popularity_threshold: PopQA only -- keep entities whose subject
            popularity is below 10^threshold, which is how HELMET restricts the
            task to genuinely long-tail entities.

    Returns:
        Dictionary with `data` (processed records) and the HELMET prompt templates.
    """
    manifest = load_kilt_manifest()
    if task not in manifest:
        raise ValueError(f"Unknown KILT task '{task}'. Available: {sorted(manifest)}")
    if length_name not in manifest[task]:
        raise ValueError(
            f"Unknown length '{length_name}' for {task}. Available: {sorted(manifest[task])}"
        )

    entry = manifest[task][length_name]
    logger.info("Fetching %s data for %s (%s)...", task, length_name, entry["test_file"])
    test_path = download_helmet_plus_file(entry["test_file"])
    demo_pool = _load_jsonl(download_helmet_plus_file(entry["demo_file"]))

    below_threshold = None
    if popularity_threshold is not None:
        import math

        def below_threshold(record: dict[str, Any]) -> bool:
            return math.log10(record["s_pop"]) < popularity_threshold

        demo_pool = [r for r in demo_pool if below_threshold(r)]

    data = _sampled_rows(test_path, max_samples, seed, keep=below_threshold)
    if not data:
        raise ValueError(f"No rows loaded for {task} at {length_name} from {test_path}")

    # some sources have no id, in which case the question stands in as the key
    key = "id" if "id" in data[0] else "question"

    def build(sample: dict[str, Any]) -> dict[str, Any]:
        demo_text = ""
        if shots > 0:
            candidates = demo_pool
            if popularity_threshold is not None:
                # PopQA's demo pool is the test file itself, so the instance
                # being answered has to be excluded from its own demos
                candidates = [d for d in demo_pool if d[key] != sample[key]]

            shuffled = list(candidates)
            random.Random(_instance_seed(sample[key])).shuffle(shuffled)
            demos = _drop_duplicates(shuffled, key)[:shots]

            demo_text = (
                "\n\n".join(
                    _DEMO_TEMPLATE.format(
                        documents=_render_passages(d["ctxs"]),
                        question=d["question"],
                        answer=d["answers"][0],
                    )
                    for d in demos
                )
                + "\n\n"
            )

        return {
            "demos": demo_text,
            "context": _render_passages(sample["ctxs"]) if sample.get("ctxs") else "",
            "question": sample["question"],
            "answer": sample["answers"],
        }

    return {
        "data": [build(r) for r in data],
        "prompt_template": _USER_TEMPLATE + "\n" + _SYSTEM_TEMPLATE,
        "user_template": _USER_TEMPLATE,
        "system_template": _SYSTEM_TEMPLATE,
    }
