#!/usr/bin/env python
"""Generate oracle captions for an image-QA task, for the `:oracle_caption` ablation.

Writes a JSONL of ``{"example_id": ..., "caption": ...}`` that
``TaskConfig.caption_source`` consumes. Running a checkpoint against these captions
instead of the real image splits its error into two parts: ``oracle - real`` is perception
debt (what the model failed to see but could have used), and ``100 - oracle`` is
knowledge/reasoning debt (what it could not use even when handed a description).

Two things make this cheap enough to run repeatedly:

* **Dedup by image content.** CharXiv descriptive has 4000 instances over only 1000
  figures (four sub-questions each), and ``charxiv_reasoning`` reuses the same figures. We
  caption each distinct image once and fan the result back out to every example_id.
* **Disk cache.** Keyed on (model, prompt version, image bytes), so re-runs and overlapping
  tasks are free. Mirrors ``olmo_eval.common.scorers.charxiv_judge``.

Captions are deliberately **question-agnostic** -- the describer never sees the question,
so it cannot shortcut to the answer. That keeps the oracle an upper bound on *perception*
rather than a second attempt at the benchmark.

Usage:
    python tools/oracle_captions/generate.py --task charxiv_descriptive \\
        --out /weka/.../captions/charxiv_descriptive.jsonl

    # Preview a handful before spending on the full set:
    python tools/oracle_captions/generate.py --task mmmu --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("oracle_captions")

DEFAULT_MODEL = "gpt-4o-2024-11-20"

#: Bump when CAPTION_PROMPT changes, so stale cache entries are not reused.
PROMPT_VERSION = "v1"

CAPTION_PROMPT = """Describe this image in complete detail, as if for someone who cannot \
see it and must answer questions about it.

Include, wherever applicable:
- The type of figure or image, and its overall layout (number of subplots and their arrangement).
- All text verbatim: titles, axis labels, tick labels, legend entries, annotations, units.
- Axis ranges and the scale of each axis (linear/log).
- For every data series: its name, its visual encoding (colour, marker, line style), and its \
values or trend across the axis, including approximate values at notable points.
- Spatial relationships, counts of distinct objects, and any structure a question might ask about.

Be exhaustive and factual. Do not interpret, summarise, or draw conclusions, and do not \
speculate about anything you cannot see. Report only what is present in the image."""

#: Cap the long edge before upload. Beyond this, detail gain is negligible relative to the
#: token cost of a high-detail image.
MAX_EDGE = 1536


def _client(model: str):
    # .strip(): a trailing newline in the key makes an illegal Authorization header, which
    # surfaces as a misleading "Connection error" rather than an auth failure.
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to generate captions.")
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai") from None
    return AsyncOpenAI(api_key=api_key)


def encode_image(image) -> tuple[str, str]:
    """Downscale and JPEG-encode a PIL image.

    :returns: ``(base64_data, content_hash)``. The hash is over the *encoded* bytes, so it
        is stable across runs and identifies the image the model actually saw.
    """
    if getattr(image, "mode", "RGB") != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    if max(width, height) > MAX_EDGE:
        scale = MAX_EDGE / max(width, height)
        from PIL import Image as PILImage

        image = image.resize((int(width * scale), int(height * scale)), PILImage.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


def _cache_path(cache_dir: Path, model: str, content_hash: str) -> Path:
    key = hashlib.sha256(f"{model}\x00{PROMPT_VERSION}\x00{content_hash}".encode()).hexdigest()
    return cache_dir / f"{key}.json"


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(".tmp", prefix=path.name, text=True, dir=str(path.parent))
    os.close(fd)
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.rename(tmp, str(path))


async def caption_image(
    client,
    b64: str,
    content_hash: str,
    *,
    model: str,
    cache_dir: Path,
    max_tokens: int,
    max_retries: int = 6,
) -> str | None:
    """Caption one image, reading through the disk cache. None if it ultimately fails."""
    cache_file = _cache_path(cache_dir, model, content_hash)
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)["caption"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
                },
            ],
        }
    ]

    delay = 2.0
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
                seed=42,
            )
            caption = (response.choices[0].message.content or "").strip()
            if not caption:
                raise ValueError("empty caption")
            _write_atomic(cache_file, {"caption": caption, "model": model})
            return caption
        except Exception as e:
            logger.warning(
                "caption failed (%s/%s) for %s: %s", attempt + 1, max_retries, content_hash[:12], e
            )
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
    return None


async def run(args: argparse.Namespace) -> int:
    from olmo_eval.evals.tasks.common.image_qa_base import load_instance_image
    from olmo_eval.evals.tasks.common.registry import get_task

    overrides = {"limit": args.limit} if args.limit else None
    task = get_task(args.task if not args.split else f"{args.task}", overrides)
    if args.split:
        task.config.split = type(task.config.split)(args.split)

    instances = list(task.instances)
    logger.info("%s: %d instances", args.task, len(instances))

    # example_id -> content hash, plus one representative encoding per distinct image.
    by_example: dict[str, str] = {}
    encoded: dict[str, str] = {}
    imageless = 0
    for instance in instances:
        example_id = str(instance.metadata.get("example_id", ""))
        if not example_id:
            raise ValueError(
                f"Instance has no example_id, which caption_source lookups need: "
                f"{instance.question[:80]!r}"
            )
        image = load_instance_image(instance)
        if image is None:
            imageless += 1
            continue
        b64, content_hash = encode_image(image)
        by_example[example_id] = content_hash
        encoded.setdefault(content_hash, b64)

    logger.info(
        "%d example_ids over %d distinct images (%d imageless, skipped)",
        len(by_example),
        len(encoded),
        imageless,
    )

    cache_dir = Path(args.cache_dir)
    cached = sum(1 for h in encoded if _cache_path(cache_dir, args.model, h).exists())
    logger.info(
        "%d/%d already cached; %d API calls needed", cached, len(encoded), len(encoded) - cached
    )

    if args.dry_run:
        logger.info("--dry-run: no API calls, no output written")
        return 0

    client = _client(args.model)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def _one(content_hash: str, b64: str) -> tuple[str, str | None]:
        async with semaphore:
            caption = await caption_image(
                client,
                b64,
                content_hash,
                model=args.model,
                cache_dir=cache_dir,
                max_tokens=args.max_tokens,
            )
            return content_hash, caption

    results = await asyncio.gather(*(_one(h, b) for h, b in encoded.items()))
    captions = {h: c for h, c in results if c is not None}
    failed = len(encoded) - len(captions)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w") as f:
        for example_id, content_hash in sorted(by_example.items()):
            caption = captions.get(content_hash)
            if caption is None:
                continue
            f.write(json.dumps({"example_id": example_id, "caption": caption}) + "\n")
            written += 1

    lengths = sorted(len(c) for c in captions.values())
    logger.info(
        "wrote %d/%d example_ids to %s (caption chars: median %d, min %d, max %d)",
        written,
        len(by_example),
        out_path,
        lengths[len(lengths) // 2] if lengths else 0,
        lengths[0] if lengths else 0,
        lengths[-1] if lengths else 0,
    )
    if failed:
        # A partial file makes the eval raise on the missing ids rather than silently
        # scoring them text-only, but it still needs to be fixed before the run counts.
        logger.error(
            "%d image(s) failed to caption; re-run to retry (cache keeps the rest)", failed
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", required=True, help="Task spec, e.g. mmmu / charxiv_descriptive / mmmu_pro"
    )
    parser.add_argument("--out", help="Output JSONL path (required unless --dry-run)")
    parser.add_argument("--split", default=None, help="Override the task's default split")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get(
            "ORACLE_CAPTION_CACHE_DIR",
            "/weka/oe-training-default/donovanc/molmofication/outputs/oracle-caption-cache",
        ),
        help="Shared on-disk caption cache (keyed by image content, so tasks share entries)",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N instances")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report instance/image/cache counts and exit without calling the API",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
