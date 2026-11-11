#!/usr/bin/env python3
"""Text-only HF model from an Olmo 3.5 multimodal HF export, for the text eval battery.

A multimodal export (``Olmo3MoeVisionForConditionalGeneration``, from scaling-ladders'
``eval_text.py convert``) holds the language model under ``language_model.``. This writes:

* ``model.safetensors``: every ``language_model.*`` tensor under its name without the prefix,
  copied byte for byte (no dtype round trip), the vision tower and connector dropped;
* ``config.json``: the export's ``text_config`` (``Olmo3MoeForCausalLM``) and the text modeling
  files beside it;
* the tokenizer: by default the Olmo 3.5 chat tokenizer with ``<think>``/``</think>`` promoted, as
  open-instruct saves it next to a text SFT export (so the battery renders the SFT chat template);
  ``--tokenizer source`` keeps the export's own tokenizer files (a mid-training checkpoint).

    python mm_hf_to_text_hf.py <multimodal hf dir> <out dir> [--tokenizer olmo35|source]

Refuses an existing ``out dir``. Writes ``TEXT_EXPORT_RECEIPT.json`` with every tensor's sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys

PREFIX = "language_model."
TEXT_FILES = ("configuration_olmo3moe.py", "modeling_olmo3moe.py", "moe_numerics.py", "generation_config.json")
SOURCE_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json")


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--tokenizer", choices=("olmo35", "source"), default="olmo35")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"refusing to overwrite {args.out}")
    config = json.load(open(os.path.join(args.src, "config.json")))
    if config.get("architectures") != ["Olmo3MoeVisionForConditionalGeneration"]:
        sys.exit(f"not a multimodal Olmo 3.5 HF export: {config.get('architectures')}")
    text_config = dict(config["text_config"])
    if text_config.get("architectures") != ["Olmo3MoeForCausalLM"]:
        sys.exit(f"unexpected text architecture {text_config.get('architectures')}")

    src_st = os.path.join(args.src, "model.safetensors")
    header, data_start = read_header(src_st)
    keys = sorted(k for k in header if k.startswith(PREFIX))
    dropped = sorted(k for k in header if k != "__metadata__" and not k.startswith(PREFIX))
    new_header: dict = {"__metadata__": header.get("__metadata__") or {"format": "pt"}}
    offset = 0
    for key in keys:
        entry = header[key]
        start, end = entry["data_offsets"]
        new_header[key[len(PREFIX) :]] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + (end - start)],
        }
        offset += end - start
    blob = json.dumps(new_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)

    tmp = args.out + ".partial"
    os.makedirs(tmp)
    hashes = {}
    with open(src_st, "rb") as src, open(os.path.join(tmp, "model.safetensors"), "wb") as dst:
        dst.write(struct.pack("<Q", len(blob)))
        dst.write(blob)
        for key in keys:
            start, end = header[key]["data_offsets"]
            src.seek(data_start + start)
            data = src.read(end - start)
            hashes[key[len(PREFIX) :]] = hashlib.sha256(data).hexdigest()
            dst.write(data)
    # Re-read every tensor from the written file and compare its hash.
    out_header, out_start = read_header(os.path.join(tmp, "model.safetensors"))
    with open(os.path.join(tmp, "model.safetensors"), "rb") as f:
        for key, digest in hashes.items():
            start, end = out_header[key]["data_offsets"]
            f.seek(out_start + start)
            if hashlib.sha256(f.read(end - start)).hexdigest() != digest:
                sys.exit(f"verify failed for {key}")

    json.dump(text_config, open(os.path.join(tmp, "config.json"), "w"), indent=2)
    for name in TEXT_FILES:
        if os.path.exists(os.path.join(args.src, name)):
            shutil.copy2(os.path.join(args.src, name), tmp)
    if args.tokenizer == "olmo35":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
        from olmo_eval.inference.providers.olmo_core_vlm.olmo35_chat import (
            load_olmo35_chat_tokenizer,
        )

        tokenizer = load_olmo35_chat_tokenizer()
        tokenizer.model_max_length = int(text_config.get("max_position_embeddings") or 65536)
        tokenizer.save_pretrained(tmp)
        tokenizer_note = "allenai/dolma2-tokenizer-olmo35@8b97170, <think>/</think> promoted (100256, 100266)"
    else:
        for name in SOURCE_TOKENIZER_FILES:
            if os.path.exists(os.path.join(args.src, name)):
                shutil.copy2(os.path.join(args.src, name), tmp)
        tokenizer_note = "copied from the source export"

    receipt = {
        "source": os.path.abspath(args.src),
        "kept": len(keys),
        "dropped": len(dropped),
        "dropped_prefixes": sorted({k.split(".")[0] for k in dropped}),
        "tokenizer": tokenizer_note,
        "tensor_sha256": hashes,
    }
    json.dump(receipt, open(os.path.join(tmp, "TEXT_EXPORT_RECEIPT.json"), "w"), indent=1)
    os.rename(tmp, args.out)
    print(f"Wrote {args.out}: kept {len(keys)}, dropped {len(dropped)} ({receipt['dropped_prefixes']})")


if __name__ == "__main__":
    main()
