"""Olmo 3.5 chat prompts for multimodal SFT checkpoints (``prompt_format="olmo35"``).

A multimodal SFT checkpoint trained by OLMo-core's olmo35 layout
(``olmo_core.data.multimodal.olmo35_layout``, allenai/OLMo-core#942) sees each conversation as the
Olmo 3.5 chat template renders it, tokenized in one piece, with the image blocks inserted at the
start of the first user turn (``Image {i+1}`` before each block when there are several). This
module builds the generation prompt of a request so its tokens are exactly the training tokens up
to the point where the model's answer starts:

* the template's generation prompt ends ``<|im_start|>assistant\\n<think>``. In training
  (``olmo35_layout.branch_segments``) ``<think>`` closes the context and ``</think>``, the answer
  and EOS are labels, so ``thinking="generate"`` (the default) stops the prompt at ``<think>`` and
  the model generates ``</think>`` and the answer; anything up to the last ``</think>`` is
  stripped from the output. ``thinking="forced"`` appends ``</think>`` to the prompt instead;
* ``<think>`` and ``</think>`` are single tokens, promoted into unused reserved slots exactly as
  open-instruct's ``promote_tokens_into_reserved_slots`` (allenai/open-instruct#1911) does for the
  text SFT recipe (``--reserved_slot_tokens '<think>' '</think>'``);
* the tokenizer splits text with ``tokenizer.json``'s own pre-tokenizer (allenai/OLMo-core#944),
  not the GPT-2 regex ``GPT2Tokenizer`` substitutes.

``tests/inference/test_olmo35_chat.py`` checks these prompt ids against the training layout.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

OLMO35_CHAT_TOKENIZER = "allenai/dolma2-tokenizer-olmo35"
OLMO35_CHAT_TOKENIZER_REVISION = "8b9717061fae09d5be814373d189919a62a9a00d"
"""The text SFT recipe's tokenizer (open-instruct ``scripts/c12/h161_gpu_launch.sh``). Vocabulary and
merges equal ``allenai/dolma2-tokenizer`` @ 5292e5d6, the mid-training tokenizer; it adds the
Olmo 3.5 chat template."""

DEFAULT_RESERVED_SLOT_TOKENS: tuple[str, ...] = ("<think>", "</think>")
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
THINKING_MODES = ("generate", "forced")
USER_HEADER = "<|im_start|>user\n"

_RESERVED_SLOT_RE = re.compile(r"<\|extra_id_\d+\|>")


def load_olmo35_chat_tokenizer(
    name_or_path: str = OLMO35_CHAT_TOKENIZER,
    *,
    revision: str | None = None,
    reserved_slot_tokens: Sequence[str] = DEFAULT_RESERVED_SLOT_TOKENS,
    **kwargs: Any,
) -> Any:
    """Load the Olmo 3.5 chat tokenizer as the SFT recipe tokenizes.

    ``tokenizer.json`` is loaded with the generic fast class, so its ``Split`` pre-tokenizer is
    kept, then ``reserved_slot_tokens`` are promoted into unused reserved slots.

    :param name_or_path: Hub id or local directory.
    :param revision: Hub revision; :data:`OLMO35_CHAT_TOKENIZER_REVISION` for the default hub id.
    :param reserved_slot_tokens: Strings to make single tokens; empty for a recipe without them.
    """
    from transformers import PreTrainedTokenizerFast
    from transformers.utils import hub

    if revision is None and name_or_path == OLMO35_CHAT_TOKENIZER:
        revision = OLMO35_CHAT_TOKENIZER_REVISION
    tokenizer = PreTrainedTokenizerFast.from_pretrained(name_or_path, revision=revision, **kwargs)
    tokenizer_file = hub.cached_file(
        name_or_path, "tokenizer.json", revision=revision, _raise_exceptions_for_missing_entries=False
    )
    if tokenizer_file is not None:
        with open(tokenizer_file) as f:
            expected = json.load(f).get("pre_tokenizer")
        actual = json.loads(tokenizer.backend_tokenizer.to_str()).get("pre_tokenizer")
        if actual != expected:
            raise ValueError(f"{name_or_path}: the loaded tokenizer lost tokenizer.json's pre-tokenizer")
    if not tokenizer.chat_template:
        raise ValueError(f"{name_or_path} has no chat template")
    promote_tokens_into_reserved_slots(tokenizer, reserved_slot_tokens)
    return tokenizer


def promote_tokens_into_reserved_slots(tokenizer: Any, tokens: Sequence[str]) -> list[tuple[str, int]]:
    """Make each of ``tokens`` a single token by renaming an unused ``<|extra_id_N|>`` slot in place.

    A port of open-instruct's ``promote_tokens_into_reserved_slots`` (``dataset_transformation.py``,
    #1911): the lowest free slots, in order, not referenced by the chat template or special-token
    map; not marked special; ``vocab_size`` unchanged. Strings already added are skipped.

    :returns: ``(content, token_id)`` of each promoted string.
    """
    from tokenizers import Tokenizer

    added = set(tokenizer.get_added_vocab())
    pending = []
    for token in tokens:
        if token in added:
            continue
        if len(tokenizer.encode(token, add_special_tokens=False)) == 1:
            raise ValueError(f"{token!r} is already one ordinary BPE token")
        pending.append(token)
    if not pending:
        return []
    spec = json.loads(tokenizer.backend_tokenizer.to_str())
    vocab = spec["model"]["vocab"]
    template = str(tokenizer.chat_template or "")
    in_use = set(tokenizer.all_special_tokens) | set(_RESERVED_SLOT_RE.findall(template))
    free = [
        entry
        for entry in sorted(spec["added_tokens"], key=lambda entry: entry["id"])
        if _RESERVED_SLOT_RE.fullmatch(entry["content"]) and entry["content"] not in in_use
    ]
    if len(free) < len(pending):
        raise ValueError(f"Not enough unused reserved slots to promote {pending}")
    sizes = (tokenizer.vocab_size, len(tokenizer))
    promoted = []
    for entry, content in zip(free, pending):
        slot_id = vocab.pop(entry["content"], None)
        if slot_id is not None:
            vocab[content] = slot_id
        entry.update(
            content=content, special=False, normalized=False, lstrip=False, rstrip=False, single_word=False
        )
        promoted.append((content, int(entry["id"])))
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(spec))
    if (tokenizer.vocab_size, len(tokenizer)) != sizes:
        raise RuntimeError("Promoting reserved-slot tokens changed the vocabulary size")
    for content, token_id in promoted:
        if tokenizer.encode(content, add_special_tokens=False) != [token_id]:
            raise RuntimeError(f"Promoted {content!r} does not encode to [{token_id}]")
    return promoted


def _render(tokenizer: Any, messages: Sequence[dict[str, Any]], *, add_generation_prompt: bool) -> str:
    text = tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
    )
    assert isinstance(text, str)
    return text


def prompt_ids(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    image_token_sequences: Sequence[Sequence[int]] = (),
    *,
    thinking: str = "generate",
) -> list[int]:
    """Token ids of the generation prompt for ``messages``, with image blocks.

    :param tokenizer: From :func:`load_olmo35_chat_tokenizer` (plus the Molmo2 image tokens).
    :param messages: ``role``/``content`` turns ending with a user turn; content is text.
    :param image_token_sequences: Expanded image token ids, one sequence per image, in order.
    :param thinking: ``"generate"`` ends the prompt at ``<think>`` (as training's context does);
        ``"forced"`` also closes the empty think block.

    :raises ValueError: If the text around the image blocks would tokenize differently from the
        whole prompt (the training layout refuses such a conversation too).
    """
    if thinking not in THINKING_MODES:
        raise ValueError(f"thinking must be one of {THINKING_MODES}, got {thinking!r}")
    messages = [{"role": m.get("role", "user"), "content": m.get("content") or ""} for m in messages]
    if image_token_sequences:
        # As the training layout (OLMo-core #942 9923dd071): the image blocks go at the start of
        # the first user turn, and leading whitespace there would merge with the user header into
        # one token, so it is dropped.
        for m in messages:
            if m["role"] == "user":
                m["content"] = m["content"].lstrip()
                break
    rendered = _render(tokenizer, messages, add_generation_prompt=True)
    if not rendered.endswith(THINK_OPEN):
        raise ValueError("the Olmo 3.5 generation prompt should end with <think>")
    if thinking == "forced":
        rendered += THINK_CLOSE
    whole = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if not image_token_sequences:
        return list(whole)
    first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
    if first_user is None:
        raise ValueError("an image request needs a user turn")
    through = _render(tokenizer, messages[: first_user + 1], add_generation_prompt=False)
    start = through.rfind(USER_HEADER)
    if start < 0 or not rendered.startswith(through):
        raise ValueError("cannot locate the first user turn's content")
    cut = start + len(USER_HEADER)
    left = tokenizer(rendered[:cut], add_special_tokens=False)["input_ids"]
    right = tokenizer(rendered[cut:], add_special_tokens=False)["input_ids"]
    if list(left) + list(right) != list(whole):
        raise ValueError("the text around the image blocks tokenizes differently on its own")
    image_ids: list[int] = []
    multi = len(image_token_sequences) > 1
    for index, sequence in enumerate(image_token_sequences):
        if multi:
            image_ids.extend(tokenizer.encode(f"Image {index + 1}", add_special_tokens=False))
        image_ids.extend(int(token) for token in sequence)
    return [*left, *image_ids, *right]


def strip_thinking(text: str) -> str:
    """The answer after the last ``</think>`` the model generated, else ``text``."""
    if THINK_CLOSE in text:
        text = text.rsplit(THINK_CLOSE, 1)[1]
    return text.strip()
