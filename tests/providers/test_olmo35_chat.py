"""The olmo35 eval prompt is exactly the training prefix of an Olmo 3.5 multimodal SFT example.

The training side is OLMo-core's olmo35 layout (``olmo_core.data.multimodal.olmo35_layout``,
allenai/OLMo-core#942), given the same tokenizer with the think tags promoted. For each
conversation the eval prompt must equal the training ids up to the answer, and the training ids
after it must be the answer and its closing EOS.

Needs the Hub tokenizer ``allenai/dolma2-tokenizer-olmo35`` (or its cache) and, for the layout
comparison, an OLMo-core with ``olmo35_layout`` on the path; skipped otherwise.
"""

from __future__ import annotations

import pytest

from olmo_eval.inference.providers.olmo_core_vlm import olmo35_chat

pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return olmo35_chat.load_olmo35_chat_tokenizer()
    except OSError as exc:  # offline without the cached tokenizer
        pytest.skip(f"olmo35 tokenizer unavailable: {exc}")


@pytest.fixture(scope="module")
def layout():
    return pytest.importorskip("olmo_core.data.multimodal.olmo35_layout")


def test_think_tags_take_open_instructs_slots(tokenizer):
    # open-instruct #1911 on dolma2-tokenizer-olmo35@8b97170: the two lowest free reserved slots.
    assert tokenizer.encode("<think>", add_special_tokens=False) == [100256]
    assert tokenizer.encode("</think>", add_special_tokens=False) == [100266]
    assert tokenizer.vocab_size == 100278


def test_generation_prompt_ends_on_closed_think(tokenizer):
    ids = olmo35_chat.prompt_ids(tokenizer, [{"role": "user", "content": "Hi"}])
    assert ids[-2:] == [100256, 100266]
    on = olmo35_chat.prompt_ids(tokenizer, [{"role": "user", "content": "Hi"}], thinking="on")
    assert on == ids[:-1]


def _image_grids():
    # (resized_h, resized_w, crop_h, crop_w) of two small images.
    import numpy as np

    return [np.array([2, 2, 1, 1]), np.array([2, 3, 1, 2])]


CASES = [
    ("text only", [{"role": "user", "content": "What is 2 + 2?"}], 0),
    ("one image", [{"role": "user", "content": "chart_qa: What is the highest value?"}], 1),
    ("two images", [{"role": "user", "content": "Which image shows a dog?\n\nA. 1\nB. 2"}], 2),
    (
        "multi-turn",
        [
            {"role": "user", "content": "Describe this image."},
            {"role": "assistant", "content": "A red bus on a street."},
            {"role": "user", "content": "How many wheels are visible?"},
        ],
        1,
    ),
]


@pytest.mark.parametrize("name,messages,n_images", CASES, ids=[c[0] for c in CASES])
def test_prompt_is_the_training_prefix(tokenizer, layout, name, messages, n_images):
    from olmo_core.nn.vision.molmo2_tokens import build_image_token_ids

    grids = _image_grids()[:n_images]
    per_image = [
        build_image_token_ids(*(int(v) for v in grid), token_ids=layout.OLMO35_TOKEN_IDS)
        for grid in grids
    ]
    answer = "The answer is 4."
    train = layout.encode_conversation(
        tokenizer,
        [*messages, {"role": "assistant", "content": answer}],
        image_ids=layout.image_block_ids(tokenizer, grids) if grids else (),
    )
    ids = train["input_ids"].tolist()
    prompt = olmo35_chat.prompt_ids(tokenizer, messages, per_image)
    assert ids[: len(prompt)] == prompt
    tail = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    assert ids[len(prompt) :] == tail
    # Every answer token and the EOS is a training label; nothing in the prompt after the last
    # user turn's generation header is, except the closing </think>.
    labels = train["labels"].tolist()
    assert labels[len(prompt) - 1 : len(prompt) - 1 + len(tail)] == tail
    assert labels[len(prompt) - 2] == 100266


def test_strip_thinking():
    assert olmo35_chat.strip_thinking("some reasoning</think> 42") == "42"
    assert olmo35_chat.strip_thinking(" 42 ") == "42"
