"""The olmo35 eval prompt is exactly the training prefix of an Olmo 3.5 multimodal SFT example.

The training side is OLMo-core's olmo35 layout at the multimodal SFT data branch
(``jasonr/olmo35-mmsft-data``, ``olmo_core.data.multimodal.olmo35_layout``): its own tokenizer
loader (``load_olmo35_sft_tokenizer``), the pieces the image-row encoder assembles
(``branch_segments``: head, then the image blocks, then context/response runs) and
``encode_conversation``. For every conversation the eval prompt must equal the training ids up to
the first label of the last assistant turn (``thinking="generate"``: the prompt ends at
``<think>``), and the labels after it must be ``</think>``, the answer and EOS.

Needs the tokenizer ``allenai/dolma2-tokenizer-olmo35`` (Hub or cache, or ``OLMO35_TOKENIZER``
pointing at a local copy) and that OLMo-core on the path; skipped otherwise.
"""

from __future__ import annotations

import os

import pytest

from olmo_eval.evals.vision.scoring.count_parsing import parse_count
from olmo_eval.inference.providers.olmo_core_vlm import olmo35_chat

THINK, END_THINK = 100256, 100266


def _tokenizer_source() -> str:
    return os.environ.get("OLMO35_TOKENIZER", olmo35_chat.OLMO35_CHAT_TOKENIZER)


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("tokenizers")
    try:
        return olmo35_chat.load_olmo35_chat_tokenizer(_tokenizer_source())
    except OSError as exc:
        pytest.skip(f"olmo35 tokenizer unavailable: {exc}")


@pytest.fixture(scope="module")
def layout():
    module = pytest.importorskip("olmo_core.data.multimodal.olmo35_layout")
    if not hasattr(module, "branch_segments"):
        pytest.skip("OLMo-core without the multimodal SFT data branch's olmo35 layout")
    return module


@pytest.fixture(scope="module")
def train_tokenizer(layout):
    source = _tokenizer_source()
    tokenizer, _ = layout.load_olmo35_sft_tokenizer(source)
    return tokenizer


def test_think_tags_take_open_instructs_slots(tokenizer):
    # open-instruct #1911 on dolma2-tokenizer-olmo35@8b97170: the two lowest free reserved slots.
    assert tokenizer.encode("<think>", add_special_tokens=False) == [THINK]
    assert tokenizer.encode("</think>", add_special_tokens=False) == [END_THINK]
    assert tokenizer.vocab_size == 100278


def test_prompt_ends_at_think_by_default(tokenizer):
    ids = olmo35_chat.prompt_ids(tokenizer, [{"role": "user", "content": "Hi"}])
    assert ids[-1] == THINK
    forced = olmo35_chat.prompt_ids(
        tokenizer, [{"role": "user", "content": "Hi"}], thinking="forced"
    )
    assert forced == [*ids, END_THINK]


def test_tokenizers_agree(tokenizer, train_tokenizer):
    probes = [
        "<think></think>The answer is 4.<|endoftext|>",
        "2024 was.\n\nchart_qa: What is it?",
        'Counting the <points coords="1 1 981 846">domain</points> shows a total of 1.',
    ]
    for text in probes:
        assert tokenizer.encode(text, add_special_tokens=False) == train_tokenizer.encode(
            text, add_special_tokens=False
        )
    assert tokenizer.chat_template == train_tokenizer.chat_template


def _grids():
    import numpy as np

    # (resized_h, resized_w, crop_h, crop_w) of two small images.
    return [np.array([2, 2, 1, 1]), np.array([2, 3, 1, 2])]


def _image_sequences(layout, grids):
    from olmo_core.nn.vision.molmo2_tokens import build_image_token_ids

    return [
        build_image_token_ids(*(int(v) for v in grid), token_ids=layout.OLMO35_TOKEN_IDS)
        for grid in grids
    ]


CASES = [
    ("text only", [("What is 2 + 2?", "4")], 0),
    ("one image", [("chart_qa: What is the highest value?", "15.3")], 1),
    ("two images", [("Which image shows a dog?\n\nA. 1\nB. 2", "B")], 2),
    (
        "multi-turn counting",
        [
            ("Count each domain", 'Counting the <points coords="1 1 981 846">domain</points> shows a total of 1.'),
            ("Point to an object", "There are none."),
        ],
        1,
    ),
]


@pytest.mark.parametrize("name,turns,n_images", CASES, ids=[c[0] for c in CASES])
def test_prompt_is_the_training_prefix(tokenizer, train_tokenizer, layout, name, turns, n_images):
    grids = _grids()[:n_images]
    images = _image_sequences(layout, grids)
    image_ids = layout.image_block_ids(train_tokenizer, grids) if grids else []

    # The image-row encoder's pieces: head + image blocks + (context, response) runs + EOS.
    head, segments = layout.branch_segments(train_tokenizer, turns)
    train_ids = list(head) + list(image_ids)
    for context, response in segments:
        train_ids += list(context) + list(response)
    train_ids.append(train_tokenizer.eos_token_id)
    last_context, last_response = segments[-1]
    first_label = len(train_ids) - 1 - len(last_response)  # the last response's first token

    messages = []
    for question, answer in turns:
        messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    prompt = olmo35_chat.prompt_ids(tokenizer, messages[:-1], images)
    assert prompt[-1] == THINK
    assert train_ids[:first_label] == prompt
    answer = turns[-1][1]
    expected = [END_THINK, *train_tokenizer.encode(answer, add_special_tokens=False)]
    assert last_response == expected
    forced = olmo35_chat.prompt_ids(tokenizer, messages[:-1], images, thinking="forced")
    assert train_ids[: first_label + 1] == forced

    # encode_conversation gives the same ids, and its labels after the prompt are </think>,
    # the answer and EOS.
    encoded = layout.encode_conversation(train_tokenizer, messages, image_ids=image_ids)
    ids = encoded["input_ids"].tolist()
    assert ids == train_ids
    labels = encoded["labels"].tolist()
    tail = [*expected, train_tokenizer.eos_token_id]
    assert labels[len(prompt) - 1 : len(prompt) - 1 + len(tail)] == tail


@pytest.mark.parametrize(
    "text,count",
    [
        ('Counting the <points coords="1 1 981 846 2 10 20">cat</points> shows a total of 2.', 2),
        ("There are none.", 0),
        ("</think>There are none.", 0),
        ("3", 3),
    ],
)
def test_count_parsing_handles_the_training_forms(text, count):
    assert parse_count(olmo35_chat.strip_thinking(text)) == count


def test_strip_thinking():
    assert olmo35_chat.strip_thinking("</think>42") == "42"
    assert olmo35_chat.strip_thinking("some reasoning</think> 42") == "42"
    assert olmo35_chat.strip_thinking(" 42 ") == "42"
