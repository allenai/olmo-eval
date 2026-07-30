"""Tests for the vLLM GPT-OSS Responses parser backport."""

import pytest

from olmo_eval.inference.patches.gpt_oss_responses_parser_patch import (
    find_harmony_parser,
    patch_parser,
)


def test_patch_removes_completed_message_channel_restriction(tmp_path):
    parser_path = tmp_path / "harmony.py"
    parser_path.write_text(
        'elif message.channel == "commentary" and recipient.startswith("functions."):\n'
        "    parse_function_call()\n"
    )

    assert patch_parser(parser_path) is True
    assert 'elif recipient.startswith("functions."):' in parser_path.read_text()
    assert patch_parser(parser_path) is False


def test_patch_rejects_unknown_parser_layout(tmp_path):
    parser_path = tmp_path / "harmony.py"
    parser_path.write_text("def unrelated(): pass\n")

    with pytest.raises(RuntimeError, match="Unsupported vLLM Responses Harmony parser"):
        patch_parser(parser_path)


def test_find_harmony_parser_in_isolated_venv(tmp_path):
    parser_path = (
        tmp_path
        / "lib"
        / "python3.12"
        / "site-packages"
        / "vllm"
        / "entrypoints"
        / "openai"
        / "responses"
        / "harmony.py"
    )
    parser_path.parent.mkdir(parents=True)
    parser_path.touch()

    assert find_harmony_parser(str(tmp_path)) == parser_path
