"""BrowseComp-Plus preparation preserves complete decrypted records."""

import base64
import hashlib
import json

import pytest

from olmo_eval.evals.tasks.browsecomp_plus.data import (
    CANARY,
    DATASET_FILENAME,
    decrypt_string,
    prepare_questions,
)


def encrypt(text):
    data = text.encode()
    digest = hashlib.sha256(CANARY.encode()).digest()
    return base64.b64encode(
        bytes(value ^ digest[i % len(digest)] for i, value in enumerate(data))
    ).decode()


def test_prepare_preserves_all_decrypted_fields_and_whitespace(tmp_path):
    expected = {
        "query_id": "007",
        "query": " \tWhich\tcity?\n",
        "answer": "São Paulo",
        "evidence_docs": [
            {"docid": "001", "text": "Full evidence text.\n東京", "url": "https://example.org/a"},
        ],
        "gold_docs": [
            {"docid": "002", "text": "Gold document text", "url": "https://example.org/b"},
        ],
        "negative_docs": [
            {"docid": "003", "text": "Negative document text", "url": "https://example.org/c"},
        ],
        "metadata": {"language": "English", "values": [True, None, 1.5]},
    }
    row = {
        "query_id": "007",
        "query": encrypt(" \tWhich\tcity?\n"),
        "answer": encrypt("São Paulo"),
        "metadata": {"language": encrypt("English"), "values": [True, None, 1.5]},
    }
    for field in ("evidence_docs", "gold_docs", "negative_docs"):
        row[field] = [
            {key: encrypt(value) for key, value in doc.items()} for doc in expected[field]
        ]
    destination = tmp_path / DATASET_FILENAME
    assert prepare_questions(iter([row]), destination, expected_count=1) == 1
    assert [json.loads(line) for line in destination.read_text().splitlines()] == [expected]
    assert row["query"] == encrypt(expected["query"])


def test_failed_preparation_does_not_replace_existing_data(tmp_path):
    destination = tmp_path / DATASET_FILENAME
    destination.write_text("previous artifact\n")
    row = {
        "query_id": "1",
        "query": encrypt("q"),
        "answer": encrypt("a"),
        "evidence_docs": [{"docid": encrypt("11")}],
    }
    with pytest.raises(ValueError, match="duplicate"):
        prepare_questions([row, row], destination, expected_count=2)
    assert destination.read_text() == "previous artifact\n"
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="Expected 830"):
        prepare_questions([row], destination)


def test_missing_evidence_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Invalid"):
        prepare_questions(
            [{"query_id": "1", "query": encrypt("q"), "answer": encrypt("a")}],
            tmp_path / DATASET_FILENAME,
        )


def test_decryption_handles_unicode_and_multiple_key_blocks():
    text = "São Paulo; 東京; " * 10
    assert decrypt_string(encrypt(text)) == text
