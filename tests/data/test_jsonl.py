"""Tests for the shared JSONL and Hub download helpers."""

import json

import numpy as np
import pytest

from olmo_eval.data import hub
from olmo_eval.data.jsonl import load_jsonl, sample_jsonl_by_key, sample_jsonl_rows


def _write(path, rows, blank_every=0):
    lines = []
    for i, row in enumerate(rows):
        lines.append(json.dumps(row))
        if blank_every and i % blank_every == 0:
            lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = _write(tmp_path / "rows.jsonl", [{"a": 1}, {"a": 2}], blank_every=1)
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_load_jsonl_rejects_non_objects_and_bad_lines(tmp_path):
    (tmp_path / "list.jsonl").write_text('{"a": 1}\n[1, 2]\n')
    with pytest.raises(ValueError, match="list.jsonl:2"):
        load_jsonl(str(tmp_path / "list.jsonl"))
    (tmp_path / "bad.jsonl").write_text("{not json}\n")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_jsonl(str(tmp_path / "bad.jsonl"))


@pytest.mark.parametrize("cap", [1, 7, 30, None])
def test_sample_jsonl_rows_matches_permuting_a_full_load(tmp_path, cap):
    rows = [{"i": i} for i in range(25)]
    path = _write(tmp_path / "rows.jsonl", rows, blank_every=4)
    if cap is None:
        expected = rows
    else:
        permutation = np.random.default_rng(3).permutation(len(rows))
        expected = [rows[int(i)] for i in permutation[:cap]]
    assert sample_jsonl_rows(path, cap, 3) == expected


def test_sample_jsonl_by_key_keeps_every_row_of_each_sampled_key(tmp_path):
    rows = [{"q": q, "depth": d} for q in range(10) for d in range(3)]
    path = _write(tmp_path / "rows.jsonl", rows)
    sampled = sample_jsonl_by_key(path, 4, 42, key=lambda r: r["q"])
    assert len({r["q"] for r in sampled}) == 4
    assert len(sampled) == 12
    kept_odd = sample_jsonl_by_key(path, None, 42, key=lambda r: r["q"], keep=lambda r: r["q"] % 2)
    assert {r["q"] for r in kept_odd} == {1, 3, 5, 7, 9}


def test_download_dataset_file_pins_the_revision(monkeypatch):
    calls = {}

    def fake_download(**kwargs):
        calls.update(kwargs)
        return "/cache/file"

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    path = hub.download_dataset_file("org/repo", "dir/file.jsonl", revision="abc123")
    assert path == "/cache/file"
    assert calls["repo_id"] == "org/repo"
    assert calls["filename"] == "dir/file.jsonl"
    assert calls["repo_type"] == "dataset"
    assert calls["revision"] == "abc123"
    with pytest.raises(TypeError):
        hub.download_dataset_file("org/repo", "file")  # type: ignore[call-arg]
