"""Tests for the DataLoader service."""

import tempfile
from pathlib import Path

import pytest

from olmo_eval.data import DataLoader, DataSource, SourceType


class TestDataLoader:
    """Tests for DataLoader."""

    def test_loader_initialization(self):
        loader = DataLoader()
        assert loader.cache_dir is None
        assert loader._backends == {}

    def test_loader_with_cache_dir(self):
        loader = DataLoader(cache_dir="/tmp/cache")
        assert loader.cache_dir == "/tmp/cache"

    def test_backend_caching(self):
        loader = DataLoader()
        backend1 = loader._get_backend(SourceType.LOCAL)
        backend2 = loader._get_backend(SourceType.LOCAL)
        assert backend1 is backend2

    def test_different_backends_for_different_types(self):
        loader = DataLoader()
        local_backend = loader._get_backend(SourceType.LOCAL)
        hf_backend = loader._get_backend(SourceType.HUGGINGFACE)
        assert local_backend is not hf_backend

    def test_load_accepts_string_uri(self):
        loader = DataLoader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"key": "value"}\n')
            f.flush()
            docs = list(loader.load(f.name))
            assert docs == [{"key": "value"}]


class TestLocalBackendIntegration:
    """Integration tests for local file loading."""

    def test_load_jsonl(self, tmp_path: Path):
        file_path = tmp_path / "data.jsonl"
        file_path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n')

        loader = DataLoader()
        source = DataSource(path=str(file_path))

        docs = list(loader.load(source))
        assert docs == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_load_json_array(self, tmp_path: Path):
        file_path = tmp_path / "data.json"
        file_path.write_text('[{"a": 1}, {"a": 2}]')

        loader = DataLoader()
        source = DataSource(path=str(file_path))

        docs = list(loader.load(source))
        assert docs == [{"a": 1}, {"a": 2}]

    def test_load_json_with_data_key(self, tmp_path: Path):
        file_path = tmp_path / "data.json"
        file_path.write_text('{"data": [{"a": 1}, {"a": 2}]}')

        loader = DataLoader()
        source = DataSource(path=str(file_path))

        docs = list(loader.load(source))
        assert docs == [{"a": 1}, {"a": 2}]

    def test_load_csv(self, tmp_path: Path):
        file_path = tmp_path / "data.csv"
        file_path.write_text("name,value\nfoo,1\nbar,2\n")

        loader = DataLoader()
        source = DataSource(path=str(file_path))

        docs = list(loader.load(source))
        assert docs == [{"name": "foo", "value": "1"}, {"name": "bar", "value": "2"}]

    def test_load_directory_with_multiple_files(self, tmp_path: Path):
        (tmp_path / "a.jsonl").write_text('{"id": 1}\n')
        (tmp_path / "b.jsonl").write_text('{"id": 2}\n')

        loader = DataLoader()
        source = DataSource(path=str(tmp_path))

        docs = list(loader.load(source))
        assert docs == [{"id": 1}, {"id": 2}]

    def test_load_nonexistent_file_raises(self):
        loader = DataLoader()
        source = DataSource(path="/nonexistent/path/data.jsonl")

        with pytest.raises(FileNotFoundError):
            list(loader.load(source))

    def test_load_unsupported_format_raises(self, tmp_path: Path):
        file_path = tmp_path / "data.xyz"
        file_path.write_text("some content")

        loader = DataLoader()
        source = DataSource(path=str(file_path))

        with pytest.raises(ValueError, match="Unsupported file format"):
            list(loader.load(source))

    def test_empty_lines_in_jsonl_skipped(self, tmp_path: Path):
        file_path = tmp_path / "data.jsonl"
        file_path.write_text('{"a": 1}\n\n{"a": 2}\n   \n{"a": 3}\n')

        loader = DataLoader()
        source = DataSource(path=str(file_path))

        docs = list(loader.load(source))
        assert docs == [{"a": 1}, {"a": 2}, {"a": 3}]
