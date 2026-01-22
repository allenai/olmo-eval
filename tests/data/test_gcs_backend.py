"""Tests for olmo_eval.data.backends.gcs module."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from olmo_eval.data.backends.gcs import GCSBackend
from olmo_eval.data.sources import DataSource


class TestGCSBackendInit:
    """Tests for GCSBackend initialization."""

    def test_default_init(self):
        """Test default initialization."""
        backend = GCSBackend()
        assert backend.project is None
        assert backend._gcs_client is None

    def test_custom_project(self):
        """Test initialization with custom project."""
        backend = GCSBackend(project="my-project")
        assert backend.project == "my-project"


class TestGCSBackendParseUri:
    """Tests for GCS URI parsing."""

    def test_parse_simple_uri(self):
        """Test parsing a simple GCS URI."""
        backend = GCSBackend()
        bucket, prefix = backend._parse_gcs_uri("gs://my-bucket/path/to/file.jsonl")
        assert bucket == "my-bucket"
        assert prefix == "path/to/file.jsonl"

    def test_parse_root_key(self):
        """Test parsing a GCS URI with root key."""
        backend = GCSBackend()
        bucket, prefix = backend._parse_gcs_uri("gs://bucket/file.jsonl")
        assert bucket == "bucket"
        assert prefix == "file.jsonl"

    def test_parse_trailing_slash(self):
        """Test parsing a GCS URI with trailing slash."""
        backend = GCSBackend()
        bucket, prefix = backend._parse_gcs_uri("gs://bucket/prefix/")
        assert bucket == "bucket"
        assert prefix == "prefix/"

    def test_parse_invalid_scheme(self):
        """Test that non-GCS URIs raise an error."""
        backend = GCSBackend()
        with pytest.raises(ValueError, match="Expected gs:// URI"):
            backend._parse_gcs_uri("s3://bucket/file.jsonl")


class TestGCSBackendIsPrefix:
    """Tests for prefix detection."""

    def test_trailing_slash_is_prefix(self):
        """Test that paths ending in / are prefixes."""
        backend = GCSBackend()
        assert backend._is_prefix("gs://bucket/path/") is True

    def test_jsonl_is_not_prefix(self):
        """Test that .jsonl files are not prefixes."""
        backend = GCSBackend()
        assert backend._is_prefix("gs://bucket/file.jsonl") is False

    def test_json_is_not_prefix(self):
        """Test that .json files are not prefixes."""
        backend = GCSBackend()
        assert backend._is_prefix("gs://bucket/file.json") is False

    def test_parquet_is_not_prefix(self):
        """Test that .parquet files are not prefixes."""
        backend = GCSBackend()
        assert backend._is_prefix("gs://bucket/file.parquet") is False

    def test_csv_is_not_prefix(self):
        """Test that .csv files are not prefixes."""
        backend = GCSBackend()
        assert backend._is_prefix("gs://bucket/file.csv") is False


class TestGCSBackendLoadJsonl:
    """Tests for JSONL loading from GCS."""

    @patch("olmo_eval.data.backends.gcs.GCSBackend._get_smart_open")
    def test_load_jsonl(self, mock_get_smart_open):
        """Test loading a JSONL file."""
        # Setup mock
        jsonl_content = '{"id": 1, "text": "hello"}\n{"id": 2, "text": "world"}\n'
        mock_file = StringIO(jsonl_content)
        mock_smart_open = MagicMock()
        mock_smart_open.__enter__ = MagicMock(return_value=mock_file)
        mock_smart_open.__exit__ = MagicMock(return_value=False)
        mock_get_smart_open.return_value = (
            MagicMock(return_value=mock_smart_open),
            {},
        )

        backend = GCSBackend()
        source = DataSource(path="gs://bucket/data.jsonl")
        docs = list(backend.load(source))

        assert len(docs) == 2
        assert docs[0] == {"id": 1, "text": "hello"}
        assert docs[1] == {"id": 2, "text": "world"}

    @patch("olmo_eval.data.backends.gcs.GCSBackend._get_smart_open")
    def test_load_jsonl_with_empty_lines(self, mock_get_smart_open):
        """Test that empty lines are skipped."""
        jsonl_content = '{"id": 1}\n\n{"id": 2}\n  \n{"id": 3}\n'
        mock_file = StringIO(jsonl_content)
        mock_smart_open = MagicMock()
        mock_smart_open.__enter__ = MagicMock(return_value=mock_file)
        mock_smart_open.__exit__ = MagicMock(return_value=False)
        mock_get_smart_open.return_value = (
            MagicMock(return_value=mock_smart_open),
            {},
        )

        backend = GCSBackend()
        source = DataSource(path="gs://bucket/data.jsonl")
        docs = list(backend.load(source))

        assert len(docs) == 3

    @patch("olmo_eval.data.backends.gcs.GCSBackend._get_smart_open")
    def test_load_jsonl_skips_invalid_json(self, mock_get_smart_open):
        """Test that invalid JSON lines are skipped with warning."""
        jsonl_content = '{"id": 1}\nnot valid json\n{"id": 2}\n'
        mock_file = StringIO(jsonl_content)
        mock_smart_open = MagicMock()
        mock_smart_open.__enter__ = MagicMock(return_value=mock_file)
        mock_smart_open.__exit__ = MagicMock(return_value=False)
        mock_get_smart_open.return_value = (
            MagicMock(return_value=mock_smart_open),
            {},
        )

        backend = GCSBackend()
        source = DataSource(path="gs://bucket/data.jsonl")
        docs = list(backend.load(source))

        assert len(docs) == 2


class TestGCSBackendLoadJson:
    """Tests for JSON loading from GCS."""

    @patch("olmo_eval.data.backends.gcs.GCSBackend._get_smart_open")
    def test_load_json_array(self, mock_get_smart_open):
        """Test loading a JSON file with array."""
        json_content = '[{"id": 1}, {"id": 2}]'
        mock_file = StringIO(json_content)
        mock_smart_open = MagicMock()
        mock_smart_open.__enter__ = MagicMock(return_value=mock_file)
        mock_smart_open.__exit__ = MagicMock(return_value=False)
        mock_get_smart_open.return_value = (
            MagicMock(return_value=mock_smart_open),
            {},
        )

        backend = GCSBackend()
        source = DataSource(path="gs://bucket/data.json")
        docs = list(backend.load(source))

        assert len(docs) == 2

    @patch("olmo_eval.data.backends.gcs.GCSBackend._get_smart_open")
    def test_load_json_with_data_key(self, mock_get_smart_open):
        """Test loading a JSON file with 'data' key."""
        json_content = '{"data": [{"id": 1}, {"id": 2}]}'
        mock_file = StringIO(json_content)
        mock_smart_open = MagicMock()
        mock_smart_open.__enter__ = MagicMock(return_value=mock_file)
        mock_smart_open.__exit__ = MagicMock(return_value=False)
        mock_get_smart_open.return_value = (
            MagicMock(return_value=mock_smart_open),
            {},
        )

        backend = GCSBackend()
        source = DataSource(path="gs://bucket/data.json")
        docs = list(backend.load(source))

        assert len(docs) == 2

    @patch("olmo_eval.data.backends.gcs.GCSBackend._get_smart_open")
    def test_load_json_with_instances_key(self, mock_get_smart_open):
        """Test loading a JSON file with 'instances' key."""
        json_content = '{"instances": [{"id": 1}]}'
        mock_file = StringIO(json_content)
        mock_smart_open = MagicMock()
        mock_smart_open.__enter__ = MagicMock(return_value=mock_file)
        mock_smart_open.__exit__ = MagicMock(return_value=False)
        mock_get_smart_open.return_value = (
            MagicMock(return_value=mock_smart_open),
            {},
        )

        backend = GCSBackend()
        source = DataSource(path="gs://bucket/data.json")
        docs = list(backend.load(source))

        assert len(docs) == 1


class TestGCSBackendLoadPrefix:
    """Tests for loading from GCS prefixes."""

    @patch("olmo_eval.data.backends.gcs.GCSBackend._get_smart_open")
    @patch("olmo_eval.data.backends.gcs.GCSBackend._list_objects")
    def test_load_prefix(self, mock_list_objects, mock_get_smart_open):
        """Test loading all files from a prefix."""
        # Mock listing objects
        mock_list_objects.return_value = [
            "gs://bucket/prefix/file1.jsonl",
            "gs://bucket/prefix/file2.jsonl",
        ]

        # Mock file content
        call_count = [0]

        def mock_open(*args, **kwargs):
            call_count[0] += 1
            content = f'{{"id": {call_count[0]}}}\n'
            mock_file = StringIO(content)
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=mock_file)
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        mock_get_smart_open.return_value = (mock_open, {})

        backend = GCSBackend()
        source = DataSource(path="gs://bucket/prefix/")
        docs = list(backend.load(source))

        assert len(docs) == 2

    @patch("olmo_eval.data.backends.gcs.GCSBackend._list_objects")
    def test_load_prefix_no_objects(self, mock_list_objects):
        """Test that empty prefix raises error."""
        mock_list_objects.return_value = []

        backend = GCSBackend()
        source = DataSource(path="gs://bucket/empty/")

        with pytest.raises(ValueError, match="No objects found"):
            list(backend.load(source))

    @patch("olmo_eval.data.backends.gcs.GCSBackend._list_objects")
    def test_load_prefix_no_supported_files(self, mock_list_objects):
        """Test that prefix with no supported files raises error."""
        mock_list_objects.return_value = [
            "gs://bucket/prefix/file.txt",
            "gs://bucket/prefix/file.md",
        ]

        backend = GCSBackend()
        source = DataSource(path="gs://bucket/prefix/")

        with pytest.raises(ValueError, match="No supported files found"):
            list(backend.load(source))

    @patch("olmo_eval.data.backends.gcs.GCSBackend._get_smart_open")
    @patch("olmo_eval.data.backends.gcs.GCSBackend._list_objects")
    def test_load_prefix_filters_by_split(self, mock_list_objects, mock_get_smart_open):
        """Test that split filtering works."""
        mock_list_objects.return_value = [
            "gs://bucket/prefix/train.jsonl",
            "gs://bucket/prefix/test.jsonl",
            "gs://bucket/prefix/validation.jsonl",
        ]

        def mock_open(*args, **kwargs):
            content = '{"id": 1}\n'
            mock_file = StringIO(content)
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=mock_file)
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx

        mock_get_smart_open.return_value = (mock_open, {})

        backend = GCSBackend()
        source = DataSource(path="gs://bucket/prefix/", split="test")
        docs = list(backend.load(source))

        # Should only load test.jsonl
        assert len(docs) == 1


class TestGCSBackendExists:
    """Tests for GCS path existence checks."""

    def test_exists_file(self):
        """Test checking if a file exists."""
        backend = GCSBackend()

        mock_blob = MagicMock()
        mock_blob.exists.return_value = True

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        backend._gcs_client = mock_client

        assert backend.exists("gs://bucket/file.jsonl") is True
        mock_bucket.blob.assert_called_once_with("file.jsonl")

    def test_exists_file_not_found(self):
        """Test checking if a non-existent file exists."""
        backend = GCSBackend()

        mock_blob = MagicMock()
        mock_blob.exists.return_value = False

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_bucket.list_blobs.return_value = iter([])  # Empty iterator

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        backend._gcs_client = mock_client

        assert backend.exists("gs://bucket/nonexistent.jsonl") is False

    def test_exists_prefix(self):
        """Test checking if a prefix exists."""
        backend = GCSBackend()

        mock_blob = MagicMock()
        mock_blob.exists.return_value = False

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        # Return an iterator with at least one blob
        mock_bucket.list_blobs.return_value = iter([MagicMock()])

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        backend._gcs_client = mock_client

        assert backend.exists("gs://bucket/prefix/") is True


class TestGCSBackendErrorHandling:
    """Tests for error handling."""

    @patch("olmo_eval.data.backends.gcs.GCSBackend._prefix_has_objects")
    def test_unsupported_format(self, mock_prefix_has_objects):
        """Test that unsupported formats raise an error."""
        mock_prefix_has_objects.return_value = False

        backend = GCSBackend()
        source = DataSource(path="gs://bucket/file.xyz")

        with pytest.raises(ValueError, match="Cannot determine file format"):
            list(backend.load(source))

    def test_missing_smart_open(self):
        """Test that missing smart_open raises ImportError."""
        backend = GCSBackend()

        with (
            patch.dict("sys.modules", {"smart_open": None}),
            pytest.raises(ImportError, match="smart_open is required"),
        ):
            backend._get_smart_open()
