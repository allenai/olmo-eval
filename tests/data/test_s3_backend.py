"""Tests for olmo_eval.data.backends.s3 module."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from olmo_eval.data.backends.s3 import S3Backend
from olmo_eval.data.sources import DataSource


class TestS3BackendInit:
    """Tests for S3Backend initialization."""

    def test_default_init(self):
        """Test default initialization."""
        backend = S3Backend()
        assert backend.endpoint_url is None
        assert backend.region_name is None
        assert backend._s3_client is None

    def test_custom_endpoint(self):
        """Test initialization with custom endpoint."""
        backend = S3Backend(
            endpoint_url="http://localhost:4566",
            region_name="us-east-1",
        )
        assert backend.endpoint_url == "http://localhost:4566"
        assert backend.region_name == "us-east-1"


class TestS3BackendParseUri:
    """Tests for S3 URI parsing."""

    def test_parse_simple_uri(self):
        """Test parsing a simple S3 URI."""
        backend = S3Backend()
        bucket, key = backend._parse_s3_uri("s3://my-bucket/path/to/file.jsonl")
        assert bucket == "my-bucket"
        assert key == "path/to/file.jsonl"

    def test_parse_root_key(self):
        """Test parsing an S3 URI with root key."""
        backend = S3Backend()
        bucket, key = backend._parse_s3_uri("s3://bucket/file.jsonl")
        assert bucket == "bucket"
        assert key == "file.jsonl"

    def test_parse_trailing_slash(self):
        """Test parsing an S3 URI with trailing slash."""
        backend = S3Backend()
        bucket, key = backend._parse_s3_uri("s3://bucket/prefix/")
        assert bucket == "bucket"
        assert key == "prefix/"

    def test_parse_invalid_scheme(self):
        """Test that non-S3 URIs raise an error."""
        backend = S3Backend()
        with pytest.raises(ValueError, match="Expected s3:// URI"):
            backend._parse_s3_uri("gs://bucket/file.jsonl")


class TestS3BackendIsPrefix:
    """Tests for prefix detection."""

    def test_trailing_slash_is_prefix(self):
        """Test that paths ending in / are prefixes."""
        backend = S3Backend()
        assert backend._is_prefix("s3://bucket/path/") is True

    def test_jsonl_is_not_prefix(self):
        """Test that .jsonl files are not prefixes."""
        backend = S3Backend()
        assert backend._is_prefix("s3://bucket/file.jsonl") is False

    def test_json_is_not_prefix(self):
        """Test that .json files are not prefixes."""
        backend = S3Backend()
        assert backend._is_prefix("s3://bucket/file.json") is False

    def test_parquet_is_not_prefix(self):
        """Test that .parquet files are not prefixes."""
        backend = S3Backend()
        assert backend._is_prefix("s3://bucket/file.parquet") is False

    def test_csv_is_not_prefix(self):
        """Test that .csv files are not prefixes."""
        backend = S3Backend()
        assert backend._is_prefix("s3://bucket/file.csv") is False


class TestS3BackendLoadJsonl:
    """Tests for JSONL loading from S3."""

    @patch("olmo_eval.data.backends.s3.S3Backend._get_smart_open")
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

        backend = S3Backend()
        source = DataSource(path="s3://bucket/data.jsonl")
        docs = list(backend.load(source))

        assert len(docs) == 2
        assert docs[0] == {"id": 1, "text": "hello"}
        assert docs[1] == {"id": 2, "text": "world"}

    @patch("olmo_eval.data.backends.s3.S3Backend._get_smart_open")
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

        backend = S3Backend()
        source = DataSource(path="s3://bucket/data.jsonl")
        docs = list(backend.load(source))

        assert len(docs) == 3

    @patch("olmo_eval.data.backends.s3.S3Backend._get_smart_open")
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

        backend = S3Backend()
        source = DataSource(path="s3://bucket/data.jsonl")
        docs = list(backend.load(source))

        assert len(docs) == 2


class TestS3BackendLoadJson:
    """Tests for JSON loading from S3."""

    @patch("olmo_eval.data.backends.s3.S3Backend._get_smart_open")
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

        backend = S3Backend()
        source = DataSource(path="s3://bucket/data.json")
        docs = list(backend.load(source))

        assert len(docs) == 2

    @patch("olmo_eval.data.backends.s3.S3Backend._get_smart_open")
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

        backend = S3Backend()
        source = DataSource(path="s3://bucket/data.json")
        docs = list(backend.load(source))

        assert len(docs) == 2

    @patch("olmo_eval.data.backends.s3.S3Backend._get_smart_open")
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

        backend = S3Backend()
        source = DataSource(path="s3://bucket/data.json")
        docs = list(backend.load(source))

        assert len(docs) == 1


class TestS3BackendLoadPrefix:
    """Tests for loading from S3 prefixes."""

    @patch("olmo_eval.data.backends.s3.S3Backend._get_smart_open")
    @patch("olmo_eval.data.backends.s3.S3Backend._list_objects")
    def test_load_prefix(self, mock_list_objects, mock_get_smart_open):
        """Test loading all files from a prefix."""
        # Mock listing objects
        mock_list_objects.return_value = [
            "s3://bucket/prefix/file1.jsonl",
            "s3://bucket/prefix/file2.jsonl",
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

        backend = S3Backend()
        source = DataSource(path="s3://bucket/prefix/")
        docs = list(backend.load(source))

        assert len(docs) == 2

    @patch("olmo_eval.data.backends.s3.S3Backend._list_objects")
    def test_load_prefix_no_objects(self, mock_list_objects):
        """Test that empty prefix raises error."""
        mock_list_objects.return_value = []

        backend = S3Backend()
        source = DataSource(path="s3://bucket/empty/")

        with pytest.raises(ValueError, match="No objects found"):
            list(backend.load(source))

    @patch("olmo_eval.data.backends.s3.S3Backend._list_objects")
    def test_load_prefix_no_supported_files(self, mock_list_objects):
        """Test that prefix with no supported files raises error."""
        mock_list_objects.return_value = [
            "s3://bucket/prefix/file.txt",
            "s3://bucket/prefix/file.md",
        ]

        backend = S3Backend()
        source = DataSource(path="s3://bucket/prefix/")

        with pytest.raises(ValueError, match="No supported files found"):
            list(backend.load(source))


class TestS3BackendExists:
    """Tests for S3 path existence checks."""

    def test_exists_file(self):
        """Test checking if a file exists."""
        backend = S3Backend()
        backend._s3_client = MagicMock()
        backend._s3_client.head_object = MagicMock(return_value={})

        assert backend.exists("s3://bucket/file.jsonl") is True
        backend._s3_client.head_object.assert_called_once_with(
            Bucket="bucket",
            Key="file.jsonl",
        )

    def test_exists_file_not_found(self):
        """Test checking if a non-existent file exists."""
        backend = S3Backend()
        backend._s3_client = MagicMock()
        backend._s3_client.head_object.side_effect = Exception("Not found")
        backend._s3_client.list_objects_v2.return_value = {"KeyCount": 0}

        assert backend.exists("s3://bucket/nonexistent.jsonl") is False

    def test_exists_prefix(self):
        """Test checking if a prefix exists."""
        backend = S3Backend()
        backend._s3_client = MagicMock()
        backend._s3_client.head_object.side_effect = Exception("Not found")
        backend._s3_client.list_objects_v2.return_value = {"KeyCount": 1}

        assert backend.exists("s3://bucket/prefix/") is True


class TestS3BackendErrorHandling:
    """Tests for error handling."""

    def test_unsupported_format(self):
        """Test that unsupported formats raise an error."""
        backend = S3Backend()
        backend._s3_client = MagicMock()
        backend._s3_client.list_objects_v2.return_value = {"KeyCount": 0}

        source = DataSource(path="s3://bucket/file.xyz")

        with pytest.raises(ValueError, match="Cannot determine file format"):
            list(backend.load(source))

    def test_missing_smart_open(self):
        """Test that missing smart_open raises ImportError."""
        backend = S3Backend()

        with (
            patch.dict("sys.modules", {"smart_open": None}),
            pytest.raises(ImportError, match="smart_open is required"),
        ):
            backend._get_smart_open()
