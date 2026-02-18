"""Tests for JSON data extraction consistency across backends."""

import pytest

from olmo_eval.data.backends.base import JSON_DATA_KEYS, extract_json_data


class TestExtractJsonData:
    """Tests for the shared extract_json_data helper."""

    def test_array_format(self):
        """Test loading JSON array format."""
        data = [{"a": 1}, {"a": 2}]
        result = list(extract_json_data(data, "test.json"))
        assert result == [{"a": 1}, {"a": 2}]

    def test_data_key(self):
        """Test loading JSON with 'data' key."""
        data = {"data": [{"a": 1}, {"a": 2}]}
        result = list(extract_json_data(data, "test.json"))
        assert result == [{"a": 1}, {"a": 2}]

    def test_instances_key(self):
        """Test loading JSON with 'instances' key."""
        data = {"instances": [{"a": 1}, {"a": 2}]}
        result = list(extract_json_data(data, "test.json"))
        assert result == [{"a": 1}, {"a": 2}]

    def test_examples_key(self):
        """Test loading JSON with 'examples' key."""
        data = {"examples": [{"a": 1}, {"a": 2}]}
        result = list(extract_json_data(data, "test.json"))
        assert result == [{"a": 1}, {"a": 2}]

    def test_items_key(self):
        """Test loading JSON with 'items' key."""
        data = {"items": [{"a": 1}, {"a": 2}]}
        result = list(extract_json_data(data, "test.json"))
        assert result == [{"a": 1}, {"a": 2}]

    def test_records_key(self):
        """Test loading JSON with 'records' key."""
        data = {"records": [{"a": 1}, {"a": 2}]}
        result = list(extract_json_data(data, "test.json"))
        assert result == [{"a": 1}, {"a": 2}]

    def test_key_priority(self):
        """Test that keys are checked in order (data takes precedence)."""
        data = {"data": [{"from": "data"}], "instances": [{"from": "instances"}]}
        result = list(extract_json_data(data, "test.json"))
        assert result == [{"from": "data"}]

    def test_invalid_dict_no_known_key(self):
        """Test error when dict has no recognized key."""
        data = {"unknown_key": [{"a": 1}]}
        with pytest.raises(ValueError, match="must contain array or object with one of"):
            list(extract_json_data(data, "test.json"))

    def test_invalid_non_list_value(self):
        """Test error when known key has non-list value."""
        data = {"data": "not a list"}
        with pytest.raises(ValueError, match="must contain array or object with one of"):
            list(extract_json_data(data, "test.json"))

    def test_invalid_type(self):
        """Test error when data is neither list nor dict."""
        with pytest.raises(ValueError, match="must contain array or object"):
            list(extract_json_data("string", "test.json"))

    def test_empty_array(self):
        """Test loading empty array."""
        data: list = []
        result = list(extract_json_data(data, "test.json"))
        assert result == []

    def test_empty_data_key(self):
        """Test loading empty array under data key."""
        data = {"data": []}
        result = list(extract_json_data(data, "test.json"))
        assert result == []

    def test_all_keys_constant(self):
        """Test that JSON_DATA_KEYS contains expected keys."""
        assert JSON_DATA_KEYS == ("data", "instances", "examples", "items", "records")
