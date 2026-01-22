"""Tests for GCS credential handling."""

import json
from unittest.mock import patch

from olmo_eval.launch.gcs import (
    GCSCredentials,
    get_local_gcs_credentials,
    is_gcs_path,
)


class TestIsGcsPath:
    """Tests for is_gcs_path function."""

    def test_gcs_path(self):
        """Test that gs:// paths are detected."""
        assert is_gcs_path("gs://bucket/path") is True
        assert is_gcs_path("gs://bucket/path/to/file.jsonl") is True
        assert is_gcs_path("gs://my-bucket") is True

    def test_s3_path(self):
        """Test that s3:// paths are not detected as GCS."""
        assert is_gcs_path("s3://bucket/path") is False

    def test_local_path(self):
        """Test that local paths are not detected as GCS."""
        assert is_gcs_path("/local/path") is False
        assert is_gcs_path("./relative/path") is False

    def test_http_path(self):
        """Test that HTTP paths are not detected as GCS."""
        assert is_gcs_path("https://example.com/file") is False
        assert is_gcs_path("http://example.com/file") is False

    def test_huggingface_path(self):
        """Test that HuggingFace paths are not detected as GCS."""
        assert is_gcs_path("meta-llama/Llama-3.1-8B") is False
        assert is_gcs_path("hf://cais/mmlu") is False


class TestGCSCredentials:
    """Tests for GCSCredentials dataclass."""

    def test_basic_creation(self):
        """Test creating GCSCredentials."""
        creds = GCSCredentials(
            json_key='{"type": "service_account"}',
            project_id="test-project",
            client_email="test@test.iam.gserviceaccount.com",
        )
        assert creds.json_key == '{"type": "service_account"}'
        assert creds.project_id == "test-project"
        assert creds.client_email == "test@test.iam.gserviceaccount.com"

    def test_optional_fields(self):
        """Test that project_id and client_email are optional."""
        creds = GCSCredentials(json_key='{"type": "service_account"}')
        assert creds.project_id is None
        assert creds.client_email is None


class TestGetLocalGcsCredentials:
    """Tests for get_local_gcs_credentials function."""

    def test_from_env_var(self, tmp_path):
        """Test reading credentials from GOOGLE_APPLICATION_CREDENTIALS env var."""
        key_file = tmp_path / "key.json"
        key_data = {
            "type": "service_account",
            "project_id": "test-project",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "private_key": "fake-key",
        }
        key_file.write_text(json.dumps(key_data))

        with patch.dict("os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": str(key_file)}):
            creds = get_local_gcs_credentials()

        assert creds is not None
        assert creds.project_id == "test-project"
        assert creds.client_email == "test@test-project.iam.gserviceaccount.com"
        assert json.loads(creds.json_key) == key_data

    def test_from_env_var_non_service_account(self, tmp_path):
        """Test that non-service-account credentials are rejected."""
        key_file = tmp_path / "key.json"
        key_data = {
            "type": "authorized_user",
            "client_id": "123",
            "client_secret": "secret",
        }
        key_file.write_text(json.dumps(key_data))

        with patch.dict("os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": str(key_file)}):
            creds = get_local_gcs_credentials()

        assert creds is None

    def test_from_env_var_file_not_found(self):
        """Test handling of missing credential file."""
        with patch.dict("os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent/path.json"}):
            creds = get_local_gcs_credentials()

        assert creds is None

    def test_no_credentials(self):
        """Test when no credentials are available."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("pathlib.Path.exists", return_value=False),
        ):
            creds = get_local_gcs_credentials()

        assert creds is None

    def test_from_gcloud_default_credentials(self, tmp_path):
        """Test reading from gcloud default credentials location."""
        # Create a mock home directory with gcloud credentials
        gcloud_dir = tmp_path / ".config" / "gcloud"
        gcloud_dir.mkdir(parents=True)
        creds_file = gcloud_dir / "application_default_credentials.json"

        key_data = {
            "type": "service_account",
            "project_id": "gcloud-project",
            "client_email": "sa@gcloud-project.iam.gserviceaccount.com",
        }
        creds_file.write_text(json.dumps(key_data))

        # Mock Path.home() to return our tmp_path
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("pathlib.Path.home", return_value=tmp_path),
        ):
            creds = get_local_gcs_credentials()

        assert creds is not None
        assert creds.project_id == "gcloud-project"

    def test_env_var_takes_precedence(self, tmp_path):
        """Test that GOOGLE_APPLICATION_CREDENTIALS takes precedence over gcloud."""
        # Create both credential files
        env_key_file = tmp_path / "env_key.json"
        env_key_data = {
            "type": "service_account",
            "project_id": "env-project",
            "client_email": "env@env-project.iam.gserviceaccount.com",
        }
        env_key_file.write_text(json.dumps(env_key_data))

        gcloud_dir = tmp_path / ".config" / "gcloud"
        gcloud_dir.mkdir(parents=True)
        gcloud_creds_file = gcloud_dir / "application_default_credentials.json"
        gcloud_key_data = {
            "type": "service_account",
            "project_id": "gcloud-project",
            "client_email": "gcloud@gcloud-project.iam.gserviceaccount.com",
        }
        gcloud_creds_file.write_text(json.dumps(gcloud_key_data))

        with (
            patch.dict("os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": str(env_key_file)}),
            patch("pathlib.Path.home", return_value=tmp_path),
        ):
            creds = get_local_gcs_credentials()

        # Should use env var credentials, not gcloud
        assert creds is not None
        assert creds.project_id == "env-project"

    def test_invalid_json_file(self, tmp_path):
        """Test handling of invalid JSON in credential file."""
        key_file = tmp_path / "invalid.json"
        key_file.write_text("not valid json")

        with patch.dict("os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": str(key_file)}):
            creds = get_local_gcs_credentials()

        assert creds is None
