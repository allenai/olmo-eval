"""Unit tests for database session management."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestCreatePostgresEngine:
    """Tests for create_postgres_engine function."""

    @patch("olmo_eval.storage.backends.postgres.session.event")
    @patch("olmo_eval.storage.backends.postgres.session.create_engine")
    def test_create_engine_default_params(self, mock_create_engine, mock_event):
        """Test creating engine with default parameters."""
        from olmo_eval.storage.backends.postgres.session import create_postgres_engine

        create_postgres_engine()

        mock_create_engine.assert_called_once()
        call_args = mock_create_engine.call_args

        # Check connection URL
        assert call_args[0][0].startswith("postgresql+psycopg://postgres:@localhost:5432/olmo_eval")

        # Check pool settings
        assert call_args[1]["pool_size"] == 5
        assert call_args[1]["max_overflow"] == 10
        assert call_args[1]["pool_timeout"] == 30.0

    @patch("olmo_eval.storage.backends.postgres.session.event")
    @patch("olmo_eval.storage.backends.postgres.session.create_engine")
    def test_create_engine_custom_params(self, mock_create_engine, mock_event):
        """Test creating engine with custom parameters."""
        from olmo_eval.storage.backends.postgres.session import create_postgres_engine

        create_postgres_engine(
            host="db.example.com",
            port=5555,
            database="test_db",
            user="testuser",
            password="testpass",
            pool_size=10,
            max_overflow=20,
        )

        call_args = mock_create_engine.call_args
        assert "db.example.com" in call_args[0][0]
        assert "5555" in call_args[0][0]
        assert "test_db" in call_args[0][0]
        assert "testuser" in call_args[0][0]
        assert call_args[1]["pool_size"] == 10
        assert call_args[1]["max_overflow"] == 20

    @patch("olmo_eval.storage.backends.postgres.session.event")
    @patch("olmo_eval.storage.backends.postgres.session.create_engine")
    def test_create_engine_password_from_env(self, mock_create_engine, mock_event):
        """Test that password can be loaded from environment variable."""
        from olmo_eval.storage.backends.postgres.session import create_postgres_engine

        with patch.dict(os.environ, {"TEST_PG_PASSWORD": "secret_password"}):
            create_postgres_engine(password_env="TEST_PG_PASSWORD")

            call_args = mock_create_engine.call_args
            assert "secret_password" in call_args[0][0]

    @patch("olmo_eval.storage.backends.postgres.session.event")
    @patch("olmo_eval.storage.backends.postgres.session.create_engine")
    def test_create_engine_echo_mode(self, mock_create_engine, mock_event):
        """Test creating engine with echo mode enabled."""
        from olmo_eval.storage.backends.postgres.session import create_postgres_engine

        create_postgres_engine(echo=True)

        call_args = mock_create_engine.call_args
        assert call_args[1]["echo"] is True

    @patch("olmo_eval.storage.backends.postgres.session.event")
    @patch("olmo_eval.storage.backends.postgres.session.create_engine")
    def test_create_engine_null_pool(self, mock_create_engine, mock_event):
        """Test creating engine with NullPool."""
        from sqlalchemy.pool import NullPool

        from olmo_eval.storage.backends.postgres.session import create_postgres_engine

        create_postgres_engine(poolclass=NullPool)

        call_args = mock_create_engine.call_args
        assert call_args[1]["poolclass"] is NullPool


class TestDatabaseSession:
    """Tests for DatabaseSession class."""

    def test_init(self):
        """Test DatabaseSession initialization."""
        from olmo_eval.storage.backends.postgres.session import DatabaseSession

        db = DatabaseSession(
            host="localhost",
            database="test_db",
            user="testuser",
            password="testpass",
        )

        assert db.host == "localhost"
        assert db.database == "test_db"
        assert db.user == "testuser"
        assert db.password == "testpass"
        assert db._engine is None
        assert db._session_factory is None

    @patch("olmo_eval.storage.backends.postgres.session.create_postgres_engine")
    @patch("olmo_eval.storage.backends.postgres.session.create_session_factory")
    def test_initialize(self, mock_session_factory, mock_engine):
        """Test DatabaseSession initialize method."""
        from olmo_eval.storage.backends.postgres.session import DatabaseSession

        db = DatabaseSession()
        db.initialize()

        mock_engine.assert_called_once()
        mock_session_factory.assert_called_once()
        assert db._engine is not None
        assert db._session_factory is not None

    @patch("olmo_eval.storage.backends.postgres.session.create_postgres_engine")
    @patch("olmo_eval.storage.backends.postgres.session.create_session_factory")
    def test_initialize_twice_warning(self, mock_session_factory, mock_engine):
        """Test that initializing twice logs a warning."""
        from olmo_eval.storage.backends.postgres.session import DatabaseSession

        db = DatabaseSession()
        db.initialize()
        db.initialize()  # Second call should log warning

        # Should only be called once
        assert mock_engine.call_count == 1

    def test_engine_property_not_initialized(self):
        """Test that accessing engine before initialization raises error."""
        from olmo_eval.storage.backends.postgres.session import DatabaseSession

        db = DatabaseSession()

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = db.engine

    def test_session_factory_property_not_initialized(self):
        """Test that accessing session_factory before initialization raises error."""
        from olmo_eval.storage.backends.postgres.session import DatabaseSession

        db = DatabaseSession()

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = db.session_factory

    @patch("olmo_eval.storage.backends.postgres.session.create_postgres_engine")
    @patch("olmo_eval.storage.backends.postgres.session.create_session_factory")
    def test_engine_property(self, mock_session_factory, mock_engine):
        """Test engine property after initialization."""
        from olmo_eval.storage.backends.postgres.session import DatabaseSession

        mock_engine.return_value = MagicMock()

        db = DatabaseSession()
        db.initialize()

        engine = db.engine
        assert engine is not None

    @patch("olmo_eval.storage.backends.postgres.session.create_postgres_engine")
    @patch("olmo_eval.storage.backends.postgres.session.create_session_factory")
    def test_session_context_manager(self, mock_session_factory, mock_engine):
        """Test session context manager."""
        from olmo_eval.storage.backends.postgres.session import DatabaseSession

        mock_session = MagicMock()
        mock_session_factory.return_value = MagicMock(return_value=mock_session)

        db = DatabaseSession()
        db.initialize()

        with db.session() as session:
            assert session is mock_session

        # Verify session lifecycle
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @patch("olmo_eval.storage.backends.postgres.session.create_postgres_engine")
    @patch("olmo_eval.storage.backends.postgres.session.create_session_factory")
    def test_dispose(self, mock_session_factory, mock_engine):
        """Test dispose method."""
        from olmo_eval.storage.backends.postgres.session import DatabaseSession

        mock_engine_instance = MagicMock()
        mock_engine.return_value = mock_engine_instance

        db = DatabaseSession()
        db.initialize()
        db.dispose()

        mock_engine_instance.dispose.assert_called_once()
        assert db._engine is None
        assert db._session_factory is None


class TestGetSession:
    """Tests for get_session context manager."""

    def test_get_session_commits_on_success(self):
        """Test that get_session commits on successful exit."""
        from olmo_eval.storage.backends.postgres.session import get_session

        mock_session = MagicMock()
        mock_factory = MagicMock(return_value=mock_session)

        with get_session(mock_factory) as session:
            assert session is mock_session

        mock_session.commit.assert_called_once()
        mock_session.rollback.assert_not_called()
        mock_session.close.assert_called_once()

    def test_get_session_rolls_back_on_exception(self):
        """Test that get_session rolls back on exception."""
        from olmo_eval.storage.backends.postgres.session import get_session

        mock_session = MagicMock()
        mock_factory = MagicMock(return_value=mock_session)

        with pytest.raises(ValueError), get_session(mock_factory):
            raise ValueError("Test error")

        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()
        mock_session.close.assert_called_once()


class TestGetTransaction:
    """Tests for get_transaction context manager."""

    def test_get_transaction_commits_on_success(self):
        """Test that get_transaction commits on successful exit."""
        from olmo_eval.storage.backends.postgres.session import get_transaction

        mock_session = MagicMock()

        with get_transaction(mock_session):
            pass

        mock_session.commit.assert_called_once()
        mock_session.rollback.assert_not_called()

    def test_get_transaction_rolls_back_on_exception(self):
        """Test that get_transaction rolls back on exception."""
        from olmo_eval.storage.backends.postgres.session import get_transaction

        mock_session = MagicMock()

        with pytest.raises(ValueError), get_transaction(mock_session):
            raise ValueError("Test error")

        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()


class TestRetryOnTransientDbError:
    """Tests for retry_on_transient_db_error decorator."""

    def test_no_error_no_retry(self):
        """Test that successful calls don't retry."""
        from olmo_eval.storage.backends.postgres.session import retry_on_transient_db_error

        call_count = 0

        @retry_on_transient_db_error(max_retries=3)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed()
        assert result == "success"
        assert call_count == 1

    def test_retries_on_connection_error(self):
        """Test that connection errors are retried."""
        from sqlalchemy.exc import OperationalError

        from olmo_eval.storage.backends.postgres.session import retry_on_transient_db_error

        call_count = 0

        @retry_on_transient_db_error(max_retries=3, base_delay=0.01)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OperationalError("connection", {}, Exception("connection reset"))
            return "success"

        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 3

    def test_retries_on_deadlock(self):
        """Test that deadlock errors are retried."""
        from sqlalchemy.exc import OperationalError

        from olmo_eval.storage.backends.postgres.session import retry_on_transient_db_error

        call_count = 0

        @retry_on_transient_db_error(max_retries=3, base_delay=0.01)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                orig = MagicMock()
                orig.pgcode = "40P01"  # deadlock_detected
                exc = OperationalError("deadlock", {}, orig)
                exc.orig = orig
                raise exc
            return "success"

        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 2

    def test_exhausts_retries(self):
        """Test that retries are exhausted and exception is raised."""
        from sqlalchemy.exc import OperationalError

        from olmo_eval.storage.backends.postgres.session import retry_on_transient_db_error

        call_count = 0

        @retry_on_transient_db_error(max_retries=2, base_delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise OperationalError("connection", {}, Exception("connection refused"))

        with pytest.raises(OperationalError):
            always_fail()

        assert call_count == 3  # Initial + 2 retries

    def test_non_retryable_error_not_retried(self):
        """Test that non-connection OperationalErrors are not retried."""
        from sqlalchemy.exc import OperationalError

        from olmo_eval.storage.backends.postgres.session import retry_on_transient_db_error

        call_count = 0

        @retry_on_transient_db_error(max_retries=3, base_delay=0.01)
        def fail_with_syntax_error():
            nonlocal call_count
            call_count += 1
            orig = MagicMock()
            orig.pgcode = "42601"  # syntax_error - not retryable
            exc = OperationalError("syntax error", {}, orig)
            exc.orig = orig
            raise exc

        with pytest.raises(OperationalError):
            fail_with_syntax_error()

        assert call_count == 1  # No retries
