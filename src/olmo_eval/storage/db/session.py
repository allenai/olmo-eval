"""SQLAlchemy session management with connection pooling."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

logger = logging.getLogger(__name__)


def create_postgres_engine(
    host: str = "localhost",
    port: int = 5432,
    database: str = "olmo_eval",
    user: str = "postgres",
    password: str = "",
    password_env: str | None = None,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: float = 30.0,
    pool_recycle: int = 3600,
    connect_timeout: int = 10,
    sslmode: str = "require",
    echo: bool = False,
    **engine_kwargs: Any,
) -> Engine:
    """Create a SQLAlchemy engine with connection pooling.

    Args:
        host: Database host.
        port: Database port.
        database: Database name.
        user: Database user.
        password: Database password (can be overridden by password_env).
        password_env: Environment variable name containing password.
        pool_size: Number of connections to maintain in the pool.
        max_overflow: Maximum number of connections to create beyond pool_size.
        pool_timeout: Seconds to wait before timing out on connection pool.
        pool_recycle: Seconds after which to recycle connections (prevents stale connections).
        connect_timeout: Seconds to wait for initial database connection (default 10).
        sslmode: SSL mode for connection (require, prefer, disable, etc.).
        echo: If True, log all SQL statements (useful for debugging).
        **engine_kwargs: Additional keyword arguments passed to create_engine.

    Returns:
        Configured SQLAlchemy Engine with connection pooling.

    Example:
        >>> engine = create_postgres_engine(
        ...     host="localhost",
        ...     database="olmo_eval",
        ...     password_env="POSTGRES_PASSWORD",
        ...     pool_size=10,
        ... )
    """
    # Get password from environment if specified
    if password_env:
        password = os.environ.get(password_env, password)

    # Build connection URL (postgresql+psycopg = psycopg3 driver)
    connection_url = (
        f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
        f"?connect_timeout={connect_timeout}&sslmode={sslmode}"
    )

    # Determine pooling strategy
    # For testing or single-threaded use, NullPool is simpler
    # For production, QueuePool with connection pooling
    pool_class = engine_kwargs.pop("poolclass", QueuePool)

    # Create engine with connection pooling
    engine = create_engine(
        connection_url,
        poolclass=pool_class,
        pool_size=pool_size if pool_class != NullPool else 0,
        max_overflow=max_overflow if pool_class != NullPool else 0,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
        echo=echo,
        future=True,  # Use SQLAlchemy 2.0 style
        connect_args={"sslmode": sslmode},
        **engine_kwargs,
    )

    # Add connection event listeners for better diagnostics
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_conn: Any, connection_record: Any) -> None:
        """Log successful connections."""
        logger.debug(f"Connected to database: {database}")

    @event.listens_for(engine, "checkin")
    def receive_checkin(dbapi_conn: Any, connection_record: Any) -> None:
        """Log connection returns to pool."""
        logger.debug("Connection returned to pool")

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to an engine.

    Args:
        engine: SQLAlchemy Engine instance.

    Returns:
        Session factory (sessionmaker) for creating new sessions.

    Example:
        >>> engine = create_postgres_engine(database="olmo_eval")
        >>> SessionFactory = create_session_factory(engine)
        >>> session = SessionFactory()
        >>> # Use session...
        >>> session.close()
    """
    return sessionmaker(
        bind=engine,
        expire_on_commit=False,  # Don't expire objects after commit (more efficient)
        autoflush=True,  # Automatically flush before queries
        autocommit=False,  # Use explicit transactions
    )


@contextmanager
def get_session(session_factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    """Context manager for database sessions with automatic cleanup.

    Provides a session that automatically commits on success and rolls back
    on exceptions. Always closes the session when done.

    Args:
        session_factory: Session factory created by create_session_factory.

    Yields:
        Database session.

    Raises:
        Exception: Any exception raised within the context will cause a rollback.

    Example:
        >>> SessionFactory = create_session_factory(engine)
        >>> with get_session(SessionFactory) as session:
        ...     eval_run = session.query(EvalRun).first()
        ...     # Session automatically commits on exit (if no exception)
    """
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_transaction(session: Session) -> Generator[None, None, None]:
    """Context manager for explicit transactions within a session.

    Useful for nested transactions or when you need fine-grained control
    over transaction boundaries.

    Args:
        session: Active database session.

    Yields:
        None (transaction is active during context).

    Raises:
        Exception: Any exception will cause a rollback of the transaction.

    Example:
        >>> with get_session(SessionFactory) as session:
        ...     # First transaction
        ...     with get_transaction(session):
        ...         session.add(eval_run1)
        ...         # Commits here if successful
        ...
        ...     # Second transaction
        ...     with get_transaction(session):
        ...         session.add(eval_run2)
        ...         # Commits here if successful
    """
    try:
        yield
        session.commit()
    except Exception:
        session.rollback()
        raise


class DatabaseSession:
    """Database session manager with connection pooling.

    Provides a high-level interface for managing database connections
    with connection pooling and session lifecycle management.

    Example:
        >>> db = DatabaseSession(
        ...     host="localhost", database="olmo_eval", password_env="POSTGRES_PASSWORD"
        ... )
        >>> db.initialize()
        >>> with db.session() as session:
        ...     eval_runs = session.query(EvalRun).all()
        >>> db.dispose()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "olmo_eval",
        user: str = "postgres",
        password: str = "",
        password_env: str | None = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        sslmode: str = "require",
        echo: bool = False,
        **engine_kwargs: Any,
    ):
        """Initialize database session manager.

        Args:
            host: Database host.
            port: Database port.
            database: Database name.
            user: Database user.
            password: Database password.
            password_env: Environment variable containing password.
            pool_size: Connection pool size.
            max_overflow: Maximum overflow connections.
            sslmode: SSL mode for connection (require, prefer, disable, etc.).
            echo: Whether to echo SQL statements.
            **engine_kwargs: Additional engine configuration.
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.password_env = password_env
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.sslmode = sslmode
        self.echo = echo
        self.engine_kwargs = engine_kwargs

        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    def initialize(self) -> None:
        """Initialize the database engine and session factory.

        Tests the connection to ensure the database is reachable.

        Raises:
            Exception: If unable to connect to the database.
        """
        if self._engine is not None:
            logger.warning("Database already initialized")
            return

        logger.info(f"Connecting to PostgreSQL: {self.host}:{self.port}/{self.database}")
        self._engine = create_postgres_engine(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            password_env=self.password_env,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            sslmode=self.sslmode,
            echo=self.echo,
            **self.engine_kwargs,
        )
        self._session_factory = create_session_factory(self._engine)

        # Test the connection to fail fast if database is unreachable
        from sqlalchemy import text

        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL: {self.host}:{self.port}/{self.database}")

    @property
    def engine(self) -> Engine:
        """Get the database engine.

        Raises:
            RuntimeError: If not initialized.
        """
        if self._engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Get the session factory.

        Raises:
            RuntimeError: If not initialized.
        """
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._session_factory

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get a database session context manager.

        Yields:
            Database session with automatic commit/rollback.

        Example:
            >>> with db.session() as session:
            ...     results = session.query(EvalRun).all()
        """
        with get_session(self.session_factory) as session:
            yield session

    def dispose(self) -> None:
        """Dispose of the engine and close all connections."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database session manager disposed")
