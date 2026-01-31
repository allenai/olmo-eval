"""Storage backend setup for the run command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from olmo_eval.runners.models import S3Config
    from olmo_eval.storage import StorageBackend

console = Console()


class StorageSetup:
    """Sets up storage backends for evaluation runs."""

    def __init__(
        self,
        store: bool = False,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "olmo_eval",
        db_user: str = "postgres",
        db_password: str = "",
        s3_bucket: str | None = None,
        s3_prefix: str | None = None,
        s3_group: str | None = None,
        s3_endpoint_url: str | None = None,
        s3_region: str = "us-east-1",
    ):
        """Initialize storage setup with configuration.

        Args:
            store: Whether to persist results to PostgreSQL.
            db_host: PostgreSQL host.
            db_port: PostgreSQL port.
            db_name: PostgreSQL database name.
            db_user: PostgreSQL user.
            db_password: PostgreSQL password.
            s3_bucket: S3 bucket for storing evaluation results.
            s3_prefix: S3 prefix/path within bucket.
            s3_group: S3 group name (used in path structure).
            s3_endpoint_url: S3 endpoint URL (for S3-compatible storage).
            s3_region: S3 region.
        """
        self.store = store
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_group = s3_group
        self.s3_endpoint_url = s3_endpoint_url
        self.s3_region = s3_region

    def setup_postgres(self) -> StorageBackend | None:
        """Set up PostgreSQL storage backend.

        Returns:
            Initialized PostgresStorage backend, or None if not enabled.

        Raises:
            SystemExit: If storage setup fails.
        """
        if not self.store:
            return None

        from olmo_eval.storage import get_backend

        try:
            storage = get_backend(
                "postgres",
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
            )
            storage.initialize()
            console.print(
                f"[green]Connected to postgres storage:[/green] "
                f"{self.db_host}:{self.db_port}/{self.db_name}"
            )
            return storage
        except ImportError as e:
            console.print(f"[red]Storage backend error:[/red] {e}")
            raise SystemExit(1) from None
        except Exception as e:
            console.print(f"[red]Failed to initialize storage backend:[/red] {e}")
            raise SystemExit(1) from None

    def setup_s3_config(self) -> S3Config | None:
        """Set up S3 configuration.

        Returns:
            S3Config if all required options are provided, None otherwise.

        Raises:
            SystemExit: If S3 configuration is incomplete.
        """
        if not self.s3_bucket and not self.s3_prefix and not self.s3_group:
            return None

        # Validate that all required S3 options are provided
        if not self.s3_bucket:
            console.print("[red]Error:[/red] --s3-bucket is required for S3 uploads")
            raise SystemExit(1)
        if not self.s3_prefix:
            console.print("[red]Error:[/red] --s3-prefix is required for S3 uploads")
            raise SystemExit(1)
        if not self.s3_group:
            console.print("[red]Error:[/red] --s3-group is required for S3 uploads")
            raise SystemExit(1)

        from olmo_eval.runners.models import S3Config

        s3_config = S3Config(
            bucket=self.s3_bucket,
            prefix=self.s3_prefix,
            group=self.s3_group,
            endpoint_url=self.s3_endpoint_url,
            region=self.s3_region,
        )
        console.print(
            f"[green]S3 uploads enabled:[/green] "
            f"s3://{self.s3_bucket}/{self.s3_prefix}/{self.s3_group}/..."
        )
        return s3_config

    def setup(self) -> tuple[list[StorageBackend], S3Config | None]:
        """Set up all storage backends.

        Returns:
            Tuple of (list of storage backends, S3Config or None).
        """
        storages: list[Any] = []

        postgres_storage = self.setup_postgres()
        if postgres_storage:
            storages.append(postgres_storage)

        s3_config = self.setup_s3_config()

        return storages, s3_config
