"""Credential management for Beaker launch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from olmo_eval.launch import BeakerLauncher, BeakerModelSpec

console = Console()


class CredentialManager:
    """Manages credential detection and setup for Beaker jobs."""

    def __init__(
        self,
        model_configs: list[BeakerModelSpec],
        store: bool,
        aws_credentials: bool | None,
        gcs_credentials: bool | None,
    ):
        """Initialize the credential manager.

        Args:
            model_configs: List of model configurations.
            store: Whether storage is enabled.
            aws_credentials: Override for AWS credential injection.
            gcs_credentials: Override for GCS credential injection.
        """
        self.model_configs = model_configs
        self.store = store
        self.aws_credentials = aws_credentials
        self.gcs_credentials = gcs_credentials

    def detect_and_setup(
        self,
        launcher: BeakerLauncher,
    ) -> tuple[bool, bool]:
        """Detect and set up credential injection.

        Args:
            launcher: BeakerLauncher instance for API calls.

        Returns:
            Tuple of (inject_aws, inject_gcs).
        """
        from olmo_eval.launch.beaker.aws import is_s3_path
        from olmo_eval.launch.beaker.gcs import get_local_gcs_credentials, is_gcs_path

        # Auto-detect S3 models
        s3_models = [m.name_or_path for m in self.model_configs if is_s3_path(m.name_or_path)]
        inject_aws = self.aws_credentials
        if inject_aws is None:
            inject_aws = bool(s3_models) or self.store

        # Auto-detect GCS models
        gcs_models = [m.name_or_path for m in self.model_configs if is_gcs_path(m.name_or_path)]
        inject_gcs = self.gcs_credentials
        if inject_gcs is None:
            inject_gcs = bool(gcs_models)

        # Display GCS credential info
        if inject_gcs:
            self._display_gcs_info(launcher, get_local_gcs_credentials())

        return inject_aws, inject_gcs

    def _display_gcs_info(self, launcher: Any, local_gcs_creds: Any) -> None:
        """Display GCS credential information."""
        beaker_user = launcher.beaker.user_name

        gcs_table = Table(show_header=False, box=None, expand=True)
        gcs_table.add_column("Key", style="blue")
        gcs_table.add_column("Value")

        if local_gcs_creds:
            gcs_table.add_row("Credentials", "[green]found[/green] (service account)")
            if local_gcs_creds.client_email:
                gcs_table.add_row("Service account", local_gcs_creds.client_email)
            if local_gcs_creds.project_id:
                gcs_table.add_row("Project", local_gcs_creds.project_id)
            gcs_table.add_row("Beaker user", beaker_user)
            gcs_table.add_row("Beaker secret", f"{beaker_user}_GOOGLE_CREDENTIALS")
        else:
            gcs_table.add_row(
                "Credentials",
                "[yellow]not found[/yellow] - job may fail if GCS access is required",
            )

        console.print()
        console.print(
            Panel(gcs_table, title="[bold]GCS Access Configuration[/bold]", border_style="magenta")
        )
        console.print()

    def display_storage_info(
        self,
        launcher: Any,
        s3_bucket: str | None,
        s3_prefix: str | None,
        s3_region: str,
        s3_endpoint_url: str | None,
        effective_groups: list[str],
        inject_aws: bool,
    ) -> None:
        """Display storage configuration information.

        Args:
            launcher: BeakerLauncher instance.
            s3_bucket: S3 bucket name.
            s3_prefix: S3 prefix.
            s3_region: S3 region.
            s3_endpoint_url: S3 endpoint URL.
            effective_groups: List of effective groups.
            inject_aws: Whether AWS credentials will be injected.
        """
        from olmo_eval.launch.beaker.aws import get_local_aws_credentials

        if not (self.store or (s3_bucket and s3_prefix) or inject_aws):
            return

        storage_lines = []

        # S3 Access credentials
        if inject_aws:
            local_creds = get_local_aws_credentials()
            beaker_user = launcher.beaker.user_name
            storage_lines.append("[bold]S3 Access:[/bold]")
            if local_creds:
                cred_type = "temporary" if local_creds.session_token else "long-term"
                storage_lines.append(f"  Credentials: [green]found[/green] ({cred_type})")
                storage_lines.append(
                    f"  Beaker secrets: {beaker_user}_AWS_ACCESS_KEY_ID, "
                    f"{beaker_user}_AWS_SECRET_ACCESS_KEY"
                )
            else:
                storage_lines.append(
                    "  Credentials: [yellow]not found[/yellow] - "
                    "job may fail if S3 access is required"
                )

        # S3 storage settings
        if s3_bucket and s3_prefix:
            storage_lines.append("[bold]S3 Storage:[/bold]")
            storage_lines.append(f"  Bucket: {s3_bucket}")
            storage_lines.append(f"  Prefix: {s3_prefix}")
            storage_lines.append(f"  Region: {s3_region}")
            if s3_endpoint_url:
                storage_lines.append(f"  Endpoint: {s3_endpoint_url}")
            if effective_groups:
                storage_lines.append(f"  Group: {effective_groups[0]}")

        # PostgreSQL storage
        if self.store:
            storage_lines.append("[bold]PostgreSQL:[/bold]")
            storage_lines.append(
                "  Credentials from Beaker secrets: olmo_eval_PGHOST, olmo_eval_PGPORT,"
            )
            storage_lines.append("    olmo_eval_PGDATABASE, olmo_eval_PGUSER, olmo_eval_PGPASSWORD")

        console.print(
            Panel(
                "\n".join(storage_lines),
                title="[bold]Storage Configuration[/bold]",
                border_style="green",
            )
        )
        console.print()
