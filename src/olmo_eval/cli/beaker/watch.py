"""Watch command for monitoring Beaker experiments."""

import click

from olmo_eval.cli.utils import console


@click.command(name="watch")
@click.option(
    "--experiment",
    "-e",
    required=True,
    help="Beaker experiment ID to watch",
)
@click.option(
    "--tail",
    "-t",
    is_flag=True,
    help="Only show recent logs (last 10 seconds). Useful for attaching to running experiments.",
)
def watch(experiment: str, tail: bool) -> None:
    """Watch an experiment's logs in real-time."""
    import sys

    try:
        from olmo_eval.launch import BeakerLauncher
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\nInstall with: pip install 'olmo-eval[beaker]'"
        )
        raise SystemExit(1) from None

    launcher = BeakerLauncher()

    try:
        exit_code = launcher.follow_experiment(experiment, tail=tail)
        sys.exit(exit_code)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None
