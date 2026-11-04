"""Offline text correctness rescoring command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from olmo_eval.runners.asynq.preparation import configure_task
from olmo_eval.runners.rescore import read_jsonl, rescore_predictions, source_identity


@click.command()
@click.option("--task", "task_spec", required=True, help="Original task spec, including variants.")
@click.option(
    "--predictions", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option(
    "--requests", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option(
    "--reward-final-answer-only/--no-reward-final-answer-only", default=False, show_default=True
)
@click.option(
    "--text-is-original",
    is_flag=True,
    help="Assert saved text is raw when original_text is absent.",
)
def rescore(
    task_spec: str,
    predictions: Path,
    requests: Path,
    output_dir: Path,
    reward_final_answer_only: bool,
    text_is_original: bool,
) -> None:
    """Rescore saved predictions and requests locally, without inference or dataset access."""
    try:
        task = configure_task(task_spec, {"reward_final_answer_only": reward_final_answer_only})
        result = asyncio.run(
            rescore_predictions(
                task,
                read_jsonl(predictions),
                read_jsonl(requests),
                text_is_original=text_is_original,
            )
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
    result["sources"] = {
        "predictions": source_identity(predictions),
        "requests": source_identity(requests),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{task_spec.replace(':', '_')}_{result['task_hash']}"
    prediction_path = output_dir / f"{stem}-predictions.jsonl"
    result_path = output_dir / f"{stem}-results.json"
    if prediction_path.exists() or result_path.exists():
        raise click.ClickException(
            f"Rescored artifacts already exist for {stem}; choose a fresh output directory"
        )
    rows = result.pop("predictions")
    with prediction_path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    click.echo(
        json.dumps(
            {
                "num_instances": result["num_instances"],
                "metrics": result["metrics"],
                "task_hash": result["task_hash"],
            }
        )
    )
