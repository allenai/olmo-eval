"""Task inspection and debugging commands."""

# Suppress HuggingFace logging before any imports that might trigger it
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")

import json

import click

from olmo_eval.cli.utils import console


@click.group()
def task() -> None:
    """Task inspection and debugging commands."""
    pass


@task.command()
@click.argument("task_spec")
@click.option("--count", "-n", default=1, type=int, help="Number of instances to display")
@click.option("--skip", "-s", default=0, type=int, help="Number of instances to skip")
@click.option("--instance", is_flag=True, help="Show instance details")
@click.option("--request", is_flag=True, help="Show the LM request")
@click.option("--tokenizer", "-T", help="Tokenizer to use for formatting/tokenization")
@click.option(
    "--formatted", is_flag=True, help="Show prompt after template applied (requires --tokenizer)"
)
@click.option("--tokens", is_flag=True, help="Show token array (requires --tokenizer)")
@click.option("--max-tokens", default=0, type=int, help="Max tokens to display (0 for no limit)")
@click.option(
    "--max-chars", default=0, type=int, help="Max chars for formatted prompt (0 for no limit)"
)
@click.option(
    "--max-string-length",
    default=0,
    type=int,
    help="Max chars for instance field values (0 for no limit, default: no limit)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON instead of rich display")
def inspect(
    task_spec: str,
    count: int,
    skip: int,
    instance: bool,
    request: bool,
    tokenizer: str | None,
    formatted: bool,
    tokens: bool,
    max_tokens: int,
    max_chars: int,
    max_string_length: int,
    as_json: bool,
) -> None:
    """Inspect instances from a task without running evaluation.

    TASK_SPEC is the task name with optional variants (e.g., arc_easy, mmlu:olmes).

    Examples:
        olmo-eval task inspect arc_easy
        olmo-eval task inspect arc_easy -n 3
        olmo-eval task inspect mmlu:olmes --request
        olmo-eval task inspect arc_easy --json
        olmo-eval task inspect humaneval -T meta-llama/Llama-3.1-8B-Instruct --formatted
        olmo-eval task inspect humaneval -T meta-llama/Llama-3.1-8B-Instruct --tokens
    """
    from olmo_eval.core.inspection import (
        format_with_chat_template,
        formatted_request_to_dict,
        inspect_formatted_request,
        inspect_instance,
        inspect_request,
        inspect_tokens,
        instance_to_dict,
        load_tokenizer,
        tokenize_request,
    )
    from olmo_eval.evals.tasks import get_task, task_exists
    from olmo_eval.evals.tasks.core.registry import get_base_task_name

    # Validate tokenizer is provided when needed
    if (formatted or tokens) and not tokenizer:
        console.print("[red]Error:[/red] --tokenizer/-T is required with --formatted or --tokens")
        raise SystemExit(1)

    # Default to showing instance if no display flags specified
    show_instance = instance or not (request or formatted or tokens)

    # Validate task exists
    base_name = get_base_task_name(task_spec)
    if not task_exists(base_name):
        console.print(f"[red]Error:[/red] Task '{base_name}' not found")
        console.print("\n[dim]Use 'olmo-eval tasks' to list available tasks[/dim]")
        raise SystemExit(1)

    # Suppress datasets logging programmatically (for cached dataset messages)
    try:
        import datasets.utils.logging as datasets_logging

        datasets_logging.set_verbosity_error()
    except ImportError:
        pass

    try:
        task_obj = get_task(task_spec)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    # Load tokenizer if needed
    tokenizer_obj = None
    if tokenizer:
        try:
            console.print(f"[dim]Loading tokenizer: {tokenizer}[/dim]")
            tokenizer_obj = load_tokenizer(tokenizer)
        except Exception as e:
            console.print(f"[red]Error loading tokenizer:[/red] {e}")
            raise SystemExit(1) from None

    # Get instances
    instances = list(task_obj.instances)

    if not instances:
        console.print(f"[yellow]Task '{task_spec}' has no instances[/yellow]")
        return

    if skip >= len(instances):
        console.print(
            f"[yellow]Skip value ({skip}) exceeds number of instances ({len(instances)})[/yellow]"
        )
        return

    # Slice instances
    end_idx = min(skip + count, len(instances))
    selected_instances = instances[skip:end_idx]

    if as_json:
        # JSON output mode
        output = []
        for i, inst in enumerate(selected_instances):
            instance_dict = instance_to_dict(inst)
            instance_dict["_index"] = skip + i
            if request or formatted or tokens:
                req = task_obj.format_request(inst)
                if request:
                    request_dict = {
                        "type": req.request_type.name,
                        "messages": list(req.messages) if req.messages else None,
                        "prompt": req.prompt if req.prompt else None,
                        "continuations": (list(req.continuations) if req.continuations else None),
                        "system_prompt": req.system_prompt if req.system_prompt else None,
                        "tools": (
                            [
                                {
                                    "name": t.name,
                                    "description": t.description,
                                    "parameters": t.parameters,
                                }
                                for t in req.tools
                            ]
                            if req.tools
                            else None
                        ),
                    }
                    instance_dict["_request"] = {
                        k: v for k, v in request_dict.items() if v is not None
                    }
                if tokenizer_obj and (formatted or tokens):
                    try:
                        formatted_prompt = format_with_chat_template(req, tokenizer_obj)
                        token_ids = tokenize_request(req, tokenizer_obj)
                        token_data = formatted_request_to_dict(
                            formatted_prompt, token_ids, tokenizer_obj
                        )
                        instance_dict.update(token_data)
                    except Exception as e:
                        instance_dict["_tokenization_error"] = str(e)
            output.append(instance_dict)

        print(json.dumps(output, indent=2, default=str))
    else:
        # Rich display mode
        console.print(f"\n[bold]Task:[/bold] {task_spec}")
        console.print(f"[dim]Showing instances {skip + 1}-{end_idx} of {len(instances)}[/dim]\n")

        for i, inst in enumerate(selected_instances):
            if show_instance:
                inspect_instance(
                    inst,
                    console=console,
                    task_name=task_spec,
                    index=skip + i,
                    max_string_length=max_string_length if max_string_length > 0 else 10000,
                )

            if request or formatted or tokens:
                req = task_obj.format_request(inst)

                if request:
                    inspect_request(
                        req,
                        console=console,
                        title=f"[bold]Request #{skip + i + 1}[/bold]",
                    )

                if tokenizer_obj:
                    if formatted:
                        try:
                            formatted_prompt = format_with_chat_template(req, tokenizer_obj)
                            inspect_formatted_request(
                                formatted_prompt,
                                console=console,
                                title=f"[bold]Formatted Prompt[/bold] ({task_spec})",
                                max_chars=max_chars,
                            )
                        except Exception as e:
                            console.print(f"[red]Error formatting request:[/red] {e}")

                    if tokens:
                        try:
                            token_ids = tokenize_request(req, tokenizer_obj)
                            inspect_tokens(
                                token_ids,
                                tokenizer_obj,
                                console=console,
                                title=f"[bold]Token IDs[/bold] ({task_spec})",
                                max_tokens=max_tokens,
                            )
                        except Exception as e:
                            console.print(f"[red]Error tokenizing request:[/red] {e}")

            console.print()
