"""The agent's Jupyter notebook: editing, rendering and listing the working directory.

Ports the parts of upstream's notebook environment that run on the host:
``NBEnvironment.edit_cell`` and ``_list_dir`` from ``src/fhda/notebook_env.py``, and
``view_notebook`` and ``limit_notebook_output`` from ``src/fhda/utils.py``, of
Future-House/data-analysis-crow at v1.5.0. The notebook is nbformat v4 JSON, edited
on the host and executed in the sandbox by ``jupyter nbconvert``, as upstream's
Docker mode does.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NOTEBOOK_NAME = "notebook.ipynb"

#: Upstream's ``NB_OUTPUT_LIMIT``: characters of one output the agent sees.
NB_OUTPUT_LIMIT = 3000

IMAGE_OUTPUT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
TABLE_OUTPUT_TYPES_TO_IGNORE = {"text/latex", "text/html", "text/markdown"}

#: The kernelspec upstream gives a new Python notebook.
PYTHON_KERNELSPEC = {"name": "python", "display_name": "Python 3 (ipykernel)", "language": "python"}


def new_notebook() -> dict[str, Any]:
    return {
        "cells": [],
        "metadata": {"kernelspec": dict(PYTHON_KERNELSPEC)},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def new_code_cell(source: str, index: int) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        # nbformat 4.5 requires cell ids; upstream's nbformat generates random ones.
        "id": f"cell-{index}",
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def read_notebook(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_notebook(notebook: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(notebook, indent=1))
    # nbconvert rewrites the notebook from inside the sandbox, whose root can be an
    # unprivileged user here (Podman's user namespaces).
    path.chmod(0o666)


def edit_cell(notebook: dict[str, Any], contents: str, idx: Any = None) -> str:
    """Append a code cell, or replace the source of an existing one, as upstream does.

    An ``idx`` that is not an integer, or past the last cell, appends. The edited
    cell keeps its old outputs until the notebook runs again.

    Returns:
        Upstream's tool reply for the edit.
    """
    if idx is not None:
        try:
            idx = int(idx)
        except (ValueError, TypeError):
            idx = None
    cells = notebook["cells"]
    if idx is None or idx >= len(cells):
        cells.append(new_code_cell(contents, len(cells)))
        return f"Appended new cell (#{len(cells) - 1})."
    cells[idx]["source"] = contents
    return f"Edited cell #{idx}."


def _text(value: str | list[str]) -> str:
    """Join a multiline field, which nbformat stores on disk as a list of lines."""
    return "".join(value) if isinstance(value, list) else value


def limit_notebook_output(output: str) -> str:
    """Keep the first and last halves of an output longer than the limit."""
    if len(output) < NB_OUTPUT_LIMIT:
        return output
    cutoff = NB_OUTPUT_LIMIT // 2
    return output[:cutoff] + "\n<...output limited...>\n" + output[-cutoff:]


def _process_output(
    output: dict[str, Any], md: list[str], images: list[str], streams: list[str]
) -> None:
    output_type = output.get("output_type")
    if output_type == "stream":
        streams.append(_text(output.get("text", "")))
    elif output_type == "execute_result":
        md.append(limit_notebook_output(_text(output.get("data", {}).get("text/plain", ""))))
    elif output_type == "error":
        md.append(limit_notebook_output("\n".join(output.get("traceback", []))))
    elif output_type == "display_data":
        data = output.get("data", {})
        data_type = next(iter(data.keys()), "")
        if data_type in TABLE_OUTPUT_TYPES_TO_IGNORE:
            return
        if data_type in IMAGE_OUTPUT_TYPES:
            md.append(f"<{len(images) + 1}>")
            images.append(f"data:image/{data_type.split('/')[-1]};base64,{_text(data[data_type])}")
        elif data_type:
            md.append(limit_notebook_output(_text(data[data_type])))


def view_notebook(notebook: dict[str, Any], language: str = "python") -> tuple[str, list[str]]:
    """Render the notebook as markdown, the way upstream shows it to the agent.

    Returns:
        The markdown, and the notebook's images as data URLs. The markdown marks
        each image's place with its 1-based number in angle brackets.
    """
    md: list[str] = []
    images: list[str] = []
    for idx, cell in enumerate(notebook.get("cells", [])):
        md.append(f"### Cell {idx}:")
        if cell.get("cell_type") == "code":
            md.extend((f"```{language}", _text(cell.get("source", "")), "```"))
            outputs = cell.get("outputs", [])
            if outputs:
                md.extend([f"### Output {idx}:", "```"])
                streams: list[str] = []
                for output in outputs:
                    _process_output(output, md, images, streams)
                if streams:
                    md.append(limit_notebook_output("\n".join(streams)))
                md.append("```")
        elif cell.get("cell_type") in {"markdown", "raw"}:
            md.append(_text(cell.get("source", "")))
    return "\n".join(md), images


def list_dir(path: Path) -> dict[str, Any]:
    """The directory tree as upstream's ``list_workdir`` reports it.

    Upstream adds an empty ``directories`` key and lists each subdirectory under
    its own name beside it; the shape is kept as it is. Entries are sorted, where
    upstream takes the file system's order, so a listing is the same on every host.
    """
    index: dict[str, Any] = {}
    for item in sorted(path.iterdir()):
        if item.is_dir():
            index.setdefault("directories", {})
            index[item.name] = list_dir(item)
        else:
            index.setdefault("files", []).append(item.name)
    return index
