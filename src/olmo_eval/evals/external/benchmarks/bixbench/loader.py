"""Loader for BixBench-Verified-50 questions and capsule data from Hugging Face."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_REPO = "phylobio/BixBench-Verified-50"
DATASET_REVISION = "c77fc8ea4611b546947b8a4710ed36ab72ee233c"
QUESTIONS_FILE = "BixBench-Verified-50.jsonl"

EVAL_MODES = ("llm_verifier", "str_verifier", "range_verifier")


@dataclass(frozen=True)
class BixBenchQuestion:
    """One open-answer question and the capsule whose data it is asked about."""

    question_id: str
    question: str
    ideal: str
    eval_mode: str
    capsule_uuid: str
    #: Zip file name of the capsule, e.g. ``CapsuleFolder-<uuid>.zip``.
    data_folder: str
    categories: str = ""

    @classmethod
    def from_row(cls, row: dict) -> BixBenchQuestion:
        if row["eval_mode"] not in EVAL_MODES:
            raise ValueError(f"{row['question_id']}: unknown eval_mode {row['eval_mode']!r}")
        return cls(
            question_id=row["question_id"],
            question=row["question"],
            ideal=str(row["ideal"]),
            eval_mode=row["eval_mode"],
            capsule_uuid=row["capsule_uuid"],
            data_folder=row["data_folder"],
            categories=str(row.get("categories", "")),
        )


def load_questions(path: str | Path) -> list[BixBenchQuestion]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [BixBenchQuestion.from_row(row) for row in rows]


def select_questions(
    questions: list[BixBenchQuestion], question_ids: list[str] | None
) -> list[BixBenchQuestion]:
    """Keep the named questions, in dataset order.

    Raises:
        ValueError: If a requested question is not in the dataset.
    """
    if not question_ids:
        return list(questions)
    known = {q.question_id for q in questions}
    missing = [q for q in question_ids if q not in known]
    if missing:
        raise ValueError(f"Unknown BixBench-Verified-50 questions: {', '.join(missing)}")
    return [q for q in questions if q.question_id in question_ids]


def download_questions(revision: str = DATASET_REVISION) -> Path:
    """Download the questions file. The dataset is gated: accept its terms on the Hub first."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(DATASET_REPO, QUESTIONS_FILE, repo_type="dataset", revision=revision)
    )


def capsule_cache_dir(revision: str = DATASET_REVISION) -> Path:
    """Where extracted capsules are kept, keyed by revision so a new pin re-extracts."""
    cache = os.environ.get("HF_DATASETS_CACHE")
    if cache is None:
        hf_home = os.environ.get("HF_HOME") or os.path.join(
            os.environ.get("XDG_CACHE_HOME", "~/.cache"), "huggingface"
        )
        cache = os.path.join(hf_home, "datasets")
    name = f"{DATASET_REPO.replace('/', '--')}-{revision[:12]}"
    return Path(cache).expanduser() / name / "capsules"


def prepare_capsule(data_folder: str, revision: str = DATASET_REVISION) -> Path:
    """Download and extract one capsule's data, once per revision, and return its directory."""
    target = capsule_cache_dir(revision) / data_folder.removesuffix(".zip")
    if target.is_dir():
        return target
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(DATASET_REPO, data_folder, repo_type="dataset", revision=revision)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Extract beside the target and rename, so concurrent runs never see half a capsule.
    staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=target.parent))
    try:
        extract_capsule(Path(zip_path), staging)
        try:
            staging.rename(target)
        except OSError:
            if not target.is_dir():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    logger.info(f"Extracted {data_folder} to {target}")
    return target


def extract_capsule(zip_path: Path, extract_dir: Path) -> None:
    """Unpack a capsule zip the way upstream's ``_extract_and_process_files`` does.

    The contents of the ``CapsuleData-*`` folder move to the top level, and the
    ``CapsuleNotebook-*`` folder, which holds the reference analysis, and any
    top-level notebook are removed, so the agent sees only the input data.

    Raises:
        ValueError: If the zip holds no data folder.
    """
    shutil.unpack_archive(zip_path, extract_dir, format="zip")
    data_folder = next(
        (p for p in sorted(extract_dir.rglob("*")) if p.is_dir() and "Data" in p.name), None
    )
    if data_folder is None:
        raise ValueError(f"{zip_path.name} has no data folder")
    for item in list(data_folder.iterdir()):
        shutil.move(str(item), str(extract_dir / item.name))
    shutil.rmtree(data_folder)
    for notebook_folder in [
        p for p in extract_dir.rglob("*") if p.is_dir() and "Notebook" in p.name
    ]:
        shutil.rmtree(notebook_folder, ignore_errors=True)
    for notebook in extract_dir.glob("*.ipynb"):
        notebook.unlink()


def stage_workdir(capsule_dir: Path, workdir: Path) -> None:
    """Copy a capsule's data into a fresh working directory for one agent run.

    Everything in it is made writable by any user: the sandbox's root can be an
    unprivileged user on this host (Podman's user namespaces), and the agent may
    write anywhere in its working directory, as root can upstream.
    """
    shutil.copytree(capsule_dir, workdir, dirs_exist_ok=True)
    for path in [workdir, *workdir.rglob("*")]:
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        path.chmod(mode | (0o777 if path.is_dir() else 0o666))
