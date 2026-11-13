"""BrowseComp-Plus installation and artifact preparation."""

import json
import os
import subprocess
from unittest.mock import Mock, call

import pytest

from olmo_eval.evals.tasks.browsecomp_plus import prepare
from olmo_eval.evals.tasks.browsecomp_plus.data import (
    DATASET_FILENAME,
    DATASET_REPO,
    INDEX_REPO,
)
from olmo_eval.evals.tasks.browsecomp_plus.retrieval import upstream_directory


def test_prepare_downloads_questions_and_both_indexes_without_git(tmp_path, monkeypatch):
    rows = [
        {
            "query_id": str(query_id),
            "query": "d0AVkObqe8BhQoI=",
            "answer": "cEkOmv0=",
            "evidence_docs": [
                {
                    "docid": "EBhN",
                    "text": "cEkOmv0=",
                    "url": "SFwIg/3wN4ZwQ9z4qt1z+7hA2oY6754kYPkvtw==",
                },
            ],
        }
        for query_id in range(830)
    ]
    load = Mock(return_value=rows)
    monkeypatch.setattr("datasets.load_dataset", load)

    def download_index(repo_id, *, repo_type, allow_patterns, local_dir):
        retriever = allow_patterns[0].split("/")[0]
        filename = "segments_1" if retriever == "bm25" else "corpus.shard0.pkl"
        path = local_dir / retriever / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(local_dir)

    download = Mock(side_effect=download_index)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    no_subprocess = Mock(side_effect=AssertionError("Preparation must not invoke Git"))
    monkeypatch.setattr(subprocess, "run", no_subprocess)
    monkeypatch.setattr(subprocess, "check_output", no_subprocess)
    monkeypatch.setattr("sys.argv", ["prepare", "--home", str(tmp_path)])

    prepare.main()
    records = [json.loads(line) for line in (tmp_path / DATASET_FILENAME).read_text().splitlines()]
    assert len(records) == 830
    assert records[0]["query"] == "Which city?"
    assert records[0]["answer"] == "Paris"
    assert records[0]["evidence_docs"] == [
        {"docid": "001", "text": "Paris", "url": "https://example.org/evidence"},
    ]
    assert "evidence_docids" not in records[0]
    load.assert_called_once_with(DATASET_REPO, split="test")
    assert download.call_args_list == [
        call(
            INDEX_REPO,
            repo_type="dataset",
            allow_patterns=[f"{retriever}/*"],
            local_dir=tmp_path / "indexes",
        )
        for retriever in ["bm25", "qwen3-embedding-8b"]
    ]
    assert (tmp_path / "indexes/bm25/segments_1").exists()
    assert (tmp_path / "indexes/qwen3-embedding-8b/corpus.shard0.pkl").exists()
    assert not (tmp_path / "reference").exists()
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "huggingface").exists()
    prepare.main()
    assert load.call_count == 1
    no_subprocess.assert_not_called()


@pytest.fixture
def installer_environment(tmp_path, monkeypatch):
    executables = tmp_path / "bin"
    executables.mkdir()
    scripts = {
        "java": (
            '#!/bin/sh\nif [ -f "$JAVA_READY" ]; then\n'
            "echo 'openjdk version \"21.0.1\"' >&2\n"
            "else echo 'openjdk version \"17.0.1\"' >&2; fi\n"
        ),
        "sudo": (
            '#!/bin/sh\nprintf "%s\\n" "$@" >> "$INSTALL_CALLS"\n'
            'if [ "${JAVA_INSTALL_FAILURE:-}" = sudo ]; then exit 127; fi\n'
            'if [ "${JAVA_INSTALL_FAILURE:-}" = "$2" ]; then exit 1; fi\n'
            'if [ "$2" = install ] && [ "${JAVA_INSTALL_FAILURE:-}" != missing_java ]; then\n'
            'touch "$JAVA_READY"\nfi\n'
        ),
        "git": "#!/bin/sh\necho 'Unexpected Git call' >&2\nexit 1\n",
        "uv": (
            '#!/bin/sh\nprintf "%s\\n" "$@" >> "$INSTALL_CALLS"\n'
            'printf "environment=%s\\n" "${UV_PROJECT_ENVIRONMENT:-}" >> "$INSTALL_CALLS"\n'
        ),
    }
    for name, script in scripts.items():
        path = executables / name
        path.write_text(script)
        path.chmod(0o755)
    log = tmp_path / "calls.txt"
    monkeypatch.setenv("PATH", f"{executables}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("INSTALL_CALLS", str(log))
    monkeypatch.setenv("JAVA_READY", str(tmp_path / "java-ready"))
    monkeypatch.delenv("JAVA_INSTALL_FAILURE", raising=False)
    home = tmp_path / "artifact home"
    monkeypatch.setenv("BROWSECOMP_PLUS_HOME", str(home))
    return home, log


@pytest.mark.parametrize("relative_home", [False, True])
def test_installer_uses_bundled_environment(installer_environment, monkeypatch, relative_home):
    home, log = installer_environment
    if relative_home:
        monkeypatch.chdir(home.parent)
        monkeypatch.setenv("BROWSECOMP_PLUS_HOME", home.name)
    installer = upstream_directory().parent / "install.sh"
    subprocess.run(["bash", str(installer)], check=True, capture_output=True)
    calls = log.read_text().splitlines()
    assert calls == [
        "apt",
        "update",
        "apt",
        "install",
        "-y",
        "openjdk-21-jdk",
        "venv",
        "--no-project",
        "--python",
        "3.10",
        "--allow-existing",
        str(home / "retrieval-venv"),
        f"environment={home}/retrieval-venv",
        "sync",
        "--project",
        str(upstream_directory()),
        "--frozen",
        f"environment={home}/retrieval-venv",
        "pip",
        "install",
        "--python",
        str(home / "retrieval-venv/bin/python"),
        "--no-build-isolation",
        "flash-attn",
        f"environment={home}/retrieval-venv",
    ]


@pytest.mark.parametrize("failure", ["sudo", "update", "install", "missing_java"])
def test_java_install_failure_explains_conda_alternative(
    installer_environment, monkeypatch, failure
):
    _, log = installer_environment
    monkeypatch.setenv("JAVA_INSTALL_FAILURE", failure)
    installer = upstream_directory().parent / "install.sh"
    result = subprocess.run(["bash", str(installer)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Java 21 installation did not succeed" in result.stderr
    assert "conda install -c conda-forge openjdk=21" in result.stderr
    assert "rerun this script" in result.stderr
    assert "venv" not in log.read_text().splitlines()
    assert "sync" not in log.read_text().splitlines()


def test_existing_java_21_skips_sudo(installer_environment, monkeypatch):
    _, log = installer_environment
    (log.parent / "java-ready").touch()
    monkeypatch.setenv("JAVA_INSTALL_FAILURE", "sudo")
    installer = upstream_directory().parent / "install.sh"
    subprocess.run(["bash", str(installer)], check=True, capture_output=True)
    calls = log.read_text().splitlines()
    assert "apt" not in calls
    assert "venv" in calls
    assert "sync" in calls
    assert "flash-attn" in calls
