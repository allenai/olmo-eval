#!/usr/bin/env bash
set -euo pipefail

java_21_available() {
    java -version 2>&1 | grep -Eq 'version "21([.]|")'
}

if ! java_21_available; then
    echo 'Java 21 was not found; installing openjdk-21-jdk with sudo apt.'
    if ! sudo apt update || ! sudo apt install -y openjdk-21-jdk || ! java_21_available; then
        echo 'Error: Java 21 installation did not succeed. You can try another installation method.' >&2
        echo 'For example, without sudo: conda install -c conda-forge openjdk=21' >&2
        echo 'Make sure Java 21 is on PATH, then rerun this script.' >&2
        exit 1
    fi
else
    echo 'Java 21 found.'
fi

retrieval_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
artifact_dir="${BROWSECOMP_PLUS_HOME:-$HOME/.cache/olmo-eval/browsecomp-plus}"
if [[ -n "${BROWSECOMP_PLUS_HOME:-}" ]]; then
    echo "Using BROWSECOMP_PLUS_HOME: $artifact_dir"
else
    echo "BROWSECOMP_PLUS_HOME is not set; using the default directory: $artifact_dir"
fi
mkdir -p -- "$artifact_dir"
export UV_PROJECT_ENVIRONMENT="$(cd -- "$artifact_dir" && pwd)/retrieval-venv"

echo "Creating the Python 3.10 retrieval environment at $UV_PROJECT_ENVIRONMENT"
uv venv --no-project --python 3.10 --allow-existing "$UV_PROJECT_ENVIRONMENT"
echo "Installing the locked retrieval dependencies from $retrieval_dir/upstream"
uv sync --project "$retrieval_dir/upstream" --frozen
echo 'Building and installing flash-attn (needs a CUDA toolchain)'
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-build-isolation flash-attn
echo "Retrieval environment ready at $UV_PROJECT_ENVIRONMENT"
