#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
git rev-parse HEAD >/dev/null

# Run inside lmsysorg/sglang:dev-Ling-3.0-tiny. Keep its GPU environment intact.
phase=${1:?Expected smoke, core, or knowledge}
output_dir=${2:-/results}
mkdir -p "$output_dir"
server_python=$(command -v python3)
if ! command -v uv > /dev/null; then
    "$server_python" -m pip install 'uv==0.11.32'
fi
"$server_python" -m pip freeze > "$output_dir/serving-packages.txt"
nvidia-smi > "$output_dir/nvidia-smi.txt"
uv sync --frozen --no-default-groups --extra clients
uv pip freeze --python .venv/bin/python > "$output_dir/eval-packages.txt"
read -r model_path model_revision <<< "$(uv run --no-sync python -c \
    'from olmo_eval.launch.peers import PEERS; p = PEERS["ling-tiny"]; print(p.model, p.revision)')"
"$server_python" -m sglang.launch_server \
    --model-path "$model_path" \
    --revision "$model_revision" \
    --host 127.0.0.1 --port 30000 --tp 1 \
    --trust-remote-code --context-length 65536 \
    > "$output_dir/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

ready=false
for ((i=0; i<240; i++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -n 80 "$output_dir/server.log"
        exit 1
    fi
    if curl --fail --silent http://127.0.0.1:30000/health > /dev/null; then
        ready=true
        break
    fi
    if (( i % 12 == 0 )); then
        tail -n 10 "$output_dir/server.log"
    fi
    sleep 5
done
if [[ "$ready" != true ]]; then
    tail -n 80 "$output_dir/server.log"
    exit 1
fi
uv run --no-sync python -m olmo_eval.launch.peers \
    --peer ling-tiny --phase "$phase" \
    --base-url http://127.0.0.1:30000/v1 --output-dir "$output_dir/eval"
