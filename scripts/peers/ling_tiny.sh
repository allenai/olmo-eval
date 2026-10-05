#!/usr/bin/env bash
set -euo pipefail

# Run inside lmsysorg/sglang:dev-Ling-3.0-tiny. Keep its GPU environment intact.
phase=${1:?Expected smoke or core}
output_dir=${2:-/results}
mkdir -p "$output_dir"
server_python=$(command -v python3)
if ! command -v uv > /dev/null; then
    "$server_python" -m pip install 'uv==0.11.32'
fi
"$server_python" -m pip freeze > "$output_dir/serving-packages.txt"
nvidia-smi > "$output_dir/nvidia-smi.txt"
"$server_python" -m sglang.launch_server \
    --model-path inclusionAI/Ling-3.0-tiny \
    --revision 9a98e35fe1c9ee255f78dd64771c7ae15a799481 \
    --host 127.0.0.1 --port 30000 --tp 1 \
    --trust-remote-code --context-length 65536 \
    > "$output_dir/server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

uv sync --frozen --no-default-groups --extra clients
uv pip install --python .venv/bin/python 'transformers==5.8.1'
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
    sleep 5
done
if [[ "$ready" != true ]]; then
    tail -n 80 "$output_dir/server.log"
    exit 1
fi
uv run --no-sync python -m olmo_eval.launch.peers \
    --peer ling-tiny --phase "$phase" \
    --base-url http://127.0.0.1:30000/v1 --output-dir "$output_dir/eval"
