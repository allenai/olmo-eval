#!/bin/bash
# Copy Beaker secrets from one workspace to another, filtered by prefixes
#
# Usage:
#   ./copy-beaker-secrets.sh <source-workspace> <dest-workspace> <prefix1> [prefix2] ...
#
# Example:
#   ./copy-beaker-secrets.sh ai2/source-ws ai2/dest-ws tylerm_ alice_

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <source-workspace> <dest-workspace> <prefix1> [prefix2] ..."
    echo ""
    echo "Example:"
    echo "  $0 ai2/source-ws ai2/dest-ws tylerm_ alice_"
    exit 1
fi

SOURCE_WS="$1"
DEST_WS="$2"
shift 2
PREFIXES=("$@")

echo "Source workspace: $SOURCE_WS"
echo "Dest workspace:   $DEST_WS"
echo "Prefixes:         ${PREFIXES[*]}"
echo ""

# List secrets from source workspace using JSON format
secrets=$(beaker secret list -w "$SOURCE_WS" --format json | jq -r '.[].name')

if [[ -z "$secrets" ]]; then
    echo "No secrets found in $SOURCE_WS"
    exit 0
fi

# Filter secrets by prefixes
filtered_secrets=()
while IFS= read -r secret; do
    [[ -z "$secret" ]] && continue
    for prefix in "${PREFIXES[@]}"; do
        if [[ "$secret" == "$prefix"* ]]; then
            filtered_secrets+=("$secret")
            break
        fi
    done
done <<< "$secrets"

if [[ ${#filtered_secrets[@]} -eq 0 ]]; then
    echo "No secrets matching prefixes: ${PREFIXES[*]}"
    exit 0
fi

echo "Found ${#filtered_secrets[@]} secrets to copy:"
printf "  - %s\n" "${filtered_secrets[@]}"
echo ""

read -p "Proceed? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Copy each secret
for secret in "${filtered_secrets[@]}"; do
    echo -n "Copying $secret... "
    value=$(beaker secret read "$secret" -w "$SOURCE_WS")
    beaker secret write "$secret" "$value" -w "$DEST_WS"
    echo "done"
done

echo ""
echo "Copied ${#filtered_secrets[@]} secrets to $DEST_WS"
