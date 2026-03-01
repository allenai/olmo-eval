#!/bin/bash
# Copy beaker secrets from one workspace to another
# Usage: ./copy_beaker_secrets.sh <source_workspace> <target_workspace> <secret_name> [secret_name...]

set -e

if [ $# -lt 3 ]; then
    echo "Error: You must specify at least one secret name to copy."
    echo "Usage: $0 <source_workspace> <target_workspace> <secret_name> [secret_name...]"
    exit 1
fi

SOURCE_WORKSPACE="$1"
TARGET_WORKSPACE="$2"
shift 2

echo "Copying secrets from '$SOURCE_WORKSPACE' to '$TARGET_WORKSPACE'..."

for secret_name in "$@"; do
    echo "Copying secret: $secret_name"

    # Read secret value from source workspace
    secret_value=$(beaker secret read "$secret_name" --workspace "$SOURCE_WORKSPACE")

    # Write secret to target workspace
    printf '%s' "$secret_value" | beaker secret write "$secret_name" --workspace "$TARGET_WORKSPACE"
done

echo "Done! Copied $# secrets."
