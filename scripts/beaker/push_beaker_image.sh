#!/usr/bin/env bash
set -euo pipefail

# Push Docker image to Beaker with versioning
#
# This script uploads a Docker image to Beaker, implementing safe version
# management by archiving the previous image before replacing it.
#
# Usage:
#   ./scripts/push_beaker_image.sh                              # Use defaults
#   ./scripts/push_beaker_image.sh --source olmo-eval:latest    # Custom source
#   ./scripts/push_beaker_image.sh --workspace ai2/oe-data      # Custom workspace
#   ./scripts/push_beaker_image.sh --dry-run                    # Preview only
#
# The script will:
#   1. Upload the source image as a temporary image (-tmp)
#   2. Rename the current Beaker image with a timestamp suffix
#   3. Rename the temporary image to the final name
#
# This ensures safe rollback capability if issues are discovered.

# Defaults matching beaker.py
SOURCE_IMAGE="olmo-eval:latest"
BEAKER_IMAGE="olmo-eval-latest"
WORKSPACE="ai2/oe-data"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE_IMAGE="$2"
            shift 2
            ;;
        --beaker-image)
            BEAKER_IMAGE="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --source IMAGE        Local Docker image (default: olmo-eval:latest)"
            echo "  --beaker-image NAME   Beaker image name (default: olmo-eval-latest)"
            echo "  --workspace WS        Beaker workspace (default: ai2/oe-data)"
            echo "  --dry-run             Preview without pushing"
            echo "  --help                Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Check dependencies
if ! command -v beaker &> /dev/null; then
    echo "Error: 'beaker' CLI not found. Install with: pip install beaker-py"
    exit 1
fi

# Generate timestamp for versioning
TIMESTAMP=$(date -u +"%Y%m%d-%H%M%S")

# Full image names
FULL_IMAGE="${WORKSPACE}/${BEAKER_IMAGE}"
TMP_IMAGE="${BEAKER_IMAGE}-tmp"
ARCHIVE_IMAGE="${BEAKER_IMAGE}-${TIMESTAMP}"

echo "Pushing image to Beaker..."
echo "  Source:      ${SOURCE_IMAGE}"
echo "  Beaker:      ${FULL_IMAGE}"
echo "  Workspace:   ${WORKSPACE}"
echo "  Timestamp:   ${TIMESTAMP}"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Would execute:"
    echo "  1. beaker image create ${SOURCE_IMAGE} --name ${TMP_IMAGE} --workspace ${WORKSPACE}"
    echo "  2. beaker image rename ${FULL_IMAGE} ${ARCHIVE_IMAGE} (if exists)"
    echo "  3. beaker image rename ${WORKSPACE}/${TMP_IMAGE} ${BEAKER_IMAGE}"
    exit 0
fi

# Step 1: Upload as temporary image
echo "Step 1/3: Uploading image as temporary..."
beaker image create \
    "${SOURCE_IMAGE}" \
    --name "${TMP_IMAGE}" \
    --workspace "${WORKSPACE}"

# Step 2: Archive existing image (if it exists)
echo "Step 2/3: Archiving existing image..."
if beaker image get "${FULL_IMAGE}" &> /dev/null; then
    beaker image rename "${FULL_IMAGE}" "${ARCHIVE_IMAGE}"
    echo "  Archived to: ${WORKSPACE}/${ARCHIVE_IMAGE}"
else
    echo "  No existing image to archive"
fi

# Step 3: Rename temporary to final
echo "Step 3/3: Promoting temporary image..."
beaker image rename "${WORKSPACE}/${TMP_IMAGE}" "${BEAKER_IMAGE}"

echo ""
echo "Success! Image available at: ${FULL_IMAGE}"
echo ""
echo "To use in Beaker jobs:"
echo "  olmo-eval launch beaker --beaker-image ${FULL_IMAGE} ..."
