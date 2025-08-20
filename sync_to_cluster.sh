#!/bin/bash

# Default cluster
CLUSTER="${1:-grenoble}"

SRC=~/projects/spartDM/
DEST="bdang@${CLUSTER}.g5k:~/project/spartDM"

echo "Syncing to: $DEST"

# List of exclude patterns
EXCLUDES=(
    "/data/PPMI/"
    ".venv/"
    "/archive/"
    "/_outputs/"
)

# Build exclude options for rsync
EXCLUDE_ARGS=()
for pattern in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=(--exclude "$pattern")
done

# Run rsync
rsync -avzPL --delete "${EXCLUDE_ARGS[@]}" "$SRC" "$DEST"
