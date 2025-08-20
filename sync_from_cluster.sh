#!/bin/bash

CLUSTER="${1:-grenoble}"
DEST=~/projects/spartDM/

echo "Syncing folders from cluster '$CLUSTER' to local '$DEST'"

# List of remote folders to sync
REMOTE_FOLDERS=(
    "workdir"
    "oarlogs"
)

for folder in "${REMOTE_FOLDERS[@]}"; do
    SRC="bdang@${CLUSTER}.g5k:~/project/spartDM/$folder"
    echo "Syncing $folder from $SRC ..."
    rsync -avxP "$SRC" "$DEST"
done
