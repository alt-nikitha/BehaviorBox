#!/bin/bash

set -e

CHECKPOINTS_FILE="/home/nsrikant/BehaviorBoxNew/checkpoints_info/olmo3_7b_checkpoints.txt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$CHECKPOINTS_FILE" ]; then
    echo "Error: Checkpoints file not found: $CHECKPOINTS_FILE"
    exit 1
fi

while IFS= read -r checkpoint; do
    checkpoint=$(echo "$checkpoint" | xargs)  # trim whitespace
    [ -z "$checkpoint" ] && continue

    echo ""
    echo "############################################"
    echo "# Running checkpoint: $checkpoint"
    echo "############################################"
    echo ""

    MODEL_REVISION="$checkpoint" bash "$SCRIPT_DIR/run_single_task.sh"

    echo ""
    echo "############################################"
    echo "# Finished checkpoint: $checkpoint"
    echo "############################################"
    echo ""
done < "$CHECKPOINTS_FILE"

echo "All checkpoints completed."
