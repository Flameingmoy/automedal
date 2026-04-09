#!/bin/bash
# AutoResearch Kaggle — Headless automation loop
# Runs OpenCode + GLM-5.1 repeatedly, each iteration does one experiment cycle.
#
# Usage:
#   bash run.sh           # 50 iterations (default)
#   bash run.sh 100       # 100 iterations
#   bash run.sh 10 fast   # 10 iterations, skip cooldown
set -euo pipefail

MAX_ITERATIONS=${1:-50}
FAST_MODE=${2:-""}
LOG_FILE="agent_loop.log"
MODEL="opencode-go/glm-5.1"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================="
echo "  AutoResearch Kaggle — Headless Loop"
echo "  Model:      $MODEL"
echo "  Iterations: $MAX_ITERATIONS"
echo "  Log:        $LOG_FILE"
echo "  Started:    $(date)"
echo "=================================================="

# Ensure data is prepared
if [ ! -f "$REPO_DIR/data/X_train.npy" ]; then
    echo "Preparing data first..."
    cd "$REPO_DIR" && python prepare.py 2>&1 | tee -a "$LOG_FILE"
fi

for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo "" | tee -a "$LOG_FILE"
    echo "========== Iteration $i / $MAX_ITERATIONS  [$(date '+%H:%M:%S')] ==========" | tee -a "$LOG_FILE"

    # Run opencode with program.md as the prompt
    cd "$REPO_DIR" && opencode run \
        -m "$MODEL" \
        --dangerously-skip-permissions \
        --title "autoresearch-iter-$i" \
        "Read program.md and execute one full experiment cycle. This is iteration $i of $MAX_ITERATIONS." \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "  [WARN] Iteration $i exited with code $EXIT_CODE" | tee -a "$LOG_FILE"
    fi

    echo "--- Iteration $i complete [$(date '+%H:%M:%S')] ---" | tee -a "$LOG_FILE"

    # Brief cooldown (skip in fast mode) to avoid API rate limits
    if [ -z "$FAST_MODE" ] && [ "$i" -lt "$MAX_ITERATIONS" ]; then
        sleep 5
    fi
done

echo ""
echo "=================================================="
echo "  AutoResearch complete — $MAX_ITERATIONS iterations"
echo "  Finished: $(date)"
echo "  Results:  cat results.tsv"
echo "=================================================="
