#!/usr/bin/env bash
# AutoMedal — Headless Three-Phase Loop
# Each iteration dispatches up to three agent phases against file-based memory:
#   Researcher (on stagnation or scheduled cadence)
#   Strategist (on empty queue or stagnation)
#   Experimenter (always — pops the next pending queue entry)
#
# Agent: pi (https://github.com/badlogic/pi-mono). Always runs in YOLO mode.
#
# Usage:
#   bash run.sh              # 50 iterations (default)
#   bash run.sh 100          # 100 iterations
#   bash run.sh 10 fast      # 10 iterations, skip cooldown

set -euo pipefail

MAX_ITERATIONS=${1:-50}
FAST_MODE=${2:-""}

STAGNATION_K=${STAGNATION_K:-3}
RESEARCH_EVERY=${RESEARCH_EVERY:-10}
COOLDOWN_SECS=${COOLDOWN_SECS:-1}
TRAIN_BUDGET_MINUTES=${TRAIN_BUDGET_MINUTES:-10}
export TRAIN_BUDGET_MINUTES
MODEL=${MODEL:-"opencode-go/minimax-m2.7"}

# ── Path resolution via Layout env vars (set by automedal/dispatch.py) ───────
# All paths fall back to the flat-repo defaults so `bash run.sh N` from the
# repo root still works exactly as before (dev mode).
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${AUTOMEDAL_CWD:-$REPO_DIR}"

LOG_FILE="${AUTOMEDAL_LOG_FILE:-${LOG_FILE:-agent_loop.log}}"

# Resolve all other layout-aware paths (flat-repo defaults preserved)
_PROMPTS_DIR="${AUTOMEDAL_PROMPTS_DIR:-${REPO_DIR}/prompts}"
_HARNESS_DIR="${AUTOMEDAL_HARNESS_DIR:-${REPO_DIR}/harness}"
_TRAIN_PY="${AUTOMEDAL_TRAIN_PY:-${REPO_DIR}/agent/train.py}"
_PREPARE_PY="${AUTOMEDAL_PREPARE_PY:-${REPO_DIR}/agent/prepare.py}"
_LAST_TRAIN_OUT="${AUTOMEDAL_LAST_TRAINING_OUTPUT:-${REPO_DIR}/harness/.last_training_output}"
_DATA_DIR="${AUTOMEDAL_DATA_DIR:-${REPO_DIR}/data}"
_PREPARE_CHECK="${AUTOMEDAL_PREPARE_PY:-${REPO_DIR}/agent/prepare.py}"

# Pi binary: use package-internal copy if available, else fall back to system `pi`
PI="${AUTOMEDAL_PI_BIN:-pi}"

# ── Graceful cancellation via SIGTERM ─────────────────────────────────────────
CANCEL=0
trap 'echo "  [harness] SIGTERM received — finishing this iteration then stopping" | tee -a "$LOG_FILE"; CANCEL=1' TERM

echo "=================================================="
echo "  AutoMedal — Three-Phase Loop"
echo "  Model:          $MODEL"
echo "  Iterations:     $MAX_ITERATIONS"
echo "  Stagnation K:   $STAGNATION_K"
echo "  Research every: $RESEARCH_EVERY"
echo "  Cooldown:       ${COOLDOWN_SECS}s"
echo "  Train budget:   ${TRAIN_BUDGET_MINUTES}m"
echo "  Log:            $LOG_FILE"
echo "  Started:        $(date)"
echo "=================================================="

# Ensure data exists (first run after bootstrap)
if [ ! -f "${_DATA_DIR}/X_train.npy" ]; then
    echo "Preparing data first..."
    python "$_PREPARE_PY" 2>&1 | tee -a "$LOG_FILE"
fi

# Ensure memory files exist (harmless if already present)
python "${_HARNESS_DIR}/init_memory.py" 2>&1 | tee -a "$LOG_FILE"

run_agent() {
    # $1 = phase name (unused, kept for log parity), $2 = prompt file, $3 = trailing context block
    local phase="$1"
    local prompt_file="$2"
    local context="$3"

    local prompt_body
    prompt_body="$(cat "$prompt_file")

---
# Runtime context

$context"

    "$PI" --no-session \
       --model "$MODEL" \
       --mode json \
       -p "$prompt_body" \
       2>>"$LOG_FILE" | tee -a "$LOG_FILE" | python3 -u "${_HARNESS_DIR}/stream_events.py"
}

for i in $(seq 1 "$MAX_ITERATIONS"); do
    # ── Check cancellation flag at top of each iteration ────────────────────
    if [[ "$CANCEL" == "1" ]]; then
        echo "  [harness] cancel requested — stopping cleanly" | tee -a "$LOG_FILE"
        break
    fi

    EXP_ID=$(python "${_HARNESS_DIR}/next_exp_id.py")

    echo "" | tee -a "$LOG_FILE"
    echo "========== Iteration $i / $MAX_ITERATIONS  exp=$EXP_ID  [$(date '+%H:%M:%S')] ==========" | tee -a "$LOG_FILE"

    # ─── Deterministic stagnation + schedule check ────
    read STAGNATING BEST <<< "$(python "${_HARNESS_DIR}/check_stagnation.py" --k "$STAGNATION_K" --both)"
    SCHEDULED_RESEARCH=0
    if [ "$RESEARCH_EVERY" -gt 0 ] && [ $((i % RESEARCH_EVERY)) -eq 0 ]; then
        SCHEDULED_RESEARCH=1
    fi

    echo "  [harness] stagnating=$STAGNATING scheduled_research=$SCHEDULED_RESEARCH best=$BEST" | tee -a "$LOG_FILE"

    # ─── Researcher phase (optional) ────
    if [ "$STAGNATING" = "1" ] || [ "$SCHEDULED_RESEARCH" = "1" ]; then
        TRIGGER="stagnation"
        [ "$SCHEDULED_RESEARCH" = "1" ] && [ "$STAGNATING" = "0" ] && TRIGGER="scheduled"

        echo "  [harness] dispatching Researcher ($TRIGGER)" | tee -a "$LOG_FILE"
        run_agent "researcher" "${_PROMPTS_DIR}/researcher.md" "Triggering experiment: $EXP_ID
Trigger type: $TRIGGER
Stagnating: $STAGNATING
Scheduled research: $SCHEDULED_RESEARCH
Current best val_loss: $BEST" || echo "  [WARN] Researcher exited non-zero"

        python "${_HARNESS_DIR}/verify_iteration.py" --phase researcher 2>&1 | tee -a "$LOG_FILE" || true
    fi

    # ─── Strategist phase (optional) ────
    QUEUE_PENDING=$(grep -c '\[STATUS: pending\]' experiment_queue.md 2>/dev/null || echo 0)
    if [ "$QUEUE_PENDING" = "0" ] || [ "$STAGNATING" = "1" ]; then
        echo "  [harness] dispatching Strategist (queue_pending=$QUEUE_PENDING, stagnating=$STAGNATING)" | tee -a "$LOG_FILE"

        # Build reflective trace (last 3 experiments, ranked by learning value)
        REFLECTIVE_TRACE=$(python "${_HARNESS_DIR}/build_trace_trailer.py" --n 3 2>/dev/null || echo "(trace unavailable)")
        RANKED_JOURNALS=$(python "${_HARNESS_DIR}/rank_journals.py" --m 30 --k 10 2>/dev/null || echo "(ranking unavailable)")

        run_agent "strategist" "${_PROMPTS_DIR}/strategist.md" "Upcoming experiment: $EXP_ID
Current iteration: $i / $MAX_ITERATIONS
Stagnating: $STAGNATING
Current best val_loss: $BEST
Pending queue entries: $QUEUE_PENDING

## Reflective trace (last 3 experiments)
$REFLECTIVE_TRACE

## Top experiments by learning value
$RANKED_JOURNALS" || echo "  [WARN] Strategist exited non-zero"

        python "${_HARNESS_DIR}/verify_iteration.py" --phase strategist 2>&1 | tee -a "$LOG_FILE" || true
    fi

    # ─── Tag the repo before the Experimenter so the journal has a stable ref ────
    if ! git rev-parse "exp/$EXP_ID" >/dev/null 2>&1; then
        git tag "exp/$EXP_ID" HEAD 2>&1 | tee -a "$LOG_FILE" || true
    fi

    # ─── Experimenter: Edit phase (agent edits code) ────
    echo "  [harness] dispatching Experimenter (edit)" | tee -a "$LOG_FILE"
    run_agent "experimenter-edit" "${_PROMPTS_DIR}/experimenter.md" "Experiment ID: $EXP_ID
Current best val_loss: $BEST" || echo "  [WARN] Experimenter-edit exited non-zero"

    # ─── Experimenter: Training (harness-managed, no agent) ────
    TRAIN_TIMEOUT=$(( TRAIN_BUDGET_MINUTES * 60 + 30 ))
    TRAIN_OUTPUT="$_LAST_TRAIN_OUT"
    echo "  [harness] running training (budget=${TRAIN_BUDGET_MINUTES}m, timeout=${TRAIN_TIMEOUT}s)..." | tee -a "$LOG_FILE"

    # Run prepare.py if it was modified
    if ! git diff --quiet -- "$_PREPARE_CHECK" 2>/dev/null; then
        echo "  [harness] prepare.py changed — running prepare first" | tee -a "$LOG_FILE"
        python "$_PREPARE_PY" 2>&1 | tee -a "$LOG_FILE" || true
    fi

    timeout --signal=TERM "$TRAIN_TIMEOUT" python "$_TRAIN_PY" 2>&1 | tee "$TRAIN_OUTPUT" | tee -a "$LOG_FILE"
    TRAIN_EXIT=${PIPESTATUS[0]}

    # Extract result
    FINAL_LOSS=$(grep -oP 'final_val_loss=\K[0-9.]+' "$TRAIN_OUTPUT" 2>/dev/null || echo "nan")
    echo "  [harness] training done: val_loss=$FINAL_LOSS exit=$TRAIN_EXIT" | tee -a "$LOG_FILE"

    # ─── Experimenter: Eval phase (agent writes journal, commits/reverts) ────
    echo "  [harness] dispatching Experimenter (eval)" | tee -a "$LOG_FILE"
    run_agent "experimenter-eval" "${_PROMPTS_DIR}/experimenter_eval.md" "Experiment ID: $EXP_ID
Current best val_loss: $BEST
Training exit code: $TRAIN_EXIT
Training val_loss: $FINAL_LOSS" || echo "  [WARN] Experimenter-eval exited non-zero"

    # Reread best_so_far after eval (the eval agent may have committed an improvement)
    read _IGNORE BEST_AFTER <<< "$(python "${_HARNESS_DIR}/check_stagnation.py" --k "$STAGNATION_K" --both 2>/dev/null || echo "0 nan")"

    # Regression gate + success_criteria check (uses new verify flags)
    VERIFY_EXIT=0
    if [[ "$FINAL_LOSS" != "nan" ]]; then
        python "${_HARNESS_DIR}/verify_iteration.py" \
            --phase experimenter --exp-id "$EXP_ID" \
            --val-loss "$FINAL_LOSS" \
            --best-before "$BEST" \
            --best-so-far "${BEST_AFTER:-$BEST}" \
            2>&1 | tee -a "$LOG_FILE"
        VERIFY_EXIT=${PIPESTATUS[0]}
    else
        python "${_HARNESS_DIR}/verify_iteration.py" --phase experimenter --exp-id "$EXP_ID" \
            2>&1 | tee -a "$LOG_FILE" || true
    fi

    # Handle regression gate in strict mode (exit code 2 → already logged by verify)
    if [ "$VERIFY_EXIT" = "2" ] && [ "${AUTOMEDAL_REGRESSION_GATE:-warn}" = "strict" ]; then
        echo "  [harness] strict regression gate triggered — reverting git tag exp/$EXP_ID" | tee -a "$LOG_FILE"
        git tag -d "exp/$EXP_ID" 2>/dev/null || true
    fi

    # Handle success_criteria near-miss (exit code 3 → one retry allowed)
    if [ "$VERIFY_EXIT" = "3" ]; then
        echo "  [harness] success_criteria near-miss — attempting one retry edit" | tee -a "$LOG_FILE"
        run_agent "experimenter-retry" "${_PROMPTS_DIR}/experimenter.md" "Experiment ID: $EXP_ID
Current best val_loss: $BEST
RETRY: Previous attempt val_loss=$FINAL_LOSS missed success_criteria by ≤1%.
Make a small targeted improvement and re-run training." || echo "  [WARN] retry exited non-zero"

        # Re-train
        timeout --signal=TERM "$TRAIN_TIMEOUT" python "$_TRAIN_PY" 2>&1 | tee "$TRAIN_OUTPUT" | tee -a "$LOG_FILE"
        RETRY_LOSS=$(grep -oP 'final_val_loss=\K[0-9.]+' "$TRAIN_OUTPUT" 2>/dev/null || echo "nan")
        echo "  [harness] retry training done: val_loss=$RETRY_LOSS" | tee -a "$LOG_FILE"

        run_agent "experimenter-eval-retry" "${_PROMPTS_DIR}/experimenter_eval.md" "Experiment ID: $EXP_ID
Current best val_loss: $BEST
Training exit code: 0
Training val_loss: $RETRY_LOSS
NOTE: This is a retry after a near-miss. Evaluate the retry result normally." || echo "  [WARN] retry eval exited non-zero"

        python "${_HARNESS_DIR}/verify_iteration.py" --phase experimenter --exp-id "$EXP_ID" \
            2>&1 | tee -a "$LOG_FILE" || true
    fi

    echo "--- Iteration $i complete  exp=$EXP_ID  [$(date '+%H:%M:%S')] ---" | tee -a "$LOG_FILE"

    if [ -z "$FAST_MODE" ] && [ "$i" -lt "$MAX_ITERATIONS" ] && [ "$COOLDOWN_SECS" -gt 0 ]; then
        sleep "$COOLDOWN_SECS"
    fi
done

echo ""
echo "=================================================="
echo "  AutoMedal complete — $MAX_ITERATIONS iterations"
echo "  Finished: $(date)"
echo "  Results:  cat "${AUTOMEDAL_RESULTS_TSV:-agent/results.tsv}""
echo "  Memory:   knowledge.md / experiment_queue.md / journal/"
echo "=================================================="
