#!/usr/bin/env bash
# Whisper RL Post-Training Ablation Sweep
#
# Usage:
#   bash scripts/run_whisper_rl_ablation.sh                    # full sweep
#   bash scripts/run_whisper_rl_ablation.sh whisper-rl-base    # single run
#   bash scripts/run_whisper_rl_ablation.sh --eval-only        # re-evaluate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="configs/train_rl_whisper.yaml"
MAX_STEPS=1500
EVAL_SAMPLES=500
RESULTS_DIR="outputs/whisper_rl_results"

mkdir -p "$RESULTS_DIR"

# Format: NAME ALPHA_FAIRNESS MAX_STEPS
ABLATIONS=(
    "whisper-rl-base     0.0  $MAX_STEPS"
    "whisper-rl-fair     2.0  $MAX_STEPS"
    "whisper-rl-fair-high 5.0 $MAX_STEPS"
)

log() { echo "$(date '+%H:%M:%S') | $*"; }

train_run() {
    local name="$1" alpha="$2" steps="$3"
    local out_dir="outputs/checkpoints/$name"

    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  TRAINING: $name  (alpha_fairness=$alpha, steps=$steps)"
    echo "══════════════════════════════════════════════════════════════"

    # Override output dir in config
    mkdir -p "$out_dir"
    sed "s|dir:.*|dir: $out_dir|" "$CONFIG" > "$out_dir/run_config.yaml"

    log "Starting training: $name"
    python ast-asr/train_rl_whisper.py \
        --config "$out_dir/run_config.yaml" \
        --max-steps "$steps" \
        --alpha-fairness "$alpha"
    log "Training complete: $name"
}

eval_run() {
    local name="$1"
    local ckpt="outputs/checkpoints/$name/final"
    local csv_out="$RESULTS_DIR/${name}_eval.csv"
    local summary="$RESULTS_DIR/${name}_summary.txt"

    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  EVAL: $name on Svarah-eval ($EVAL_SAMPLES samples)"
    echo "──────────────────────────────────────────────────────────────"

    if [ ! -d "$ckpt" ]; then
        log "ERROR: Checkpoint not found at $ckpt — skipping eval"
        return 1
    fi

    python ast-asr/pipeline.py \
        --model whisper-small-rl \
        --model-path "$ckpt" \
        --svarah-split eval \
        --max-samples "$EVAL_SAMPLES" \
        --output "$csv_out" \
        2>&1 | tee "$summary"

    log "Eval complete: $name → $summary"
}

main() {
    local filter="${1:-}"

    log "Whisper RL Ablation Sweep"
    log "Config: $CONFIG"
    log "Max steps: $MAX_STEPS"

    if [ "$filter" = "--eval-only" ]; then
        log "Re-evaluating existing checkpoints..."
        for ablation in "${ABLATIONS[@]}"; do
            read -r name _ <<< "$ablation"
            eval_run "$name" || true
        done
        return
    fi

    for ablation in "${ABLATIONS[@]}"; do
        read -r name alpha steps <<< "$ablation"

        if [ -n "$filter" ] && [ "$filter" != "$name" ]; then
            continue
        fi

        train_run "$name" "$alpha" "$steps"
        eval_run "$name" || log "WARNING: Eval failed for $name"
    done

    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  SWEEP COMPLETE"
    echo "══════════════════════════════════════════════════════════════"
    log "Results in: $RESULTS_DIR/"
    ls -la "$RESULTS_DIR/"
}

main "${1:-}"
