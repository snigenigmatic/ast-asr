#!/usr/bin/env bash
# RL Post-Training Ablation Sweep
# Runs GRPO ablations varying KL penalty, fairness strength, and synthetic data.
#
# Usage:
#   bash scripts/run_rl_ablation.sh           # full sweep (6 runs)
#   bash scripts/run_rl_ablation.sh rl-base   # single run
#   bash scripts/run_rl_ablation.sh --eval-only  # re-evaluate existing checkpoints

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG_BASE="configs/train_rl.yaml"
EVAL_SAMPLES=500
MAX_STEPS=3000
RESULTS_DIR="outputs/rl_ablation_results"

mkdir -p "$RESULTS_DIR"

# ── Ablation configurations ──────────────────────────────────────────────────
# Format: NAME BETA_KL ALPHA_FAIRNESS SYNTH_ST MAX_STEPS
ABLATIONS=(
    "rl-base     0.1  0.0  no  $MAX_STEPS"
    "rl-fair     0.1  2.0  no  $MAX_STEPS"
    "rl-synth    0.1  2.0  yes $MAX_STEPS"
    "rl-kl-low   0.01 2.0  yes $MAX_STEPS"
    "rl-kl-high  0.5  2.0  yes $MAX_STEPS"
    "rl-fair-high 0.1 5.0  yes $MAX_STEPS"
)

# ── Helper functions ─────────────────────────────────────────────────────────

log() {
    echo "$(date '+%H:%M:%S') | $*"
}

train_run() {
    local name="$1" beta_kl="$2" alpha_fair="$3" synth="$4" steps="$5"
    local out_dir="outputs/checkpoints/$name"

    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  TRAINING: $name"
    echo "  beta_kl=$beta_kl  alpha_fairness=$alpha_fair  synth=$synth"
    echo "  steps=$steps  output=$out_dir"
    echo "══════════════════════════════════════════════════════════════"

    # Create per-run config by overriding parameters
    local run_config="$out_dir/run_config.yaml"
    mkdir -p "$out_dir"

    python -c "
import yaml
with open('$CONFIG_BASE') as f:
    cfg = yaml.safe_load(f)

cfg['rl']['beta_kl'] = $beta_kl
cfg['rl']['max_steps'] = $steps
cfg['reward']['alpha_fairness'] = $alpha_fair
cfg['output']['dir'] = '$out_dir'

# If no synthetic data, remove ST from hard stage
if '$synth' == 'no':
    for stage in cfg['curriculum']['stages']:
        if 'Sino-Tibetan' in stage.get('families', []):
            stage['families'].remove('Sino-Tibetan')
            stage['family_weights'].pop('Sino-Tibetan', None)

with open('$run_config', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f'Config written: $run_config')
"

    log "Starting training: $name"
    python ast-asr/train_rl.py --config "$run_config" --max-steps "$steps"
    log "Training complete: $name"
}

eval_run() {
    local name="$1"
    local ckpt="outputs/checkpoints/$name/final"
    local csv_out="$RESULTS_DIR/${name}_eval.csv"
    local summary="$RESULTS_DIR/${name}_summary.txt"

    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  EVAL: $name on Svarah ($EVAL_SAMPLES samples)"
    echo "──────────────────────────────────────────────────────────────"

    if [ ! -d "$ckpt" ]; then
        log "ERROR: Checkpoint not found at $ckpt — skipping eval"
        return 1
    fi

    python ast-asr/pipeline.py \
        --model rl-grpo \
        --model-path "$ckpt" \
        --max-samples "$EVAL_SAMPLES" \
        --output "$csv_out" \
        2>&1 | tee "$summary"

    log "Eval complete: $name → $summary"
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    local filter="${1:-}"

    log "RL Ablation Sweep"
    log "Config base: $CONFIG_BASE"
    log "Max steps: $MAX_STEPS"
    log "Eval samples: $EVAL_SAMPLES"

    if [ "$filter" = "--eval-only" ]; then
        log "Re-evaluating existing checkpoints..."
        for ablation in "${ABLATIONS[@]}"; do
            read -r name _ <<< "$ablation"
            eval_run "$name" || true
        done
        return
    fi

    for ablation in "${ABLATIONS[@]}"; do
        read -r name beta_kl alpha_fair synth steps <<< "$ablation"

        # Filter to a single run if requested
        if [ -n "$filter" ] && [ "$filter" != "$name" ]; then
            continue
        fi

        # Skip runs requiring synthetic data if it doesn't exist
        if [ "$synth" = "yes" ] && [ ! -f "data/synthetic-st/manifest.csv" ]; then
            log "WARNING: Synthetic ST data not found — skipping $name"
            log "  Generate with: python ast-asr/synth/tts_generator.py"
            continue
        fi

        train_run "$name" "$beta_kl" "$alpha_fair" "$synth" "$steps"
        eval_run "$name" || log "WARNING: Eval failed for $name"
    done

    # Final summary
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  ABLATION SWEEP COMPLETE"
    echo "══════════════════════════════════════════════════════════════"
    log "Results in: $RESULTS_DIR/"
    ls -la "$RESULTS_DIR/"
}

main "${1:-}"
