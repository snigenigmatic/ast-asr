#!/usr/bin/env bash
# scripts/run_ablation.sh
#
# Full ablation sweep: 4 runs, fully sequential.
# Each run = TRAIN (on SPIRE-SIES) then EVAL (on Svarah 500 samples).
#
# Runs:
#   1. ft-w2v2        LoRA fine-tune only, NO adversary/GRL  (pure CTC baseline)
#   2. hybrid-lam0.05  LoRA + GRL adversary, target λ=0.05
#   3. hybrid-lam0.10  LoRA + GRL adversary, target λ=0.10
#   4. hybrid-lam0.30  LoRA + GRL adversary, target λ=0.30
#
# Outputs:
#   outputs/checkpoints/<run>/final/   — saved LoRA adapter + adversary head
#   outputs/logs/train_<run>.log       — training log (loss per step)
#   outputs/logs/eval_<run>.log        — Svarah eval log
#   outputs/summary_<run>.csv          — one-row fairness summary
#
# Usage:
#   bash scripts/run_ablation.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

SVARAH_SAMPLES=500
BASE_CFG=configs/train_hybrid.yaml
LOG_DIR=outputs/logs
mkdir -p "$LOG_DIR" outputs/checkpoints

# ─────────────────────────────────────────────────────────────────────────────
# train_run NAME USE_ADVERSARY TARGET_LAMBDA
#   Patches the base config, trains the model, saves checkpoint to
#   outputs/checkpoints/NAME/final/
# ─────────────────────────────────────────────────────────────────────────────
train_run() {
    local name="$1"
    local use_adv="$2"   # "true" or "false"
    local lam="$3"       # float, e.g. 0.05
    local ckpt_dir="outputs/checkpoints/$name"

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  TRAINING: $name"
    echo "    use_adversary = $use_adv"
    echo "    target_lambda = $lam"
    echo "    checkpoint    → $ckpt_dir/final"
    echo "    log           → $LOG_DIR/train_${name}.log"
    echo "══════════════════════════════════════════════════════"

    local tmp_cfg
    tmp_cfg=$(mktemp /tmp/train_cfg_XXXX.yaml)
    cp "$BASE_CFG" "$tmp_cfg"
    sed -i "s/use_adversary:.*/use_adversary: $use_adv/" "$tmp_cfg"
    sed -i "s/target_lambda:.*/target_lambda: $lam/" "$tmp_cfg"

    rm -rf "$ckpt_dir"

    python3 ast-asr/train.py \
        --config "$tmp_cfg" \
        --output-dir "$ckpt_dir" \
        2>&1 | tee "$LOG_DIR/train_${name}.log"

    rm -f "$tmp_cfg"
    echo "  ✓ Training complete → $ckpt_dir/final"
}

# ─────────────────────────────────────────────────────────────────────────────
# eval_run NAME CHECKPOINT_DIR
#   Runs Svarah fairness eval on a saved checkpoint.
#   Writes outputs/summary_NAME.csv
# ─────────────────────────────────────────────────────────────────────────────
eval_run() {
    local name="$1"
    local ckpt_dir="$2"

    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  EVAL: $name"
    echo "    checkpoint → $ckpt_dir"
    echo "    log        → $LOG_DIR/eval_${name}.log"
    echo "──────────────────────────────────────────────────────"

    python3 ast-asr/pipeline.py \
        --model hybrid-w2v2-grl \
        --model-path "$ckpt_dir" \
        --max-samples "$SVARAH_SAMPLES" \
        --output "outputs/results_${name}_clean.csv" \
        2>&1 | tee "$LOG_DIR/eval_${name}.log"

    if [ -f "outputs/summary_hybrid-w2v2-grl.csv" ]; then
        cp "outputs/summary_hybrid-w2v2-grl.csv" "outputs/summary_${name}.csv"
        echo "  ✓ Fairness summary → outputs/summary_${name}.csv"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# SWEEP
# ─────────────────────────────────────────────────────────────────────────────

# 1. ft-w2v2: LoRA only, GRL disabled
train_run "ft-w2v2"        "false" "0.0"
eval_run  "ft-w2v2"        "outputs/checkpoints/ft-w2v2/final"

# 2. hybrid λ=0.05
train_run "hybrid-lam0.05" "true"  "0.05"
eval_run  "hybrid-lam0.05" "outputs/checkpoints/hybrid-lam0.05/final"

# 3. hybrid λ=0.10
train_run "hybrid-lam0.10" "true"  "0.1"
eval_run  "hybrid-lam0.10" "outputs/checkpoints/hybrid-lam0.10/final"

# 4. hybrid λ=0.30
train_run "hybrid-lam0.30" "true"  "0.3"
eval_run  "hybrid-lam0.30" "outputs/checkpoints/hybrid-lam0.30/final"

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  ABLATION RESULTS"
echo "══════════════════════════════════════════════════════"
python3 - <<'EOF'
import pandas as pd, glob, os

order = ["ft-w2v2", "hybrid-lam0.05", "hybrid-lam0.10", "hybrid-lam0.30"]
rows = []
for name in order:
    f = f"outputs/summary_{name}.csv"
    if not os.path.exists(f):
        continue
    try:
        df = pd.read_csv(f)
        row = df.iloc[0].to_dict()
        row["run"] = name
        rows.append(row)
    except Exception as e:
        print(f"  WARNING: could not read {f}: {e}")

if rows:
    out = pd.DataFrame(rows).set_index("run")
    cols = ["overall_wer", "wer_dravidian", "wer_indo_aryan", "wer_sino_tibetan",
            "delta_dp", "delta_eo", "max_noise_gap", "poisson_p", "systematic_gap"]
    cols = [c for c in cols if c in out.columns]
    print(out[cols].round(4).to_string())
else:
    print("  No summary files found.")
EOF
