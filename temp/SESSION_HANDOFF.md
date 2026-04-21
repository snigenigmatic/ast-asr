# SESSION_HANDOFF.md

Persistence doc for a future Claude session. Read this first, then the auto-memory index at `~/.claude/projects/-home-mluser-ast-asr/memory/MEMORY.md`.

**Last updated:** 2026-04-17
**Branch:** `ast-adversery`
**HEAD:** `20a224e results: Whisper RL ablation sweep + zero-shot baseline on Svarah-eval`

---

## 1. Current state summary

Working on Indian-English ASR fairness research (see auto-memory `project_asr_fairness.md`). The ablation sweep in `outputs/whisper_rl_results/` is the last fully-committed work. Since then:

- **v3 Whisper GRPO run (LR=1e-4, 32 rank, 4 modules, wd=0):** diverged — KL exploded to 1951 by step 80. Log in `outputs/whisper_rl_results/whisper-rl-fair-v3_train.log`.
- **v4 Whisper GRPO run (LR=2e-5, r=16, 4 modules, beta_kl=0.1):** was working as intended — KL stable 0.3–18 range, step 200 decode probe showed 1/2 samples starting to differ from reference (first real output change). **Killed at step ~210 by a power cut on the remote.** Log in `outputs/whisper_rl_results/whisper-rl-fair-v4_train.log`.
- **v5 plan (deferred):** shortened 1800-step run with `save_interval=150`, launched via `nohup setsid` (not tmux — see gotcha #2). See `~/.claude/plans/sorted-petting-lecun.md` for full plan.

### Uncommitted changes at time of handoff

```
 M ast-asr/rl/reward.py
 M ast-asr/rl/whisper_grpo.py
 M ast-asr/train_rl_whisper.py
 M configs/train_rl_whisper.yaml
 M outputs/checkpoints/rl-base/final/adapter_config.json
 M outputs/checkpoints/whisper-rl-fair/metrics.csv
 M outputs/whisper_rl_results/summary_whisper-small-rl.csv
 M outputs/whisper_rl_results/whisper-rl-fair_eval.csv
 M outputs/whisper_rl_results/whisper-rl-fair_summary.txt
 M scripts/run_whisper_rl_ablation.sh
 D uv.lock
?? download_indictts.py
```

These modifications contain the v3/v4 fixes (decode probe, weight_decay=0, alpha-fairness CLI override, case-norm WER). They're in the tar but NOT pushed to origin. `uv.lock` was deleted at some point — restore via `uv sync` if needed.

All commits before HEAD are pushed to `github.com/snigenigmatic/ast-asr.git`.

---

## 2. Project structure

```
/home/mluser/ast-asr/
├── ast-asr/                        # Python source (400K)
│   ├── data_loader.py              # load_svarah() — HF Svarah loader with 70/30 split
│   ├── spire_loader.py             # SPIRE-SIES loader (expects data/spire-sies/raw/ layout)
│   ├── pipeline.py                 # eval entrypoint (all model types)
│   ├── train.py                    # hybrid Wav2Vec2+LoRA+GRL SFT training
│   ├── train_rl.py                 # Wav2Vec2 GRPO (early RL pivot)
│   ├── train_rl_whisper.py         # Whisper-small GRPO (current RL pipeline)
│   ├── rl/
│   │   ├── reward.py               # multi-component reward (WER+CER+fairness)
│   │   ├── whisper_grpo.py         # GRPO step for Whisper
│   │   └── grpo.py                 # GRPO step for Wav2Vec2
│   ├── synth/
│   │   └── tts_generator.py        # synthetic ST-family data plan (stub)
│   └── noise_augment.py
├── configs/
│   ├── train.yaml                  # hybrid SFT config
│   ├── train_rl.yaml               # Wav2Vec2 RL config
│   └── train_rl_whisper.yaml       # Whisper RL config (v5 plan modifies this)
├── scripts/
│   ├── run_ablation_sweep.sh       # hybrid SFT ablation (λ∈{0.05,0.10,0.30})
│   └── run_whisper_rl_ablation.sh  # Whisper RL ablation (α_fair ∈ {0, 2, 5})
├── outputs/                        # CHECKPOINTS + LOGS + CSVs (9.4G, in tar)
│   ├── checkpoints/
│   │   ├── ft-w2v2/                      1.9G — Wav2Vec2 SFT baseline
│   │   ├── hybrid-lam0.{05,10,30}/       1.9G each — GRL ablation
│   │   ├── hybrid_w2v2_lora_grl/         1.9G — main GRL run
│   │   ├── whisper-rl-fair/              42M  — v4 (210 steps before power cut)
│   │   ├── whisper-rl-fair-high/         21M  — α_fair=5 ablation
│   │   ├── whisper-rl-base/              21M  — α_fair=0 ablation
│   │   └── rl-base/, rl-fair/            Wav2Vec2 RL early attempts
│   ├── whisper_rl_results/         3.8M — eval CSVs + training logs
│   └── logs/, results_*.csv, summary_*.csv
├── data/                           # 115G — NOT IN TAR (re-downloadable)
│   ├── spire-sies/                 80G  — from VectorSigma389/spire-sies (HF private)
│   └── svarah_split/               40K  — train_uids.txt + eval_uids.txt (IN TAR)
├── cache/                          # 1.1G — HF datasets Arrow cache (regenerated)
├── .venv/                          # 7.4G — Python 3.12 env (regenerated via uv sync)
├── .git/                           # 6.8G — loose objects (IN TAR)
├── .env                            # HF_TOKEN=... (IN TAR — sensitive!)
├── .python-version                 # 3.12
├── pyproject.toml                  # uv-managed deps
├── download_indictts.py            # untracked, planned IndicTTS integration
└── SESSION_HANDOFF.md              # this file
```

---

## 3. Size breakdown (snapshot 2026-04-17)

| Path | Size | In tar? | Reason |
|------|------|---------|--------|
| `data/spire-sies/` | **80 G** | ❌ | Re-download from `VectorSigma389/spire-sies` (HF private) |
| `data/svarah_split/` | 40 K | ✅ | The 70/30 split manifest — critical for reproducibility |
| `.venv/` | **7.4 G** | ❌ | `uv sync` recreates |
| `outputs/checkpoints/` | **9.4 G** | ✅ | All model weights — irreplaceable |
| `.git/` | **6.8 G** | ✅ | Full history including unpushed work |
| `cache/` | 1.1 G | ❌ | Regenerated on first `load_dataset` |
| source/configs/scripts/root | <1 M | ✅ | Code |

`~/.cache/huggingface/` (18 G — Mistral-7B 14G, Svarah 1.1G, Whisper-small 927M, Wav2Vec2-base 361M, MiniLM, Whisper-tiny) is **outside** the project and **not** in tar. All re-downloadable.

**Tar total: ~15 GB gzipped.**

---

## 4. Data re-download instructions

### Svarah (public, AI4Bharat)

```python
# Already wired into ast-asr/data_loader.py
from data_loader import load_svarah
df = load_svarah(cache_dir="cache", svarah_split="train")  # or "eval"
```

The 70/30 split is determined by `data/svarah_split/{train_uids,eval_uids}.txt` (speaker-stratified, committed to the tar). If those files go missing, the loader will generate a new split — **this breaks reproducibility with existing eval results**. Keep them.

Alternatively raw:
```bash
huggingface-cli download ai4bharat/svarah --repo-type dataset --local-dir data/svarah
```

### SPIRE-SIES (private — yours)

Pushed by you on 2026-04-09 as `VectorSigma389/spire-sies` (HF private dataset).

```bash
# Requires HF_TOKEN in .env or login
huggingface-cli login  # or: export HF_TOKEN=...
huggingface-cli download VectorSigma389/spire-sies \
    --repo-type dataset \
    --local-dir data/spire-sies
```

Expected local layout (per `ast-asr/spire_loader.py`):
```
data/spire-sies/
├── raw/
│   ├── IISc_SPIRE_SIES_Transcription.csv
│   └── IISc_SPIRE_SIES_<Language>/<speaker_id>/<utt>.wav
└── splits.json                # speaker-level train/val/test split
```

If the HF repo's file layout differs, adjust or re-run `ast-asr/spire_loader.py` to rebuild `splits.json`.

### IndicTTS (planned — not yet integrated)

```bash
# Requires Kaggle credentials (~/.kaggle/kaggle.json)
pip install kagglehub
python download_indictts.py
# Downloads tuannguyenvananh/indictts-english
```

Purpose (per Sarvam researcher advice in auto-memory): synthesize Sino-Tibetan-family training data to close the ΔDP gap. Not yet wired into training pipeline — future work.

---

## 5. Rebuild on a fresh machine (or after restore)

```bash
# 1. Extract
cd ~ && tar -xzf /path/to/ast-asr-backup-*.tar.gz
cd ast-asr

# 2. Python env (Python 3.12 required)
pip install uv
uv sync            # reads pyproject.toml, builds .venv/
source .venv/bin/activate

# 3. Secrets
# .env should contain HF_TOKEN (already in tar — verify it's readable)
cat .env

# 4. Data
# Svarah: auto-downloads on first load (data_loader.py)
# SPIRE-SIES:
huggingface-cli download VectorSigma389/spire-sies \
    --repo-type dataset --local-dir data/spire-sies
# IndicTTS (optional, future):
python download_indictts.py

# 5. Verify
python -c "from ast_asr.data_loader import load_svarah; print(load_svarah(max_samples=5, cache_dir='cache', svarah_split='train'))"
```

---

## 6. Auto-memory pointers

The Claude auto-memory system lives at `~/.claude/projects/-home-mluser-ast-asr/memory/`. Read `MEMORY.md` first (the index), then individual files as relevant:

- `user_role.md`, `user_hf_account.md` — user profile (VectorSigma389)
- `feedback_plans_before_code.md` — always use ExitPlanMode before non-trivial code
- `project_asr_fairness.md` — project overview
- `project_training_dynamics.md` — hybrid GRL tuning history (λ=0.1 diverges etc.)
- `project_ablation_results.md` — ft-w2v2 wins; GRL widens ΔDP due to ST OOD
- `project_sarvam_advice.md` — synthetic data + RL post-training strategy
- `project_rl_posttraining_design.md` — 4-pillar RL framework
- `reference_remote_tmux.md` — **critical:** no systemd lingering on this remote; use `nohup setsid` for long runs

These persist across Claude sessions automatically.

---

## 7. v5 run plan (deferred — run when ready)

Plan file: `~/.claude/plans/sorted-petting-lecun.md` (earlier version — current plan file is the tar/handoff plan)

**Config changes needed:**
```yaml
# configs/train_rl_whisper.yaml
rl:
  max_steps: 1800          # was 3000
  save_interval: 150       # was 500 — 12 intermediate checkpoints
  # keep rest: lr=2e-5, beta_kl=0.1, r=16, 4 modules, weight_decay=0
```
```bash
# scripts/run_whisper_rl_ablation.sh
MAX_STEPS=1800            # was 3000
```

**Launch (NOT in tmux — use nohup+setsid):**
```bash
cd /home/mluser/ast-asr
nohup setsid bash -c "source .venv/bin/activate && bash scripts/run_whisper_rl_ablation.sh whisper-rl-fair" \
    > outputs/whisper_rl_results/whisper-rl-fair-v5_train.log 2>&1 < /dev/null &
disown
```

Expected runtime: 1800 steps × 7.5 s ≈ 3.75 hr train + 10 min eval. Budget: ~7 hr allocation remaining at handoff time.

---

## 8. Known gotchas

1. **`.venv` is Python 3.12.** `pyproject.toml` requires `>=3.12,<3.14`. On Windows, use WSL Ubuntu with pyenv-installed 3.12, or skip `.venv` reuse and recreate with `uv sync`.
2. **This remote has no systemd user lingering.** `loginctl show-user` returns empty Linger. Consequence: tmux servers die when the last SSH session closes. For multi-hour jobs use `nohup setsid` (see `reference_remote_tmux.md`). Previous v4 run loss (pre-power-cut analysis) suspected tmux death until a later power cut confirmed it was actually hardware. Either way, the rule stands.
3. **`data/spire-sies/` is NOT in the tar** (80 G). You MUST re-download from `VectorSigma389/spire-sies` (HF private) before running SPIRE-based training.
4. **HF auth required** for SPIRE-SIES pull. `.env` contains `HF_TOKEN` — make sure it's exported or use `huggingface-cli login`.
5. **Kaggle auth required** for IndicTTS (when added) — `~/.kaggle/kaggle.json`.
6. **`.env` is in the tar** and contains `HF_TOKEN`. Treat the tarball as sensitive; don't share publicly.
7. **`.git/` has no packfile** (all loose objects, 6.8 G). After restoring you may want to run `git gc --aggressive` to pack — drops size significantly. Not required for correctness.
8. **WER was case-normalized in commit `9f1efca`.** Older eval CSVs pre-dating this commit are NOT comparable to newer ones. See `project_training_dynamics.md` for context.
9. **GRL hybrid models widen ΔDP rather than narrowing it** on Svarah because Sino-Tibetan (ST) speakers are absent from SPIRE-SIES training data — OOD gap dominates the fairness signal. See `project_ablation_results.md`. Next research direction: synthetic ST data via IndicTTS + synth pipeline, or RL post-training on ft-w2v2 checkpoint.

---

## 9. Quick sanity checks after restore

```bash
# Branch and HEAD
git status                          # should show same modified files listed in §1
git log --oneline -3                # should show 20a224e at top

# Python env works
python -c "import torch, transformers, peft, datasets; print('ok')"

# Data loader works (no actual Svarah download if already cached)
python -c "from ast_asr.data_loader import load_svarah; df = load_svarah(max_samples=5, cache_dir='cache', svarah_split='eval'); print(df.shape, df.columns.tolist())"

# Checkpoints accessible
ls outputs/checkpoints/whisper-rl-fair/final/adapter_model.safetensors

# HF auth
python -c "from huggingface_hub import whoami; print(whoami())"
```
