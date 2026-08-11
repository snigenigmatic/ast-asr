# FR-CISPO: fair and robust tiny Indian-English ASR

This branch restores the capstone's original research goal: adapt pinned
`openai/whisper-tiny` checkpoints for lower language-family disparity and
better noise robustness on Indian-English speech.

The new implementation lives in `src/ast_asr`. The historical `ast-asr/`
scripts are retained only as evidence and are never added to `sys.path` or
imported by the new training path.

## Research contract

- Svarah is the only transcribed speech corpus. MUSAN supplies unlabeled speech
  noise only.
- All splits are speaker-disjoint, deterministic, and require exactly 117
  authoritative `speaker_id` values from Svarah's official
  `meta_speaker_stats.csv`.
- The model and dataset revisions are pinned in
  `configs/fr_cispo_tiny.json`.
- Fairness means reporting and improving worst language-family by acoustic-
  condition WER. The project does not use demographic-parity or equal-
  opportunity terminology.
- Historical GRPO/CISPO numbers are hypotheses, not results of this framework.

Svarah's public prose says four language families, while its published table of
19 primary languages identifies three represented families. The exact mapping
and this discrepancy are stored in `src/ast_asr/taxonomy.py`; data preparation
fails on any unseen language rather than silently assigning it.

## Setup

```bash
uv sync --extra dev
uv run pytest -q
```

The official Svarah archive must contain:

```text
data/raw/Svarah/
  audio/
  svarah_manifest.json
  meta_speaker_stats.csv
```

MUSAN must be extracted under `data/raw/musan/musan/speech` (or
`data/raw/musan/speech`). Its pinned OpenSLR archive MD5 is recorded in the
experiment config.

## Reproducible commands

Run the isolated Modal storage audit and L4 CUDA smoke before any paid
experiment. This exercises all five policy objectives with the real pinned
Whisper-tiny model, four inner updates, and a checkpoint round trip. It uses
synthetic audio and is therefore runtime validation, not research evidence.

```bash
uvx modal run scripts/modal_fr_cispo.py \
  --run-name fr-cispo-smoke-YYYYMMDD --seed 2026
```

The runner reuses the existing `ast-asr-cache` and `ast-asr-data` volumes but
writes only to the separate `ast-asr-fr-cispo-runs` volume. Search every
accessible official Svarah repository revision for speaker metadata with:

```bash
uvx modal run scripts/modal_fr_cispo.py::inspect_svarah_history \
  --run-name fr-cispo-smoke-YYYYMMDD
```

Do not start fold training unless the resulting storage audit reports
`speaker_fold_ready: true` and data preparation independently verifies exactly
117 speakers.

### Development-only profile fallback

The currently cached public Parquet release does not expose `speaker_id`. An
explicit fallback can derive 115 stable IDs from the eight demographic profile
columns in `meta_speaker_stats.csv`. These are **profile clusters, not verified
speakers**: their folds and results are useful for exercising the pipeline, but
are invalid for the paper's speaker-disjoint claims and must not be combined
with authoritative runs.

Upload the exact source CSV, extract the embedded Parquet audio, and freeze the
five fallback folds on Modal:

```bash
uvx modal volume put ast-asr-data meta_speaker_stats.csv \
  /fr_cispo/source/meta_speaker_stats.csv
uvx modal run scripts/modal_fr_cispo.py::prepare_profile_cluster_data \
  --run-name profile-dev-YYYYMMDD
```

Run bounded real-audio SFT and FR-CISPO checks:

```bash
uvx modal run scripts/modal_fr_cispo.py::run_profile_sft_smoke \
  --run-name profile-dev-YYYYMMDD --seed 2026
uvx modal run scripts/modal_fr_cispo.py::run_profile_fr_cispo_smoke \
  --run-name profile-dev-extended-YYYYMMDD --seed 2026 \
  --sft-run-name profile-dev-YYYYMMDD --rollout-cycles 10 \
  --probe-examples 18 --maximum-new-tokens 64
```

Only after those movement checks pass, run one complete development-fold SFT:

```bash
uvx modal run --detach \
  scripts/modal_fr_cispo.py::run_profile_sft_development \
  --run-name profile-dev-full-sft-YYYYMMDD --seed 2026
```

Prepare content-hashed data and five immutable speaker folds:

```bash
uv run ast-asr prepare-data \
  --archive-root data/raw/Svarah \
  --dataset-revision ebbf7777fe771490696a3f7b007097606fa8c924 \
  --output-dir data/fr_cispo
```

Train the development-fold SFT adapter:

```bash
uv run ast-asr train-sft \
  --config configs/fr_cispo_tiny.json \
  --fold 0 --seed 11 \
  --output-dir runs/dev/seed-11/sft
```

Train one post-training arm from the selected SFT checkpoint:

```bash
uv run ast-asr train-policy \
  --config configs/fr_cispo_tiny.json \
  --fold 0 --seed 11 \
  --arm fr-cispo --learning-rate 0.00003 \
  --sft-checkpoint runs/dev/seed-11/sft/checkpoint-epoch-SELECTED \
  --output-dir runs/dev/seed-11/fr-cispo
```

The policy arms are `live-grpo`, `cispo-mwer`, `sequence-cispo-mwer`,
`fair-cispo`, and `fr-cispo`. Zero-shot and SFT are evaluated directly rather
than routed through the policy trainer.

Evaluate clean, white-noise 10 dB, and MUSAN-babble 10 dB:

```bash
uv run ast-asr evaluate-fold \
  --config configs/fr_cispo_tiny.json \
  --fold 0 --arm fr-cispo \
  --checkpoint runs/dev/seed-11/fr-cispo/checkpoint-final \
  --output-dir runs/dev/seed-11/fr-cispo/evaluation
```

Use `--checkpoint base` for the zero-shot Whisper-tiny arm.

Aggregate completed out-of-fold predictions with 10,000 paired
speaker-clustered bootstrap samples:

```bash
uv run ast-asr aggregate-oof \
  --predictions runs/oof/fold-*/evaluation/predictions.jsonl \
  --output-dir runs/oof/aggregate
```

## Gates

Every rollout stores the old model revision, hypotheses, token masks, FP32 old
token/sequence log-probabilities, WERs, groups, and conditions. Each frozen
rollout receives four optimizer passes. Update-zero ratios must be one; later
updates record live ratio movement.

Folds 1-4 are blocked unless `--development-gate development_gate.json` is
provided. That immutable artifact must record three development seeds, the
selected safe learning rate, at least 0.02 mean worst-group WER improvement,
and no more than 0.01 mean clean-WER degradation. A failed gate is a result; it
does not authorize a new method.

The local test suite verifies fold isolation, sequence-ratio arithmetic,
stop-gradient CISPO weights, group-weight placement, live ratio movement,
dropout-free scoring, corruption SNR, attention masks, batch-invariant greedy
inference, edit-count aggregation, bootstrap behavior, and literal gates.
