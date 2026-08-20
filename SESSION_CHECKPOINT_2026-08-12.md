# Session checkpoint — 2026-08-12 (restart from here tomorrow)

## How to restart
1. Read `docs/plain-language-walkthrough.md` (this worktree) — the framework, in plain terms.
2. Read this file.
3. Recovery plan: `C:\Users\C_Kaustubh\.claude\plans\ok-so-all-the-happy-sphinx.md`.
4. Memory: `fair-cispo-pivot-2026-08-12.md` + `spire-sies-corpus-2026-08-12.md`.

## HARD CONSTRAINTS (do not violate)
- **NEVER download the SPIRE dataset (or any large corpus) to the laptop — it will crash.** All heavy
  data + compute runs on **Modal** (into a Modal volume). Local machine only does code, tests, small peeks.
- Work stays in the isolated worktree `C:/Kaustubh/ast-asr-worktrees/fair-cispo-work` (branch
  `fair-cispo-work`, off `codex/fair-cispo-tiny`). Do not disturb `codex/fair-cispo-tiny` or `ast-adversery`.
- Quality over speed. Math + experiments must be genuine and understood to the basics. No mid-tier rush.

## Locked decisions
1. Foundation = codex FR-CISPO framework (`src/ast_asr/`), in this worktree.
2. Paper = "replicate first, then decide" — run H6 before choosing the story.
3. Data blocker = "defensible split" — do not wait on the missing authoritative 117 Svarah speaker IDs.

## DONE this session
- Worktree created; `uv sync --extra dev` (torch is CPU on Windows); **all 66 tests green** (8s).
- Wrote `docs/plain-language-walkthrough.md` (objective math, live-ratio/μ=1 fix, KL safety, gates,
  where the missing speaker IDs bite). Correction captured: **H6 uses seeds 2027/2028** (exploratory,
  profile-cluster), NOT 11/17/23 (those are the later confirmatory gate needing authoritative speakers).
- Assessed SPIRE-SIES and confirmed it via the private HF repo (see next section).

## SPIRE-SIES — everything needed to materialize it ON MODAL tomorrow
- Repo: `VectorSigma389/spire-sies` (dataset). Auth: `HF_TOKEN` (37 chars) in `C:/Kaustubh/ast-asr/.env`.
  **On Modal, add it as a Modal Secret — do NOT ship the .env.**
- Files: 17 language parquets under `data/<language>.parquet` + `splits.json`.
- `splits.json`: `{"train": 1126 speakers, "val": 198 speakers}` — speaker IDs like `F11`, `F1124`
  (prefix F/M/O = gender). Speaker-disjoint, ready-made. **Use this directly — no heuristic split needed.**
- Parquet schema (confirmed on `data/sindhi.parquet`), columns:
  `uid, accent, language_family, speaker_id, gender, age, reference, duration, src_sr, split, audio_bytes`
  - `split` is a per-row label ('train'/'val') already consistent with `splits.json` — filter on it directly.
  - `audio_bytes` = raw WAV bytes; `src_sr` = **48000** → resample to 16 kHz for Whisper.
  - `reference` is UPPERCASE as stored (fine — normalize at scoring time with Whisper's EnglishTextNormalizer).
  - Families present: **Dravidian + Indo-Aryan only (no Sino-Tibetan)** → cross-corpus contrast is 2-way.
  - `age` = all "Unknown" (no age axis); `gender` present + balanced.
- Val stats target: 198 speakers, ~15.3 h. Pool the long language tail to family level.

### Modal materialization approach (write tomorrow; run with `--detach`)
A Modal function that: mounts a Modal Volume (e.g. `spire-sies-val`), streams the 17 parquets from HF with
`hf_hub_download` (HF_TOKEN via Modal Secret), and for rows where `split == "val"` decodes `audio_bytes`
(`soundfile.read(io.BytesIO(...))`), resamples 48k→16k (`librosa.resample`), writes `<uid>.wav` to the
volume + a `manifest.csv` (uid, speaker_id, accent, language_family, gender, duration, reference). Keep the
RAW reference. ~29 GB is scanned on Modal, but only ~a few GB of val WAVs are written — laptop untouched.
Then a cross-corpus eval Modal function reads the volume, runs `greedy_transcribe` + WER (EnglishTextNormalizer),
family-pooled (2-way), gender axis, speaker-clustered bootstrap. Reuse codex `src/ast_asr/{evaluation,inference,
metrics,taxonomy}.py`; port ONLY the parquet-row → record logic (NOT `spire_loader.normalize_transcript`).

## Next actions (in order)
1. **Phase 2 — split reframe.** SPIRE: use provided `split`/`splits.json` (done-for-us). Svarah in-domain:
   port `extract_svarah_speaker_id`/`make_speaker_split` (ast-adversery `ast-asr/data_loader.py`) into codex
   `src/ast_asr/folds.py`; add a non-"authoritative" identity mode (see `policy_training.py:266`
   `publication_valid` gate) and document the heuristic limitation honestly.
2. **SPIRE cross-corpus eval on Modal** (materialize val → eval). Never local.
3. **Phase 3 — H6** (seeds 2027/2028, whisper-tiny, LR 1e-5, 40 cycles) per
   `experiments/H6-replication/protocol.md`, Svarah primary, on Modal `--detach`.
4. **Phase 4** — let the numbers pick the paper story.

## Gotchas / ops
- Don't anchor the paper on the old 16.4% v5 WER (invalid comparison — `docs/prior-results-benchmark.md`).
- Modal budget ~$25 free tier; `modal app list` → confirm 0 live tasks before launching; `uvx modal run --detach`.
- `$env:PYTHONUTF8='1'` on Windows; invoke Modal via `uvx`/`uv run`.
