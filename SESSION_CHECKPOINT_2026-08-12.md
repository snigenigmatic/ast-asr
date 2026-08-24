# Session checkpoint — 2026-08-12 (restart from here tomorrow)

## ⚠ Status correction — codex branch advanced 12 commits past this branch point (read first)
After this checkpoint was drafted, `codex/fair-cispo-tiny` moved on (tip `c5a7baf`, we branched at `b178e95`):
- **H6 replication RAN and FAILED the safety gate.** β=0.04 reliably reduced divergence (mechanism holds),
  but the effect did NOT replicate: seed 2026 +2.2 pp worst-group improvement, seed 2027 **−0.4 pp (worse)**,
  seed 2028 β=0 control **tripped KL 0.111 > 0.1 at cycle 27 and stopped**. `H6 = failed_safety`; no 3-seed
  efficacy claim. Also `H0 refuted`: no LR runs 300 cycles safely.
- **H7 sentinel-KL diagnostic** is locked but NOT authorized to run — tests whether the stop is real
  instability or a candidate-bank/measurement artifact (`experiments/H7-sentinel-kl/protocol.md`).
- **`fair-cispo-work` is 12 commits behind codex → rebase onto latest codex before continuing.**
- Sources of truth: `research-state.yaml`, `experiments/H6-replication/result-20260812.md` on codex.

## SPIRE evaluation: smoke PASSED, 4 matched-pair arms launched (2026-08-20)

**Materialization COMPLETE.** `spire-val-20260820` reproduced the audit numbers exactly (5,214 utterances,
198/198 speakers, 15.3192 h, families {Dravidian, Indo-Aryan}) — an independent agreement between the
metadata-streaming and full-download paths. Volume `ast-asr-spire` now holds `/val/wav`, `/val/manifest.csv`
(sha256 `7a7c0021d1a9c0c395a274cad40212d47630c80c9e442267010f6d185358b39f`), `/val/preparation-report.json`.

**GPU smoke PASSED** (`spire-eval-smoke-zeroshot-20260820`, `--limit 40`, exit 0): overall WER 0.1693 over
40 utterances / 756 reference words. NOTE: those 40 rows are only **2 speakers, one family, one gender**, so
`family_gap` is 0.0 by construction. It validates plumbing, NOT science.

**Four matched-pair arms launched detached** (5,214 utterances each, greedy FP32, L4):

| Eval run name | Arm | Checkpoint run / output |
| --- | --- | --- |
| `spire-eval-h5-beta0-20260820` | `h5-beta0-s2026` | `profile-h5-refkl-beta0-r1-s2026-20260812` / `h5-beta0-fr-cispo` |
| `spire-eval-h5-beta004-20260820` | `h5-beta004-s2026` | `profile-h5-refkl-beta004-s2026-20260812` / `h5-beta004-fr-cispo` |
| `spire-eval-h6-beta0-20260820` | `h6-beta0-s2027` | `profile-h6-refkl-beta0-s2027-20260812` / `h6-beta0-fr-cispo` |
| `spire-eval-h6-beta004-20260820` | `h6-beta004-s2027` | `profile-h6-refkl-beta004-s2027-20260812` / `h6-beta004-fr-cispo` |

All use `--checkpoint-name checkpoint-last-safe`. Coordinates and traps are in
`experiments/SPIRE-crosscorpus/arms.md` — **the H5 beta-zero control is the `-r1-` run**; the non-r1
directory has no adapter output.

### When they finish
1. **Verify provenance before reporting any number.** Each `metrics.json` records
   `adapter_checkpoint_revision`; it must equal the revision published in
   `experiments/H6-replication/result-20260812.md` (table in `arms.md`). A mismatch means stop.
2. Run the two paired speaker-clustered bootstraps:
```powershell
$env:PYTHONIOENCODING='utf-8'
uvx modal run scripts/modal_spire_eval.py::compare_spire_arms `
  --control-run-name spire-eval-h5-beta0-20260820 `
  --treatment-run-name spire-eval-h5-beta004-20260820
uvx modal run scripts/modal_spire_eval.py::compare_spire_arms `
  --control-run-name spire-eval-h6-beta0-20260820 `
  --treatment-run-name spire-eval-h6-beta004-20260820
```
3. Fetch results: `uvx modal volume get ast-asr-spire /eval/<run>/metrics.json -`
4. **The scientific question:** in-domain, beta=0.04 improved seed 2026's worst group by 2.22 pp but
   worsened seed 2027's by 0.40 pp. If the cross-corpus deltas disagree in sign across seeds too, the
   non-replication is a property of the method rather than of the Svarah profile-cluster folds. Write the
   result up as `experiments/SPIRE-crosscorpus/result-<date>.md` either way.

## SPIRE cross-corpus: audit PASSED, materialization launched (2026-08-20)

**Audit `spire-val-audit-20260820`** (app `ap-MnNC7dxDvoHL7QDdupLKWb`, 86 s end-to-end, metadata-only
streaming, no audio transferred). Result recorded in `experiments/SPIRE-crosscorpus/audit-20260820.md`:
- **198 declared = 198 observed** validation speakers; **15.3192 h**; families exactly
  {Dravidian 7.2961 h, Indo-Aryan 8.0231 h}; 5,214 accepted utterances of 35,299 scanned.
- **Zero contract violations across all 35,299 rows** — the per-row `split` label agreed with `splits.json`
  everywhere and every language resolved against `ast_asr.taxonomy`. The speaker-disjoint guarantee is
  verified, not assumed.
- Disclosed constraints: only **13 of 17** shipped languages appear in val (Dogri, Gujarati, Maithili,
  Sindhi contribute none), and the tail is severe (**Kashmiri = 2 utterances**, Nepali 34, Punjabi 50).
  Analysis pools to family; per-language cells are descriptive only.

**Materialization `spire-val-20260820`** launched detached: app `ap-L4lXx6uF1WYAdxuL7sMeRh`, 1 task.
Writes 16 kHz mono PCM-16 WAV + `manifest.csv` to volume `ast-asr-spire` under `/val`. Check with
`uvx modal app list --json`; on completion read
`uvx modal volume get ast-asr-spire /runs/spire-val-20260820/preparation.json -`.

**Evaluation path is built and committed but NOT yet run** (`scripts/modal_spire_eval.py` +
`scripts/spire_eval_entry.py`). Once the manifest exists:
```powershell
$env:PYTHONIOENCODING='utf-8'
# zero-shot baseline first
uvx modal run --detach scripts/modal_spire_eval.py::evaluate_spire `
  --run-name spire-eval-zeroshot-20260820 --arm zero-shot
# then any FR-CISPO checkpoint, e.g. the completed H5 seed-2026 pair
uvx modal run --detach scripts/modal_spire_eval.py::evaluate_spire `
  --run-name spire-eval-h5-beta004-20260820 --arm h5-beta004 `
  --checkpoint-run-name <training-run> --checkpoint-output-name <output> `
  --checkpoint-name checkpoint-last-safe
# then the paired speaker-clustered bootstrap
uvx modal run scripts/modal_spire_eval.py::compare_spire_arms `
  --control-run-name spire-eval-zeroshot-20260820 `
  --treatment-run-name spire-eval-h5-beta004-20260820
```
Tip: use `--limit 40` on the first eval as a cheap smoke before the full 5,214 utterances.
Verify checkpoint run/output names against the H5/H6 result docs before launching.

## H7 r1 outcome (2026-08-20) — terminal, and the encoding fix DID work

A parallel Codex thread authorized (`39a4508`, tag `h7-r1-authorized-20260812`) and ran H7 r1 today, then
recorded a **terminal measurement failure** (`e9c4e6b`,
`experiments/H7-sentinel-kl/failure-r1-20260820.md`). Read that file before touching H7.

- The **launcher encoding fix worked**: unlike the 3.4 s charmap crash, this attempt reached remote
  execution (task `ta-01M0F56FBP7HJ7QXV8RF47KAXR`, image `im-T5smJqHtvyR2slZGuNwOU1`) and validated banks.
- It then **fail-closed** on `ValueError: H7 input-lock noisy waveform hash mismatch` at cycle 000, pair 0.
- **Root cause:** the input lock froze *bitwise* hashes of a `torch.randn` noise waveform plus float SNR
  mixing. The lock was built on Windows / `torch 2.13.0+cpu`; Modal executed Linux / `torch 2.11.0+cu128`.
  PyTorch does not guarantee bitwise-identical results across versions or platforms. Seed, SNR, row
  identity, and source-audio bytes were all ruled out; the clean tensor hash matched, only the noisy one
  differed.
- **The authorization is consumed and cannot be retried.** No forward pass, decode, or WER happened, so it
  is a non-result, not evidence about KL.
- **Design lesson (carry into the paper):** a bitwise cross-platform tensor contract is unachievable. Fix by
  persisting the noise tensors as data in the lock, or verifying within a numerical tolerance, or generating
  noise with a platform-stable RNG. Do not simply re-freeze new hashes on a different machine.

## Session 2 progress (same day, after the status correction)

- **Rebased `fair-cispo-work` onto codex tip `c5a7baf`** — now 0 commits behind; docs sit on top.
- **H7 blocker diagnosed as a Windows console bug, not science.** The consumed H7 attempt died on
  `'charmap' codec can't encode '✓'` (Modal's Rich tick mark on a cp1252/CP437 console); zero remote
  work ran. The recovery protocol's prescribed fix is `PYTHONIOENCODING=utf-8`, and **its required
  no-Modal preflight probe now PASSES on this machine** (`utf-8 utf-8 cp1252`, exit 0). H7 r1 preflight
  gate #4 is therefore satisfied; gates #1-#3, #5, #6 (reviewed commit + tag + r1 authorization doc +
  coordinate-only renames + absent run root + `retries=0`) remain outstanding.
- **Built the SPIRE cross-corpus workstream** (new, does not touch H7's frozen scope):
  - `experiments/SPIRE-crosscorpus/protocol.md` — pre-registered contract, endpoints, forbidden actions.
  - `scripts/spire_crosscorpus.py` — pure logic: family resolution, split validation, fail-closed row
    acceptance, error-weighted pooling, worst-group, **paired speaker-clustered bootstrap** (real speakers).
  - `scripts/modal_spire_prepare.py` — Modal app `ast-asr-spire-crosscorpus`, isolated volume
    `ast-asr-spire`, `audit_spire_val` (metadata only, cheap) and `prepare_spire_val` (writes 16 kHz WAV).
  - `tests/test_spire_crosscorpus.py` — 33 tests, incl. taxonomy-agreement and a subprocess test proving
    the preparation path works with `torch` imports **blocked**.
- **Verified:** ruff clean, ruff-formatted, `compileall` OK, `git diff --check` clean, **135 tests pass**.

### Design decisions worth remembering
- WER uses the repo's own `ast_asr.metrics.normalize_for_wer` (NOT Whisper's `EnglishTextNormalizer`) so
  SPIRE numbers stay directly comparable to every existing Svarah number. SPIRE's uppercase references
  normalize cleanly. This supersedes the earlier note in this file.
- All SPIRE code lives in `scripts/`, **never `src/` or `configs/`**, because the H7 authorization freezes
  those trees' SHA-256 manifests and forbids source changes. Refactor into `src/` only after H7 r1.
- The prep path is stdlib-only because `ast_asr/__init__` imports `.objectives` -> torch; a local
  17-language family table is used, with a test asserting it matches `ast_asr.taxonomy` exactly.

### Immediate next action
Run the cheap metadata audit on Modal (no audio written, verifies split + taxonomy + counts/hours):

```powershell
$env:PYTHONIOENCODING='utf-8'
uvx modal run scripts/modal_spire_prepare.py::audit_spire_val --run-name spire-val-audit-20260812
```
Expect ~198 observed val speakers, ~15.3 h, families exactly {Dravidian, Indo-Aryan}. Then
`prepare_spire_val` with `--detach`. After that, build the GPU cross-corpus eval function.

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
   CAVEAT (codex `forbidden_shortcuts`): a heuristic Svarah speaker split is EXPLORATORY-only and must never
   be presented as authoritative. SPIRE-SIES (real speaker IDs) is the legitimate publication-valid route.

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
3. ~~Phase 3 — H6~~ **SUPERSEDED: H6 already ran on codex and failed safety** (see status banner above).
   The real frontier is the **H7 sentinel-KL diagnostic** (locked, not yet authorized) to explain the
   seed-2028 KL trip before any further training. Reconcile the codex thread (H7) with this session's SPIRE
   cross-corpus plan.
4. **Phase 4** — let the numbers pick the paper story.

## Gotchas / ops
- Don't anchor the paper on the old 16.4% v5 WER (invalid comparison — `docs/prior-results-benchmark.md`).
- Modal budget ~$25 free tier; `modal app list` → confirm 0 live tasks before launching; `uvx modal run --detach`.
- `$env:PYTHONUTF8='1'` on Windows; invoke Modal via `uvx`/`uv run`.
