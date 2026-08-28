# Handoff for direction-setting — Indian-English ASR fairness

**Date:** 2026-08-25. **Purpose:** give a new agent everything needed to brainstorm
and weigh directions. This is a decision aid, not a plan. Nothing here is a
commitment.

**Provenance rule used throughout.** Items marked **[V]** were verified directly
against the repository, Modal, or a committed artifact during the 2026-08-24/25
session. Items marked **[R]** are *reported* from a strategy handoff written
elsewhere and were **not** independently verified — treat them as claims to check.
Several lines of that handoff arrived truncated; where I inferred intent I say so.

---

## 0. Why this project is being redone [V — from `docs/legacy/PAPER_SUGGESTIONS.md`]

A paper was already **submitted to ACL SRW and rejected**. The criticism was
structural and, per the author's own assessment, correct: it was a protocol-only
survey with no empirical results, on a problem reviewers considered already
recognised. A second criticism was that the prose read as machine-written (heavy
em-dash use). **Everything since is the response to that rejection**, which is why
the live strategic question is "audit versus method" rather than "which method".

The rewrite plan ranked its intended contributions as:

1. **The GRL failure mode** — gradient-reversal adversarial de-biasing *widens*
   accent disparity when the minority group (Sino-Tibetan) is out-of-distribution
   for the adversary. Judged the cleanest, most reviewer-proof finding because it is
   a concrete falsifiable negative result.
2. **The fairness protocol actually applied** to an Indian-English corpus
   (dDP, dEO, d_noise, Poisson significance).
3. **The CISPO baseline** — reported as a baseline pending a corrected reward.
4. **The noise-robustness gap** — Sino-Tibetan +16.9 pp WER under white noise vs
   +6.6 pp Dravidian; clean-condition and noise-condition fairness are distinct axes.

### Two traps in that document

**Trap 1 — different model.** Those v5-era numbers (zero-shot WER 18.4%, CISPO
16.4%, dDP p 0.020 -> 0.078) are **Whisper-small**. The entire FR-CISPO programme
and every SPIRE number in this handoff are **`whisper-tiny`**. The two threads are
not numerically comparable, and `docs/prior-results-benchmark.md` explains why the
16.4% figure is not a citable comparison point at all.

**Trap 2 — a hedge that is now a finding.** The rewrite plan honestly hedged that
the v5 improvement "may be attributable to the balanced sampler rather than the
GRPO fairness objective." The SPIRE cross-corpus result (section 5) settles this:
the explicit fairness term is inert while the pipeline as a whole produces large
gains. That hedge can be promoted to a supported claim.

Full documents, with a staleness warning, are in `docs/legacy/`.

## 1. The problem, in one paragraph

Automatic speech recognition (ASR) is trained mostly on Western English and
performs worse on Indian English — unevenly across speaker groups. The project's
goal is to measure that unevenness honestly and, if possible, reduce it for the
worst-off group without degrading everyone else. Primary metric is Word Error
Rate (WER); lower is better. The model under study is `openai/whisper-tiny`
(revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`), adapted with LoRA. Two
corpora are in play: **Svarah** (`ai4bharat/Svarah`, in-domain, used for training)
and **SPIRE-SIES** (`VectorSigma389/spire-sies`, private, held out, evaluation
only, never trained on).

## 2. Where the code lives [V]

Private repo `github.com/snigenigmatic/ast-asr`. Four branches, all pushed.

| Branch | Worktree | Contents |
| --- | --- | --- |
| `master` | — | old baseline |
| `ast-adversery` | `C:/Kaustubh/ast-asr` | April-era code **plus a large uncommitted WIP**: an MWER-vs-GRPO "ladder" refactor, 105 tests green. Parked, not deleted. |
| `codex/fair-cispo-tiny` | `C:/Kaustubh/ast-asr-worktrees/fair-cispo-tiny` | The rigorous FR-CISPO framework (`src/ast_asr/`). **A parallel Codex agent works here and on the same Modal account** — re-check `git log` and `modal app list` before assuming status. |
| `fair-cispo-work` | `C:/Kaustubh/ast-asr-worktrees/fair-cispo-work` | Branched off codex. Adds the SPIRE cross-corpus workstream. Tip `d7e26de`. 137 tests green. |

Key documents on `fair-cispo-work`:

- `docs/plain-language-walkthrough.md` — the method explained against the code.
- `SESSION_CHECKPOINT_2026-08-12.md` — running status log with commands.
- `experiments/SPIRE-crosscorpus/` — `protocol.md`, `arms.md`, `audit-20260820.md`,
  `result-20260824.md`, `result-20260825.md`.
- `experiments/{H1,H5,H6,H7}-*/` — protocols, results, and failure records.
- `research-state.yaml` — **the live source of truth for hypothesis states.**
- `docs/legacy/` — the pre-rejection paper plan and bug diagnosis, with a staleness note.

**A remote agent should use `fair-cispo-work` and nothing else.** It is a strict
superset of every other branch's committed work (15 commits ahead of remote codex,
0 behind), so every file referenced in this document is reachable from it.

## 3. History: what was fixed and why it matters [V]

An early run ("v5", commit `7fbdcd6`) reported WER 18.4% → 16.4% and looked like a
fairness win. Two real bugs were later found (documented in
`docs/legacy/RL_TRAINING_FIXES.md`):

1. **The fairness reward cancelled itself.** Under group-relative advantage
   normalization the family-need weight divides out exactly, so the intended
   fairness signal contributed nothing. The v5 gain was more plausibly caused by
   the family-balanced sampler.
2. **The importance-ratio axis was inert.** With one optimizer step per rollout
   batch, `pi_theta / pi_old == 1` identically, so any PPO/CISPO clipping
   machinery measured nothing.

**Do not anchor any new claim on the 16.4% number** — the evaluation protocol
differed (`docs/prior-results-benchmark.md`).

The `codex/fair-cispo-tiny` framework fixes both: group weighting is applied
*outside* candidate centering with a mean-one invariant (cannot cancel), and each
rollout batch gets **four inner updates against a frozen reference** so ratios are
genuinely live — enforced by a check that the ratio is ~1 at update 0 and has
moved by update 1. It also adds hard safety stops (per-cycle KL/token < 0.1,
ratio p99 < 2.0), FP32 evaluation invariance (an earlier FP16 bug made greedy
decoding depend on batch shape; solo-vs-batch now verified byte-identical on
5,772 predictions), pre-registered protocols, and immutable failure records.

## 4. Evidence ledger: what the RL programme actually produced [V]

From `research-state.yaml` and the committed result documents.

| Hypothesis | Status | Substance |
| --- | --- | --- |
| **H0** | **refuted** | No candidate learning rate survives 300 cycles inside the KL limit. 1e-5 → KL/token **1.16** (11.6x over); 3e-5 → **43.99**; 1e-4 → tripped a 10-cycle probe (ratio p99 2.48, KL 0.30). |
| **H1** | safe short-horizon signal | 20 cycles, ratio p99 1.173, peak KL 0.0092, 5/32 probe predictions changed. Exploratory only. |
| **H5** | mechanism supported, efficacy unconfirmed | Matched 40-cycle beta=0 vs beta=0.04, seed 2026. Peak KL 0.0249 → 0.0164 (penalty works). Worst family-by-condition improved **2.22 pp**, but a 10,000-sample bootstrap over 39 profile clusters included zero/harm for every registered delta. |
| **H6** | **failed_safety** | Replication failed. Seed 2027: worst group **worsened 0.40 pp**. Seed 2028: beta=0 control tripped KL **0.1110 at cycle 27** and stopped; its treatment was never launched per the locked stop rule. Three-seed rule not evaluable. |
| **H7** | **measurement_failed (terminal)** | A diagnostic to explain the seed-2028 stop. Its sole authorization was consumed by an infrastructure failure, no retry permitted. |
| **H8** | **DRAFT, not approved** | A matched clean-vs-noisy SFT test. Gated behind implementation, tests, and independent review before one seed runs. |

**Net:** roughly six weeks of careful, safety-gated work; four hypotheses; **no
publication-valid result on Svarah.** Everything there carries
`publication_valid: false` because the authoritative 117 Svarah speaker IDs are
unobtainable, so folds use 115 *demographic profile clusters* — and a profile
cluster is not a speaker.

### The structural bind (important for any "just train longer" proposal)

- Train hard enough to move WER → the policy drifts and trips the KL gate.
- Stay inside the gate → almost no movement (an early low-LR calibration moved
  greedy WER by a single deletion, 533 → 532 errors).
- Escape only by raising the KL ceiling — which is what produced an earlier
  blow-up (KL 1951 by step 80) and discards the safety contract that makes the
  numbers trustworthy.

**A longer run is not an untried idea. H0 *was* that experiment, at 300 cycles,
for all three learning rates, and it failed.**

### Two methodological findings worth carrying into any paper [V]

1. **Worst-group endpoints are unreliable when groups are near-tied.** On SPIRE the
   two family gaps are 0.56–2.33 pp and the "worst family" label *flips* between
   Indo-Aryan and Dravidian across arms. A worst-group delta then estimates
   something close to a coin flip. Check group separation before adopting such an
   endpoint.
2. **Bitwise cross-platform tensor contracts are unachievable.** H7 died on
   `H7 input-lock noisy waveform hash mismatch`: the lock froze *bitwise* hashes of
   a `torch.randn` + SNR-mixed waveform, built on Windows/`torch 2.13.0+cpu` and
   executed on Linux/`torch 2.11.0+cu128`. Seed, SNR, row identity and source audio
   were all ruled out; the clean tensor hash matched, only the noisy one differed.
   Fix by persisting noise tensors as data, or verifying within a numerical
   tolerance, or using a platform-stable RNG. (A prior H7 attempt died separately
   on a Windows console encoding bug — `'charmap' codec can't encode '\u2713'` —
   fixed by `PYTHONIOENCODING=utf-8`, verified.)

## 5. The banked positive result: SPIRE-SIES cross-corpus [V]

This is complete, committed, and pushed. It is the one defensible positive result
currently in hand, and it cost no further training.

**Why it is different.** SPIRE-SIES ships **real speaker identifiers and a
ready-made speaker-disjoint split** (1126 train / 198 val, zero overlap). So the
grouping unit is a genuine speaker, not a profile cluster. It is used for
**evaluation only** and was never trained on.

**Split audit** (`audit-20260820.md`): 198 declared = **198 observed** validation
speakers; **15.3192 h**; 5,214 accepted utterances of 35,299 scanned; families
exactly {Dravidian 7.2961 h, Indo-Aryan 8.0231 h}; **zero contract violations
across all 35,299 rows** (each row's split label agreed with `splits.json` and
every language resolved against the repo taxonomy).

**Disclosed constraints:** only **13 of 17** shipped languages appear in the
validation split (Dogri, Gujarati, Maithili, Sindhi contribute none); the
per-language tail is severe (**Kashmiri = 2 utterances**, Nepali 34, Punjabi 50);
**no Sino-Tibetan** (so a 2-family contrast, versus Svarah's 3); **age is
Unknown corpus-wide**; clean read speech only.

### The six-arm ladder

5,214 utterances, 198 real speakers, 106,057 reference words, clean, greedy FP32,
identical manifest. All adapter revisions were verified against the published
H5/H6 evidence *before any number was read*.

| Arm | overall | vs base | vs SFT | family gap | gender gap | worst-20% spk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `zero-shot` | 29.89% | — | — | 1.70 pp | 11.46 pp | 85.28% |
| `sft-epoch1` | 26.10% | −3.79 | — | 0.95 pp | 6.52 pp | 65.57% |
| **`h5-beta0-s2026`** | **24.79%** | **−5.10** | **−1.32** | 0.56 pp | 6.69 pp | **61.41%** |
| `h6-beta0-s2027` | 25.50% | −4.39 | −0.60 | 0.75 pp | 6.93 pp | 65.02% |
| `h6-beta004-s2027` | 25.50% | −4.39 | −0.60 | 0.67 pp | 6.71 pp | 64.76% |
| `h5-beta004-s2026` | 25.60% | −4.29 | −0.50 | 2.33 pp | 7.70 pp | 65.97% |

"gender gap" = Male WER minus the better of Female/Other; Male is worst in every
arm. The `zero-shot` arm was validated as a true baseline: with no checkpoint the
loader attaches a *fresh* LoRA, which is an identity only because all **48
`lora_B` tensors are zero-initialised** (checked directly, so `delta W = 0`).
Re-verify if `peft` is upgraded.

### Three findings

1. **The pipeline transfers, and helps the worst-off most.** Overall −5.10 pp
   (17% relative). Worst-20%-speaker WER **85.28% → 61.41% = −23.87 pp**, i.e.
   **4.7x the average gain**. Gender gap roughly halves (11.46 → ~6.5 pp).
2. **The fairness term is inert.** Within each seed `beta=0.04` never beats its
   control; `h5-beta004` is the *worst* RL arm while its own control is the *best*.
   Paired speaker-clustered bootstraps (10,000 resamples, 198 real clusters):
   worst-family delta **+1.69 pp [−1.17, +7.27]** (seed 2026) and **−0.03 pp
   [−1.98, +1.72]** (seed 2027). Neither excludes zero → **no demonstrated
   benefit**, not established harm. So the equity gains come from the *pipeline*,
   not the fairness objective — the cross-corpus, real-speaker confirmation of the
   original reward-cancellation finding.
3. **Family is the least informative axis, at every stage.** Ordering
   **speaker >> gender >> family** holds in the **base model too** (family 1.70 pp,
   gender 11.46 pp, worst-20% speakers 55 pp above overall), so it is a property of
   the corpus and task, not induced by training. The objective optimizes worst
   *family*. Note this does not contradict Svarah, where family gaps were large
   (41 pp zero-shot, 14 pp after SFT) — family-gap prominence is corpus dependent.

### Direction reversal worth noting

The same SFT checkpoint **failed** its clean-regression constraint in-domain on
Svarah (clean 22.08% → 24.17%) yet **improves** clean WER by 3.79 pp
out-of-domain. Fitting Svarah harder cost in-domain accuracy while producing a
better-generalizing model.

### Known holes in this result

1. **No continued-SFT control matched on optimizer steps.** So the SFT → RL gain
   (−0.50 to −1.32 pp) is not yet attributable to the RL *objective* rather than
   simply more optimization. Both beta arms improve similarly, consistent with a
   generic RL-stage effect. **This is the weakest joint in the positive half.**
2. **No bootstrap intervals on the ladder contrasts.** Only the beta pairs have
   intervals. The base/SFT/RL contrasts are point estimates on a fixed 198-speaker
   set — large, but not interval-estimated.
3. Clean speech only; no robustness claim. Two families, no age axis. One
   generalization corpus. Checkpoints still carry profile-cluster *training*
   provenance even though the *evaluation* grouping is a real speaker.

## 6. The proposed pivot [R — from the strategy handoff, unverified by me]

The recommendation from the project guide is to **pivot away from racing new RL
results** and instead ship a **fairness audit** of existing systems, using the RL
work as motivating evidence ("aggregate WER hides subgroup movement") rather than
as the contribution.

Claimed context — please verify each of these:

- **The competitive gap:** Sarvam's Saaras V3 reports a single aggregate number,
  with no subgroup breakdown and no significance test.
- **Scope:** 11 planned systems. **Six need zero new training** (checkpoints
  already on disk: `ft-w2v2`, three `hybrid-lam` GRL variants,
  `hybrid_w2v2_lora_grl`, `whisper-rl-fair`).
- **Known bug:** the `rl-grpo` registry entry points at a checkpoint directory
  that no longer exists.
- **Adding API systems is clean.** `asr_inference.py`'s registry already branches
  on a `model_type` string to build an inference closure, so an `"api"` branch
  (Saaras V3, Qwen3-ASR) slots in without touching `pipeline.py`.
- **Blocker A (per-family rate ratios + standard errors) is nearly free.** The
  Poisson GLM already computes per-group log-rate coefficients internally; they are
  simply never exponentiated into rate ratios or paired with confidence intervals
  before being discarded.
- **Blocker B (train/eval sentence-overlap check) must be built from scratch.** The
  Svarah loader exposes raw reference text but no sentence/transcript ID, and the
  existing split verification only checks speaker/UID disjointness, never text.
  Proposed approach: group by exact reference string across the committed split.
- **The current significance test is not naive.** It already uses cluster-robust
  standard errors clustered on real per-utterance speaker IDs — a legitimate
  partial answer to the speaker-random-effects critique, though not literal random
  intercepts. `statsmodels` is already a locked dependency, so a true mixed model
  is feasible.
- **Code drift since the Phase-2 report:** the CLI flag is `--model`, not
  `--model_type`; `compute_fairness_summary()` was split into separate
  `delta_dp` / `delta_eo` / `delta_noise` / `poisson_significance` calls. Cosmetic.
- **Unverified dependency:** Qwen3-ASR support in the currently pinned
  `transformers` version.
- **A stretch idea:** the noise-aware training curriculum (targeting the Δnoise
  gap, reported as +6.6 pp Dravidian / +15.3 pp Indo-Aryan / +16.9 pp
  Sino-Tibetan) is already implemented and unit-tested but never enabled.
- **Deadline:** reported as ~5 days. Confirm.

## 7. Options on the table, with trade-offs

### A. Ship the fairness audit (the guide's recommendation)
*Multi-system subgroup breakdown with significance testing, including API systems.*
- **For:** no training risk; directly targets a real gap in how competitors report;
  six systems need no training; Blocker A is nearly free; deadline-feasible.
- **Against:** Blocker B is from scratch; Qwen3-ASR support unverified; needs API
  access and budget; the contribution is an audit of others' systems, which some
  venues weight lower than a method.

### B. Longer RL training run
- **Verdict: not viable.** H0 already ran this at 300 cycles for all three learning
  rates and it was refuted; even 40 cycles is seed-unstable (2028 halted at cycle
  27). Returns are also small: SFT gave −3.79 pp while the whole RL stage adds
  −0.50 to −1.32 pp, and the spread *between* RL arms (0.81 pp) is about the size
  of the gain. Scaling a small, noisy effect whose fairness mechanism is provably
  inert does not produce a better WER story.

### C. Enable the noise-aware curriculum (stretch)
- **For:** already implemented and unit-tested; targets a real measured Δnoise gap;
  could yield a fresh positive result.
- **Against:** it is a training run, and this cycle is 0-for-4 on training runs. Its
  stated purpose — "one genuine positive result rather than only an audit" — is
  **already satisfied** by the SPIRE ladder, at zero further compute. Time-boxed
  bonus at best; must not be load-bearing.

### D. Continued-SFT control on the SPIRE split
*One training run matched to the RL arms on optimizer steps, then one eval.*
- **For:** closes hole #1, the weakest joint in the only positive result. Small and
  bounded (~$1–2 estimated). Directly pre-empts the question a reviewer will press
  hardest: "is this the RL objective, or just more optimization?"
- **Against:** still a training run. If continued-SFT matches or beats the RL arms,
  it weakens the RL narrative — though that is an honest, publishable outcome and
  arguably strengthens the "gains are not from the fairness objective" thesis.

### E. Statistics: true mixed-effects vs cluster-robust
- **Recommendation:** ship cluster-robust, clearly labelled. It is defensible, and
  the SPIRE bootstrap also clusters on real speakers, so both halves of the paper
  are methodologically consistent. Upgrade only if a reviewer asks.

### F. Cheap extensions to the banked SPIRE result
- **F1 — bootstrap intervals on the ladder contrasts.** Pure CPU post-processing on
  predictions already saved to the volume; essentially free. Closes hole #2.
  **Highest value-per-cost item on the board.**
- **F2 — noise conditions on SPIRE.** Adds a robustness axis cross-corpus; costs GPU.
- **F3 — a third corpus.** Highest credibility gain, clearly out of scope for a
  5-day window.

## 8. My assessment

Load-bearing: **A** (the audit), because it is the only option with no training
risk on a hard deadline. Add **F1** because it is nearly free and upgrades the
banked positive result from point estimates to interval estimates. Consider **D**
only if the audit lands early. Skip **B** entirely and treat **C** as abandoned
unless the audit finishes with days to spare.

The important reframing: **you are not pivoting empty-handed.** The SPIRE ladder is
a finished, committed, real-speaker result showing the pipeline cuts WER 29.89% →
24.79% and cuts the worst-20% speakers' WER by 23.87 pp. That *demonstrates* the
"aggregate WER hides subgroup movement" argument rather than asserting it — which
is exactly the gap a single aggregate number cannot cover.

A candidate thesis the current evidence supports:

> A supervised-plus-RL post-training pipeline delivers large, cross-corpus,
> speaker-level equity gains on Indian-English ASR — but the explicit fairness
> mechanism it contains contributes nothing measurable, and the language-family
> group axis it optimizes is the least informative axis available. Aggregate WER
> conceals all of this.

## 9. Open questions for the brainstorm

1. **What is the contribution?** The audit, the SPIRE cross-corpus equity finding,
   or both? If both, which leads?
2. **Which group axis does the paper foreground?** The evidence says speaker tail
   and gender dominate family by an order of magnitude — but the existing pipeline,
   prior report, and much of the literature are built around family.
3. Mixed-effects vs cluster-robust (see E).
4. Is exact-reference-string matching sufficient for Blocker B, or is
   near-duplicate detection needed?
5. Does Saaras V3 API access exist, and what does it cost per hour of audio?
6. What is the real venue and deadline, and does it weight audits vs methods?
7. Do the parallel threads stay frozen? Specifically: the `ast-adversery` ladder
   WIP, and the Codex agent's H8 draft. If a Codex agent is still active it may
   move `codex/fair-cispo-tiny` under you.
8. Is the Δnoise gap worth foregrounding as future work, given the curriculum is
   built but untested in anger?

## 10. Practical notes and traps [V]

**Environment.** `uv sync --extra dev` for pytest. On Windows `torch` resolves to
the CPU wheel, so local runs are CPU-only; all GPU work is on Modal.

**Hard rule.** Never download a large corpus to the laptop — it has crashed the
machine. All heavy data and compute runs on Modal into a Modal volume.

**Modal.** Volumes: `ast-asr-cache`, `ast-asr-data`, `ast-asr-fr-cispo-runs`,
`ast-asr-spire`. Secret `huggingface` holds `HF_TOKEN`. Budget: free tier, roughly
$15 remaining — **verify on the dashboard**, and confirm zero live tasks with
`modal app list` before launching.

Traps that have already cost time:

- Always set `PYTHONIOENCODING='utf-8'` before `uvx modal ...` on Windows, or the
  launcher dies rendering a tick mark.
- **Do not pipe `modal run` through `Select-Object`** — PowerShell buffers the whole
  stream and it looks like the job produces no logs. Use
  `modal app logs <app-id> --tail 200` or `-f`.
- Anything importing `ast_asr` on Modal must run via `uv run`; the container
  interpreter has no project install. This killed a comparison job once with
  `ModuleNotFoundError: No module named 'ast_asr'`.
- `ast_asr/__init__` imports `.objectives`, which imports torch — so any
  lightweight CPU path must avoid importing `ast_asr`.
- **Do not add files to `src/` or `configs/`** while any H7 authorization is live:
  those trees' SHA-256 manifests are frozen. The whole SPIRE workstream lives in
  `scripts/` for this reason.
- The audited H5 beta-zero control is the **`-r1-`** run
  (`profile-h5-refkl-beta0-r1-s2026-20260812`); the non-r1 directory has no adapter
  output at all. Policy runs expose `checkpoint-last-safe`, not `checkpoint-final`.
- Score WER with `ast_asr.metrics.normalize_for_wer` — **not** Whisper's
  `EnglishTextNormalizer` and **not** `spire_loader.normalize_transcript` (an
  uppercase CTC leftover) — or numbers stop being comparable across the project.

**Reproducing the SPIRE result:**

```powershell
$env:PYTHONIOENCODING='utf-8'
uvx modal run scripts/modal_spire_eval.py::compare_spire_arms `
  --control-run-name spire-eval-h5-beta0-20260820 `
  --treatment-run-name spire-eval-h5-beta004-20260820
uvx modal volume get ast-asr-spire /eval/<run>/metrics.json -
```

## 11. Working principles that have paid off

- Pre-register the protocol, then run, then record — including failures. Failed
  runs are committed as evidence, never deleted.
- Verify provenance before reading numbers. Every SPIRE arm's adapter revision was
  checked against previously published hashes first.
- Prefer an honest null to a flattering claim. Both beta intervals include zero, so
  the result is stated as "no demonstrated benefit", not "harm".
- State what a result does *not* license, explicitly, in the result document.
