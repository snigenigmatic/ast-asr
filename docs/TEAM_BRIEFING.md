# Project update — Kaustubh

Update on the fairness-ASR work and where I'd like the team to pick things up. (You already know the
project premise; this is what's changed and what's next.)

**Live status is always `research-state.yaml` on `codex/fair-cispo-tiny`** — that file is the source of
truth for hypothesis states. This document is the narrative.

## What I've done recently

- **Rebuilt onto a clean, tested framework (FR-CISPO).** The earlier exploratory code had two real bugs —
  the fairness signal cancelled itself out under group-relative normalization, and the CISPO/PPO importance
  ratio was inert (one update per rollout, so it measured nothing). The rebuild fixes both: a frozen SFT
  reference plus several inner updates per rollout makes the ratio genuinely live, and group weighting is
  applied *outside* candidate centering so it cannot cancel. Pre-registered protocols, hard safety stops
  (KL and importance-ratio limits that halt training automatically), and **135 tests green**.
- **Ran a matched, pre-registered 3-seed experiment (H5 + H6).** The honest result:
  - The β=0.04 reference-KL penalty **reliably reduces divergence from the base model** — the *mechanism*
    works on both seeds where it ran.
  - **The fairness benefit did not replicate.** Seed 2026 improved the worst group by 2.2 pp; seed 2027
    *worsened* it by ~0.4 pp; seed 2028 **tripped the KL safety gate at cycle 27 and stopped itself.**
  - Separately, **no tested learning rate can train long-horizon (300 cycles) inside the safety limits.**
- **Diagnosed why, twice, without faking a result.** A follow-up diagnostic (H7) was designed to test
  whether that safety trip was real instability or a measurement artifact. Both attempts failed *before
  measuring anything*, and both were recorded as non-results rather than quietly retried:
  1. a Windows console encoding bug in the launcher (now fixed and verified);
  2. the diagnostic froze **bitwise** hashes of a generated noise waveform, but the lock was built on
     Windows/CPU torch and executed on Linux/CUDA torch — and PyTorch does not guarantee bitwise identical
     results across versions and platforms, so it fail-closed on a hash mismatch.
- **Built the SPIRE-SIES cross-corpus workstream.** 102 h of Indian-English with **real speaker IDs** and a
  ready-made speaker-disjoint split (1126 train / 198 val). Written this week: a pre-registered protocol,
  the scoring/bootstrap logic (33 tests), and a Modal-only preparation job. Evaluation-only; never trained on.

## What this means (the honest read)

We do **not** have a clean "our method improves fairness" result, and I am not going to manufacture one.
What we do have is a rigorous **characterization**: a mechanism that works, an effect that does not robustly
replicate, real stability limits, and a concrete reproducibility lesson about RL-for-ASR experiments. That is
publishable, and it is exactly the empirical rigor the earlier rejected paper lacked.

The strongest remaining upside is **cross-corpus generalization on real speakers** (SPIRE), because every
Svarah result is capped at `publication_valid: false` — Svarah's authoritative 117 speaker identities are
unobtainable, so its folds use demographic *profile clusters*, which are not speakers.

## Decisions locked

- **Honesty over hype** — a rigorous negative or nuanced result beats an overclaimed positive. Failed runs
  are committed as evidence, not deleted.
- **Never present heuristic speaker groups as authoritative.** SPIRE's real IDs are the only defensible
  route to a speaker-level fairness claim.
- **Cloud-only heavy compute (Modal).** No large data or training on personal machines.
- **Pre-register before running.** Protocol first, then execute, then record — including failures.

## Next steps

1. **SPIRE cross-corpus evaluation** — cheap metadata audit, then materialize the 198-speaker val split, then
   evaluate the existing checkpoints on it. All on Modal.
2. **Re-design the H7 diagnostic contract** so it is portable: persist the noise tensors as data, or verify
   with a numerical tolerance, instead of asserting cross-platform bitwise equality.
3. **Decide the paper framing** from the cross-corpus evidence plus the two limit findings, then write.

## Where I'd like help

- **Cloud/ML eng:** own the Modal jobs (SPIRE prepare + evaluate). Scoped and ready.
- **Stats / methodology:** seed-instability and honest uncertainty are the crux — make the speaker-clustered
  bootstrap and the reporting bulletproof. Reviewers will attack exactly this.
- **Reproducibility:** help fix the H7 lock design, and independently re-run the suite plus one experiment.
- **Writing / lit review:** related work, and a skeleton for the characterization framing.

Deeper detail: `docs/plain-language-walkthrough.md`, `SESSION_CHECKPOINT_2026-08-12.md`,
`experiments/H6-replication/result-20260812.md`, `experiments/H7-sentinel-kl/failure-r1-20260820.md`,
`experiments/SPIRE-crosscorpus/protocol.md`.
