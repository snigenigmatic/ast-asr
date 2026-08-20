# Project update — 2026-08-12 (Kaustubh)

Quick update on the fairness-ASR work and where I'd like the team to pick things up. (You already know the
project premise; this is what's changed and what's next.)

## What I've done recently
- **Rebuilt onto a clean, tested framework (FR-CISPO).** The earlier exploratory code had two real bugs —
  the fairness signal cancelled itself out under group-relative normalization, and the CISPO/PPO importance
  ratio was inert (one update per rollout, so it measured nothing). The rebuild fixes both (frozen SFT
  reference + several inner updates per rollout → the ratio is genuinely live; group weighting applied
  *outside* candidate centering so it can't cancel). Isolated branch, **66 tests green**, pre-registered
  protocols, and hard safety stops (KL and importance-ratio limits that halt training automatically).
- **Ran a matched, pre-registered experiment across three seeds (H5 + H6).** The honest result:
  - The β=0.04 reference-KL penalty **reliably reduces the model's divergence from the base** — the
    *mechanism* works on both seeds where it ran.
  - **But the fairness benefit did not replicate.** Seed 2026 improved the worst group by 2.2 pp; seed 2027
    *worsened* it by ~0.4 pp; and seed 2028 **tripped the KL safety gate at cycle 27 and stopped itself**.
    So the single promising result is seed-dependent, and even short-horizon *safety* isn't stable.
  - Separately: **no tested learning rate can train long-horizon (300 cycles) inside the safety limits** —
    a clear, useful negative finding.
- **Onboarded a second dataset (SPIRE-SIES)** — 102 h Indian-English with **real speaker IDs** and a
  ready-made speaker-disjoint split. Unlike our first corpus (no official speaker list), this supports a
  *legitimate* speaker-level fairness evaluation.
- Wrote a plain-language method walkthrough + a checkpoint doc; committed and pushed everything to the
  private repo.

## What this means (the honest read)
We do **not** have a clean "our method improves fairness" result. We have a **mechanism that works but an
effect that doesn't robustly replicate, plus real stability limits.** That is a legitimate, publishable
*characterization* — "when and why fair-RL post-training helps or fails on Indian-English ASR" — and it's
exactly the empirical rigor the earlier rejected paper lacked. We lock the final framing once the
diagnostics below are in.

## Decisions locked
- **Honesty over hype** — a rigorous negative/nuanced result beats an overclaimed positive.
- **Don't present heuristic speaker groups as authoritative** — our first corpus lacks official speaker IDs,
  so any heuristic grouping is exploratory only; **SPIRE (real IDs) is the path to a defensible claim.**
- **Cloud-only heavy compute (Modal)** — no large data or training on personal machines.

## Next steps
1. **Diagnose the safety failure (H7).** Determine whether seed 2028's KL trip is genuine instability or a
   measurement/candidate-bank artifact, *before* spending more on training. Protocol is locked; the run is
   not yet authorized.
2. **Cross-corpus fairness eval on SPIRE** — on Modal, using its real speaker-disjoint split.
3. **Decide the paper story** from the diagnostics + cross-corpus evidence, then write. Quality over speed.

## Where I'd like help
- **Cloud/ML eng:** own the Modal jobs (the H7 diagnostic runner; SPIRE materialization + eval). Scoped.
- **Stats / methodology:** seed-instability and honest uncertainty are the crux — help make the
  significance/bootstrap analysis and reporting bulletproof (reviewers will attack exactly this).
- **Writing / lit review:** related work + a skeleton for the "characterization" framing.
- **Repro check:** re-run the tests + one experiment from scratch and confirm the numbers match.

Deeper detail: `docs/plain-language-walkthrough.md`, `SESSION_CHECKPOINT_2026-08-12.md`, and
`experiments/H6-replication/result-20260812.md`.
