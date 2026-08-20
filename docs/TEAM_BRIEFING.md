# Project update — 2026-08-12 (Kaustubh)

Quick update on the fairness-ASR work and where I'd like the team to pick things up. (You already know the
project premise; this is just what's changed and what's next.)

## What I've done recently
- **Rebuilt onto a clean, tested framework (FR-CISPO).** The earlier exploratory code had two real bugs —
  the fairness signal cancelled itself out under group-relative normalization, and the CISPO/PPO importance
  ratio was inert (one update per rollout, so it measured nothing). The rebuild fixes both (frozen SFT
  reference + several inner updates per rollout → the ratio is genuinely live; group weighting applied
  *outside* candidate centering so it can't cancel). It lives on an isolated branch, **66 tests green**.
- **Ran the first real experiment (H5)** — matched β=0 vs β=0.04 reference-KL, 40 cycles. The KL safety
  mechanism works as intended, and worst-group WER improved **−2.22 pp** on one seed — **but** the paired
  bootstrap CI still includes harm and noise-robustness dipped slightly. So: **promising, not proven.**
- **Onboarded a second dataset (SPIRE-SIES)** — 102 h of Indian-English with a ready-made, clean
  speaker-disjoint split — to test cross-corpus generalization.
- Wrote a plain-language method walkthrough + a checkpoint/handoff doc, and got everything committed and
  tracked in the repo.

## Decisions I've locked
- **Replicate before claiming anything** — run more seeds, then let the data pick the paper's story
  (a working method vs. an honest trade-off analysis).
- **Defensible data split** — don't block on an unavailable "official" speaker list.
- **Cloud-only heavy compute (Modal)** — no large data or training on personal machines.

## Next steps
1. Set up the train/test speaker splits for both corpora.
2. SPIRE cross-corpus evaluation — **on Modal**.
3. Replication run (H6, additional seeds) — **on Modal**.
4. Decide the paper story from the numbers, then write it. Target ~May 2026 — quality over speed.

## Where I'd like help
- **Cloud/ML eng:** own the Modal jobs (SPIRE materialization + cross-corpus eval). Scoped and ready to go.
- **Experiments:** run the H6 replication under the locked protocol.
- **Stats:** vet the speaker-clustered bootstrap and how we report uncertainty.
- **Writing / lit review:** related work + paper skeleton.
- **Repro check:** re-run the tests + one experiment from scratch and confirm the numbers match.

Deeper detail if you want it: `docs/plain-language-walkthrough.md` and `SESSION_CHECKPOINT_2026-08-12.md`.
