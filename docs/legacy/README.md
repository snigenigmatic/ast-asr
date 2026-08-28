# Legacy documents — read for motivation, not as current instructions

These two files were untracked on a laptop and existed on no branch until
2026-08-25. They are committed here so a remote agent can read them. They are
**historical context**, and parts are stale. Do not follow their code snippets.

## What they are

- **`RL_TRAINING_FIXES.md`** — diagnoses the bugs in the "v5" GRPO run: the
  fairness-reward cancellation, double normalization, KL instability, and a broken
  probe set. This is the motivation for the entire FR-CISPO redesign.
- **`PAPER_SUGGESTIONS.md`** — the rewrite plan after the ACL SRW rejection, with
  the ranked contribution list, the v5-era numbers, and LaTeX-level fixes.

## Why they are stale

1. **Different model.** These documents describe **Whisper-small**. The current
   FR-CISPO programme and every SPIRE number use **`openai/whisper-tiny`**. Numbers
   are therefore *not* comparable across the two threads.
2. **Different code layout.** They reference `ast-asr/rl/*.py` from the
   `ast-adversery` branch. The current framework lives in `src/ast_asr/`, which is
   hash-frozen by an experiment authorization and must not be edited.
3. **The fixes were already implemented differently.** `RL_TRAINING_FIXES.md`
   proposes patches; the codex framework solved the same problems its own way
   (group weighting applied outside candidate centering, four inner updates against
   a frozen reference). Read `docs/plain-language-walkthrough.md` for what actually
   shipped.
4. **`docs/prior-results-benchmark.md` explains why the 16.4% WER figure quoted in
   `PAPER_SUGGESTIONS.md` is not a valid comparison point.** Do not cite it.

## The one claim that has since been settled

`PAPER_SUGGESTIONS.md` flags an honest caveat: the v5 WER improvement and p-value
flip "may be attributable to the balanced sampler rather than the GRPO fairness
objective."

Cross-corpus evidence now supports that caveat directly. On SPIRE-SIES with real
speaker-disjoint identities, the explicit fairness term is inert while the pipeline
as a whole produces large gains — see
`experiments/SPIRE-crosscorpus/result-20260825.md`. The caveat can be upgraded from
a hedge to a finding.
