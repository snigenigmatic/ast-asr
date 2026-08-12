# FR-CISPO findings

## Current conclusion

The project has a usable, testable Whisper-tiny training framework, but it does
not yet have publication-grade evidence for FR-CISPO. The main obstacles are
specific: authoritative Svarah speaker identities are missing, the original
learning-rate grid violates the KL safety gate, and SFT has a sharp fairness/
robustness trade-off rather than a uniformly better result.

The speaker-identity gap is now confirmed rather than merely suspected. The
official repository documents an original speaker table, but does not retain it
in reachable history; every accessible official Parquet revision removes the
identifier. Publication-valid speaker claims therefore require an artifact from
the Svarah maintainers, not further inference from filenames or demographics.

## What is real so far

The corrected SFT checkpoint is reproducible. On the current development fold,
SFT reduces clean worst-family WER from 0.6028 to 0.3649 and white-noise overall
WER from 0.6218 to 0.5725. Those are useful signals that adaptation can change
the intended failure modes.

The same checkpoint worsens clean overall WER from 0.2208 to 0.2417, unseen
MUSAN-babble WER from 0.5733 to 0.6073, and worst family-condition WER from
1.2339 to 1.4617. It therefore fails the proposed development contract.

FR-CISPO produces genuine policy movement when the frozen rollout is reused for
four optimizer passes. This fixes the inert-ratio problem from the earlier
mu=1 implementation. However, none of the prespecified learning rates keeps
per-token KL below 0.1 for the full 300-cycle schedule. A small ratio p99 is not
sufficient evidence of global policy stability.

The runner itself also obscured this failure: it checked KL only after training
and labelled a running maximum like a cycle value. The repaired path now stops
at the first KL or ratio breach, records both cycle and cumulative diagnostics,
and preserves the adapter from immediately before a failing later cycle. This
is an observability and safety repair; it is not evidence that FR-CISPO works.

After that repair, one preregistered 20-cycle engineering pilot at `1e-5`
completed inside the trust region: peak ratio p99 1.1732 and peak sampled K3 KL
0.00916, with all update-zero ratios exactly one and 5/32 probe predictions
changing. This supports the narrow claim that genuine short-horizon movement is
possible. It says nothing yet about WER or fairness. The run also exposed an
artifact-naming deviation (`checkpoint-final` instead of
`checkpoint-last-safe`), so it cannot be promoted to a confirmatory result.

The same immutable 20-cycle adapter has now been evaluated in FP32 on all three
conditions. It improves clean WER from 0.2417 to 0.2166 and the provisional
worst family-condition WER from 1.4617 to 1.0343 versus matched SFT. It also
beats zero-shot on clean and white-noise WER, but is 1.62 WER points worse than
zero-shot on unseen MUSAN babble. This is the first useful policy-level WER
signal in the repaired pipeline, while remaining a one-seed profile-cluster
engineering result rather than evidence of general fairness or robustness.

A matched 40-cycle H5 comparison now isolates a fixed-SFT sampled-K3 penalty.
Both beta-zero and beta `0.04` runs completed safely with live ratios and exact
checkpoint reloads. Beta `0.04` reduced peak sampled K3 by 34.1% and improved
the provisional worst family-condition WER from 0.8528 to 0.8306 while clean
overall WER changed only from 0.2128 to 0.2134. This is the intended proximal
mechanism and the one-seed primary point gate is met.

The result is not a uniform robustness win: white-noise WER worsened from
0.5155 to 0.5214 and MUSAN-babble WER from 0.5641 to 0.5815. The 10,000-sample
paired interval over 39 fold-0 demographic-profile clusters also includes no
improvement for the registered worst-group delta. H5 therefore supports
replication of a fairness/average-robustness trade-off; it does not confirm
FR-CISPO efficacy.

## Invalid or provisional evidence

The current 115 profile-cluster identities are not the authoritative 117 Svarah
speakers. Any speaker-disjoint folds, worst-20%-speaker metrics, or clustered
bootstrap results based on those identifiers are engineering diagnostics only.

Earlier FP16 evaluation was not batch invariant. Correct FP32 predictions now
replace it; earlier FP16 metric tables should not be cited.

## Immediate research questions

1. Can authoritative speaker IDs be recovered from a pinned official artifact
   without inventing a heuristic mapping?
2. At what cycle and under which group/condition does KL escape, and is the
   movement gradual or caused by a localized update?
3. Does the one-seed worst-group improvement from beta `0.04` replicate over
   two additional fixed seeds without repeating the average-noise regressions?
4. If authoritative IDs remain unavailable, what narrower claim can be made
   honestly without speaker-level fairness conclusions?

## Publication stance

The defensible paper is a failure-decomposition and repaired-method study only
if the final experiment clears its stated gates. If it does not, the project
should still produce a rigorous capstone report documenting why apparent RL
gains vanished under correct ratios, masks, checkpointing, identity handling,
and evaluation precision. Venue choice must follow completed evidence, not drive
another silent scope change.
