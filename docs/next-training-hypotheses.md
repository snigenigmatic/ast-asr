# Next training hypotheses after the bounded H1 pilot

All items below are ranked by the present failure evidence, not by a desired
story. Until an official utterance-to-117-speaker mapping is obtained, every
run is profile-cluster engineering evidence (`publication_valid: false`). The
first action is the locked checkpoint evaluation in
`experiments/H1-checkpoint-eval/protocol.md`; no new training is justified
until it is recorded.

## Evidence constraining the next move

- SFT selected for validation macro-family WER improved white-10-dB WER and
  clean worst-family WER, but worsened clean overall, MUSAN, and provisional
  worst-family-condition WER.
- The original FR-CISPO loss moved live ratios at all three registered learning
  rates, but each 300-cycle run crossed the KL ceiling. At `1e-5`, peak ratio
  p99 was only 1.2069 while KL still reached 1.1638; clipping the ratio alone
  is not a global trust-region control.
- The same `1e-5` method was safe for 20 cycles (peak ratio p99 1.1732; peak
  sampled K3 KL/token 0.00916). The sharp evidence is therefore about horizon
  and accumulated drift, not an inert policy or numerical failure.

## 1. H3 — validate the KL-budgeted bounded horizon before extending it

**Claim to test.** A policy limited to a predeclared KL-safe horizon can alter
the SFT error distribution without crossing the trust region; any useful
provisional WER movement should already be visible in the existing 20-cycle
checkpoint.

| Field | Design |
| --- | --- |
| Minimal change | No new training code. Evaluate the existing immutable H1 checkpoint under the locked FP32 protocol. If and only if that shows an interpretable signal, a new protocol may use the existing hard stop to train from a fresh SFT checkpoint until the first KL boundary rather than force 300 cycles. |
| Compute | One fold-0, three-condition FP32 evaluation now; a later horizon-limited run would be one L4 run, not a learning-rate sweep. |
| Expected mechanism | The hard KL budget prevents cumulative likelihood drift while retaining the sequence-level MWER learning signal that produced 5/32 changed probe predictions. |
| Falsifier | The existing H1 adapter provides no favourable provisional movement on worst family x condition WER while respecting the clean-WER limit, or reaches the KL stop before a nontrivial retained checkpoint in a fresh run. |
| Method status | The loss, candidates, four frozen-rollout passes, ratio unit, group weighting, and corruption policy are preserved. The finite horizon is an explicit controller/method change and must be reported as such; it is not the original 300-cycle FR-CISPO contract. |

**Why first.** It extracts all information already paid for and distinguishes
“safe early movement is useless” from “the full-horizon schedule is the
problem.” It must not be expanded into a horizon sweep.

## 2. H4 — align SFT checkpoint selection with the robustness endpoint

**Claim to test.** Selecting an SFT epoch by clean validation macro-family WER
is misaligned with the study endpoint and is responsible for part of the
clean/noise trade-off before any policy update occurs.

| Field | Design |
| --- | --- |
| Minimal change | Re-evaluate the five already-saved SFT epoch checkpoints on the validation partition under deterministic clean and white-10-dB FP32 decoding. Select one epoch by a predeclared validation worst family x `{clean, white_10db}` WER subject to clean validation WER no worse than the epoch-1 value by 0.01. Do not retrain the SFT models. |
| Compute | At most five validation-only FP32 evaluations; no additional optimizer steps. |
| Expected mechanism | The selected adapter is optimized for the same worst-group/clean-regression trade-off used later, rather than a clean macro-family proxy that may ignore noise amplification. |
| Falsifier | No saved epoch meets the clean constraint and improves the validation robust endpoint, or the selected epoch fails to improve the held-out exploratory endpoint versus epoch 1. |
| Method status | LoRA architecture, training examples, optimizer, and training duration are unchanged. This explicitly changes the **model-selection criterion**, so it is a new SFT baseline protocol rather than a re-analysis of the original SFT result. |

**Why second.** The existing evidence shows a baseline-selection mismatch. A
policy method cannot convincingly repair an SFT starting point whose intended
endpoint is already worsened; this is a low-cost, interpretable correction.

## 3. H5 — impose a reference-KL proximal update for the 300-cycle objective

**Claim to test.** The 300-cycle instability is caused by unbounded cumulative
departure from the SFT reference; adding an explicit reference-KL proximal term
over the same frozen rollout tokens can maintain useful sequence-MWER updates
without relying on ratio clipping as a surrogate trust region.

| Field | Design |
| --- | --- |
| Minimal change | Add one separately configured reference-KL penalty to the policy loss, measured on the exact response masks and frozen rollout tokens already used for the sequence ratio. Retain all FR-CISPO components otherwise. Set its coefficient by a single predeclared one-cycle calibration to target a nonzero ratio movement below the existing 0.1 KL ceiling; lock the derived coefficient before the bounded run. |
| Compute | One synthetic/unit calibration plus one short (for example 20-cycle) L4 mechanism run. A 300-cycle run is forbidden until the short run passes every safety invariant. |
| Expected mechanism | Directly penalizing deviation from the SFT distribution constrains accumulated KL, whereas clipping a stop-gradient importance weight only bounds the weight in the policy-gradient estimator. |
| Falsifier | The calibrated run still crosses the KL ceiling, produces no post-update ratio movement, or improves KL only by collapsing candidate/WER diversity. |
| Method status | This **changes the FR-CISPO objective** and must be named as a proximal FR-CISPO variant, not claimed as the original sequence-CISPO method. The calibration rule and coefficient are part of the protocol, not a hidden hyperparameter sweep. |

**Why third.** This is the first justified loss-level intervention if a
horizon-controlled controller cannot produce a useful signal. It has a clear
mechanism and falsifier; random learning-rate trials below `1e-5`, new reward
terms, or changes to dual weights are not justified by the current evidence.

## Explicitly rejected next moves

- Do not rerun the `{1e-5, 3e-5, 1e-4}` grid or add lower rates merely because
  the 300-cycle runs failed.
- Do not alter family dual weights: the 20-cycle H1 run kept all group weights
  in a moderate 0.13694–0.20262 range, so there is no evidence that monopoly
  caused the KL escape.
- Do not launch multiple seeds, new folds, bootstraps, or publication claims
  until authoritative speaker IDs and the FP32/checkpoint gates are satisfied.
