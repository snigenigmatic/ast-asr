# FR-CISPO research log

This file is append-only. It records decisions and failures, not just successful
runs. Engineering smoke tests using non-authoritative speaker identities must be
labelled as such.

## 2026-08-10 — Data identity audit

- Svarah audio and metadata could be resolved to 6,656 utterances.
- The available fallback formed 115 demographic profile clusters; it did not
  recover the dataset's documented 117 speaker identities.
- Decision: retain these manifests for engineering only. Do not use them for
  publication claims, speaker bootstrap confidence intervals, or final folds.

## 2026-08-10 — SFT baseline

- Trained five epochs of rank-8 Whisper-tiny LoRA SFT with dropout disabled.
- Selected epoch 1 by validation macro-family WER (0.1766945).
- Reloaded checkpoint reproduced its saved predictions exactly.
- Corrected FP32 development evaluation later showed a trade-off rather than a
  general win: clean WER worsened by 2.09 points, white-noise WER improved by
  4.94 points, and the worst family-condition WER worsened by 22.78 points.

## 2026-08-10 — FR-CISPO learning-rate gate

- 1e-4 failed in a 10-cycle probe: ratio p99 2.477 and KL 0.2958.
- 3e-5 completed 300 cycles but failed the trust region: KL 43.9888; the first
  recorded crossing of 0.1 occurred at cycle 10.
- 1e-5 completed 300 cycles with ratio p99 1.2069 but KL 1.1638.
- All runs were finite, changed adapters and predictions, and reloaded exactly.
- Decision: none of the preregistered learning rates is development-safe. Do not
  proceed to five-fold training.

## 2026-08-10 — Evaluation invariance repair

- FP16 Whisper decoding produced one deterministic solo-versus-batch mismatch
  in an eight-example diagnostic, even though features and masks were identical.
- Casting the model to FP32 restored equality; evaluation now does this and has
  regression coverage plus a diagnostic CLI.
- Recomputed zero-shot and SFT predictions in FP32; both now pass solo/batched
  equality across 5,772 predictions.

## 2026-08-11 — Research program reset

- Preserved the original objective: fairness and robustness adaptation of
  Whisper-tiny for Indian English.
- Split the next work into three independent streams: authoritative speaker
  recovery, policy-stability instrumentation, and a bounded execution/publication
  runbook.
- Decision: the next paid run must be preceded by a frozen exploratory protocol.
  A failed gate is evidence and will be reported rather than triggering an
  unrecorded method change.

## 2026-08-11 — Authoritative identity recovery result

- Audited every reachable official Svarah GitHub commit and all seven
  data-bearing Hugging Face revisions.
- The GitHub README documents an original `meta_speaker_stats.csv` with
  `speaker_id`, but the file is absent from reachable Git history. Every public
  Parquet schema omits `speaker_id`.
- Hardened the storage gate: a filename is insufficient; a candidate table must
  contain 117 non-empty distinct IDs and pass later utterance alignment checks.
- Decision: publication-valid speaker folds remain externally blocked. Continue
  only with explicitly non-publication engineering diagnostics while requesting
  the original mapping from the maintainers.

## 2026-08-11 — Policy safety instrumentation

- Identified that KL was checked only after the complete run and that a
  per-cycle-looking field stored the running maximum.
- Added current-cycle KL, running maximum KL, full ratio distributions including
  the state after the fourth update, group-risk/weight trajectories, and a
  deterministic source/config hash.
- Added hard stops for KL `>= 0.1`, ratio p99 `>= 2`, and non-finite ratios. A
  failed later cycle restores and saves the immediately preceding safe adapter
  state without emitting `checkpoint-final`.
- Decision: review and freeze the 20-cycle H1 hard-stop pilot before launching a
  new Modal task. The controller does not change the FR-CISPO objective.

## 2026-08-11 — H1 bounded engineering pilot

- Froze the protocol in commit `2cb4312` and launched the single registered
  Modal run `profile-h1-klstop-s2026-20260811/h1-klstop-fr-cispo`.
- Completed all 20 cycles. Peak ratio p99 was 1.1731905; peak sampled K3 KL was
  0.0091612 at cycle 14; every update-zero ratio distribution was exactly one;
  adapter drift was 0.1139263; and 5 of 32 probe predictions changed.
- The checkpoint reloaded exactly. The source/config hash in the run matched a
  local recomputation using the exact invoked remote config.
- Protocol deviation: the successful bounded checkpoint was stored as
  `checkpoint-final` rather than the required `checkpoint-last-safe`. The
  original artifact remains untouched and is labelled engineering-only.
- Decision: H1 provides a positive short-horizon safety signal but does not
  select a learning rate, establish WER gains, unlock three seeds, or authorize
  five folds. Repair bounded-checkpoint naming before any later run.

## 2026-08-11 — H1 FP32 checkpoint evaluation

- The first registered Modal invocation failed before model loading because the
  remote wrapper imported `ast_asr` outside the project environment. The failed
  app, exact exception, and absence of output artifacts were preserved.
- A separately preregistered launcher-only recovery completed 5,772 predictions
  and passed FP32 solo-versus-batched equality with the exact historical adapter
  hash.
- Clean WER was 0.216564, white-10-dB WER 0.547073, MUSAN-babble WER 0.589434,
  and worst family-condition WER 1.034274.
- Against matched SFT, clean improved by 2.52 points and worst family-condition
  improved by 42.74 points. Against zero-shot, unseen MUSAN worsened by 1.62
  points while clean, white noise, worst-group, and tail metrics improved.
- Decision: this is a useful single-seed, 20-cycle engineering signal. Select
  H5 explicit SFT-reference KL as the next bounded training hypothesis; do not
  unlock three seeds, five folds, or a publication claim.

## 2026-08-12 — H5 matched fixed-reference KL experiment

- Froze H5 before execution as a matched 40-cycle beta-zero control versus a
  beta `0.04` sampled-K3 SFT-reference penalty. Seed, SFT checkpoint, data,
  batches, LR `1e-5`, candidates, inner updates, and decoding were identical.
- The first beta-zero invocation failed before cycle zero because the Modal
  wrapper imported Torch outside the project environment. Preserved the failed
  app as launcher-only evidence, committed a one-retry recovery protocol, and
  moved the probe under `uv run --frozen` without changing H5.
- Both matched training arms completed 40/40 cycles and 160/160 optimizer
  steps. Beta `0.04` reduced peak/cycle-40 sampled K3 from `0.0249191` to
  `0.0164290` while retaining movement in every cycle. Peak ratio p99 remained
  safe (`1.17210` control, `1.18770` treatment).
- Both immutable FP32 evaluations produced 5,772 predictions and passed
  solo-versus-batched equality. Treatment versus control changed clean WER
  `0.212756 -> 0.213422`, white-10 WER `0.515516 -> 0.521418`, MUSAN WER
  `0.564065 -> 0.581485`, and worst family-condition WER `0.852823 -> 0.830645`.
- The one-seed point estimate therefore clears the proposed two-point
  worst-group threshold and clean-regression tolerance, but aggregate noise
  robustness worsens. A 10,000-sample paired bootstrap over the 39 fold-0
  demographic-profile clusters includes zero or harm for every registered
  delta; these are not authoritative speakers.
- Decision: H5 supports the proximal mechanism and justifies drafting a locked
  two-seed replication protocol. It does not establish efficacy, unlock five
  folds, or support a publication-valid fairness claim.

## 2026-08-12 — H6 matched replication stopped on safety

- Locked H6 before execution as two additional matched seed pairs. It froze
  the H5 code/data/checkpoint hashes, prohibited performance-based early
  stopping, and required a stop if either arm failed safety.
- Seed 2027 completed safely. Beta `0.04` reduced peak sampled K3 from
  `0.0762783` to `0.0294708`, so the proximity mechanism repeated. Its primary
  WER did not: worst family-condition WER changed from `1.112903` to `1.116935`.
  Clean and white WER improved slightly, while MUSAN WER worsened by 1.49
  points.
- The seed-2027 control Modal client reported a local `OSError(22)` after the
  remote app had completed and committed all artifacts. Independent audit
  accepted the run and did not retry it.
- Seed 2028 beta zero failed closed at cycle 27 when sampled K3/token reached
  `0.1109513`, above the registered `0.1` limit. It saved the preceding
  last-safe adapter. The beta-`0.04` treatment and both seed-2028 evaluations
  were not launched.
- Trajectory audit shows the seed-2028 control first exceeded `0.05` at cycle
  13, reached `0.0999123` at cycle 23, dipped, and crossed the limit at cycle
  27 while ratio p99 remained small. This is cumulative reference drift, not a
  ratio-cap failure.
- Decision: H6 fails its safety prerequisite. The fixed three-seed efficacy
  rule is not evaluable, no bootstrap/mean gate is computed, and no additional
  paid policy run is authorized without a new question and locked protocol.
