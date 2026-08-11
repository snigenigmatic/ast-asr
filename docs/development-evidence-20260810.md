# FR-CISPO development evidence — 2026-08-10

## Scope limitation

These runs use 115 deterministic demographic-profile clusters because the
cached public Svarah Parquet release does not expose authoritative speaker IDs.
They are pipeline and method-development evidence only. They do not establish
speaker-disjoint research results and must not be combined with future
authoritative 117-speaker runs.

## Revisions and data

- Model: `openai/whisper-tiny` at
  `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`.
- Svarah: `ebbf7777fe771490696a3f7b007097606fa8c924`.
- Supplied metadata SHA-256:
  `e2daa48863581eb41befd1826b7b14cd80e05f3a1bab9b72d08e7814248f1f94`.
- MUSAN: official OpenSLR archive, MD5
  `0c472d4fc0c5141eca47ad1ffeb2a7df`; 426 speech WAVs extracted.
- Fold 0 contains 1,924 held-out utterances; every evaluated arm produced
  5,772 predictions over clean, white-noise 10 dB, and MUSAN-babble 10 dB.

## SFT checkpoint selection

Five epochs completed with exact checkpoint round-trip predictions. Epoch 1
was selected by validation macro-family WER (`0.1766945`). Later epochs reduced
training loss but did not improve the selection metric.

Selected adapter revision:
`d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e`.

## Policy learning-rate gate

| Learning rate | Scope | Ratio p99 | KL/token | Result |
| ---: | --- | ---: | ---: | --- |
| `1e-4` | 10-cycle probe | 2.4770 | 0.2958 | fail |
| `3e-5` | 300 cycles | 1.8218 | 43.9888 | fail |
| `1e-5` | 300 cycles | 1.2069 | 1.1638 | fail |

All runs remained finite, skipped no steps, moved all rollout ratios, changed
greedy predictions, and reproduced predictions after checkpoint reload. They
fail because KL/token exceeds the preregistered `0.1` ceiling. At `3e-5`, the
cumulative maximum first crosses `0.1` at cycle 10. No learning rate is
selected, and the failed policy checkpoints are not candidate results.

## Corrected held-out baseline evaluation

FP16 greedy decoding of the SFT adapter differed for one of eight utterances
between solo and batch-8 inference. The processor features and attention masks
were bit-identical, batch repeats were deterministic, and FP32 restored exact
solo/batched equality. Both arms below were therefore rerun entirely in FP32.

| Metric | Zero-shot | SFT epoch 1 | SFT delta (WER points) |
| --- | ---: | ---: | ---: |
| Clean overall WER | 0.2208 | 0.2417 | +2.09 |
| White 10 dB overall WER | 0.6218 | 0.5725 | -4.94 |
| MUSAN babble 10 dB overall WER | 0.5733 | 0.6073 | +3.41 |
| Worst-family clean WER | 0.6028 | 0.3649 | -23.79 |
| Clean family gap | 0.4130 | 0.1405 | -27.25 |
| Worst family × condition WER | 1.2339 | 1.4617 | +22.78 |
| Worst-20%-speaker WER | 0.4475 | 0.5540 | +10.65 |

SFT improves the seen white-noise condition and substantially narrows the clean
family gap, but it degrades clean overall WER, unseen MUSAN robustness, the
primary worst-group endpoint, and the worst-speaker tail. This seed therefore
does not satisfy the proposed development gate.

## Decision

- Do not launch additional policy seeds or folds from these checkpoints.
- Do not report the failed policy checkpoints as FR-CISPO results.
- Preserve these artifacts as failure-decomposition evidence.
- Recover authoritative Svarah speaker IDs before any publication-valid fold
  experiment.
- Any change to training duration, KL control, or the dual-weight update is a
  new explicit experiment, not an automatic continuation of this protocol.
