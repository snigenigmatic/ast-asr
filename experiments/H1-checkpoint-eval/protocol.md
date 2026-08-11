# H1 checkpoint evaluation: bounded FR-CISPO adapter

**Status:** proposed exploratory evaluation. This protocol must be committed
before execution. It evaluates an already-existing 20-cycle engineering
checkpoint; it does not train a model, repair its historical naming deviation,
or promote the checkpoint to a completed FR-CISPO run.

## Question

Does the immutable adapter emitted by the bounded H1 KL-control pilot change
fold-0 FP32 WER relative to the already-evaluated zero-shot and SFT models?

This is a deliberately narrow, **exploratory engineering** question. The input
uses 115 demographic-profile clusters rather than the required authoritative
117 Svarah speakers. Therefore neither a positive nor a negative answer may be
reported as a speaker-disjoint fairness, robustness, or publication result.

## Frozen input identity

| Item | Required value |
| --- | --- |
| Historical policy run | `profile-h1-klstop-s2026-20260811/h1-klstop-fr-cispo` |
| Historical checkpoint directory | `checkpoint-final` (legacy name; it is an immutable bounded-smoke artifact, **not** a 300-cycle final model) |
| Expected H1 adapter content hash | `4654e4116cb1a9ebce142e3ccdc11ddd36ec3bb323c9a0ff879b6b0b3c987fe8` |
| H1 starting SFT adapter hash | `d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e` |
| Base model | `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af` |
| Dataset | `ai4bharat/Svarah@ebbf7777fe771490696a3f7b007097606fa8c924` |
| Profile metadata CSV | SHA-256 `e2daa48863581eb41befd1826b7b14cd80e05f3a1bab9b72d08e7814248f1f94` |
| MUSAN archive | MD5 `0c472d4fc0c5141eca47ad1ffeb2a7df` |
| Historical volume config | SHA-256 `94c5312c70353b1bc597ffe95c1a1f4c32166a03187e0636ddae9f2df9e6317c` |
| H1 result record | `experiments/H1-kl-control/result-20260811.md`, SHA-256 `4183ae08b71c319e5db2152fb993faaa9183b2e9d4324087c56c51b94947cf65` |

The source tree used to submit this evaluation must be committed. The current
launcher hash before this protocol commit is
`e06fa2bd00c8eaede31d2ad822a21d7d25d7f5a65c1d956497eba3aaac04cec8`;
record the post-commit Git revision and the submitted-image source hashes in
the result artifact instead of assuming this value remains current.

## Preconditions and stop rules

Do not submit the command unless all are true:

1. The historical checkpoint exists at the exact three-component Modal-volume
   path below and `directory_content_hash` resolves to the expected H1 hash.
2. The volume-side config resolves to the historical SHA-256 above. A local
   config with different bytes is not a substitute.
3. The output directory for the fixed run name is absent. Immutable writes
   intentionally make rerunning the name an error.
4. The evaluation path retains FP32 model casting and its eight-utterance
   solo-versus-batched greedy-prediction equality check.
5. The result record carries `publication_valid: false` and states that the
   115 identities are demographic-profile clusters.

Stop and record a failed evaluation if any checkpoint/config hash differs, any
condition is missing, the expected prediction count is not `5,772` (1,924
held-out utterances x 3 conditions), metrics contain a non-finite value, or the
solo/batched equality assertion fails. Do not substitute a newly trained
adapter, a renamed checkpoint, a different fold, or a different decoding
precision. A failed evaluation is a result; it does not authorize a retrain.

## Fixed execution

Run exactly once from the isolated `codex/fair-cispo-tiny` worktree, after the
protocol commit:

```powershell
uvx modal run scripts/modal_fr_cispo.py::run_profile_evaluation `
  --run-name profile-h1-checkpoint-eval-20260811 `
  --arm fr-cispo `
  --checkpoint-run-name profile-h1-klstop-s2026-20260811 `
  --checkpoint-output-name h1-klstop-fr-cispo `
  --checkpoint-name checkpoint-final
```

`run_profile_evaluation` records the supplied arm label. `--arm fr-cispo`
therefore writes `evaluation-fr-cispo` and prediction records with
`arm: fr-cispo`; zero-shot remains the only arm that uses the base model.
Every adapter arm, including SFT and policy arms, requires the three checkpoint
coordinates above. The legacy checkpoint directory is not renamed.

The command evaluates all configured conditions on fold 0:
`clean`, `white_10db`, and `musan_babble_10db`. `evaluate-fold` casts the
loaded adapter to FP32 before greedy decoding, so this command is valid despite
the legacy checkpoint directory name.

## Registered comparisons

Use identical normalization and `summarize_prediction_records` metrics. The
comparison table must include clean overall WER, white-10-dB overall WER,
MUSAN-babble-10-dB overall WER, worst-family clean WER, clean family gap,
worst family x condition WER, and worst-20%-profile-cluster WER.

| Existing exploratory reference | Clean | White 10 dB | MUSAN babble 10 dB | Worst family x condition |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot | 0.2208472 | 0.6218467 | 0.5732508 | 1.2338710 |
| SFT epoch 1 | 0.2417420 | 0.5724893 | 0.6073298 | 1.4616935 |
| H1 bounded checkpoint | **measured by this protocol** | **measured by this protocol** | **measured by this protocol** | **measured by this protocol** |

The first two rows are copied solely as engineering comparators from
`docs/development-evidence-20260810.md` (SHA-256
`9b08fe612dc680e84a15b30af4beed213694403445c32292591c22ecff1010c9`). They
are not speaker-disjoint baselines. Report absolute WER deltas to both rows;
do not choose a favourable condition after observing the outcome.

## Required result record

Store a new immutable result markdown file beside this protocol containing:

- the full command, submitting commit, Modal app/run identity, timestamps, and
  exact output-volume path;
- all input hashes above plus `run.json` checkpoint revision and the SHA-256 of
  `resolved_config.json`, `metrics.json`, `predictions.jsonl`, and
  `edit_counts.json`;
- the full registered comparison table and the exact solo/batched equality
  status;
- confirmation that the immutable output and prediction records retain
  `arm: fr-cispo`; and
- this fixed conclusion sentence: **“This is a profile-cluster exploratory
  evaluation of a 20-cycle bounded checkpoint and is not evidence of
  publication-valid fairness, robustness, or full-horizon FR-CISPO efficacy.”**

No bootstrap interval, accent-level result, speaker-level claim, development
gate, learning-rate selection, five-fold launch, or paper claim is permitted
from this run.
