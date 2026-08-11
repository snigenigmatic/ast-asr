# FR-CISPO execution runbook

**Status:** active research-control document. This runbook is written from the
FP32-corrected development evidence on 2026-08-10. It is not evidence that
FR-CISPO has improved ASR, and it does not authorize a paid run by itself.

## 1. Operating contract

The question is narrowly fixed: can `openai/whisper-tiny`, adapted with Svarah
transcripts and MUSAN only as unlabeled noise, lower **worst language-family ×
acoustic-condition WER** without increasing clean overall WER by more than
0.01? All publication-facing claims must use speaker-disjoint folds built from
the authoritative 117 Svarah speakers.

The immutable inputs for a run are:

| Input | Required value / record |
| --- | --- |
| Base model | `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af` |
| Transcribed corpus | `ai4bharat/Svarah@ebbf7777fe771490696a3f7b007097606fa8c924` |
| Noise-only corpus | MUSAN speech archive, MD5 `0c472d4fc0c5141eca47ad1ffeb2a7df` |
| Supplied speaker-statistics CSV | SHA-256 `e2daa48863581eb41befd1826b7b14cd80e05f3a1bab9b72d08e7814248f1f94` |
| SFT specification | LoRA rank 8 / alpha 16 on q/k/v/out projections, dropout 0, LR `1e-4`, at most 5 epochs |
| Policy specification | `K=4`, four inner updates, frozen rollouts, live `mu=4`, and the declared objective arm |
| Evaluation | FP32 greedy decoding; clean, white 10 dB, and MUSAN babble 10 dB |

Never overwrite a run directory. Each run must save its resolved config,
manifest hash, fold manifest hash, seed, model and dataset revisions, stdout
log, diagnostics, checkpoint revision, predictions, edit counts, and a command
record. A changed input or command produces a new run ID.

## 2. Gates, in dependency order

```text
authoritative data --> reproducible baselines --> bounded H1 safety test
        |                       |                         |
        v                       v                         v
  117 speakers, folds      FP32 + round-trip          select LR only if
  and no overlap           equality holds              all three seeds safe
                                                           |
                                                           v
                                                    3-seed dev ladder
                                                           |
                                                           v
                                                   five folds and OOF
```

### Gate A — authoritative data (hard block)

Before any publication-valid training:

1. Recover `speaker_id` from an official, pinned Svarah artifact. The result
   must contain exactly 117 distinct values.
2. Build five deterministic speaker folds and save the input/content hashes,
   group counts, and train/validation/test speaker lists in immutable manifests.
3. Verify no speaker overlaps within an outer fold; verify every test speaker
   occurs exactly once across the five out-of-fold test partitions.
4. Reconcile the language-to-family taxonomy explicitly. An unseen language is
   a preparation failure, not an invitation to infer a family.

The current 115 demographic-profile clusters fail this gate. They may be used
only for synthetic/runtime or engineering diagnostics and every resulting
artifact must carry `publication_valid: false`. They cannot be pooled with
authoritative results, used for speaker-clustered confidence intervals, or
described as speaker-disjoint fairness evidence.

### Gate B — baseline and reproducibility (hard block)

For each authoritative development seed (11, 17, 23), create or reuse the
fold-0 SFT checkpoint selected by validation macro-family WER. Before comparing
methods, require all of the following:

- attention masks pass the existing regression tests;
- FP32 solo and batched greedy predictions are byte-for-byte equal for the
  recorded probe and for the saved evaluation predictions;
- reloading each SFT checkpoint reproduces its saved probe predictions;
- zero-shot and SFT predictions include utterance ID, authoritative speaker,
  family, condition, hypothesis, reference, and edit counts;
- no cross-seed checkpoint, prediction, or manifest directory is reused.

The existing corrected profile-cluster results are engineering evidence only:
zero-shot clean WER 0.2208 and SFT clean WER 0.2417; SFT improves white-10-dB
WER (0.6218 to 0.5725) while worsening the provisional worst
family-condition endpoint (1.2339 to 1.4617). They must not be treated as the
baseline for the authoritative study.

### Gate C — movement and trust region (hard stop)

At inner update zero, every importance ratio must equal one. At a later inner
update the live ratio must move while all values remain finite. A run is
immediately unsafe, and must stop rather than emit a candidate checkpoint, if
any of the following occurs:

- non-finite loss, parameter, ratio, or diagnostic;
- an optimizer step is skipped;
- ratio p99 is `>= 2.0`;
- sampled per-token K3 KL from the SFT reference is `>= 0.1`;
- checkpoint round trip or FP32 solo/batched equality fails.

The stop record must preserve the last safe cycle (if one exists), the first
failing cycle, the diagnostic trajectory, and the reason. A failed run remains
an immutable failure-decomposition artifact; it is never retagged as a
successful checkpoint.

Current evidence fails this gate: `1e-4` reached ratio p99 2.477 and KL 0.2958
in a 10-cycle probe; `3e-5` and `1e-5` completed 300 cycles but reached KL
43.9888 and 1.1638 respectively. Thus no learning rate is presently selected.

## 3. Compute-conscious experiment matrix

Every row has a predeclared purpose. A row marked **confirmatory** may be used
in the final comparison only when Gates A and B have passed; **exploratory**
rows are engineering or mechanism checks and cannot be promoted after seeing
their result.

| Order | Label | Classification | Scope and fixed inputs | Advance only when |
| ---: | --- | --- | --- | --- |
| 0 | Storage/runtime smoke | Exploratory engineering | Modal storage audit and synthetic four-update smoke for all objective arms | outputs are immutable and all mathematical tests pass |
| 1 | Authoritative preparation | Confirmatory data gate | One pinned Svarah artifact; five speaker folds | exactly 117 speakers, hashes stable, no overlap |
| 2 | FP32 zero-shot/SFT baselines | Confirmatory | Fold 0, seeds 11/17/23; zero-shot plus selected SFT | Gate B passes for every seed |
| 3 | H1 KL-control pilot | Exploratory | The locked protocol in `experiments/H1-kl-control/protocol.md` | a root-reviewed stop artifact establishes whether H1 is worth a three-seed test |
| 4 | Learning-rate safety selection | Confirmatory only after an approved protocol | Each of `1e-5`, `3e-5`, `1e-4` × seeds 11/17/23; no unlisted LR | every selected LR seed passes Gate C; otherwise no policy ladder |
| 5 | Development ladder | Confirmatory | Fold 0 × seeds 11/17/23: zero-shot, SFT, live-GRPO, CISPO-MWER, sequence-CISPO-MWER, Fair-CISPO, FR-CISPO | FR-CISPO passes the three-seed development gate below |
| 6 | Five-fold confirmation | Confirmatory | Folds 0–4 for SFT, CISPO-MWER, Fair-CISPO, FR-CISPO at the frozen selected LR | all fold artifacts and evaluations pass |
| 7 | OOF aggregation | Confirmatory analysis | Concatenated OOF predictions; 10,000 paired speaker-clustered bootstrap draws | every test speaker appears once and only once |

For cost control, run the next row only after the previous row emits its
immutable pass artifact. Do not parallelize a full fold with an unresolved
lower-numbered gate. Bounded smokes may reduce runtime uncertainty, but never
count as a result. If Modal capacity is unavailable, pause paid work and use
local unit/synthetic tests; do not substitute a smaller dataset or an
unrecorded hyperparameter.

## 4. Stop and decision rules

### Learning-rate selection

The candidate grid is exactly `{1e-5, 3e-5, 1e-4}`. For each value, all three
development seeds must pass Gate C. Select the largest passing value and write
`development_learning_rate_gate.json` with all nine movement records. If none
pass, report H1 as unsupported and stop policy training; do not add a lower
learning rate, change the objective, or increase/decrease the horizon without
a new protocol.

### Three-seed development gate

With the selected LR frozen, compare FR-CISPO against its fold-0 SFT checkpoint
for seeds 11, 17, and 23. The gate is passed only when the mean:

- improvement in worst family × condition WER is at least **0.02** absolute;
- degradation in clean overall WER is at most **0.01** absolute.

All three runs must also pass Gate B and Gate C. Save an immutable
`development_gate.json` containing the seed-level WERs, the selected LR,
movement diagnostics, and the decision. Failure blocks folds 1–4. It is a
result, not a trigger to change methods silently.

### Five-fold and out-of-fold gate

Only a passed `development_gate.json` unlocks folds 1–4. Evaluate clean, seen
white noise, and unseen MUSAN babble in FP32. The primary endpoint is the
worst family × condition WER from concatenated OOF predictions. Secondary
reports are clean overall WER, worst-family clean WER, family gap, noise
amplification, worst-20%-speaker WER, and 19 accent-level WER estimates. Use
10,000 paired speaker-clustered bootstrap draws; report uncertainty rather than
per-accent hypothesis tests. Do not use demographic-parity or
equal-opportunity terminology.

## 5. Immutable artifact checklist

| Artifact | Produced at | Minimum contents |
| --- | --- | --- |
| `dataset_manifest.json` | preparation | source revisions, hashes, utterance IDs, authoritative speakers |
| `fold-<n>.json` | splitting | train/validation/test speakers, balance summary, content hash |
| `resolved_config.json` and command record | every run | config, CLI, package versions, seed, device, start/end time |
| rollout and diagnostic JSON | every policy cycle | frozen old scores, masks, candidate WERs, ratios, KL, group probabilities |
| stop artifact | every stopped policy run | first violating cycle, last safe checkpoint, reason, diagnostic hashes |
| checkpoint + content hash | SFT/policy | adapter, processor, revision, round-trip probe output |
| prediction JSONL + edit counts | evaluation | arm, utterance, speaker, group, condition, reference, hypothesis, counts |
| gate JSON | each gate | literal inputs, threshold checks, pass/fail, links to artifacts |
| OOF summary + bootstrap draws | final analysis | seed, cluster method, primary/secondary endpoints and intervals |

## 6. Publication and venue decision gate

Do not claim a venue, deadline, acceptance likelihood, or publication status in
project material until it has been verified from that venue's official call for
papers. The mentor's practical constraint is reflected as a decision rule: a
low-cost local/Indian conference, a journal, or a later archival venue are
options only after the evidence package is complete and travel/reimbursement
is feasible.

The paper path is selected at an outer-loop review only if all are true:

1. authoritative 117-speaker data and all reproducibility artifacts are
   available;
2. the locked development gate and five-fold OOF gate have completed;
3. the narrative makes a narrow, evidence-backed claim (positive or negative),
   rather than advertising preliminary numbers;
4. a complete draft, artifact index, limitations section, and contribution
   ledger exist;
5. the supervisor and team decide authorship and budget from the evidence.

Otherwise, write a capstone report around the verified failure decomposition:
identity leakage risk, numerical evaluation invariance, correct frozen-rollout
ratios, and observed KL instability. That report must distinguish engineering
diagnostics from publication-valid estimates.

For a future paper, the one-sentence claim, all citations, and the target
venue's current requirements must be verified before drafting. Keep explicit
limitations: Svarah coverage, tiny-model scope, noise conditions, group sample
sizes, and any unsupported H1 result.
