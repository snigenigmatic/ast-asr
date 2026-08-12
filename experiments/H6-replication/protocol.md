# H6 two-seed matched replication protocol

Status: **locked; launch authorized only after this protocol is committed**

Classification: exploratory demographic-profile-cluster engineering;
`publication_valid: false`.

## First-principles decision

H5 established one narrow causal fact: with every other setting matched,
`reference_kl_beta=0.04` reduced sampled fixed-SFT divergence over 40 cycles.
It did not establish efficacy. Its one-seed worst family-condition improvement
was concentrated in Sino-Tibetan MUSAN, all eight other family-condition cells
worsened, and the paired profile-cluster interval included harm.

The minimum experiment that can test whether the point result is repeatable is
two additional **matched** beta-zero versus beta-`0.04` pairs. Running only the
treatment cannot separate the penalty from seed-dependent rollouts. Sweeping a
new beta, LR, horizon, checkpoint, or corruption would answer a different
question and is prohibited.

H6 adds seeds 2027 and 2028 to the completed seed-2026 H5 pair. It does not add
a method, dataset, fold, endpoint, or hyperparameter.

## Frozen contract

| Setting | Value |
| --- | --- |
| Code path | H5 implementation at `4f447e9`; launcher recovery at `d5d3367` |
| Model | `openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af` |
| SFT source | `profile-dev-full-sft-20260810/profile-sft-development/checkpoint-epoch-1` |
| SFT revision | `d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e` |
| Data | Svarah fold 0, 115 prepared demographic-profile clusters |
| Seeds | completed 2026 plus new 2027 and 2028 |
| Pair | beta `0.0` control then beta `0.04` treatment within each seed |
| Training | LR `1e-5`, 40 cycles, `K=4`, four inner updates, max tokens 225 |
| Evaluation | FP32 clean, white 10 dB, MUSAN babble 10 dB |
| Checkpoint | successful bounded `checkpoint-last-safe` only |

Before every paid invocation, both `git diff --exit-code d5d3367 --
scripts/modal_fr_cispo.py src configs uv.lock` and `git status --porcelain --
scripts/modal_fr_cispo.py src configs uv.lock` must be empty. The second check
also rejects untracked executable inputs that Modal would otherwise ship. This
proves that documentation/result commits after H5 did not change executable
inputs. After each policy run, audit its launcher artifact against these
literal H5 values before launching the paired arm or evaluation:

| Invariant | Required value |
| --- | --- |
| Launcher script SHA-256 | `caee47837b58d3cd7c8477a518c89798258c25752358b55ec35a89098af8c28d` |
| Remote source-directory SHA-256 | `98d9755a65965608624989feab2b1a3756f7fdf87a7ab52ebb1efb354342ed66` |
| `uv.lock` SHA-256 | `f6ed29f46ad81e91637368cc91bf7b30134a08ab6dc0efdb8e590f374275312a` |
| Immutable source-config SHA-256 | `dce57ae19c9a08845a76492e3faee4377564eeb3e6339809aad070cefdb3d090` |
| Prepared-manifest SHA-256 | `65bdd8cf87f5db0f815e742739be815d2306ddd2b9977ee5687774feb1a18b56` |
| Fold-0 manifest SHA-256 | `22e9ab64006fe8a33bac37f5f2b98887df6aed061e158252778c29c6d928a1f0` |
| Source SFT revision | `d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e` |
| Beta-zero resolved-config SHA-256 | `3673abefc4322f4951ee067c8b6ed2c2fef93008b3f85c2cf66afd5abd406ae5` |
| Beta-.04 resolved-config SHA-256 | `00b1257018e4dc8fe926714ada6d361da5ba7fae6895e43aa71736f230eba6d3` |

The per-policy `source_tree_content_hash` is expected to differ from H5 because
it deliberately includes the run-specific invoked-config filename. It must be
identical within repeated runs that share an output/config filename and beta;
do not compare it across beta arms. The only resolved-policy content difference
within a seed is `policy.reference_kl_beta: 0.0 -> 0.04`.

## Run matrix

| Seed | Arm | Training run / output | Evaluation run |
| ---: | --- | --- | --- |
| 2027 | beta zero | `profile-h6-refkl-beta0-s2027-20260812` / `h6-beta0-fr-cispo` | `profile-h6-refkl-beta0-eval-s2027-20260812` |
| 2027 | beta .04 | `profile-h6-refkl-beta004-s2027-20260812` / `h6-beta004-fr-cispo` | `profile-h6-refkl-beta004-eval-s2027-20260812` |
| 2028 | beta zero | `profile-h6-refkl-beta0-s2028-20260812` / `h6-beta0-fr-cispo` | `profile-h6-refkl-beta0-eval-s2028-20260812` |
| 2028 | beta .04 | `profile-h6-refkl-beta004-s2028-20260812` / `h6-beta004-fr-cispo` | `profile-h6-refkl-beta004-eval-s2028-20260812` |

Every run and output directory must be absent before its sole invocation.
Execute seed 2027 completely before seed 2028. Within a seed, train the control
first, audit it, then train the treatment. Evaluate both safe checkpoints; do
not evaluate only the apparent winner.

## Frozen training commands

```powershell
uvx modal run scripts/modal_fr_cispo.py::run_profile_fr_cispo_smoke `
  --run-name profile-h6-refkl-beta0-s2027-20260812 --seed 2027 `
  --sft-run-name profile-dev-full-sft-20260810 `
  --sft-output-name profile-sft-development `
  --output-name h6-beta0-fr-cispo --learning-rate 0.00001 `
  --reference-kl-beta 0.0 --rollout-cycles 40 `
  --probe-examples 32 --maximum-new-tokens 225

uvx modal run scripts/modal_fr_cispo.py::run_profile_fr_cispo_smoke `
  --run-name profile-h6-refkl-beta004-s2027-20260812 --seed 2027 `
  --sft-run-name profile-dev-full-sft-20260810 `
  --sft-output-name profile-sft-development `
  --output-name h6-beta004-fr-cispo --learning-rate 0.00001 `
  --reference-kl-beta 0.04 --rollout-cycles 40 `
  --probe-examples 32 --maximum-new-tokens 225

uvx modal run scripts/modal_fr_cispo.py::run_profile_fr_cispo_smoke `
  --run-name profile-h6-refkl-beta0-s2028-20260812 --seed 2028 `
  --sft-run-name profile-dev-full-sft-20260810 `
  --sft-output-name profile-sft-development `
  --output-name h6-beta0-fr-cispo --learning-rate 0.00001 `
  --reference-kl-beta 0.0 --rollout-cycles 40 `
  --probe-examples 32 --maximum-new-tokens 225

uvx modal run scripts/modal_fr_cispo.py::run_profile_fr_cispo_smoke `
  --run-name profile-h6-refkl-beta004-s2028-20260812 --seed 2028 `
  --sft-run-name profile-dev-full-sft-20260810 `
  --sft-output-name profile-sft-development `
  --output-name h6-beta004-fr-cispo --learning-rate 0.00001 `
  --reference-kl-beta 0.04 --rollout-cycles 40 `
  --probe-examples 32 --maximum-new-tokens 225
```

## Frozen evaluation command template

After both arms for a seed pass the training audit, evaluate each once:

```powershell
uvx modal run scripts/modal_fr_cispo.py::run_profile_evaluation `
  --run-name <evaluation-run-from-matrix> --arm fr-cispo `
  --checkpoint-run-name <training-run-from-matrix> `
  --checkpoint-output-name <training-output-from-matrix> `
  --checkpoint-name checkpoint-last-safe
```

The matrix supplies every placeholder; no alternative name or checkpoint role
is allowed.

An evaluation is accepted only when its checkpoint revision equals the audited
training `checkpoint-last-safe` revision, no `failure.json` exists, the run
records exactly the three frozen conditions, `predictions.jsonl` and
`edit_counts.json` each contain 5,772 records, and FP32 solo-versus-batched
predictions are equal. Reject partial or differently labeled evaluations before
aggregation.

## Safety and stopping rules

Each training arm retains every H5 fail-closed rule: finite loss, gradients,
parameters, ratios and K3; 160/160 applied steps; update-zero ratio identity;
post-first-update movement; ratio p99 `<2`; post-cycle K3 `<0.1`; cycle-zero
fixed-reference K3 `<=1e-6`; changed greedy probe; checkpoint round-trip; and
immutable provenance.

- If a beta-zero arm fails a safety gate, do not launch its treatment or any
  later H6 seed. Record H6 as a failed replication.
- If a beta-`0.04` arm fails a safety gate, do not launch a later H6 seed.
  Record H6 as a failed proximal replication.
- A launcher failure before model loading is not a policy result, but it does
  not authorize a retry. A separately committed recovery protocol is required.
- Performance cannot stop H6 early. If seed 2027 is safe, seed 2028 must run
  even if seed-2027 WER is favorable or unfavorable.

## Registered aggregation and decisions

For each seed, report treatment-minus-control deltas for the primary endpoint,
clean overall, white overall, MUSAN overall, worst-family clean, family gap,
noise amplification, worst-20%-profile-cluster WER, and all nine
family-condition cells.

The completed seed-2026 H5 pair and the two new pairs form the fixed H6
three-seed exploratory set. This is **not** the original confirmatory
development gate, whose frozen seeds are 11, 17, and 23 and whose grouping unit
requires authoritative Svarah speakers. H6 can only guide engineering method
selection while those identities remain unavailable.

Within each seed and arm, pool word edit counts by family-condition, compute
all nine cell WERs, and take that arm's maximum cell as its worst
family-condition WER. Then compute the paired treatment-minus-control delta
within that seed. The H6 summary is the equal-weight mean of the three paired
seed deltas; never pool predictions across seeds as if they were new speech.

The H6 exploratory point rule is met only if:

1. mean control-minus-treatment worst family-condition WER is at least `0.02`;
2. mean treatment-minus-control clean overall WER is at most `0.01`; and
3. all six training arms pass their safety and movement gates.

Aggregate-noise behavior is a separate exploratory interpretation, not a
hidden selection rule. A positive aggregate-noise signal additionally requires
non-positive mean treatment-minus-control deltas for both white and MUSAN
overall WER. If the primary rule passes but either noisy condition worsens,
describe the result as a worst-group/average-robustness trade-off. Neither
outcome authorizes a general robustness claim.

For each of 10,000 resamples, draw 39 fold-0 profile-cluster IDs with
replacement once and apply those identical multiplicities to every arm and
all three seeds. Recompute per-arm cell WERs, per-arm maxima, within-seed paired
deltas, and finally their equal-seed mean. Use seed 2026 for the bootstrap RNG.
Report percentile intervals descriptively. This is a profile-cluster
sensitivity analysis, never a speaker-level interval. Do not treat seeds as
independent speakers, do not use `p>=0.05` as a fairness claim, and do not
rename profile clusters as speakers.

Write one immutable `h6_replication_gate.json` containing all three seeds'
checkpoint revisions, artifact hashes, safety results, per-seed metrics and
deltas, equal-seed means, both decision-rule booleans, robustness
classification, and bootstrap intervals. No H6 decision is valid without this
artifact and a checkpoint/prediction count audit.

Even a passed H6 gate does not authorize folds 1--4 while the authoritative 117
Svarah identities remain missing. It can support the capstone's engineering
method selection and a request to recover the official metadata; it cannot
produce publication-valid fairness evidence by itself.
