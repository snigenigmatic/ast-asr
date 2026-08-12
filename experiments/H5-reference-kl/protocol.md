# H5: matched fixed-reference KL test for FR-CISPO

**Status: LOCKED / NOT YET EXECUTED.** This is an exploratory, single-seed,
profile-clustered engineering experiment. It is not a development comparison,
does not select a final method, and is explicitly `publication_valid: false`.
This file must be committed unchanged before either command is run.

## Question

**H5.** Holding the FR-CISPO design, seed, and random-number schedule fixed,
does adding a small,
fixed-SFT sampled-K3 penalty (`beta = 0.04`) keep a 40-cycle policy trajectory
at least as safe as the otherwise identical `beta = 0` control while preserving
real policy movement?

The intervention is deliberately one-dimensional. Candidate sampling, reward,
sequence-CISPO ratio, family x condition weighting, clean/noisy pairing,
optimizer, seed, SFT checkpoint, fold, decoding settings, and cycle count are
all fixed. Only `reference_kl_beta` changes.

This transfer hypothesis is inspired by the fixed-reference KL term used in
[GRPO for Speech Recognition](https://arxiv.org/abs/2509.01939) and the
fixed-reference proximal-policy machinery documented by the
[TRL GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer). Neither source
establishes that `0.04` is optimal for Whisper-tiny, Svarah, sequence-CISPO,
or this sampled estimator. They provide precedent for the *causal mechanism*,
not evidence for this setting.

## First-principles derivation

### Assumptions challenged

1. A 300-cycle failure means FR-CISPO has no usable learning signal. This is
   not entailed: the existing 20-cycle beta-zero checkpoint was safe and moved.
2. A KL term should be added because it is conventional. It need not be; it
   must earn its cost by changing the divergence trajectory under a matched
   comparison.
3. A lower WER after one run validates the mechanism. It does not, because
   decoding noise, data sampling, and profile-cluster pseudo-speakers remain
   confounders without matched control and authoritative IDs.
4. Cycle-start KL must always be near zero. Only cycle zero scores the policy
   against the identical SFT initialization. Later rollouts correctly begin
   from an already moved policy and may have nonzero SFT-reference KL.

### Irreducible facts and the bounded test

- The policy gradient can change the adapter only through a nonzero loss
  gradient; unchanged ratios/predictions would show no practical learning
  movement.
- The frozen SFT reference assigns a reproducible FP32 score to each sampled
  response token. Given the same valid-token mask, the penalty is fully
  determined by those scores and the current policy scores.
- H1 reached 20 cycles safely at beta zero (peak sampled-K3 KL/token
  `0.0091612`) and its evaluated checkpoint had clean WER `0.216564`, white
  10 dB WER `0.547073`, MUSAN babble 10 dB WER `0.589434`, and provisional
  worst family-condition WER `1.034274`.
- A 300-cycle beta-zero trajectory eventually violated the predeclared
  KL ceiling. Therefore 40 cycles is intentionally **beyond the demonstrated
  20-cycle safe horizon** but still bounded to limit paid compute and prevent
  a silently long, uninformative divergence. It is a mechanism probe, not a
  shortened substitute for the 300-cycle research contract.

### Exact additional loss

For the sampled response tokens whose valid-token mask is `m_t`, let
`d_t = log pi_ref(y_t|x) - log pi_theta(y_t|x)`, where `pi_ref` is the frozen
SFT adapter and reference log probabilities are stored FP32. The additional
term is the global, **unweighted** valid-token mean:

```text
KL_K3(theta || ref) = (1 / sum_t m_t) * sum_t m_t * (exp(d_t) - d_t - 1)
L_total = L_FR-CISPO + beta * KL_K3
```

`L_FR-CISPO` is otherwise unchanged. In particular, family x condition dual
weights remain outside candidate centering in the policy term and do **not**
weight this global reference penalty. This prevents the proximal constraint
from being accidentally redefined as a group-weighted reward.

The implementation uses the numerically stable identity
`expm1(d_t) - d_t`. Non-finite scored tokens or any valid-token
`abs(d_t) > 20` fail closed; the estimator is not silently clipped.

At every rollout's inner update zero, current and reference models are the
same only at **cycle 0**, so K3 must be numerically near zero there. For later
cycles, nonzero cycle-start KL is expected and must be recorded, not rejected
as an identity failure. The reference remains immutable and in evaluation mode
for every cycle and all four inner updates.

## Frozen matched runs

| Field | Beta-zero control | Fixed-reference treatment |
| --- | --- | --- |
| Run name | `profile-h5-refkl-beta0-s2026-20260812` | `profile-h5-refkl-beta004-s2026-20260812` |
| Output name | `h5-beta0-fr-cispo` | `h5-beta004-fr-cispo` |
| `reference_kl_beta` | `0.0` | `0.04` |
| Fold / seed | `0` / `2026` | `0` / `2026` |
| SFT source | `profile-dev-full-sft-20260810/profile-sft-development/checkpoint-epoch-1` | identical |
| SFT revision | `d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e` | identical |
| Learning rate | `1e-5` | identical |
| Rollout cycles | `40` | identical |
| Candidates / inner updates | `K=4` / `4` | identical |
| Probe examples / max tokens | `32` / `225` | identical |
| Objective | live FR-CISPO, sequence ratio cap `2` | identical plus the frozen term above |
| Execution mode | `exploratory_bounded` | identical |
| Publication validity | `false` | `false` |

No hyperparameter sweep is authorized. The paired beta-zero control is needed
to distinguish the effect of beta from the effect of simply extending H1 from
20 to 40 cycles.

## Preconditions

Before either run starts:

1. The committed wrapper exposes the explicit `--reference-kl-beta` parameter
   and records its received value in immutable `resolved_config.json` and
   `run.json`; it must reject negative or non-finite values.
2. The beta-zero path remains loss-identical to its prior implementation, and
   tests prove that no reference score is required when `beta=0`.
3. Positive-beta tests prove: FP32 frozen reference log probabilities; valid
   token mask alignment; global (not group-weighted) K3 average; no reference
   gradient; and numerically near-zero K3 at cycle-zero inner update zero.
4. Each run writes code revision, config hash, fold-manifest hash, data hash,
   model revision, source SFT hash, Modal image/environment identity, exact
   command, RNG seed, and the rollout/corruption realization identifiers.
5. The current 115 profile clusters are stated in all artifacts. No inferred
   ID may be called an authoritative Svarah speaker; the required 117 IDs are
   still absent.

## Frozen Modal commands

Run the beta-zero control first, then the treatment, once each, from
`C:\Kaustubh\ast-asr-worktrees\fair-cispo-tiny`. The commands rely on the
wrapper parameter specified above; a different parameter name, changed flag,
or reused run/output name is a protocol deviation.

```powershell
uvx modal run scripts/modal_fr_cispo.py::run_profile_fr_cispo_smoke `
  --run-name profile-h5-refkl-beta0-s2026-20260812 `
  --seed 2026 `
  --sft-run-name profile-dev-full-sft-20260810 `
  --sft-output-name profile-sft-development `
  --output-name h5-beta0-fr-cispo `
  --learning-rate 0.00001 `
  --reference-kl-beta 0.0 `
  --rollout-cycles 40 `
  --probe-examples 32 `
  --maximum-new-tokens 225
```

```powershell
uvx modal run scripts/modal_fr_cispo.py::run_profile_fr_cispo_smoke `
  --run-name profile-h5-refkl-beta004-s2026-20260812 `
  --seed 2026 `
  --sft-run-name profile-dev-full-sft-20260810 `
  --sft-output-name profile-sft-development `
  --output-name h5-beta004-fr-cispo `
  --learning-rate 0.00001 `
  --reference-kl-beta 0.04 `
  --rollout-cycles 40 `
  --probe-examples 32 `
  --maximum-new-tokens 225
```

## Safety, movement, and comparison gates

For each arm, fail closed and save `failure.json`, diagnostics through the
first violating cycle, and the preceding last-safe checkpoint if one exists
when any of the following occurs:

- non-finite loss, parameters, gradients, ratios, or K3;
- skipped optimizer step;
- sequence-ratio p99 `>= 2.0`;
- sampled per-token SFT-reference K3 `>= 0.1` after a cycle;
- cycle-zero update-zero K3 is not numerically near zero (`<= 1e-6`);
- update-zero ratio differs from one;
- no post-first-update ratio movement;
- checkpoint round-trip or FP32 solo/batched probe equality failure.

A completed arm must also document adapter drift and at least one changed
greedy probe prediction. Movement is evidence that the proximal term did not
merely freeze the adapter; it is not evidence of improved WER.

The matched comparison table must report both arms' cycle-by-cycle maximum
ratio p99, cycle-start K3, post-cycle K3, K3 loss, gradient norm, skipped-step
status, adapter drift, and probe changes. H5 is provisionally supported only
if the beta-0.04 arm completes safely with genuine movement **and** has lower
or equal peak/cycle-40 K3 than beta zero. A beta-0.04 WER advantage cannot
override a safety failure. Conversely, equal safety with no measurable
movement falsifies the practical proximal variant.

## Evaluation trigger and result interpretation

Only if an arm completes all 40 cycles safely, run the existing immutable
FP32 fold evaluator on that arm's `checkpoint-last-safe` over clean, seen
white noise at 10 dB, and unseen MUSAN babble at 10 dB. Store predictions,
word edit counts, WER by family x condition, checkpoint hash, and solo/batch
equality result. Evaluate **both** completed arms; do not evaluate only the
winner.

This follow-up reports the paired deltas against the matched beta-zero arm and
the H1 20-cycle reference. It must say that the `1.034274` H1 worst
family-condition result is a single-seed profile-cluster estimate, and it must
not claim superiority to the historical Whisper-small HTML/PDF results or
publication-level fairness. No three-seed or five-fold launch is unlocked by
H5 alone.

## Falsifiers and deviations

H5 is falsified if beta `0.04` fails a safety gate, does not retain genuine
movement, or does not improve/equal divergence relative to the paired beta-0
control. If beta zero itself fails between cycles 21--40, that is useful
confirmation that H1's 20-cycle safety did not extend; it does not license a
new beta/LR/cycle count. If neither arm completes, record the negative evidence
and stop. Any changed SFT, seed, data realization, objective, metric, beta,
reference estimator, or threshold requires a new versioned protocol before
another run.
