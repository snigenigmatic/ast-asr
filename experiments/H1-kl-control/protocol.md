# H1: KL-control pilot for FR-CISPO

**Status: LOCKED / APPROVED FOR ONE ENGINEERING RUN.** This document must be
frozen in version control before execution. Results from this protocol are
**exploratory**; they are not the planned
three-seed development comparison and cannot unlock five-fold training.

## Question and prediction

**H1.** Can an explicit hard KL stop preserve genuine live off-policy movement
for a short FR-CISPO trajectory while keeping the emitted checkpoint inside the
existing per-token trust-region ceiling?

The existing policy loss is retained exactly: centered MWER advantages,
sequence-level CISPO ratio, upper ratio cap 2, dual family × condition weights,
paired clean/white-noise inputs, `K=4`, and four optimizer passes over each
frozen rollout. The intervention is only a run controller: measure sampled K3
KL from the frozen SFT reference after each rollout cycle and **stop before
accepting a checkpoint whose KL is `>= 0.1`**. It neither adds a KL penalty nor
changes advantages, ratios, group weights, corruption, optimizer, or learning
rate grid.

Prediction: at least one noninitial inner update will have a ratio different
from one, while a retained last-safe checkpoint remains finite, has no skipped
steps, has ratio p99 `< 2`, and K3 KL `< 0.1`. A stop before there is a
meaningful retained checkpoint refutes this pilot's practical premise; it is not
rescued by a new learning rate or a changed loss.

## Preconditions

All must be true before the command is made runnable:

1. The policy trainer writes per-cycle (not only cumulative) KL and ratio
   diagnostics, retains the last-safe checkpoint, and writes an immutable stop
   artifact with the first violating cycle and reason.
2. Unit tests cover update-zero ratios, post-update movement, KL stop semantics,
   stop-gradient behavior, and checkpoint round trip.
3. The run directory is new and its command, code revision, config, data/fold
   manifest hashes, base model revision, SFT checkpoint revision, and Modal
   environment are recorded.
4. The current data source is labelled precisely. If it still uses 115
   profile clusters, the run is a non-publication engineering pilot and is not
   evaluated with speaker-level claims.

Failure of any precondition blocks execution. Resolving authoritative 117
speaker IDs is still required before a publication-valid experiment.

## Fixed pilot

| Field | Value |
| --- | --- |
| Classification | Exploratory mechanism/safety pilot |
| Arm | `fr-cispo` only |
| Fold | 0 development fold |
| Seed | 2026 (reuses the existing engineering SFT checkpoint; it is not a confirmatory development seed) |
| Learning rate | `1e-5` (smallest member of the frozen grid; not selected for later experiments) |
| Maximum rollout cycles | 20 |
| Candidates / inner updates | 4 / 4 |
| SFT starting point | `profile-dev-full-sft-20260810/profile-sft-development/checkpoint-epoch-1`, revision `d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e` |
| Decoding / evaluation precision | FP32 greedy decoding |
| Primary pilot measurements | per-cycle K3 KL, per-update ratio p99, finite/skip state, adapter drift, probe prediction change |
| Permitted conclusion | whether this exact hard-stop controller can emit a safe short-horizon diagnostic checkpoint |
| Prohibited conclusion | WER improvement, fairness improvement, learning-rate selection, or publication-ready FR-CISPO efficacy |

The 20-cycle limit is deliberately a safety diagnostic rather than a shortened
claim of a 300-cycle method. It is below the 300-cycle contract and therefore
must be stored as `execution_mode: exploratory_bounded`; it cannot be compared
as an arm in the confirmatory ladder.

## Execution and stop logic

1. Snapshot the SFT reference in evaluation mode and preserve its FP32 old
   log-probabilities throughout each frozen rollout batch.
2. For cycles 0 through 19, generate a new frozen rollout and perform exactly
   four inner updates.
3. Assert update-zero ratios are one. Record every inner update's loss,
   gradient norm, ratio p99, and the cycle's KL versus the SFT reference.
4. If any safety condition below fails, do not continue to another cycle. Save
   the violating diagnostic, a stop artifact, and (only if it exists) the
   preceding safe checkpoint. Do not write a `checkpoint-final` that could be
   mistaken for a completed policy run.
5. If all 20 cycles are safe, save the cycle-19 checkpoint as
   `checkpoint-last-safe`, reload it, and verify its probe predictions exactly.

Hard stops:

- NaN/Inf in loss, parameters, ratios, gradients, or KL;
- skipped optimizer step;
- any ratio p99 `>= 2.0`;
- any sampled per-token K3 KL `>= 0.1`;
- update-zero ratio differs from one;
- no ratio movement after the first optimizer update;
- checkpoint reload or FP32 solo/batched prediction mismatch.

## Planned analysis

The pilot report contains a trajectory plot/table with cycle, max ratio p99,
cycle KL, cumulative max KL, group probability range, and stop status. It also
contains before/after probe predictions only to demonstrate movement and
round-trip reproducibility. It must display the profile-cluster limitation if
authoritative IDs are not yet available.

Pass for this pilot requires all 20 cycles to complete safely and at least one
post-update ratio movement. A stopped run is reported as a negative result. A
pass only justifies root consideration of a new, separately locked three-seed
safety protocol; it does not select `1e-5` or authorize development/five-fold
training.

## Frozen execution command

Run exactly once, from the isolated `codex/fair-cispo-tiny` worktree after the
protocol commit exists:

```powershell
uvx modal run scripts/modal_fr_cispo.py::run_profile_fr_cispo_smoke `
  --run-name profile-h1-klstop-s2026-20260811 `
  --seed 2026 `
  --sft-run-name profile-dev-full-sft-20260810 `
  --sft-output-name profile-sft-development `
  --output-name h1-klstop-fr-cispo `
  --learning-rate 0.00001 `
  --rollout-cycles 20 `
  --probe-examples 32 `
  --maximum-new-tokens 225
```

The run directory did not exist when this protocol was locked. Reusing the name
or changing any flag is prohibited.

## Deviations

Any change to cycle count, LR, data identity source, KL estimator, retained
checkpoint rule, objective, group weighting, decoding precision, or success
criterion creates a new protocol version before another run. Record why the
change is necessary and label its result exploratory unless it is part of a
newly approved confirmatory plan.
