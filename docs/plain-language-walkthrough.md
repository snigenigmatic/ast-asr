# FR-CISPO in plain language — a walkthrough you can defend

Purpose: understand this framework *to the basics*, tied to the exact code. Written for the
`fair-cispo-work` branch (forked from `codex/fair-cispo-tiny`). Every claim points at a file:line so you
can check it yourself. Nothing here is aspirational — it describes what the code actually does today.

---

## 1. What this project is, in one sentence

We take a Whisper model that was lightly fine-tuned (SFT) on Indian-English speech, and we *nudge* it with
reinforcement-style updates so it makes fewer errors on the **worst-off** language-family × acoustic
condition — **without** letting it drift dangerously far from the SFT model or wreck its clean-speech
accuracy.

The method is **FR-CISPO** = "Fair & Robust" CISPO. CISPO is a policy-gradient variant that reweights
sampled hypotheses by a *clipped, stop-gradient* importance ratio.

---

## 2. The three models you must never confuse

Almost all confusion in RL-for-ASR comes from blurring these. Keep them separate:

| Name | What it is | Moves during training? | Role |
|---|---|---|---|
| **Policy** `policy_model` | The model we're training (LoRA adapter on Whisper-tiny) | **Yes** | The thing we optimize |
| **π_old** (rollout policy) | A *snapshot* of the policy at the moment we sampled candidates | No (frozen per cycle, refreshed next cycle) | Denominator of the importance ratio |
| **SFT reference** `reference_model` | A second, independent copy of the original SFT adapter | **Never** | Fixed anchor for the KL trust-region penalty |

- Policy vs π_old → `policy_training.py:576` (policy, trainable) vs the frozen rollout at
  `policy_training.py:660` (`generate_frozen_rollout`, which stores `old_token_log_probs`).
- SFT reference → `policy_training.py:582` built with `trainable=False`, then
  `_freeze_sft_reference_model` (`policy_training.py:164`): `.eval()` + `requires_grad_(False)`.

**Why two frozen things?** π_old defines "how much did the policy move *this cycle*" (the ratio). The SFT
reference defines "how far has the policy wandered from where it started" (the KL penalty). Different jobs.

---

## 3. The training loop, one cycle at a time

The loop is `for cycle in range(rollout_cycles)` at `policy_training.py:649` (40 cycles for H5/H6). Each cycle:

1. **Snapshot** the current adapter to memory (so we can roll back if this cycle turns unsafe) —
   `_clone_trainable_state`, `policy_training.py:650`.
2. **Pick a balanced batch**: one utterance per language family, so no family dominates —
   `_balanced_batch`, `policy_training.py:55`.
3. **Generate a frozen rollout**: sample K=4 hypotheses per utterance at temperature, and record the
   policy's log-probs for those exact tokens. These are π_old. `policy_training.py:660`.
4. **Score with the SFT reference** (no grad) on the *same* hypotheses → the KL anchor log-probs —
   `policy_training.py:673`.
5. **Cycle-zero identity check**: at cycle 0 the policy and SFT reference are identical copies, so their
   sampled-K3 divergence must be ~0 (≤ 1e-6). If not, something is wrong and we stop. `policy_training.py:189`.
6. **4 inner updates** on this frozen batch (`optimize_frozen_rollout`, `policy_training.py:781`). This is
   where the policy actually changes.
7. **Post-cycle safety**: recompute KL from SFT; if ratio p99 ≥ 2.0 or KL/token ≥ 0.1, **fail closed** —
   save the last-safe checkpoint and raise. `policy_training.py:901` (ratio) and `:925` (KL).

At the end: save the checkpoint, re-transcribe a probe set, and assemble the **movement gate**
(`policy_training.py:967`) + a **checkpoint round-trip** equality check (`policy_training.py:991`).

---

## 4. Why the importance ratio is *live* here (the μ=1 fix)

This is the single most important thing to understand, because it's the exact bug that made the old
`ast-adversery` "ladder" measure nothing.

- The importance ratio is `π_θ(candidate) / π_old(candidate)`. If the policy hasn't changed since the
  rollout, this ratio is exactly 1 and any clipping/PPO/CISPO machinery is inert.
- In the old trainer there was effectively **one** optimizer step per rollout, so within that step the
  weights were frozen → ratio ≡ 1 → the "clipped ratio" experiment was a no-op.
- Here we do **4 inner updates on the same frozen batch** (`config.policy.inner_updates`,
  `policy_training.py:788`). After the first update the policy has moved but π_old has not, so the ratio is
  genuinely ≠ 1.
- The code *forces* this to be true: it asserts ratio ≈ 1 at update 0 and ratio ≠ 1 at update 1
  (`ratio_safety_check`, `policy_training.py:766`; enforced in `_ratio_protocol_violation`,
  `policy_training.py:304`). If the ratio never moves, the run is rejected. So you can trust that the ratio
  axis is measuring real policy change.

---

## 5. The objective, knob by knob (`objectives.py`)

Everything funnels through one function: `policy_objective` (`objectives.py:176`). An `ObjectiveSpec`
(`objectives.py:48`) picks values along independent axes. This decoupling is why each design choice can be
tested in isolation.

- **Advantage** — how we assign credit among the 4 candidates of one utterance:
  - `CENTERED_MWER` (`objectives.py:81`): reward = −WER (clipped at 2.0), then subtract the utterance mean.
    Better-than-average hypotheses get positive credit; worse-than-average get negative. This is literally
    the MWER objective.
  - `STANDARDIZED` (`objectives.py:157`): same but also divide by the per-utterance std (the GRPO move).
- **Ratio unit** — `NONE`, `TOKEN`, or `SEQUENCE`. `SEQUENCE` uses `exp(mean_t(logπ − logπ_old))`
  (`objectives.py:93`).
- **Clip rule**:
  - `PPO_SYMMETRIC` (`objectives.py:241`): the usual `min(r·A, clip(r)·A)`.
  - `CISPO_UPPER` (`objectives.py:256`): clip the ratio's *upper* side and **stop its gradient**
    (`ratios.clamp(max=...).detach()`), then multiply by advantage and the log-prob. Gradient flows through
    the log-prob, not the ratio — this is what makes CISPO stable.
- **Group weighting** — `UNIFORM` or `DUAL`. **This is the fairness lever, and it's the fix for the
  "fairness cancels out" bug.** The weights multiply the *whole utterance's* objective **after** the
  within-utterance advantage centering (`objectives.py:268`, `candidate_objectives * weights.unsqueeze(1)`),
  and are constrained to **mean 1** (`objectives.py:187`). Because they act outside the centering, they
  reweight *which utterances matter*, and can never algebraically divide out inside a group the way the old
  per-candidate fairness reward did.
- **Corruption** — `CLEAN` or `PAIRED_WHITE` (train on clean + a white-noise copy).

---

## 6. The reference-KL penalty (β) — the thing H5/H6 actually test

Total loss = base policy loss + **β · K3(policy ‖ SFT reference)**.

- The K3 estimator is `expm1(log_ratio) − log_ratio` (`sampled_k3_reference_kl`, `objectives.py:118`). It is
  non-negative, numerically stable near zero, computed in FP32, and **fails closed** if any token log-ratio
  exceeds 20 in magnitude (`objectives.py:147`).
- β is `config.policy.reference_kl_beta`, passed at `policy_training.py:793`.
- **H5 finding (one seed):** β=0.04 vs β=0 reduced divergence from SFT by ~34% while keeping the policy
  moving. Worst-group WER improved −2.22 pp, but the bootstrap interval `[−2.22, +5.24] pp` still includes
  harm, and noise robustness slightly worsened. **Mechanism works; efficacy unproven.**

---

## 7. The safety gates — why you can trust the numbers

These are hard stops, not knobs (`gates.py:10`): `MAX_RATIO_P99 = 2.0`, `MAX_KL_PER_TOKEN = 0.1`.

- **Per-cycle**: non-finite anything, ratio p99 ≥ 2, or KL ≥ 0.1 → save last-safe checkpoint, write
  `failure.json`, stop.
- **Movement gate** (`gates.py:23`): a run only "counts" if it had finite values, zero skipped optimizer
  steps, real adapter drift, *changed greedy predictions*, ratio p99 < 2, and KL < 0.1.
- **Development gate** (`gates.py:69`): needs **3 seeds** with mean worst-group improvement ≥ 0.02 **and**
  mean clean-WER degradation ≤ 0.01. Folds 1–4 are *blocked* until a passed `development_gate.json` exists
  (`gates.py:100`).

The point of all this: no silent hyperparameter pivots, no "it exploded but we kept the good part," no
comparing checkpoints that were saved under different rules.

---

## 8. Where the missing speaker IDs bite (and our reframe)

`policy_training.py:266` defines `publication_valid = (identity_mode == "authoritative" and
identity_count == 117)`. Right now the public Svarah data doesn't expose those 117 authoritative speaker
IDs, so every run so far is labelled **`publication_valid: false`** and uses "demographic-profile clusters"
as a stand-in grouping.

Two experiment tiers follow from this (do not confuse them):

- **Exploratory replication = H6**, seeds **2027 + 2028** added to the seed-2026 H5 pair → a 3-seed
  *profile-cluster* check. Cheap, engineering-only, `publication_valid: false`. Its bootstrap is explicitly a
  "profile-cluster sensitivity analysis, never a speaker-level interval"
  (`experiments/H6-replication/protocol.md:186`).
- **Confirmatory development gate**, seeds **11 / 17 / 23**, which *requires authoritative speakers* and is
  what would license a real fairness claim.

> **Correction to the recovery plan:** H6 uses seeds 2027/2028 (not 11/17/23). 11/17/23 belong to the later
> confirmatory gate. This matters because "replicate first" = run H6 (cheap, profile-cluster) *before*
> investing in the confirmatory path.

**Our chosen reframe** (Phase 2): instead of waiting on Svarah maintainers, introduce a transparent,
reproducible **heuristic speaker grouping** (derived from filenames) and a **speaker-clustered bootstrap**,
then state the limitation openly. Concretely that means adding a non-"authoritative" identity mode and being
honest in the paper's Threats-to-Validity that grouping is heuristic. This lets us produce a defensible
speaker-level analysis without a third party, at the cost of a clearly-labelled caveat.

---

## 9. The honest bottom line right now

- The **machinery is sound and trustworthy**: live ratios, a real trust region, fail-closed safety, FP32
  eval invariance, provenance hashing. This is publishable-quality *engineering*.
- The **scientific result is not yet established**: one seed shows a fragile worst-group gain with a
  trade-off against noise robustness. That's why the next step is *replicate* (H6), then let the data pick
  the paper's story.
- **Do not** anchor anything on the old 16.4% WER number (different eval protocol — see
  `docs/prior-results-benchmark.md`).
