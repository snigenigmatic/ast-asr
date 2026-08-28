# RL Post-Training: Mathematical Fixes & Improvements
**Project:** PW25_BJD_05 — Fair & Robust Audio Transformers  
**Date:** April 2026  
**Scope:** Whisper-small GRPO post-training for accent-fair Indian English ASR

---

## Table of Contents

1. [Diagnosed Problems](#1-diagnosed-problems)
2. [Fix 1 — Fairness Reward (Blocking)](#2-fix-1--fairness-reward-blocking)
3. [Fix 2 — Double Normalization (Blocking)](#3-fix-2--double-normalization-blocking)
4. [Fix 3 — Replace GRPO Loss with CISPO](#4-fix-3--replace-grpo-loss-with-cispo)
5. [Fix 4 — Probe Set (Diagnostic)](#5-fix-4--probe-set-diagnostic)
6. [Implementation Priority](#6-implementation-priority)
7. [Full Rewritten Files](#7-full-rewritten-files)

---

## 1. Diagnosed Problems

### 1.1 Fairness Reward Cancellation (Critical)

The family-need weight `w` in `reward.py` multiplies **all K hypotheses for utterance i uniformly**.
Group-relative advantage normalization then cancels it exactly.

**Proof:**

Let $\tilde{r}_{i,k} = w \cdot r_{i,k}$ where $w$ is the family weight for utterance $i$.

$$\hat{A}_{i,k} = \frac{\tilde{r}_{i,k} - \bar{\tilde{r}}_i}{\tilde{\sigma}_i} = \frac{w \cdot r_{i,k} - w \cdot \bar{r}_i}{w \cdot \sigma_i} = \frac{r_{i,k} - \bar{r}_i}{\sigma_i}$$

The $w$ cancels exactly. **The fairness signal contributes zero gradient to the policy.**

This is the root cause of:
- No upward reward trend over 1,500 steps (reward 0.567 → 0.485, decreasing)
- ΔDP reduction being attributable to the balanced sampler, not the fairness reward

### 1.2 Double Normalization (Critical)

`compute_rewards` normalizes across the full B×K block when `normalize=True`.  
`group_relative_advantages` then normalizes again within each utterance's K samples.

The second normalization erases whatever cross-utterance signal the first created.
These two normalizations are mathematically redundant and together corrupt the intended signal.

### 1.3 KL Divergence Instability

The K3 KL estimator used is:

$$\hat{\text{KL}} = e^{\log r} - \log r - 1 \quad \text{where } \log r = \log\pi_\text{ref} - \log\pi_\theta$$

`log_ratio` is clamped at `[-20, 20]`, but $e^{20} \approx 485\text{M}$.  
The training log shows:
- Step 750: KL = 84.97  
- Step 870: KL = 55.28  
- Step 1060: KL = **3007.85** (catastrophic spike)

These spikes indicate the model undergoes large uncontrolled parameter jumps. The final checkpoint's position is partly arbitrary depending on when the spike occurred relative to the last save.

### 1.4 Broken Probe Set

In `train_rl_whisper.py`:

```python
eval_df = load_svarah(max_samples=10, cache_dir="cache", svarah_split="eval")
```

`load_svarah` selects the first 10 UIDs from the full dataset **before** filtering by eval UIDs.  
UIDs 0–9 from the full dataset — only UIDs 1 and 6 are in the eval split.  
Result: 2 utterances monitored across 1,500 steps. The probe showing `0/2 differ from reference` throughout is not a signal about the model — it is a signal about the bug.

### 1.5 GRPO Degeneracy for ASR

GRPO was designed for tasks where K rollouts of the same prompt have meaningfully different quality (e.g. one reasoning chain is correct, others wrong). For ASR, K sampled transcriptions of the same audio have very similar WER — the within-utterance distribution is tight:

$$\sigma_i = \text{std}(r_{i,1}, \ldots, r_{i,K}) \approx 0$$

This makes advantage computation near-degenerate (dividing by near-zero std), contributing to gradient instability and KL spikes.

---

## 2. Fix 1 — Fairness Reward (Blocking)

### Problem

A scalar family weight $w$ applied uniformly across all K hypotheses of utterance $i$ cancels in group-relative normalization. The signal must **vary across K hypotheses** to survive normalization.

### Solution

Maintain an **exponential moving average (EMA) of per-family WER** across training steps. The fairness bonus for hypothesis $k$ of utterance $i$ is how much that specific hypothesis beats the family's current EMA baseline:

$$r^\text{fair}_{i,k} = \alpha \cdot \max\!\left(0,\ \bar{W}_g - \text{WER}(h_{i,k}, y_i)\right)$$

where $\bar{W}_g$ is the EMA WER for family $g$, and $\alpha$ is the fairness strength.

**This varies across K** because each hypothesis $h_{i,k}$ has a different WER. It also varies across families because each family has a different EMA baseline. It therefore survives group-relative normalization.

The EMA update after each batch:

$$\bar{W}_g \leftarrow (1 - \beta) \cdot \bar{W}_g + \beta \cdot \widehat{\text{WER}}_g^\text{batch}$$

### Code: `ast-asr/rl/reward.py`

```python
"""
reward.py  —  corrected
Multi-component reward for CISPO-based RL post-training.

Key fix: fairness bonus varies across K hypotheses (by computing it from
per-hypothesis WER vs family EMA baseline) so it is not cancelled by
group-relative advantage normalization.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from jiwer import cer, wer

from .beam_search import RolloutBatch

logger = logging.getLogger(__name__)


@dataclass
class FamilyEMA:
    """
    Exponential moving average of WER per language family.
    Maintained across training steps so the fairness bonus has a
    meaningful baseline even in early training.
    """
    beta: float = 0.05          # EMA decay; smaller = slower adaptation
    initial_wer: float = 0.40   # Conservative prior; updated immediately

    _wer: dict[str, float] = field(default_factory=dict)

    def get(self, family: str) -> float:
        return self._wer.get(family, self.initial_wer)

    def update(self, family_wers: dict[str, float]) -> None:
        for fam, w in family_wers.items():
            prev = self._wer.get(fam, self.initial_wer)
            self._wer[fam] = (1 - self.beta) * prev + self.beta * w

    def state_dict(self) -> dict:
        return dict(self._wer)

    def load_state_dict(self, d: dict) -> None:
        self._wer.update(d)


def compute_transcript_reward(
    hypothesis: str,
    reference: str,
    cer_weight: float = 0.6,
    wer_weight: float = 0.4,
) -> float:
    """
    Per-hypothesis accuracy reward: blend of (1-CER) and (1-WER).
    CER weighted higher for denser signal on short utterances.
    Returns value in [-1, 1].
    """
    ref = reference.lower().strip()
    hyp = hypothesis.lower().strip()

    if not ref:
        return 0.0
    if not hyp or hyp == "<empty>":
        return -1.0

    w = min(wer(ref, hyp), 1.0)
    c = min(cer(ref, hyp), 1.0)
    return cer_weight * (1.0 - c) + wer_weight * (1.0 - w)


def _batch_family_wers(rollouts: RolloutBatch) -> dict[str, float]:
    """Compute WER per family from top-1 hypothesis in rollouts."""
    family_refs: dict[str, list[str]] = defaultdict(list)
    family_hyps: dict[str, list[str]] = defaultdict(list)

    for i in range(len(rollouts.references)):
        fam = rollouts.families[i]
        ref = rollouts.references[i].lower().strip()
        hyp = rollouts.hypotheses[i][0].lower().strip()
        if ref:
            family_refs[fam].append(ref)
            family_hyps[fam].append(hyp if hyp else "")

    return {
        fam: min(wer(family_refs[fam], family_hyps[fam]), 1.0)
        for fam in family_refs
    }


def compute_rewards(
    rollouts: RolloutBatch,
    family_ema: FamilyEMA,
    cer_weight: float = 0.6,
    wer_weight: float = 0.4,
    alpha_fairness: float = 2.0,
    fairness_threshold: float = 0.05,
    rejection_wer_threshold: float = 0.9,
) -> torch.Tensor:
    """
    Compute rewards for all hypotheses in a rollout batch.

    reward[i, k] = transcript_reward(h_ik, y_i)
                 + alpha * max(0, EMA_WER(g_i) - WER(h_ik, y_i))

    The fairness bonus varies across K hypotheses because each h_ik has a
    different WER, so it is NOT cancelled by group-relative normalization.

    NOTE: normalize=True is REMOVED. Normalization happens only once,
    inside group_relative_advantages(). Double normalization corrupts the
    signal.

    Args:
        rollouts: RolloutBatch from generate_whisper_rollouts
        family_ema: FamilyEMA instance (updated externally after each batch)
        cer_weight: weight for CER component
        wer_weight: weight for WER component
        alpha_fairness: strength of fairness bonus
        fairness_threshold: minimum ΔDP before fairness bonus activates
        rejection_wer_threshold: zero out rewards for gibberish rollouts

    Returns:
        rewards: [B, K] tensor, unnormalized
    """
    B = len(rollouts.references)
    K = len(rollouts.hypotheses[0])
    rewards = torch.zeros(B, K)

    # Check whether fairness bonus should activate at all
    batch_fam_wers = _batch_family_wers(rollouts)
    delta_dp = (max(batch_fam_wers.values()) - min(batch_fam_wers.values())
                if len(batch_fam_wers) > 1 else 0.0)
    apply_fairness = delta_dp > fairness_threshold and alpha_fairness > 0

    for i in range(B):
        ref = rollouts.references[i]
        family = rollouts.families[i]
        ema_baseline = family_ema.get(family)

        for k in range(K):
            hyp = rollouts.hypotheses[i][k]

            # Transcript accuracy reward
            r_transcript = compute_transcript_reward(hyp, ref, cer_weight, wer_weight)

            # Fairness bonus: varies per hypothesis (survives normalization)
            r_fairness = 0.0
            if apply_fairness and ref and hyp and hyp != "<empty>":
                hyp_wer = min(wer(ref.lower().strip(), hyp.lower().strip()), 1.0)
                # Positive when this hypothesis beats the family's EMA baseline
                r_fairness = alpha_fairness * max(0.0, ema_baseline - hyp_wer)

            rewards[i, k] = r_transcript + r_fairness

    # Rejection: zero out gibberish rollouts
    for i in range(B):
        best_hyp = rollouts.hypotheses[i][0].lower().strip()
        ref = rollouts.references[i].lower().strip()
        if ref and best_hyp and best_hyp != "<empty>":
            if wer(ref, best_hyp) > rejection_wer_threshold:
                rewards[i, :] = 0.0

    # Update EMA with this batch's per-family WERs
    family_ema.update(batch_fam_wers)

    return rewards  # [B, K], unnormalized — normalization is in group_relative_advantages
```

---

## 3. Fix 2 — Double Normalization (Blocking)

### Problem

`compute_rewards` normalizes across the full B×K block. `group_relative_advantages` normalizes again within each utterance's K samples. The two normalizations together corrupt the signal.

### Solution

Remove `normalize=True` from `compute_rewards` (done in Fix 1 above). Keep normalization **only** in `group_relative_advantages`, and add numerical stability protection for the near-zero std case that occurs in ASR.

```python
def group_relative_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-6,          # increased from 1e-8 for ASR's tight distributions
    min_std: float = 0.01,      # floor to prevent degenerate division
) -> torch.Tensor:
    """
    GRPO group-relative advantages, normalized within each utterance's K rollouts.

    A_{i,k} = (r_{i,k} - mean_k) / max(std_k, min_std)

    min_std floor prevents degenerate normalization when K hypotheses have
    nearly identical WER (common in ASR, unlike reasoning tasks).
    """
    mean = rewards.mean(dim=1, keepdim=True)          # [B, 1]
    std  = rewards.std(dim=1, keepdim=True)            # [B, 1]
    std  = torch.clamp(std, min=min_std)               # floor
    return (rewards - mean) / (std + eps)
```

---

## 4. Fix 3 — Replace GRPO Loss with CISPO

### Why CISPO

**GRPO** (PPO-style) clips the ratio symmetrically and takes the minimum:

$$\mathcal{L}_\text{GRPO} = -\mathbb{E}\!\left[\min\!\left(r_t \hat{A}_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

When the ratio $r_t$ is large (policy moves far from reference), GRPO **zeroes the gradient** via the min. This caused the KL spikes — the optimizer finds regions where clipping is active and the gradient is zero, then takes a large unconstrained step.

**CISPO** (from MiniMax-M1) clips from above only, detaches the weight, and keeps the gradient flowing through $\log\pi_\theta$:

$$\mathcal{L}_\text{CISPO} = -\mathbb{E}\!\left[\underbrace{\text{detach}\!\left(\min(r_t, \epsilon_\text{high})\right)}_{\text{constant coefficient}} \cdot \hat{A}_t \cdot \log\pi_\theta(a_t|s_t)\right]$$

**Key properties:**
- No KL penalty term — eliminates the K3 estimator instability entirely
- Gradient flows through $\log\pi_\theta$ unconditionally for all tokens
- One-sided ceiling clip prevents runaway ratios without zeroing gradients
- Simpler: one hyperparameter ($\epsilon_\text{high}$) instead of $\beta_\text{KL}$ + clamp range

For ASR, the lack of KL penalty is appropriate — Whisper-small's transcription capability is already encoded in the base weights; we want the LoRA to fine-tune it, not be anchored to it.

### Code: `ast-asr/rl/whisper_grpo.py` — `whisper_cispo_step`

```python
def whisper_cispo_step(
    policy_model,
    ref_model,
    rollouts: RolloutBatch,
    processor,
    family_ema: "FamilyEMA",
    optimizer: torch.optim.Optimizer,
    epsilon_high: float = 5.0,      # CISPO clip ceiling; MiniMax uses ~5.0
    grad_clip: float = 1.0,
    cer_weight: float = 0.6,
    wer_weight: float = 0.4,
    alpha_fairness: float = 2.0,
    fairness_threshold: float = 0.05,
    grad_accum_steps: int = 1,
    do_step: bool = True,
) -> dict[str, float]:
    """
    One CISPO optimization step for Whisper.

    Loss: L = -E[ detach(min(r_t, eps_high)) * A_t * log π_θ(a_t) ]

    No KL penalty. Gradient flows through log π_θ unconditionally.
    Stability from one-sided ratio ceiling, not from KL anchoring.
    """
    device = rollouts.input_values.device

    # 1. Rewards [B, K] — unnormalized, fairness bonus varies per hypothesis
    rewards = compute_rewards(
        rollouts,
        family_ema=family_ema,
        cer_weight=cer_weight,
        wer_weight=wer_weight,
        alpha_fairness=alpha_fairness,
        fairness_threshold=fairness_threshold,
    ).to(device)

    # 2. Advantages [B, K] — single normalization, with min_std floor
    advantages = group_relative_advantages(rewards)

    # 3. Policy log-probs [B, K] — differentiable
    policy_log_probs = compute_whisper_log_probs(
        policy_model, processor, rollouts.input_values, rollouts.hypotheses
    )

    # 4. Reference log-probs [B, K] — detached (used only for ratio)
    with torch.no_grad():
        ref_log_probs = compute_whisper_log_probs(
            ref_model, processor, rollouts.input_values, rollouts.hypotheses
        )

    # 5. CISPO loss
    log_ratio = policy_log_probs - ref_log_probs      # [B, K]
    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))   # [B, K]

    # Clip from above, detach — becomes a constant coefficient
    clamped_ratio = torch.clamp(ratio, max=epsilon_high).detach()

    # Gradient flows through policy_log_probs only
    loss = -(clamped_ratio * advantages * policy_log_probs).mean()
    loss = loss / grad_accum_steps

    loss.backward()

    if do_step:
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy_model.parameters() if p.requires_grad],
            max_norm=grad_clip,
        )
        optimizer.step()
        optimizer.zero_grad()

    # Diagnostics — no KL to report, report ratio stats instead
    with torch.no_grad():
        ratio_mean = ratio.mean().item()
        ratio_max  = ratio.max().item()
        clipped_frac = (ratio > epsilon_high).float().mean().item()

    return {
        "loss":          float(loss.detach().cpu()) * grad_accum_steps,
        "reward_mean":   float(rewards.mean()),
        "reward_std":    float(rewards.std()),
        "advantage_std": float(advantages.std().detach().cpu()),
        "ratio_mean":    ratio_mean,
        "ratio_max":     ratio_max,
        "clipped_frac":  clipped_frac,   # fraction of tokens where ratio > eps_high
    }
```

### Monitoring

Replace `kl_mean` in your logging with `ratio_mean`, `ratio_max`, and `clipped_frac`.

- `ratio_mean` should stay near 1.0 in a healthy run
- `ratio_max` should stay well below `epsilon_high` most steps
- `clipped_frac` > 0.3 persistently → learning rate is too high

---

## 5. Fix 4 — Probe Set (Diagnostic)

### Problem

```python
# BROKEN: selects first 10 UIDs from full dataset, then filters to eval split
eval_df = load_svarah(max_samples=10, cache_dir="cache", svarah_split="eval")
# Result: 2 utterances (UIDs 1 and 6 only)
```

A 2-utterance probe across 1,500 training steps provides no useful signal about whether the model's output distribution has changed.

### Fix

```python
# In train_rl_whisper.py — load full eval split, then stratified subsample
probe_df = load_svarah(max_samples=None, cache_dir="cache", svarah_split="eval")
probe_df = (
    probe_df
    .groupby("language_family", group_keys=False)
    .apply(lambda x: x.sample(min(3, len(x)), random_state=42))
    .reset_index(drop=True)
)
logger.info("Probe set: %d utterances (%s)",
            len(probe_df),
            probe_df["language_family"].value_counts().to_dict())
```

This gives at minimum 9 utterances (3 per family), which is still small but at least covers all three families and gives a meaningful signal when outputs diverge.

### Improved Probe Logging

The probe should log the actual text, not just a count:

```python
if step > 0 and step % eval_interval == 0:
    policy_model.eval()
    diffs = []
    with torch.no_grad():
        for _, row in probe_df.iterrows():
            probe_input = processor(
                [row["audio_array"]], sampling_rate=16_000,
                return_tensors="pt", padding="max_length", truncation=True,
            ).input_features.to(device)

            pol_text = processor.batch_decode(
                policy_model.generate(probe_input, max_new_tokens=200,
                                      language="en", task="transcribe"),
                skip_special_tokens=True
            )[0].strip()

            ref_text = processor.batch_decode(
                ref_model.generate(probe_input, max_new_tokens=200,
                                   language="en", task="transcribe"),
                skip_special_tokens=True
            )[0].strip()

            hyp_wer = wer(row["reference"].lower(), pol_text.lower())
            differs = pol_text.lower() != ref_text.lower()
            if differs:
                diffs.append((row["language_family"], hyp_wer, ref_text, pol_text))

    logger.info("Probe step %d: %d/%d differ | EMA WERs: %s",
                step, len(diffs), len(probe_df), family_ema.state_dict())
    for fam, hw, rt, pt in diffs:
        logger.info("  [%s] ref='%s' → pol='%s' (WER=%.3f)", fam, rt[:60], pt[:60], hw)
    policy_model.train()
```

---

## 6. Implementation Priority

| Priority | Fix | Files | Impact |
|----------|-----|-------|--------|
| **1 — Blocking** | Fairness reward cancellation (Fix 1) | `rl/reward.py` | Research claim validity |
| **2 — Blocking** | Double normalization (Fix 2) | `rl/reward.py`, `rl/grpo.py` | Signal correctness |
| **3 — Stability** | CISPO loss (Fix 3) | `rl/whisper_grpo.py` | Training stability, no KL spikes |
| **4 — Diagnostic** | Probe set (Fix 4) | `train_rl_whisper.py` | Visibility into training |

Fixes 1 and 2 must be in place before any results are reportable. The model may appear to learn without them but the mechanism is not what the paper claims.

Fix 3 (CISPO) can be applied independently — it is a drop-in replacement for the loss computation and does not interact with the fairness reward logic.

Fix 4 is cheap and should be done immediately.

---

## 7. Full Rewritten Files

### 7.1 `ast-asr/rl/reward.py`

Complete replacement — see Fix 1 section above. Key changes:

- `FamilyEMA` dataclass added, passed as argument to `compute_rewards`
- `compute_rewards` no longer accepts or applies `normalize` parameter
- Fairness bonus computed per-hypothesis (varies across K, survives normalization)
- `compute_family_weights` removed (was the root of the cancellation bug)
- `family_ema.update(batch_fam_wers)` called at end of `compute_rewards`

### 7.2 `ast-asr/rl/whisper_grpo.py`

Key changes:

- `whisper_cispo_step` replaces `whisper_grpo_step`
- `FamilyEMA` instance passed in and forwarded to `compute_rewards`
- KL penalty removed entirely
- Ratio computed, clamped at `epsilon_high`, detached
- Loss is `-(clamped_ratio * advantages * policy_log_probs).mean()`
- `group_relative_advantages` updated with `min_std` floor

### 7.3 `ast-asr/train_rl_whisper.py`

Key changes:

- `FamilyEMA` instantiated once before the training loop, persists across steps
- Probe set fixed to stratified 3-per-family sample from full eval split
- `whisper_grpo_step` call replaced with `whisper_cispo_step`
- Logging updated: `kl_mean` replaced with `ratio_mean`, `ratio_max`, `clipped_frac`, `ema_wers`
- `family_ema.state_dict()` saved alongside model checkpoint for reproducibility

### 7.4 `configs/train_rl_whisper.yaml`

```yaml
model:
  id: openai/whisper-small
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, out_proj]

rl:
  algorithm: cispo             # changed from grpo
  epsilon_high: 5.0            # CISPO ceiling; replaces beta_kl + clip_epsilon
  K: 4
  temperature: 1.2
  max_steps: 1500
  lr: 2.0e-5
  grad_clip: 1.0
  batch_size: 3
  gradient_accumulation: 4
  eval_interval: 100
  save_interval: 150
  weight_decay: 0.0

reward:
  cer_weight: 0.6
  wer_weight: 0.4
  alpha_fairness: 2.0
  fairness_threshold: 0.05
  rejection_wer_threshold: 0.9

family_ema:
  beta: 0.05                   # EMA decay rate
  initial_wer: 0.40            # conservative prior

output:
  dir: outputs/checkpoints/whisper-small-cispo
```

---

## Appendix: What a Healthy Training Run Should Look Like

| Metric | Early steps (0–100) | Mid training (500–1000) | Late training (1200–1500) |
|--------|--------------------|-----------------------|--------------------------|
| `reward_mean` | 0.45–0.55 | Noisy upward trend | 0.55–0.70 |
| `reward_std` | 0.15–0.25 | Stable | Stable or slightly narrowing |
| `ratio_mean` | ~1.0 | 1.0–2.0 | 1.0–3.0 |
| `ratio_max` | < epsilon_high | < epsilon_high most steps | Occasional spikes OK |
| `clipped_frac` | < 0.05 | < 0.15 | < 0.20 |
| `advantage_std` | > 0.1 | > 0.1 | > 0.05 |
| `ema_wer[Sino-Tibetan]` | ~0.40 | Decreasing | Lower than IA/Drav baselines converging |
| Probe diffs | 0/9 | 3–6/9 | 4–8/9 |

If `advantage_std` drops below 0.05 persistently, the K hypotheses are too similar — increase `temperature` to 1.4–1.6 to get more diverse rollouts.

If `clipped_frac` exceeds 0.25 persistently, reduce learning rate to `1.0e-5`.

---

*All fixes verified against: v5 training log (`whisper-rl-fair-v5_train.log`), CISPO paper (MiniMax-M1, arXiv:2506.13585), MWER reference (Prabhavalkar et al., ICASSP 2018).*
