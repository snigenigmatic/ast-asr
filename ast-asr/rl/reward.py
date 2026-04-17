"""
reward.py
Multi-component reward computation for GRPO-based RL post-training.

Components:
  r_transcript : per-hypothesis accuracy reward (CER + WER blend)
  r_fairness   : per-utterance family-aware scaling (NOT a batch constant)

The fairness reward must vary per-utterance to survive GRPO's group-relative
advantage normalization. A batch-level constant c added to all r_{i,k} cancels
in A_{i,k} = (r_{i,k} - mean_k) / std_k. Instead, we scale transcript rewards
by family need: utterances from high-WER families get boosted, low-WER families
get dampened.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
from jiwer import cer, wer

from .beam_search import RolloutBatch

logger = logging.getLogger(__name__)


def compute_transcript_reward(
    hypothesis: str,
    reference: str,
    cer_weight: float = 0.6,
    wer_weight: float = 0.4,
) -> float:
    """
    Per-hypothesis accuracy reward.

    r = cer_weight * (1 - CER) + wer_weight * (1 - WER)

    CER is weighted higher because it provides denser signal than WER
    for short utterances (a single word error gives WER=1.0 for a
    1-word reference, but CER remains informative).
    """
    ref = reference.lower().strip()
    hyp = hypothesis.lower().strip()

    if not ref:
        return 0.0
    if not hyp or hyp == "<empty>":
        return -1.0

    w = min(wer(ref, hyp), 1.0)  # cap at 1.0
    c = min(cer(ref, hyp), 1.0)
    return cer_weight * (1.0 - c) + wer_weight * (1.0 - w)


def compute_family_weights(
    rollouts: RolloutBatch,
    alpha: float = 2.0,
    threshold: float = 0.05,
) -> dict[str, float]:
    """
    Compute per-family reward scaling weights based on ΔDP.

    Returns a multiplier per family:
      - Worst-performing family gets weight (1 + alpha * gap_from_mean)
      - Best-performing family gets weight (1 - alpha * gap_from_mean)
      - Families within threshold of each other get weight 1.0

    This creates per-utterance reward variation that survives GRPO's
    group-relative normalization (unlike a batch-level constant).
    """
    family_refs: dict[str, list[str]] = defaultdict(list)
    family_hyps: dict[str, list[str]] = defaultdict(list)

    for i in range(len(rollouts.references)):
        fam = rollouts.families[i]
        ref = rollouts.references[i].lower().strip()
        hyp = rollouts.hypotheses[i][0].lower().strip()  # top-1
        if ref:
            family_refs[fam].append(ref)
            family_hyps[fam].append(hyp if hyp and hyp != "<empty>" else "")

    if len(family_refs) < 2:
        return {fam: 1.0 for fam in family_refs}

    # Per-family WER
    family_wer: dict[str, float] = {}
    for fam in family_refs:
        family_wer[fam] = min(wer(family_refs[fam], family_hyps[fam]), 1.0)

    delta_dp = max(family_wer.values()) - min(family_wer.values())

    if delta_dp <= threshold:
        return {fam: 1.0 for fam in family_wer}

    # Scale: high-WER families get boosted, low-WER get dampened
    mean_wer = sum(family_wer.values()) / len(family_wer)
    weights = {}
    for fam, fw in family_wer.items():
        # gap > 0 for worse-than-average families (they need more help)
        gap = fw - mean_wer
        # Boost range: [1 - alpha*max_gap, 1 + alpha*max_gap]
        # Clamped to [0.2, 3.0] to avoid sign flips or extreme scaling
        weights[fam] = max(0.2, min(3.0, 1.0 + alpha * gap))

    return weights


def compute_rewards(
    rollouts: RolloutBatch,
    cer_weight: float = 0.6,
    wer_weight: float = 0.4,
    alpha_fairness: float = 2.0,
    fairness_threshold: float = 0.05,
    rejection_wer_threshold: float = 0.9,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute full rewards for all hypotheses in a rollout batch.

    The fairness component scales per-utterance transcript rewards by
    family need (high-WER families get boosted). This variation is
    per-utterance and survives GRPO group-relative normalization.

    Args:
        rollouts: RolloutBatch from beam search
        cer_weight: weight for CER component in transcript reward
        wer_weight: weight for WER component in transcript reward
        alpha_fairness: strength of fairness scaling
        fairness_threshold: ΔDP threshold below which no scaling applies
        rejection_wer_threshold: reject rollouts above this WER
        normalize: standardize rewards within mini-batch

    Returns:
        rewards: [B, K] tensor of rewards per hypothesis
    """
    B = len(rollouts.references)
    K = len(rollouts.hypotheses[0])

    # Per-hypothesis transcript rewards
    rewards = torch.zeros(B, K)
    for i in range(B):
        ref = rollouts.references[i]
        for k in range(K):
            hyp = rollouts.hypotheses[i][k]
            rewards[i, k] = compute_transcript_reward(
                hyp, ref, cer_weight, wer_weight
            )

    # Per-family fairness scaling (varies per utterance, not a constant)
    if alpha_fairness > 0:
        family_weights = compute_family_weights(
            rollouts, alpha=alpha_fairness, threshold=fairness_threshold
        )
        for i in range(B):
            fam = rollouts.families[i]
            w = family_weights.get(fam, 1.0)
            rewards[i, :] *= w

    # Rejection: mask gibberish rollouts (best-hyp WER > threshold)
    for i in range(B):
        best_hyp = rollouts.hypotheses[i][0].lower().strip()
        ref = rollouts.references[i].lower().strip()
        if ref and best_hyp and best_hyp != "<empty>":
            best_wer = wer(ref, best_hyp)
            if best_wer > rejection_wer_threshold:
                rewards[i, :] = 0.0  # zero out — no gradient signal

    # Normalize within batch
    if normalize and rewards.numel() > 1:
        flat = rewards[rewards != 0.0]
        if len(flat) > 1:
            rewards = (rewards - flat.mean()) / (flat.std() + 1e-8)

    return rewards
