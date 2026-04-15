"""
reward.py
Multi-component reward computation for GRPO-based RL post-training.

Components:
  r_transcript : per-hypothesis accuracy reward (CER + WER blend)
  r_fairness   : batch-level demographic parity penalty
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


def compute_fairness_reward(
    rollouts: RolloutBatch,
    alpha: float = 2.0,
    threshold: float = 0.05,
) -> float:
    """
    Batch-level fairness penalty based on demographic parity gap (ΔDP).

    r_fairness = -alpha * max(0, ΔDP - threshold)

    Only penalizes when the WER gap between the best and worst family
    in the batch exceeds the threshold (default 5pp). Uses the top-1
    beam hypothesis per utterance to compute per-family WER.

    Returns a scalar applied uniformly to all utterances in the batch.
    """
    # Group best hypotheses by family
    family_refs: dict[str, list[str]] = defaultdict(list)
    family_hyps: dict[str, list[str]] = defaultdict(list)

    for i in range(len(rollouts.references)):
        fam = rollouts.families[i]
        ref = rollouts.references[i].lower().strip()
        hyp = rollouts.hypotheses[i][0].lower().strip()  # top-1 beam
        if ref:
            family_refs[fam].append(ref)
            family_hyps[fam].append(hyp if hyp and hyp != "<empty>" else "")

    if len(family_refs) < 2:
        return 0.0  # can't compute fairness gap with < 2 families

    # Compute per-family WER
    family_wer: dict[str, float] = {}
    for fam in family_refs:
        family_wer[fam] = min(wer(family_refs[fam], family_hyps[fam]), 1.0)

    delta_dp = max(family_wer.values()) - min(family_wer.values())
    penalty = -alpha * max(0.0, delta_dp - threshold)
    return penalty


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

    Args:
        rollouts: RolloutBatch from beam search
        cer_weight: weight for CER component in transcript reward
        wer_weight: weight for WER component in transcript reward
        alpha_fairness: strength of fairness penalty
        fairness_threshold: ΔDP threshold below which no penalty applies
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

    # Batch-level fairness reward (same for all utterances)
    r_fair = compute_fairness_reward(
        rollouts, alpha=alpha_fairness, threshold=fairness_threshold
    )
    rewards += r_fair

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
