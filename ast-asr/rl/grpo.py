"""
grpo.py
Group Relative Policy Optimization (GRPO) for CTC-based ASR.

Adapts GRPO (from DeepSeek-R1) for CTC models. The key difference from
autoregressive GRPO: hypotheses come from CTC beam search, and
log-probabilities are computed via differentiable F.ctc_loss(reduction='none').
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .beam_search import RolloutBatch, compute_ctc_log_probs
from .reward import compute_rewards

logger = logging.getLogger(__name__)


def group_relative_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute group-relative advantages for GRPO.

    For each utterance i, the advantage of hypothesis k is:
        A_ik = (r_ik - mean(r_i*)) / (std(r_i*) + eps)

    This normalizes within the group (utterance), so only the relative
    ranking of hypotheses matters, not absolute reward scale.

    Args:
        rewards: [B, K] per-hypothesis rewards
        eps: small constant for numerical stability

    Returns:
        advantages: [B, K] group-relative advantages
    """
    mean = rewards.mean(dim=1, keepdim=True)  # [B, 1]
    std = rewards.std(dim=1, keepdim=True)    # [B, 1]
    return (rewards - mean) / (std + eps)


def grpo_step(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    rollouts: RolloutBatch,
    processor,
    optimizer: torch.optim.Optimizer,
    beta_kl: float = 0.1,
    grad_clip: float = 1.0,
    cer_weight: float = 0.6,
    wer_weight: float = 0.4,
    alpha_fairness: float = 2.0,
    fairness_threshold: float = 0.05,
) -> dict[str, float]:
    """
    Perform one GRPO optimization step.

    1. Compute rewards for beam search rollouts
    2. Compute group-relative advantages
    3. Forward pass through policy for differentiable CTC log-probs
    4. Forward pass through frozen reference for KL penalty
    5. Compute GRPO loss and backprop

    Args:
        policy_model: the model being optimized (HybridAdversarialASR)
        ref_model: frozen reference model (ft-w2v2 checkpoint)
        rollouts: beam search hypotheses from generate_rollouts()
        processor: Wav2Vec2Processor for tokenizing hypotheses
        optimizer: optimizer for policy model parameters
        beta_kl: KL penalty coefficient
        grad_clip: max gradient norm for clipping
        cer_weight: CER weight in transcript reward
        wer_weight: WER weight in transcript reward
        alpha_fairness: fairness penalty strength
        fairness_threshold: ΔDP threshold for fairness penalty

    Returns:
        dict with loss, reward_mean, kl_mean, advantage_std metrics
    """
    device = rollouts.input_values.device
    B = len(rollouts.references)
    K = len(rollouts.hypotheses[0])

    # Use eval mode for deterministic log-prob computation (disables dropout
    # and SpecAugment). Gradients still flow — eval() only affects stochastic
    # layers, not autograd tracking. This ensures policy and ref log-probs
    # are directly comparable for the KL penalty.
    was_training = policy_model.training
    policy_model.eval()

    # 1. Compute rewards [B, K]
    rewards = compute_rewards(
        rollouts,
        cer_weight=cer_weight,
        wer_weight=wer_weight,
        alpha_fairness=alpha_fairness,
        fairness_threshold=fairness_threshold,
        normalize=True,
    ).to(device)

    # 2. Group-relative advantages [B, K]
    advantages = group_relative_advantages(rewards)

    # 3. Differentiable CTC log-probs under the current policy [B, K]
    policy_log_probs = compute_ctc_log_probs(
        policy_model,
        rollouts.input_values,
        rollouts.attention_mask,
        rollouts.hypotheses,
        processor,
    )

    # 4. Reference log-probs (detached) [B, K]
    with torch.no_grad():
        ref_log_probs = compute_ctc_log_probs(
            ref_model,
            rollouts.input_values,
            rollouts.attention_mask,
            rollouts.hypotheses,
            processor,
        )

    # 5. GRPO loss: -advantage * log_pi + beta * KL
    #    KL(pi || pi_ref) ≈ log_pi - log_pi_ref (per-sample)
    kl = policy_log_probs - ref_log_probs  # [B, K]

    # Policy gradient: weighted by advantage
    pg_loss = -(advantages * policy_log_probs).mean()

    # KL penalty
    kl_loss = beta_kl * kl.mean()

    total_loss = pg_loss + kl_loss

    # 6. Backward + step
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in policy_model.parameters() if p.requires_grad],
        max_norm=grad_clip,
    )
    optimizer.step()

    # Restore training mode
    if was_training:
        policy_model.train()

    # Metrics
    with torch.no_grad():
        raw_rewards = compute_rewards(
            rollouts,
            cer_weight=cer_weight,
            wer_weight=wer_weight,
            alpha_fairness=alpha_fairness,
            fairness_threshold=fairness_threshold,
            normalize=False,
        )

    return {
        "loss": float(total_loss.detach().cpu()),
        "pg_loss": float(pg_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "reward_mean": float(raw_rewards.mean()),
        "reward_std": float(raw_rewards.std()),
        "kl_mean": float(kl.mean().detach().cpu()),
        "advantage_std": float(advantages.std().detach().cpu()),
    }
