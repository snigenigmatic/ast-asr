"""Policy objectives used by the FR-CISPO training ladder."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class AdvantageKind(StrEnum):
    """How candidate-level credit is constructed within an utterance."""

    CENTERED_MWER = "centered_mwer"
    STANDARDIZED = "standardized"


class RatioUnit(StrEnum):
    """Unit at which current and rollout policies are compared."""

    NONE = "none"
    TOKEN = "token"
    SEQUENCE = "sequence"


class ClipRule(StrEnum):
    """How importance ratios enter the objective."""

    NONE = "none"
    PPO_SYMMETRIC = "ppo_symmetric"
    CISPO_UPPER = "cispo_upper"


class GroupWeighting(StrEnum):
    """Risk weighting applied after within-utterance advantage construction."""

    UNIFORM = "uniform"
    DUAL = "dual"


class CorruptionPolicy(StrEnum):
    """Acoustic conditions represented in each rollout batch."""

    CLEAN = "clean"
    PAIRED_WHITE = "paired_white"


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    """Independent choices defining one arm of the training ladder."""

    advantage: AdvantageKind
    ratio_unit: RatioUnit
    clip_rule: ClipRule
    group_weighting: GroupWeighting = GroupWeighting.UNIFORM
    corruption: CorruptionPolicy = CorruptionPolicy.CLEAN
    clip_lower: float = 0.8
    clip_upper: float = 2.0
    wer_cap: float = 2.0
    advantage_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.clip_lower <= 0 or self.clip_upper <= 0:
            raise ValueError("ratio clip limits must be positive")
        if self.clip_lower > self.clip_upper:
            raise ValueError("clip_lower cannot exceed clip_upper")
        if self.wer_cap <= 0:
            raise ValueError("wer_cap must be positive")


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    """Differentiable loss plus diagnostics from the same calculation."""

    loss: torch.Tensor
    ratios: torch.Tensor
    advantages: torch.Tensor
    candidate_objectives: torch.Tensor


def centered_mwer_advantages(
    candidate_wers: torch.Tensor,
    *,
    wer_cap: float = 2.0,
) -> torch.Tensor:
    """Return negative, candidate-centered clipped WER for each utterance."""
    if candidate_wers.ndim != 2:
        raise ValueError("candidate WERs must have shape [utterance, candidate]")
    clipped = candidate_wers.clamp(max=wer_cap)
    return -(clipped - clipped.mean(dim=1, keepdim=True))


def sequence_importance_ratio(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Return ``exp(mean_t(log pi - log pi_old))`` for each sequence.

    All three tensors have shape ``[utterance, candidate, token]``. Masked
    positions never contribute to the sequence statistic.
    """
    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("current and old log-probability shapes must match")
    if token_mask.shape != current_log_probs.shape:
        raise ValueError("token mask must match log-probability shape")

    mask = token_mask.to(dtype=current_log_probs.dtype)
    lengths = mask.sum(dim=-1)
    if torch.any(lengths == 0):
        raise ValueError("every candidate must contain at least one scored token")

    mean_log_ratio = ((current_log_probs - old_log_probs) * mask).sum(dim=-1)
    mean_log_ratio = mean_log_ratio / lengths
    return torch.exp(mean_log_ratio)


def sampled_k3_reference_kl(
    current_token_log_probs: torch.Tensor,
    reference_token_log_probs: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a masked sampled-K3 penalty against a frozen reference policy.

    The rollout hypotheses are sampled once and remain fixed during every
    inner update.  This is therefore a response-token penalty, rather than a
    full-vocabulary KL.  ``reference_token_log_probs`` is deliberately
    detached: the SFT reference is evidence, never an optimization target.
    The K3 form is non-negative pointwise and has useful gradients on both
    sides of the reference log-probability.
    """
    if current_token_log_probs.shape != reference_token_log_probs.shape:
        raise ValueError("current and reference log-probability shapes must match")
    if token_mask.shape != current_token_log_probs.shape:
        raise ValueError("token mask must match log-probability shape")
    if reference_token_log_probs.dtype != torch.float32:
        raise ValueError("reference token log-probabilities must be FP32")
    if not bool(token_mask.any()):
        raise ValueError("reference KL requires at least one scored token")

    log_ratio = (
        reference_token_log_probs.detach().float() - current_token_log_probs.float()
    )
    selected = log_ratio[token_mask]
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError("reference KL received non-finite scored tokens")
    if bool((selected.abs() > 20.0).any()):
        raise FloatingPointError(
            "reference KL token log-ratio exceeded the numerical safety bound"
        )
    # expm1 is stable around zero and preserves K3's mathematical
    # non-negativity at the cycle-zero identity check.
    k3 = torch.expm1(log_ratio) - log_ratio
    return k3[token_mask].mean()


def _standardized_advantages(
    candidate_wers: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    rewards = -candidate_wers
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    scale = rewards.std(dim=1, keepdim=True, unbiased=False)
    return centered / scale.clamp_min(epsilon)


def _token_importance_ratio(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    log_ratio = current_log_probs - old_log_probs
    return torch.where(token_mask, torch.exp(log_ratio), torch.ones_like(log_ratio))


def policy_objective(
    spec: ObjectiveSpec,
    *,
    current_token_log_probs: torch.Tensor,
    old_token_log_probs: torch.Tensor,
    token_mask: torch.Tensor,
    candidate_wers: torch.Tensor,
    utterance_weights: torch.Tensor | None = None,
) -> ObjectiveResult:
    """Calculate an objective arm through one stable, testable interface.

    ``utterance_weights`` must have mean one. Keeping this scaling invariant
    makes uniform dual weights exactly equivalent to the unweighted objective.
    """
    if current_token_log_probs.ndim != 3:
        raise ValueError("token log-probabilities must have shape [B, K, T]")
    if candidate_wers.shape != current_token_log_probs.shape[:2]:
        raise ValueError("candidate WERs must have shape [B, K]")
    if old_token_log_probs.dtype != torch.float32:
        raise ValueError("old token log-probabilities must be stored in FP32")

    mask = token_mask.to(dtype=current_token_log_probs.dtype)
    lengths = mask.sum(dim=-1)
    if torch.any(lengths == 0):
        raise ValueError("every candidate must contain at least one scored token")
    mean_current = (current_token_log_probs * mask).sum(dim=-1) / lengths

    if spec.advantage is AdvantageKind.CENTERED_MWER:
        advantages = centered_mwer_advantages(candidate_wers, wer_cap=spec.wer_cap)
    else:
        advantages = _standardized_advantages(
            candidate_wers,
            spec.advantage_epsilon,
        )

    if utterance_weights is None:
        weights = torch.ones(
            current_token_log_probs.shape[0],
            dtype=current_token_log_probs.dtype,
            device=current_token_log_probs.device,
        )
    else:
        if utterance_weights.shape != (current_token_log_probs.shape[0],):
            raise ValueError("utterance weights must have shape [B]")
        weights = utterance_weights.to(
            dtype=current_token_log_probs.dtype,
            device=current_token_log_probs.device,
        )

    if spec.ratio_unit is RatioUnit.SEQUENCE:
        ratios = sequence_importance_ratio(
            current_token_log_probs,
            old_token_log_probs,
            token_mask,
        )
    elif spec.ratio_unit is RatioUnit.TOKEN:
        ratios = _token_importance_ratio(
            current_token_log_probs,
            old_token_log_probs,
            token_mask,
        )
    else:
        ratios = torch.ones_like(mean_current)

    advantage_tokens = advantages.unsqueeze(-1)
    if spec.clip_rule is ClipRule.PPO_SYMMETRIC:
        if spec.ratio_unit is RatioUnit.NONE:
            raise ValueError("PPO clipping requires token or sequence ratios")
        clipped = ratios.clamp(spec.clip_lower, spec.clip_upper)
        if spec.ratio_unit is RatioUnit.TOKEN:
            surrogate = torch.minimum(
                ratios * advantage_tokens,
                clipped * advantage_tokens,
            )
            candidate_objectives = (surrogate * mask).sum(dim=-1) / lengths
        else:
            candidate_objectives = torch.minimum(
                ratios * advantages,
                clipped * advantages,
            )
    elif spec.clip_rule is ClipRule.CISPO_UPPER:
        if spec.ratio_unit is RatioUnit.NONE:
            raise ValueError("CISPO clipping requires token or sequence ratios")
        stopped = ratios.clamp(max=spec.clip_upper).detach()
        if spec.ratio_unit is RatioUnit.TOKEN:
            weighted_log_probs = stopped * advantage_tokens * current_token_log_probs
            candidate_objectives = (weighted_log_probs * mask).sum(dim=-1) / lengths
        else:
            candidate_objectives = stopped * advantages * mean_current
    else:
        candidate_objectives = advantages * mean_current

    loss = -(candidate_objectives * weights.unsqueeze(1)).mean()
    return ObjectiveResult(
        loss=loss,
        ratios=ratios,
        advantages=advantages,
        candidate_objectives=candidate_objectives,
    )
