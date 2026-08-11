from __future__ import annotations

import math

import pytest
import torch

from ast_asr.objectives import (
    AdvantageKind,
    ClipRule,
    ObjectiveSpec,
    RatioUnit,
    centered_mwer_advantages,
    policy_objective,
    sequence_importance_ratio,
)


def test_sequence_importance_ratio_matches_mean_token_log_ratio() -> None:
    current = torch.tensor([[[-0.5, -1.0, -9.0]]], dtype=torch.float32)
    old = torch.tensor([[[-1.0, -1.5, -7.0]]], dtype=torch.float32)
    mask = torch.tensor([[[True, True, False]]])

    ratio = sequence_importance_ratio(current, old, mask)

    assert ratio.shape == (1, 1)
    assert ratio.item() == pytest.approx(math.exp(0.5))


def test_centered_mwer_clips_wer_without_standardizing_candidates() -> None:
    wers = torch.tensor([[0.0, 1.0, 3.0, 2.0]], dtype=torch.float32)

    advantages = centered_mwer_advantages(wers)

    torch.testing.assert_close(
        advantages,
        torch.tensor([[1.25, 0.25, -0.75, -0.75]]),
    )
    assert advantages.mean().item() == pytest.approx(0.0)


def test_sequence_cispo_detaches_ratio_and_weights_after_centering() -> None:
    current = torch.full((2, 2, 1), -0.2, requires_grad=True)
    old = current.detach().clone()
    mask = torch.ones_like(current, dtype=torch.bool)
    wers = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    utterance_weights = torch.tensor([1.0, 3.0])
    spec = ObjectiveSpec(
        advantage=AdvantageKind.CENTERED_MWER,
        ratio_unit=RatioUnit.SEQUENCE,
        clip_rule=ClipRule.CISPO_UPPER,
        clip_upper=2.0,
    )

    result = policy_objective(
        spec,
        current_token_log_probs=current,
        old_token_log_probs=old,
        token_mask=mask,
        candidate_wers=wers,
        utterance_weights=utterance_weights,
    )
    result.loss.backward()

    torch.testing.assert_close(result.ratios, torch.ones(2, 2))
    torch.testing.assert_close(
        current.grad,
        torch.tensor([[[-0.125], [0.125]], [[-0.375], [0.375]]]),
    )
