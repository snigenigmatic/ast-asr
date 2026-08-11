from __future__ import annotations

import pytest
import torch

from ast_asr.objectives import AdvantageKind, ClipRule, ObjectiveSpec, RatioUnit
from ast_asr.optimization import (
    InnerUpdateSafetyStop,
    deterministic_model_mode,
    optimize_frozen_rollout,
)


def test_four_inner_updates_reuse_old_scores_and_move_live_ratios() -> None:
    parameters = torch.nn.Parameter(torch.zeros(1, 2, 1))
    optimizer = torch.optim.SGD([parameters], lr=0.2)
    old = torch.zeros(1, 2, 1, dtype=torch.float32)
    old_before = old.clone()
    mask = torch.ones_like(old, dtype=torch.bool)
    wers = torch.tensor([[0.0, 1.0]])
    spec = ObjectiveSpec(
        advantage=AdvantageKind.CENTERED_MWER,
        ratio_unit=RatioUnit.SEQUENCE,
        clip_rule=ClipRule.CISPO_UPPER,
        clip_upper=2.0,
    )

    diagnostics = optimize_frozen_rollout(
        spec=spec,
        score_current=lambda: parameters,
        old_token_log_probs=old,
        token_mask=mask,
        candidate_wers=wers,
        optimizer=optimizer,
        inner_updates=4,
    )

    torch.testing.assert_close(diagnostics[0].ratios, torch.ones(1, 2))
    assert not torch.equal(diagnostics[1].ratios, torch.ones(1, 2))
    assert diagnostics[0].ratio_p01 == 1.0
    assert diagnostics[0].ratio_is_finite is True
    assert diagnostics[0].ratio_median == 1.0
    assert diagnostics[0].ratio_p99 == 1.0
    assert diagnostics[0].ratio_max == 1.0
    assert diagnostics[1].ratio_p01 <= diagnostics[1].ratio_median
    assert diagnostics[1].ratio_median <= diagnostics[1].ratio_p99
    assert diagnostics[1].ratio_p99 <= diagnostics[1].ratio_max
    torch.testing.assert_close(old, old_before)
    assert len(diagnostics) == 5
    assert [item.update for item in diagnostics] == [0, 1, 2, 3, 4]


def test_final_optimizer_step_cannot_escape_ratio_safety_measurement() -> None:
    parameters = torch.nn.Parameter(torch.zeros(1, 2, 1))
    optimizer = torch.optim.SGD([parameters], lr=0.2)
    old = torch.zeros(1, 2, 1, dtype=torch.float32)
    mask = torch.ones_like(old, dtype=torch.bool)
    wers = torch.tensor([[0.0, 1.0]])
    spec = ObjectiveSpec(
        advantage=AdvantageKind.CENTERED_MWER,
        ratio_unit=RatioUnit.SEQUENCE,
        clip_rule=ClipRule.CISPO_UPPER,
        clip_upper=2.0,
    )

    with pytest.raises(InnerUpdateSafetyStop) as error:
        optimize_frozen_rollout(
            spec=spec,
            score_current=lambda: parameters,
            old_token_log_probs=old,
            token_mask=mask,
            candidate_wers=wers,
            optimizer=optimizer,
            inner_updates=4,
            update_safety_check=lambda item: (
                "unsafe post-fourth-step ratio" if item.update == 4 else None
            ),
        )

    assert [item.update for item in error.value.diagnostics] == [0, 1, 2, 3, 4]


def test_deterministic_scoring_disables_dropout_without_disabling_gradients() -> None:
    model = torch.nn.Sequential(torch.nn.Dropout(p=0.9), torch.nn.Linear(4, 1, bias=False))
    inputs = torch.ones(3, 4)
    model.train()

    with deterministic_model_mode(model):
        first = model(inputs)
        second = model(inputs)
        second.sum().backward()

    torch.testing.assert_close(first, second)
    assert model.training is True
    assert model[1].weight.grad is not None
