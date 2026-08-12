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
    assert [item.optimizer_step_applied for item in diagnostics] == [
        True,
        True,
        True,
        True,
        False,
    ]


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


def test_non_finite_parameters_after_an_optimizer_step_fail_closed() -> None:
    class CorruptingOptimizer(torch.optim.Optimizer):
        def __init__(self, values: list[torch.nn.Parameter]) -> None:
            super().__init__(values, {})

        def step(self, closure=None):
            del closure
            with torch.no_grad():
                for group in self.param_groups:
                    for parameter in group["params"]:
                        parameter.fill_(float("nan"))

    parameters = torch.nn.Parameter(torch.zeros(1, 2, 1))
    old = torch.zeros_like(parameters, dtype=torch.float32)
    mask = torch.ones_like(old, dtype=torch.bool)
    wers = torch.tensor([[0.0, 1.0]])
    spec = ObjectiveSpec(
        advantage=AdvantageKind.CENTERED_MWER,
        ratio_unit=RatioUnit.SEQUENCE,
        clip_rule=ClipRule.CISPO_UPPER,
    )

    with pytest.raises(FloatingPointError, match="non-finite trainable parameter"):
        optimize_frozen_rollout(
            spec=spec,
            score_current=lambda: parameters,
            old_token_log_probs=old,
            token_mask=mask,
            candidate_wers=wers,
            optimizer=CorruptingOptimizer([parameters]),
            inner_updates=1,
        )


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


def test_zero_reference_kl_beta_is_exactly_the_existing_policy_loss() -> None:
    baseline = torch.nn.Parameter(torch.tensor([[[-1.2], [-0.8]]]))
    with_zero_beta = torch.nn.Parameter(baseline.detach().clone())
    old = torch.tensor([[[-1.2], [-0.8]]], dtype=torch.float32)
    reference = torch.tensor([[[-1.0], [-1.0]]], dtype=torch.float32)
    mask = torch.ones_like(old, dtype=torch.bool)
    wers = torch.tensor([[0.0, 1.0]])
    spec = ObjectiveSpec(
        advantage=AdvantageKind.CENTERED_MWER,
        ratio_unit=RatioUnit.SEQUENCE,
        clip_rule=ClipRule.CISPO_UPPER,
    )

    baseline_diagnostics = optimize_frozen_rollout(
        spec=spec,
        score_current=lambda: baseline,
        old_token_log_probs=old,
        token_mask=mask,
        candidate_wers=wers,
        optimizer=torch.optim.SGD([baseline], lr=0.1),
        inner_updates=1,
    )
    zero_beta_diagnostics = optimize_frozen_rollout(
        spec=spec,
        score_current=lambda: with_zero_beta,
        old_token_log_probs=old,
        token_mask=mask,
        candidate_wers=wers,
        optimizer=torch.optim.SGD([with_zero_beta], lr=0.1),
        inner_updates=1,
        reference_token_log_probs=reference,
        reference_kl_beta=0.0,
    )

    torch.testing.assert_close(baseline, with_zero_beta)
    assert [item.loss for item in baseline_diagnostics] == [
        item.loss for item in zero_beta_diagnostics
    ]
    assert all(item.reference_kl_evaluated for item in zero_beta_diagnostics)
    assert all(item.reference_kl_value > 0.0 for item in zero_beta_diagnostics)
    assert all(item.reference_kl_loss == 0.0 for item in zero_beta_diagnostics)
    assert [item.total_loss for item in zero_beta_diagnostics] == [
        item.loss for item in zero_beta_diagnostics
    ]


def test_positive_reference_kl_adds_gradients_without_unstopping_cispo_ratio() -> None:
    parameters = torch.nn.Parameter(torch.tensor([[[-1.2], [-1.2]]]))
    old = parameters.detach().clone().float()
    reference = torch.tensor([[[-1.0], [-1.0]]], dtype=torch.float32)
    mask = torch.ones_like(old, dtype=torch.bool)
    # Equal WERs make the centered CISPO policy gradient exactly zero.
    wers = torch.tensor([[1.0, 1.0]])
    spec = ObjectiveSpec(
        advantage=AdvantageKind.CENTERED_MWER,
        ratio_unit=RatioUnit.SEQUENCE,
        clip_rule=ClipRule.CISPO_UPPER,
    )

    diagnostics = optimize_frozen_rollout(
        spec=spec,
        score_current=lambda: parameters,
        old_token_log_probs=old,
        token_mask=mask,
        candidate_wers=wers,
        optimizer=torch.optim.SGD([parameters], lr=0.1),
        inner_updates=1,
        reference_token_log_probs=reference,
        reference_kl_beta=1.0,
    )

    assert diagnostics[0].base_policy_loss == pytest.approx(0.0)
    assert diagnostics[0].reference_kl_value > 0.0
    assert diagnostics[0].reference_kl_loss == pytest.approx(
        diagnostics[0].reference_kl_value
    )
    assert diagnostics[0].total_loss == pytest.approx(
        diagnostics[0].base_policy_loss + diagnostics[0].reference_kl_loss
    )
    assert torch.all(parameters.detach() > old)
    # CISPO ratios remain the detached rollout/current diagnostic, independent
    # of the separately differentiated reference penalty.
    torch.testing.assert_close(diagnostics[0].ratios, torch.ones_like(diagnostics[0].ratios))


def test_reference_kl_diagnostic_is_isolated_from_group_weights() -> None:
    old = torch.tensor([[[-1.2]], [[-0.8]]], dtype=torch.float32)
    reference = torch.tensor([[[-1.0]], [[-1.0]]], dtype=torch.float32)
    mask = torch.ones_like(old, dtype=torch.bool)
    wers = torch.tensor([[0.0], [1.0]])
    spec = ObjectiveSpec(
        advantage=AdvantageKind.CENTERED_MWER,
        ratio_unit=RatioUnit.SEQUENCE,
        clip_rule=ClipRule.CISPO_UPPER,
    )

    values = []
    for weights in (torch.ones(2), torch.tensor([0.5, 1.5])):
        parameters = torch.nn.Parameter(old.clone())
        diagnostics = optimize_frozen_rollout(
            spec=spec,
            score_current=lambda current=parameters: current,
            old_token_log_probs=old,
            token_mask=mask,
            candidate_wers=wers,
            optimizer=torch.optim.SGD([parameters], lr=0.1),
            inner_updates=1,
            utterance_weights=weights,
            reference_token_log_probs=reference,
            reference_kl_beta=0.04,
        )
        values.append(diagnostics[0].reference_kl_value)

    assert values[0] == pytest.approx(values[1])
