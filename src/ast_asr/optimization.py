"""Live multi-update optimization over immutable rollout evidence."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch

from .objectives import ObjectiveSpec, policy_objective, sampled_k3_reference_kl


@dataclass(frozen=True, slots=True)
class InnerUpdateDiagnostics:
    update: int
    loss: float
    ratios: torch.Tensor
    ratio_is_finite: bool
    ratio_p01: float
    ratio_median: float
    ratio_p99: float
    ratio_max: float
    gradient_norm: float
    base_policy_loss: float = 0.0
    reference_kl_value: float = 0.0
    reference_kl_loss: float = 0.0
    total_loss: float = 0.0
    reference_kl_evaluated: bool = False
    optimizer_step_applied: bool = False


class InnerUpdateSafetyStop(RuntimeError):
    """An update guard stopped optimization with its measured trajectory."""

    def __init__(
        self,
        message: str,
        diagnostics: Sequence[InnerUpdateDiagnostics],
    ) -> None:
        super().__init__(message)
        self.diagnostics = tuple(diagnostics)


def _ratio_diagnostics(
    *,
    update: int,
    loss: torch.Tensor,
    ratios: torch.Tensor,
    token_mask: torch.Tensor,
    gradient_norm: float,
    base_policy_loss: torch.Tensor,
    reference_kl_value: torch.Tensor,
    reference_kl_loss: torch.Tensor,
    reference_kl_evaluated: bool,
) -> InnerUpdateDiagnostics:
    valid_ratios = ratios[token_mask] if ratios.ndim == token_mask.ndim else ratios.flatten()
    valid_ratios_cpu = valid_ratios.detach().float().cpu()
    return InnerUpdateDiagnostics(
        update=update,
        loss=float(loss.detach().cpu()),
        ratios=ratios.detach().cpu().clone(),
        ratio_is_finite=bool(torch.isfinite(valid_ratios_cpu).all()),
        ratio_p01=float(torch.quantile(valid_ratios_cpu, 0.01)),
        ratio_median=float(torch.quantile(valid_ratios_cpu, 0.5)),
        ratio_p99=float(torch.quantile(valid_ratios_cpu, 0.99)),
        ratio_max=float(valid_ratios_cpu.max()),
        gradient_norm=gradient_norm,
        base_policy_loss=float(base_policy_loss.detach().cpu()),
        reference_kl_value=float(reference_kl_value.detach().cpu()),
        reference_kl_loss=float(reference_kl_loss.detach().cpu()),
        total_loss=float(loss.detach().cpu()),
        reference_kl_evaluated=reference_kl_evaluated,
    )


def _total_policy_loss(
    *,
    base_policy_loss: torch.Tensor,
    current_token_log_probs: torch.Tensor,
    reference_token_log_probs: torch.Tensor | None,
    token_mask: torch.Tensor,
    reference_kl_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Add an optional fixed-reference penalty without changing beta-zero loss."""
    if reference_kl_beta == 0.0:
        zero = base_policy_loss.detach().new_zeros(())
        if reference_token_log_probs is None:
            value = zero
            evaluated = False
        else:
            with torch.no_grad():
                value = sampled_k3_reference_kl(
                    current_token_log_probs,
                    reference_token_log_probs,
                    token_mask,
                )
            evaluated = True
        # Returning the original tensor, rather than adding a numerical zero,
        # makes all existing beta-zero objectives exactly identical.
        return base_policy_loss, value, zero, evaluated
    if reference_token_log_probs is None:
        raise ValueError("positive reference KL beta requires reference scores")
    value = sampled_k3_reference_kl(
        current_token_log_probs,
        reference_token_log_probs,
        token_mask,
    )
    penalty = value * reference_kl_beta
    return base_policy_loss + penalty, value, penalty, True


@contextmanager
def deterministic_model_mode(model: torch.nn.Module) -> Iterator[None]:
    """Temporarily disable stochastic layers while preserving autograd."""
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in states:
            module.training = training


def optimize_frozen_rollout(
    *,
    spec: ObjectiveSpec,
    score_current: Callable[[], torch.Tensor],
    old_token_log_probs: torch.Tensor,
    token_mask: torch.Tensor,
    candidate_wers: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    inner_updates: int,
    utterance_weights: torch.Tensor | None = None,
    max_gradient_norm: float = 1.0,
    update_safety_check: Callable[[InnerUpdateDiagnostics], str | None] | None = None,
    reference_token_log_probs: torch.Tensor | None = None,
    reference_kl_beta: float = 0.0,
) -> tuple[InnerUpdateDiagnostics, ...]:
    """Run ``mu`` optimizer passes while keeping old evidence fixed."""
    if inner_updates < 1:
        raise ValueError("inner_updates must be positive")
    if old_token_log_probs.dtype != torch.float32:
        raise ValueError("old token log-probabilities must be FP32")
    if reference_kl_beta < 0 or not torch.isfinite(torch.tensor(reference_kl_beta)):
        raise ValueError("reference KL beta must be finite and nonnegative")
    if reference_token_log_probs is not None:
        if reference_token_log_probs.dtype != torch.float32:
            raise ValueError("reference token log-probabilities must be FP32")
        if reference_token_log_probs.shape != old_token_log_probs.shape:
            raise ValueError("reference and old log-probability shapes must match")
    frozen_old = old_token_log_probs.detach().clone()
    frozen_reference = (
        None
        if reference_token_log_probs is None
        else reference_token_log_probs.detach().clone()
    )
    diagnostics = []
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    for update in range(inner_updates):
        optimizer.zero_grad(set_to_none=True)
        current = score_current()
        result = policy_objective(
            spec,
            current_token_log_probs=current,
            old_token_log_probs=frozen_old,
            token_mask=token_mask,
            candidate_wers=candidate_wers,
            utterance_weights=utterance_weights,
        )
        (
            total_loss,
            reference_kl_value,
            reference_kl_loss,
            reference_kl_evaluated,
        ) = _total_policy_loss(
            base_policy_loss=result.loss,
            current_token_log_probs=current,
            reference_token_log_probs=frozen_reference,
            token_mask=token_mask,
            reference_kl_beta=reference_kl_beta,
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite policy loss at inner update {update}")
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient at inner update {update}")
        diagnostic = _ratio_diagnostics(
            update=update,
            loss=total_loss,
            ratios=result.ratios,
            token_mask=token_mask,
            gradient_norm=float(gradient_norm.detach().cpu()),
            base_policy_loss=result.loss,
            reference_kl_value=reference_kl_value,
            reference_kl_loss=reference_kl_loss,
            reference_kl_evaluated=reference_kl_evaluated,
        )
        diagnostics.append(diagnostic)
        if update_safety_check is not None:
            message = update_safety_check(diagnostic)
            if message is not None:
                raise InnerUpdateSafetyStop(message, diagnostics)
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in parameters):
            raise FloatingPointError(
                f"non-finite trainable parameter at inner update {update}"
            )
        diagnostics[-1] = replace(diagnostic, optimizer_step_applied=True)

    with torch.no_grad():
        final_current = score_current()
        final_result = policy_objective(
            spec,
            current_token_log_probs=final_current,
            old_token_log_probs=frozen_old,
            token_mask=token_mask,
            candidate_wers=candidate_wers,
            utterance_weights=utterance_weights,
        )
        (
            final_loss,
            final_reference_kl_value,
            final_reference_kl_loss,
            final_reference_kl_evaluated,
        ) = _total_policy_loss(
            base_policy_loss=final_result.loss,
            current_token_log_probs=final_current,
            reference_token_log_probs=frozen_reference,
            token_mask=token_mask,
            reference_kl_beta=reference_kl_beta,
        )
    final_diagnostic = _ratio_diagnostics(
        update=inner_updates,
        loss=final_loss,
        ratios=final_result.ratios,
        token_mask=token_mask,
        gradient_norm=0.0,
        base_policy_loss=final_result.loss,
        reference_kl_value=final_reference_kl_value,
        reference_kl_loss=final_reference_kl_loss,
        reference_kl_evaluated=final_reference_kl_evaluated,
    )
    diagnostics.append(final_diagnostic)
    if update_safety_check is not None:
        message = update_safety_check(final_diagnostic)
        if message is not None:
            raise InnerUpdateSafetyStop(message, diagnostics)
    torch.testing.assert_close(old_token_log_probs, frozen_old)
    if frozen_reference is not None:
        torch.testing.assert_close(reference_token_log_probs, frozen_reference)
    return tuple(diagnostics)
