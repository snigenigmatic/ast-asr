"""Live multi-update optimization over immutable rollout evidence."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .objectives import ObjectiveSpec, policy_objective


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
    )


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
) -> tuple[InnerUpdateDiagnostics, ...]:
    """Run ``mu`` optimizer passes while keeping old evidence fixed."""
    if inner_updates < 1:
        raise ValueError("inner_updates must be positive")
    if old_token_log_probs.dtype != torch.float32:
        raise ValueError("old token log-probabilities must be FP32")
    frozen_old = old_token_log_probs.detach().clone()
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
        if not torch.isfinite(result.loss):
            raise FloatingPointError(f"non-finite policy loss at inner update {update}")
        result.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient at inner update {update}")
        diagnostic = _ratio_diagnostics(
            update=update,
            loss=result.loss,
            ratios=result.ratios,
            token_mask=token_mask,
            gradient_norm=float(gradient_norm.detach().cpu()),
        )
        diagnostics.append(diagnostic)
        if update_safety_check is not None:
            message = update_safety_check(diagnostic)
            if message is not None:
                raise InnerUpdateSafetyStop(message, diagnostics)
        optimizer.step()

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
    final_diagnostic = _ratio_diagnostics(
        update=inner_updates,
        loss=final_result.loss,
        ratios=final_result.ratios,
        token_mask=token_mask,
        gradient_norm=0.0,
    )
    diagnostics.append(final_diagnostic)
    if update_safety_check is not None:
        message = update_safety_check(final_diagnostic)
        if message is not None:
            raise InnerUpdateSafetyStop(message, diagnostics)
    torch.testing.assert_close(old_token_log_probs, frozen_old)
    return tuple(diagnostics)
