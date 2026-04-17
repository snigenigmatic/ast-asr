"""
Gradient Reversal Layer (Ganin & Lempitsky, 2015).

Forward: identity.
Backward: multiplies the incoming gradient by `-lambda_`.

Used to train a domain/accent adversary that fights the encoder: the encoder
sees reversed gradients from the adversary's classification loss and learns
representations that are invariant to the protected attribute.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Functional API: reverses gradients flowing back through `x`."""
    return _GradReverseFn.apply(x, lambda_)


class GradientReversalLayer(nn.Module):
    """
    Module wrapper around `grad_reverse` with a mutable `lambda_` attribute.

    The training loop updates `lambda_` on a schedule (warmup from 0 to the
    target value over the first portion of training steps), so callers should
    write directly to `.lambda_` rather than reconstructing the module.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambda_)

    def extra_repr(self) -> str:
        return f"lambda_={self.lambda_}"
