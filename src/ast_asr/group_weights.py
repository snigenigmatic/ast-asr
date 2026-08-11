"""Worst-group risk tracking for family by acoustic-condition groups."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class RiskGroup:
    family: str
    condition: str

    def __post_init__(self) -> None:
        if not self.family or not self.condition:
            raise ValueError("risk groups require family and condition")

    @property
    def key(self) -> str:
        return f"{self.family}::{self.condition}"


class DualRiskWeights:
    """EMA risks and exponentiated-gradient dual weights behind one interface."""

    def __init__(
        self,
        groups: Sequence[RiskGroup],
        *,
        ema_decay: float = 0.9,
        dual_learning_rate: float = 0.1,
        uniform_mix: float = 0.2,
    ) -> None:
        unique_groups = tuple(dict.fromkeys(groups))
        if not unique_groups:
            raise ValueError("at least one risk group is required")
        if len(unique_groups) != len(groups):
            raise ValueError("risk groups must be unique")
        if not 0 <= ema_decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        if dual_learning_rate <= 0:
            raise ValueError("dual learning rate must be positive")
        if not 0 <= uniform_mix <= 1:
            raise ValueError("uniform mix must be in [0, 1]")

        self._groups = unique_groups
        self._ema_decay = ema_decay
        self._dual_learning_rate = dual_learning_rate
        self._uniform_mix = uniform_mix
        self._ema_risks = {group: 0.0 for group in unique_groups}
        self._log_weights = {group: 0.0 for group in unique_groups}
        uniform = 1.0 / len(unique_groups)
        self._probabilities = {group: uniform for group in unique_groups}

    @property
    def probabilities(self) -> dict[RiskGroup, float]:
        return dict(self._probabilities)

    @property
    def ema_risks(self) -> dict[RiskGroup, float]:
        return dict(self._ema_risks)

    def update(self, observed_risks: Mapping[RiskGroup, float]) -> dict[RiskGroup, float]:
        missing = set(self._groups) - set(observed_risks)
        extra = set(observed_risks) - set(self._groups)
        if missing or extra:
            raise ValueError(f"risk keys must match configured groups; missing={missing}, extra={extra}")

        for group in self._groups:
            risk = float(observed_risks[group])
            if not math.isfinite(risk) or risk < 0:
                raise ValueError("group risks must be finite and non-negative")
            ema = (
                self._ema_decay * self._ema_risks[group]
                + (1.0 - self._ema_decay) * risk
            )
            self._ema_risks[group] = ema
            self._log_weights[group] += self._dual_learning_rate * ema

        largest = max(self._log_weights.values())
        unnormalized = {
            group: math.exp(self._log_weights[group] - largest)
            for group in self._groups
        }
        denominator = sum(unnormalized.values())
        uniform = 1.0 / len(self._groups)
        self._probabilities = {
            group: (1.0 - self._uniform_mix) * unnormalized[group] / denominator
            + self._uniform_mix * uniform
            for group in self._groups
        }
        return self.probabilities

    def loss_weights(self, groups: Sequence[RiskGroup]) -> tuple[float, ...]:
        """Return weights scaled so a uniform dual distribution yields one."""
        scale = len(self._groups)
        try:
            return tuple(self._probabilities[group] * scale for group in groups)
        except KeyError as error:
            raise ValueError(f"unknown risk group: {error.args[0]}") from error
