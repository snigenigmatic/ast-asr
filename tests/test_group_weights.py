from __future__ import annotations

import pytest

from ast_asr.group_weights import DualRiskWeights, RiskGroup


def test_dual_weights_start_uniform_then_raise_the_higher_risk_group() -> None:
    clean = RiskGroup("Dravidian", "clean")
    noisy = RiskGroup("Dravidian", "noisy")
    weights = DualRiskWeights(
        (clean, noisy),
        ema_decay=0.9,
        dual_learning_rate=0.1,
        uniform_mix=0.2,
    )

    assert weights.loss_weights((clean, noisy)) == pytest.approx((1.0, 1.0))

    probabilities = weights.update({clean: 0.2, noisy: 1.2})

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities[noisy] > probabilities[clean]
    assert min(probabilities.values()) >= 0.1
