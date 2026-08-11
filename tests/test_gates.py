from __future__ import annotations

import pytest

from ast_asr.gates import (
    DevelopmentSeedResult,
    MovementMetrics,
    evaluate_development_gate,
    require_development_gate,
    select_largest_safe_learning_rate,
)


def _movement(ratio_p99: float, kl: float) -> MovementMetrics:
    return MovementMetrics(
        has_non_finite_values=False,
        skipped_steps=0,
        adapter_drift=0.01,
        greedy_predictions_changed=True,
        ratio_p99=ratio_p99,
        kl_per_token=kl,
    )


def test_learning_rate_and_development_gates_are_literal() -> None:
    selected = select_largest_safe_learning_rate(
        {
            1e-5: (_movement(1.2, 0.02),) * 3,
            3e-5: (_movement(1.8, 0.08),) * 3,
            1e-4: (_movement(2.1, 0.08),) * 3,
        }
    )
    assert selected == 3e-5

    decision = evaluate_development_gate(
        
            DevelopmentSeedResult(seed, 0.30, 0.27, 0.18, 0.185)
            for seed in (11, 17, 23)
        
    )
    assert decision.passed is True
    assert decision.worst_group_improvement == pytest.approx(0.03)
    assert decision.clean_wer_degradation == pytest.approx(0.005)

    with pytest.raises(RuntimeError, match="folds 1-4 are blocked"):
        require_development_gate(1, None)
