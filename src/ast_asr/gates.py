"""Literal movement and development gates that prevent silent method pivots."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MAX_RATIO_P99 = 2.0
MAX_KL_PER_TOKEN = 0.1


@dataclass(frozen=True, slots=True)
class MovementMetrics:
    has_non_finite_values: bool
    skipped_steps: int
    adapter_drift: float
    greedy_predictions_changed: bool
    ratio_p99: float
    kl_per_token: float

    @property
    def passed(self) -> bool:
        return (
            not self.has_non_finite_values
            and self.skipped_steps == 0
            and self.adapter_drift > 0
            and self.greedy_predictions_changed
            and self.ratio_p99 < MAX_RATIO_P99
            and self.kl_per_token < MAX_KL_PER_TOKEN
        )


def select_largest_safe_learning_rate(
    trials: Mapping[float, Sequence[MovementMetrics]],
) -> float:
    required = {1e-5, 3e-5, 1e-4}
    if set(trials) != required:
        raise ValueError(f"learning-rate grid must be exactly {sorted(required)}")
    safe = []
    for learning_rate, seed_metrics in trials.items():
        if len(seed_metrics) != 3:
            raise ValueError("every learning rate requires exactly three development seeds")
        if all(metrics.passed for metrics in seed_metrics):
            safe.append(learning_rate)
    if not safe:
        raise RuntimeError("no learning rate satisfies every movement gate")
    return max(safe)


@dataclass(frozen=True, slots=True)
class DevelopmentSeedResult:
    seed: int
    sft_worst_family_condition_wer: float
    fr_worst_family_condition_wer: float
    sft_clean_overall_wer: float
    fr_clean_overall_wer: float


@dataclass(frozen=True, slots=True)
class DevelopmentGateDecision:
    passed: bool
    worst_group_improvement: float
    clean_wer_degradation: float
    reason: str


def evaluate_development_gate(
    seed_results: Iterable[DevelopmentSeedResult],
) -> DevelopmentGateDecision:
    results = tuple(seed_results)
    if len(results) != 3 or len({result.seed for result in results}) != 3:
        raise ValueError("development gate requires exactly three distinct seeds")
    improvement = sum(
        result.sft_worst_family_condition_wer - result.fr_worst_family_condition_wer
        for result in results
    ) / len(results)
    degradation = sum(
        result.fr_clean_overall_wer - result.sft_clean_overall_wer
        for result in results
    ) / len(results)
    passed = improvement >= 0.02 and degradation <= 0.01
    reason = (
        "development gate passed"
        if passed
        else (
            "development gate failed: require mean worst-group improvement >= 0.02 "
            "and mean clean-WER degradation <= 0.01; do not launch five-fold training"
        )
    )
    return DevelopmentGateDecision(
        passed=passed,
        worst_group_improvement=improvement,
        clean_wer_degradation=degradation,
        reason=reason,
    )


def require_development_gate(fold: int, path: Path | None) -> None:
    """Block non-development folds unless a passed gate artifact is supplied."""
    if fold == 0:
        return
    if path is None or not path.is_file():
        raise RuntimeError(
            "folds 1-4 are blocked until a development_gate.json artifact is supplied"
        )
    gate = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "passed",
        "worst_group_improvement",
        "clean_wer_degradation",
        "development_seeds",
        "selected_learning_rate",
    }
    if not required.issubset(gate):
        raise ValueError("development gate artifact is missing required evidence")
    if gate["passed"] is not True:
        raise RuntimeError("development gate failed; folds 1-4 remain blocked")
    if len(gate["development_seeds"]) != 3:
        raise ValueError("development gate must contain three seeds")
    if float(gate["worst_group_improvement"]) < 0.02:
        raise ValueError("development gate improvement is below 0.02")
    if float(gate["clean_wer_degradation"]) > 0.01:
        raise ValueError("development gate clean degradation exceeds 0.01")
