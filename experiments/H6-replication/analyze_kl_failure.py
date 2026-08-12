r"""Offline localization of the H6 sampled-K3 safety stop.

This script consumes read-only copies of Modal policy artifacts.  It never
imports Modal or a model, and it deliberately reports only batch-level facts:
the stored records have no per-token or per-utterance reference-K3 values.

Example (PowerShell):
    uv run --frozen python experiments/H6-replication/analyze_kl_failure.py `
      --artifacts-root C:\path\to\downloaded-artifacts `
      --output C:\path\to\localization.json

Expected ``--artifacts-root`` children are the five labels in ``ARMS`` below.
Each label contains ``diagnostics/cycle-*.json`` and ``rollouts/cycle-*.json``
downloaded from the immutable Modal Volume artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ARMS = (
    "h5_s2026_beta0",
    "h5_s2026_beta004",
    "h6_s2027_beta0",
    "h6_s2027_beta004",
    "h6_s2028_beta0",
)
ARM_SOURCES = {
    "h5_s2026_beta0": {
        "run_name": "profile-h5-refkl-beta0-r1-s2026-20260812",
        "output_name": "h5-beta0-fr-cispo",
        "expected_cycles": 40,
    },
    "h5_s2026_beta004": {
        "run_name": "profile-h5-refkl-beta004-s2026-20260812",
        "output_name": "h5-beta004-fr-cispo",
        "expected_cycles": 40,
    },
    "h6_s2027_beta0": {
        "run_name": "profile-h6-refkl-beta0-s2027-20260812",
        "output_name": "h6-beta0-fr-cispo",
        "expected_cycles": 40,
    },
    "h6_s2027_beta004": {
        "run_name": "profile-h6-refkl-beta004-s2027-20260812",
        "output_name": "h6-beta004-fr-cispo",
        "expected_cycles": 40,
    },
    "h6_s2028_beta0": {
        "run_name": "profile-h6-refkl-beta0-s2028-20260812",
        "output_name": "h6-beta0-fr-cispo",
        "expected_cycles": 28,
    },
}
EXPECTED_GROUPS = {
    "Dravidian::clean",
    "Dravidian::white_train",
    "Indo-Aryan::clean",
    "Indo-Aryan::white_train",
    "Sino-Tibetan::clean",
    "Sino-Tibetan::white_train",
}
FAILURE_ARM = "h6_s2028_beta0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_manifest(root: Path, arm: str) -> dict[str, Any]:
    arm_root = root / arm
    entries = []
    for kind in ("diagnostics", "rollouts"):
        for path in sorted((arm_root / kind).glob("cycle-*.json")):
            entries.append(
                {
                    "path": path.relative_to(arm_root).as_posix(),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    source = ARM_SOURCES[arm]
    payload = {
        "arm": arm,
        "modal_volume_path": (
            f"/artifacts/{source['run_name']}/{source['output_name']}"
        ),
        "files": entries,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return {
        **payload,
        "diagnostic_file_count": sum(
            entry["path"].startswith("diagnostics/") for entry in entries
        ),
        "rollout_file_count": sum(
            entry["path"].startswith("rollouts/") for entry in entries
        ),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty collection")
    return float(statistics.mean(values))


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = _mean(left), _mean(right)
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    if left_variance == 0.0 or right_variance == 0.0:
        return None
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    return covariance / math.sqrt(left_variance * right_variance)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks for ties; adequate for descriptive Spearman values."""
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        for _, index in ordered[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _linear_residuals(values: list[float]) -> list[float]:
    """Remove a least-squares linear cycle trend without external packages."""
    xs = [float(index) for index in range(len(values))]
    mean_x, mean_y = _mean(xs), _mean(values)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator == 0.0:
        return values
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(xs, values, strict=True)
    ) / denominator
    intercept = mean_y - slope * mean_x
    return [value - (intercept + slope * index) for index, value in enumerate(values)]


def _descriptive_correlation(
    rows: list[dict[str, Any]], metric: str
) -> dict[str, float | None]:
    k3 = [float(row["k3_post"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    return {
        "pearson": _pearson(k3, values),
        "spearman": _pearson(_ranks(k3), _ranks(values)),
        "linear_cycle_detrended_pearson": _pearson(
            _linear_residuals(k3), _linear_residuals(values)
        ),
    }


def _cycle_row(diagnostic: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
    utterances = rollout.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        raise ValueError("rollout has no utterances")
    trajectory = diagnostic.get("loss_trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("diagnostic has no loss trajectory")
    group_state = diagnostic.get("group_weight_trajectory")
    if not isinstance(group_state, dict):
        raise TypeError("diagnostic has no group trajectory")
    probabilities = diagnostic.get("group_probabilities")
    risks = group_state.get("observed_risks")
    loss_weights = group_state.get("utterance_loss_weights")
    if not isinstance(probabilities, dict) or not isinstance(risks, dict):
        raise TypeError("diagnostic has no group probabilities or risks")
    if not isinstance(loss_weights, list) or len(loss_weights) != 6:
        raise TypeError("diagnostic has no six-utterance loss-weight vector")
    if set(probabilities) != EXPECTED_GROUPS or set(risks) != EXPECTED_GROUPS:
        raise ValueError("unexpected family-condition group layout")

    token_lengths: list[int] = []
    candidate_wers: list[float] = []
    reference_word_counts: list[int] = []
    utterance_groups: list[str] = []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            raise TypeError("rollout utterance is not an object")
        family, condition = utterance.get("family"), utterance.get("condition")
        group = f"{family}::{condition}"
        utterance_groups.append(group)
        candidates = utterance.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("rollout utterance has no candidates")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise TypeError("rollout candidate is not an object")
            mask = candidate.get("token_mask")
            if not isinstance(mask, list):
                raise TypeError("rollout candidate has no token mask")
            token_lengths.append(sum(bool(value) for value in mask))
            candidate_wers.append(float(candidate["wer"]))
        reference_word_counts.append(len(str(utterance.get("reference", "")).split()))

    if set(utterance_groups) != EXPECTED_GROUPS or len(utterance_groups) != 6:
        raise ValueError("rollout is not one utterance for each expected group")
    if len(token_lengths) != 24:
        raise ValueError("expected K=4 candidates for each of six utterances")

    k3_post = float(diagnostic["current_cycle_sampled_k3_kl_per_token_from_sft"])
    k3_pre = float(diagnostic["reference_kl"]["update_zero_k3_kl_per_token"])
    ratio_trajectory = diagnostic.get("ratio_trajectory")
    if not isinstance(ratio_trajectory, list) or not ratio_trajectory:
        raise ValueError("diagnostic has no ratio trajectory")
    return {
        "cycle": int(diagnostic["cycle"]),
        "k3_pre_update": k3_pre,
        "k3_post": k3_post,
        "k3_within_cycle_change": k3_post - k3_pre,
        "final_ratio_p99": float(ratio_trajectory[-1]["p99"]),
        "max_group_probability": max(float(value) for value in probabilities.values()),
        "min_group_probability": min(float(value) for value in probabilities.values()),
        "sino_tibetan_white_probability": float(probabilities["Sino-Tibetan::white_train"]),
        "max_observed_risk": max(float(value) for value in risks.values()),
        "sino_tibetan_white_risk": float(risks["Sino-Tibetan::white_train"]),
        "min_utterance_loss_weight": min(float(value) for value in loss_weights),
        "max_utterance_loss_weight": max(float(value) for value in loss_weights),
        "candidate_count": len(token_lengths),
        "candidate_mean_valid_tokens": _mean(float(value) for value in token_lengths),
        "candidate_min_valid_tokens": min(token_lengths),
        "candidate_max_valid_tokens": max(token_lengths),
        "candidate_total_valid_tokens": sum(token_lengths),
        "candidate_mean_wer": _mean(candidate_wers),
        "candidate_max_wer": max(candidate_wers),
        "mean_reference_words": _mean(float(value) for value in reference_word_counts),
        "min_reference_words": min(reference_word_counts),
        "max_reference_words": max(reference_word_counts),
        "group_probabilities": {key: float(value) for key, value in probabilities.items()},
        "observed_risks": {key: float(value) for key, value in risks.items()},
    }


def _load_arm(root: Path, arm: str) -> list[dict[str, Any]]:
    diagnostics_dir = root / arm / "diagnostics"
    rollouts_dir = root / arm / "rollouts"
    diagnostic_paths = sorted(diagnostics_dir.glob("cycle-*.json"))
    if not diagnostic_paths:
        raise FileNotFoundError(f"no diagnostics found for {arm}: {diagnostics_dir}")
    rows = []
    for diagnostic_path in diagnostic_paths:
        rollout_path = rollouts_dir / diagnostic_path.name
        if not rollout_path.is_file():
            raise FileNotFoundError(f"missing rollout for {diagnostic_path}")
        row = _cycle_row(_read_json(diagnostic_path), _read_json(rollout_path))
        rows.append(row)
    cycles = [int(row["cycle"]) for row in rows]
    if cycles != list(range(len(cycles))):
        raise ValueError(f"non-contiguous cycles for {arm}: {cycles}")
    expected_cycles = int(ARM_SOURCES[arm]["expected_cycles"])
    if len(rows) != expected_cycles:
        raise ValueError(
            f"unexpected cycle count for {arm}: {len(rows)} != {expected_cycles}"
        )
    return rows


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "max_group_probability",
        "sino_tibetan_white_probability",
        "max_observed_risk",
        "sino_tibetan_white_risk",
        "candidate_mean_valid_tokens",
        "candidate_max_valid_tokens",
        "candidate_mean_wer",
        "candidate_max_wer",
        "mean_reference_words",
    )
    return {
        "cycle_count": len(rows),
        "k3": {
            "first": rows[0]["k3_post"],
            "last": rows[-1]["k3_post"],
            "maximum": max(float(row["k3_post"]) for row in rows),
            "maximum_cycle": max(rows, key=lambda row: float(row["k3_post"]))["cycle"],
        },
        "descriptive_correlations_with_post_cycle_k3": {
            metric: _descriptive_correlation(rows, metric) for metric in metrics
        },
        "top_k3_cycles": [
            {
                key: row[key]
                for key in (
                    "cycle",
                    "k3_pre_update",
                    "k3_post",
                    "k3_within_cycle_change",
                    "sino_tibetan_white_probability",
                    "max_group_probability",
                    "sino_tibetan_white_risk",
                    "candidate_mean_valid_tokens",
                    "candidate_total_valid_tokens",
                    "candidate_mean_wer",
                )
            }
            for row in sorted(rows, key=lambda row: float(row["k3_post"]), reverse=True)[:6]
        ],
    }


def _failure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failing = rows[-1]
    prior = rows[-2]
    changes = [
        float(rows[index]["k3_post"]) - float(rows[index - 1]["k3_post"])
        for index in range(1, len(rows))
    ]
    return {
        "failure_cycle": failing["cycle"],
        "post_cycle_k3": failing["k3_post"],
        "pre_update_k3": failing["k3_pre_update"],
        "within_cycle_k3_change": failing["k3_within_cycle_change"],
        "final_ratio_p99": failing["final_ratio_p99"],
        "previous_cycle_post_k3": prior["k3_post"],
        "cross_candidate_cycle_change_from_previous": (
            float(failing["k3_post"]) - float(prior["k3_post"])
        ),
        "largest_adjacent_k3_increase": max(changes),
        "largest_adjacent_k3_decrease": min(changes),
        "group_probability_at_failure": failing["group_probabilities"],
        "observed_risk_at_failure": failing["observed_risks"],
        "utterance_loss_weight_range_at_failure": {
            "minimum": failing["min_utterance_loss_weight"],
            "maximum": failing["max_utterance_loss_weight"],
        },
        "candidate_lengths_at_failure": {
            key: failing[key]
            for key in (
                "candidate_count",
                "candidate_total_valid_tokens",
                "candidate_mean_valid_tokens",
                "candidate_min_valid_tokens",
                "candidate_max_valid_tokens",
                "mean_reference_words",
            )
        },
        "candidate_wer_at_failure": {
            key: failing[key] for key in ("candidate_mean_wer", "candidate_max_wer")
        },
        "interpretation_limits": [
            "The saved K3 is one token-mean over the full six-utterance frozen rollout batch.",
            "There are no per-token, candidate, utterance, or group K3 contributions; group-level causal attribution is not identifiable.",
            "A fresh rollout is generated every cycle, so adjacent post-cycle K3 values use different candidate sets and do not isolate parameter drift.",
            "Correlations are descriptive only; they are not significance tests and cannot establish causation.",
        ],
    }


def analyze(root: Path) -> dict[str, Any]:
    records = {arm: _load_arm(root, arm) for arm in ARMS}
    return {
        "input_manifests": {arm: _input_manifest(root, arm) for arm in ARMS},
        "artifact_schema": {
            "available": [
                "cycle-level pre/post sampled fixed-SFT K3",
                "per-update ratio, loss, gradient, and fixed-reference K3 trajectories",
                "six family-by-condition observed risks, EMA-derived probabilities, and per-utterance loss weights",
                "utterance identity/family/condition/reference plus four candidate hypotheses, WERs, token masks, and old log probabilities",
            ],
            "not_available": [
                "per-token current-versus-reference log-probabilities or K3 contributions",
                "per-candidate, per-utterance, or per-group sampled K3",
                "counterfactual rescoring of one fixed candidate bank across adjacent model states",
                "optimizer-state or gradient contributions partitioned by group",
            ],
        },
        "arms": {arm: _arm_summary(rows) for arm, rows in records.items()},
        "failed_seed_2028": _failure_summary(records[FAILURE_ARM]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.artifacts_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
