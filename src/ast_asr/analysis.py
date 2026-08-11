"""Out-of-fold metrics and paired speaker-clustered bootstrap analysis."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from .artifacts import write_immutable_json
from .metrics import EditCounts, PredictionRecord, word_edit_counts


@dataclass(frozen=True, slots=True)
class AnalysisSummary:
    metrics: dict[str, object]
    bootstrap: dict[str, tuple[float, float]]


@dataclass(slots=True)
class _SpeakerAggregate:
    condition: dict[str, EditCounts]
    family_condition: dict[tuple[str, str], EditCounts]
    family_clean: dict[str, EditCounts]
    accent: dict[str, EditCounts]
    clean: EditCounts


def _add(target: dict, key: object, counts: EditCounts) -> None:
    target[key] = target.get(key, EditCounts()) + counts


def _speaker_aggregates(
    records: Iterable[PredictionRecord],
) -> dict[str, _SpeakerAggregate]:
    aggregates: dict[str, _SpeakerAggregate] = {}
    for record in records:
        aggregate = aggregates.setdefault(
            record.speaker_id,
            _SpeakerAggregate({}, {}, {}, {}, EditCounts()),
        )
        counts = word_edit_counts(record.reference, record.hypothesis)
        _add(aggregate.condition, record.condition, counts)
        _add(aggregate.family_condition, (record.family, record.condition), counts)
        _add(aggregate.accent, record.primary_language, counts)
        if record.condition == "clean":
            _add(aggregate.family_clean, record.family, counts)
            aggregate.clean = aggregate.clean + counts
    return aggregates


def _merge_weighted(
    speakers: Mapping[str, _SpeakerAggregate],
    multiplicities: Mapping[str, int],
) -> dict[str, object]:
    by_condition: dict[str, EditCounts] = {}
    by_family_condition: dict[tuple[str, str], EditCounts] = {}
    by_family_clean: dict[str, EditCounts] = {}
    by_accent: dict[str, EditCounts] = {}
    speaker_clean_wers: list[float] = []
    for speaker_id, multiplier in multiplicities.items():
        aggregate = speakers[speaker_id]
        for key, counts in aggregate.condition.items():
            _add(by_condition, key, counts.scaled(multiplier))
        for key, counts in aggregate.family_condition.items():
            _add(by_family_condition, key, counts.scaled(multiplier))
        for key, counts in aggregate.family_clean.items():
            _add(by_family_clean, key, counts.scaled(multiplier))
        for key, counts in aggregate.accent.items():
            _add(by_accent, key, counts.scaled(multiplier))
        speaker_clean_wers.extend([aggregate.clean.wer] * multiplier)

    family_condition_wers = {
        f"{family}::{condition}": counts.wer
        for (family, condition), counts in sorted(by_family_condition.items())
    }
    family_clean_wers = {
        family: counts.wer for family, counts in sorted(by_family_clean.items())
    }
    accent_wers = {accent: counts.wer for accent, counts in sorted(by_accent.items())}
    clean_overall = by_condition.get("clean", EditCounts()).wer
    noisy_conditions = [condition for condition in by_condition if condition != "clean"]
    noise_amplification = max(
        (by_condition[condition].wer - clean_overall for condition in noisy_conditions),
        default=0.0,
    )
    worst_count = max(1, math.ceil(0.2 * len(speaker_clean_wers)))
    worst_speakers = sorted(speaker_clean_wers, reverse=True)[:worst_count]
    return {
        "worst_family_condition_wer": max(family_condition_wers.values(), default=0.0),
        "clean_overall_wer": clean_overall,
        "worst_family_clean_wer": max(family_clean_wers.values(), default=0.0),
        "family_gap": (
            max(family_clean_wers.values()) - min(family_clean_wers.values())
            if family_clean_wers
            else 0.0
        ),
        "noise_amplification": noise_amplification,
        "worst_20_percent_speaker_wer": sum(worst_speakers) / len(worst_speakers),
        "family_condition_wer": family_condition_wers,
        "family_clean_wer": family_clean_wers,
        "accent_wer": accent_wers,
        "condition_wer": {
            condition: counts.wer for condition, counts in sorted(by_condition.items())
        },
    }


def aggregate_prediction_records(
    records: Iterable[PredictionRecord],
    *,
    expected_speakers: int = 117,
    bootstrap_samples: int = 10_000,
    seed: int = 2026,
) -> AnalysisSummary:
    records = tuple(records)
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    records_by_arm: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        records_by_arm[record.arm].append(record)
    if not records_by_arm:
        raise ValueError("no prediction records were supplied")
    speaker_ids_by_arm: dict[str, tuple[str, ...]] = {}
    speakers_by_arm: dict[str, dict[str, _SpeakerAggregate]] = {}
    observed_by_arm: dict[str, dict[str, object]] = {}
    for arm, arm_records in sorted(records_by_arm.items()):
        speaker_folds: dict[str, set[int]] = defaultdict(set)
        for record in arm_records:
            speaker_folds[record.speaker_id].add(record.fold)
        if len(speaker_folds) != expected_speakers:
            raise ValueError(
                f"expected {expected_speakers} out-of-fold speakers for {arm}, "
                f"received {len(speaker_folds)}"
            )
        leaking = [speaker for speaker, folds in speaker_folds.items() if len(folds) != 1]
        if leaking:
            raise ValueError(f"speaker appears in more than one test fold: {leaking[:5]}")
        if set().union(*speaker_folds.values()) != set(range(5)):
            raise ValueError("out-of-fold predictions must cover test folds 0 through 4")
        speakers = _speaker_aggregates(arm_records)
        speaker_ids = tuple(sorted(speakers))
        speaker_ids_by_arm[arm] = speaker_ids
        speakers_by_arm[arm] = speakers
        observed_by_arm[arm] = _merge_weighted(
            speakers,
            {speaker: 1 for speaker in speaker_ids},
        )
    common_speakers = next(iter(speaker_ids_by_arm.values()))
    if any(speakers != common_speakers for speakers in speaker_ids_by_arm.values()):
        raise ValueError("paired bootstrap requires identical speakers in every arm")

    arms = tuple(sorted(observed_by_arm))
    single_arm = len(arms) == 1
    baseline_arm = "sft" if "sft" in observed_by_arm else arms[0]
    endpoints = (
        "worst_family_condition_wer",
        "clean_overall_wer",
        "worst_family_clean_wer",
        "family_gap",
        "noise_amplification",
        "worst_20_percent_speaker_wer",
    )
    nested_metrics = (
        "family_condition_wer",
        "family_clean_wer",
        "accent_wer",
        "condition_wer",
    )
    samples: dict[str, list[float]] = {}

    def metric_key(arm: str, metric: str) -> str:
        return metric if single_arm else f"arm::{arm}::{metric}"

    for arm, observed in observed_by_arm.items():
        for endpoint in endpoints:
            samples[metric_key(arm, endpoint)] = []
        for metric in nested_metrics:
            for group in observed[metric]:
                samples[metric_key(arm, f"{metric}::{group}")] = []
    if not single_arm:
        for arm in arms:
            if arm != baseline_arm:
                for endpoint in endpoints:
                    samples[f"difference::{arm}-{baseline_arm}::{endpoint}"] = []

    generator = random.Random(seed)
    for _ in range(bootstrap_samples):
        multiplicities = Counter(
            generator.choices(common_speakers, k=len(common_speakers))
        )
        bootstrap_by_arm = {
            arm: _merge_weighted(speakers_by_arm[arm], multiplicities)
            for arm in arms
        }
        for arm, bootstrap_metrics in bootstrap_by_arm.items():
            for endpoint in endpoints:
                samples[metric_key(arm, endpoint)].append(
                    float(bootstrap_metrics[endpoint])
                )
            for metric in nested_metrics:
                for group, value in bootstrap_metrics[metric].items():
                    key = metric_key(arm, f"{metric}::{group}")
                    if key in samples:
                        samples[key].append(float(value))
        if not single_arm:
            baseline = bootstrap_by_arm[baseline_arm]
            for arm, bootstrap_metrics in bootstrap_by_arm.items():
                if arm != baseline_arm:
                    for endpoint in endpoints:
                        samples[f"difference::{arm}-{baseline_arm}::{endpoint}"].append(
                            float(bootstrap_metrics[endpoint]) - float(baseline[endpoint])
                        )
    intervals = {
        endpoint: (
            float(torch.quantile(torch.tensor(values), 0.025)),
            float(torch.quantile(torch.tensor(values), 0.975)),
        )
        for endpoint, values in samples.items()
        if values
    }
    if single_arm:
        metrics: dict[str, object] = observed_by_arm[arms[0]]
    else:
        metrics = {
            "by_arm": observed_by_arm,
            "paired_differences_from_baseline": {
                f"{arm}-{baseline_arm}": {
                    endpoint: float(observed_by_arm[arm][endpoint])
                    - float(observed_by_arm[baseline_arm][endpoint])
                    for endpoint in endpoints
                }
                for arm in arms
                if arm != baseline_arm
            },
        }
    return AnalysisSummary(metrics=metrics, bootstrap=intervals)


def summarize_prediction_records(records: Iterable[PredictionRecord]) -> dict[str, object]:
    records = tuple(records)
    speakers = _speaker_aggregates(records)
    return _merge_weighted(speakers, {speaker: 1 for speaker in speakers})


def _read_predictions(paths: Iterable[Path]) -> tuple[PredictionRecord, ...]:
    records = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(PredictionRecord.from_dict(json.loads(line)))
    return tuple(records)


def aggregate_oof(args: argparse.Namespace) -> None:
    summary = aggregate_prediction_records(
        _read_predictions(args.predictions),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_immutable_json(
        args.output_dir / "oof_metrics.json",
        {"metrics": summary.metrics, "bootstrap_95_percent": summary.bootstrap},
    )
