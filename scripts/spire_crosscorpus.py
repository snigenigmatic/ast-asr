"""Pure, testable logic for the SPIRE-SIES cross-corpus evaluation.

This module imports no Modal, and imports nothing beyond the standard
library on the corpus-preparation path, so preparation can run in a light
CPU container and every rule here is unit testable. Scoring lazily imports
the repository's own WER arithmetic so SPIRE numbers stay directly
comparable with every existing Svarah number.

See experiments/SPIRE-crosscorpus/protocol.md for the registered contract.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ast_asr.metrics import EditCounts

# The 17 primary languages SPIRE-SIES actually ships. Declared locally so the
# corpus-preparation path stays stdlib-only: importing ast_asr pulls in torch
# via its package __init__, which a CPU preparation container should not need.
# test_spire_crosscorpus.py asserts this table agrees with ast_asr.taxonomy,
# so the two can never silently diverge.
SPIRE_LANGUAGE_FAMILIES: dict[str, str] = {
    "Bengali": "Indo-Aryan",
    "Dogri": "Indo-Aryan",
    "Gujarati": "Indo-Aryan",
    "Hindi": "Indo-Aryan",
    "Kannada": "Dravidian",
    "Kashmiri": "Indo-Aryan",
    "Konkani": "Indo-Aryan",
    "Maithili": "Indo-Aryan",
    "Malayalam": "Dravidian",
    "Marathi": "Indo-Aryan",
    "Nepali": "Indo-Aryan",
    "Odia": "Indo-Aryan",
    "Punjabi": "Indo-Aryan",
    "Sindhi": "Indo-Aryan",
    "Tamil": "Dravidian",
    "Telugu": "Dravidian",
    "Urdu": "Indo-Aryan",
}

TARGET_SAMPLE_RATE = 16_000
SOURCE_SAMPLE_RATE = 48_000

# SPIRE-SIES covers 17 of Svarah's 19 primary languages. It omits Assamese
# (Indo-Aryan) and Bodo, and Bodo is Svarah's only Sino-Tibetan language, so
# this corpus is structurally two-family. Encoding that expectation makes any
# future drift fail loudly instead of silently changing the endpoint.
SPIRE_EXPECTED_FAMILIES: tuple[str, ...] = ("Dravidian", "Indo-Aryan")

MINIMUM_DURATION_SECONDS = 1.0
MAXIMUM_DURATION_SECONDS = 30.0
EXPECTED_VAL_SPEAKERS = 198
EXPECTED_TRAIN_SPEAKERS = 1126


@lru_cache(maxsize=1)
def _edit_counts_class() -> type:
    """Import the repository's edit-count type only when scoring is needed."""
    from ast_asr.metrics import EditCounts as _EditCounts

    return _EditCounts


@lru_cache(maxsize=1)
def _word_edit_counts():
    """Import the repository's WER arithmetic only when scoring is needed."""
    from ast_asr.metrics import word_edit_counts as _word_edit_counts_impl

    return _word_edit_counts_impl


def zero_counts() -> EditCounts:
    """Return an empty edit-count accumulator."""
    return _edit_counts_class()()


class SpireContractError(RuntimeError):
    """Raised when observed corpus data contradicts the frozen contract."""


def resolve_family(language: str, declared_family: str | None = None) -> str:
    """Map a primary language to its family, failing closed on disagreement."""
    key = str(language).strip().title()
    if key not in SPIRE_LANGUAGE_FAMILIES:
        raise SpireContractError(f"unknown primary language: {language!r}")
    family = SPIRE_LANGUAGE_FAMILIES[key]
    if family not in SPIRE_EXPECTED_FAMILIES:
        raise SpireContractError(
            f"language {key!r} resolves to unexpected family {family!r}"
        )
    if declared_family is not None:
        declared = str(declared_family).strip()
        if declared != family:
            raise SpireContractError(
                f"corpus family {declared!r} disagrees with taxonomy "
                f"{family!r} for language {key!r}"
            )
    return family


def validate_splits(splits: Mapping[str, Sequence[str]]) -> frozenset[str]:
    """Validate the corpus split file and return the evaluation speaker set."""
    if set(splits) != {"train", "val"}:
        raise SpireContractError(
            f"splits must contain exactly train and val, got {sorted(splits)}"
        )
    train = {str(speaker) for speaker in splits["train"]}
    val = {str(speaker) for speaker in splits["val"]}
    if len(train) != len(splits["train"]) or len(val) != len(splits["val"]):
        raise SpireContractError("split speaker lists contain duplicates")
    overlap = train & val
    if overlap:
        raise SpireContractError(
            f"split is not speaker-disjoint; {len(overlap)} shared speakers"
        )
    if not val:
        raise SpireContractError("validation split is empty")
    if len(val) != EXPECTED_VAL_SPEAKERS or len(train) != EXPECTED_TRAIN_SPEAKERS:
        raise SpireContractError(
            f"expected {EXPECTED_TRAIN_SPEAKERS}/{EXPECTED_VAL_SPEAKERS} "
            f"train/val speakers, observed {len(train)}/{len(val)}"
        )
    return frozenset(val)


@dataclass(frozen=True, slots=True)
class SpireUtterance:
    """One accepted validation utterance, before transcription."""

    uid: str
    speaker_id: str
    language: str
    family: str
    gender: str
    duration: float


def accept_row(
    row: Mapping[str, object],
    val_speakers: frozenset[str],
) -> SpireUtterance | None:
    """Return a validation utterance, or None when the row is out of scope.

    Fails closed when the per-row split label and the split file disagree,
    because that would mean the speaker-disjoint guarantee is not real.
    """
    speaker = str(row["speaker_id"]).strip()
    label = str(row["split"]).strip()
    in_val_set = speaker in val_speakers
    if label == "val" and not in_val_set:
        raise SpireContractError(
            f"row {row['uid']!r} is labelled val but speaker {speaker!r} "
            "is absent from the split file"
        )
    if label != "val":
        if in_val_set:
            raise SpireContractError(
                f"speaker {speaker!r} is a val speaker but row "
                f"{row['uid']!r} is labelled {label!r}"
            )
        return None

    duration = float(row["duration"])
    if not MINIMUM_DURATION_SECONDS <= duration <= MAXIMUM_DURATION_SECONDS:
        return None
    reference = str(row["reference"]).strip()
    if not reference:
        return None

    return SpireUtterance(
        uid=str(row["uid"]),
        speaker_id=speaker,
        language=str(row["accent"]).strip().title(),
        family=resolve_family(row["accent"], row.get("language_family")),
        gender=str(row.get("gender", "Unknown")).strip() or "Unknown",
        duration=duration,
    )


@dataclass(frozen=True, slots=True)
class UtteranceResult:
    """One scored validation utterance."""

    uid: str
    speaker_id: str
    family: str
    gender: str
    counts: EditCounts


def score_utterance(
    utterance: SpireUtterance,
    reference: str,
    hypothesis: str,
) -> UtteranceResult:
    """Score one utterance with the repository's standard WER arithmetic."""
    return UtteranceResult(
        uid=utterance.uid,
        speaker_id=utterance.speaker_id,
        family=utterance.family,
        gender=utterance.gender,
        counts=_word_edit_counts()(reference, hypothesis),
    )


def pool(results: Iterable[UtteranceResult], key: str) -> dict[str, EditCounts]:
    """Pool edit counts by an attribute name, so WER is error-weighted."""
    grouped: dict[str, EditCounts] = defaultdict(_edit_counts_class())
    for result in results:
        grouped[getattr(result, key)] += result.counts
    return dict(grouped)


def group_wers(counts: Mapping[str, EditCounts]) -> dict[str, float]:
    return {group: value.wer for group, value in counts.items()}


def worst_group(wers: Mapping[str, float]) -> tuple[str, float]:
    """Return the worst group, breaking ties by name for determinism."""
    if not wers:
        raise SpireContractError("cannot take the worst of zero groups")
    name = max(sorted(wers), key=lambda group: wers[group])
    return name, wers[name]


def worst_fraction_speaker_wer(
    results: Iterable[UtteranceResult],
    fraction: float = 0.2,
) -> float:
    """Pooled WER of the worst-performing tail of speakers."""
    if not 0 < fraction <= 1:
        raise SpireContractError("fraction must lie in (0, 1]")
    by_speaker = pool(results, "speaker_id")
    if not by_speaker:
        raise SpireContractError("no speakers to summarize")
    ranked = sorted(
        by_speaker,
        key=lambda speaker: (-by_speaker[speaker].wer, speaker),
    )
    take = max(1, round(len(ranked) * fraction))
    total = zero_counts()
    for speaker in ranked[:take]:
        total += by_speaker[speaker]
    return total.wer


def summarize_arm(results: Sequence[UtteranceResult]) -> dict[str, object]:
    """Compute every registered endpoint for one checkpoint."""
    if not results:
        raise SpireContractError("cannot summarize an empty arm")
    total = zero_counts()
    for result in results:
        total += result.counts
    family_wers = group_wers(pool(results, "family"))
    observed = set(family_wers)
    if not observed <= set(SPIRE_EXPECTED_FAMILIES):
        raise SpireContractError(f"unexpected families present: {sorted(observed)}")
    worst_family, worst_family_wer = worst_group(family_wers)
    return {
        "utterances": len(results),
        "speakers": len({result.speaker_id for result in results}),
        "reference_words": total.reference_words,
        "overall_wer": total.wer,
        "wer_by_family": dict(sorted(family_wers.items())),
        "worst_family": worst_family,
        "worst_family_wer": worst_family_wer,
        "family_gap": max(family_wers.values()) - min(family_wers.values()),
        "wer_by_gender": dict(sorted(group_wers(pool(results, "gender")).items())),
        "worst_20_percent_speaker_wer": worst_fraction_speaker_wer(results, 0.2),
    }


def _speaker_index(
    results: Sequence[UtteranceResult],
) -> dict[str, dict[str, EditCounts]]:
    """Pre-pool each speaker's counts by family plus a total, for resampling."""
    counts_class = _edit_counts_class()
    index: dict[str, dict[str, EditCounts]] = defaultdict(
        lambda: defaultdict(counts_class)
    )
    for result in results:
        index[result.speaker_id][result.family] += result.counts
        index[result.speaker_id]["__total__"] += result.counts
    return {speaker: dict(cells) for speaker, cells in index.items()}


def _resampled_metrics(
    index: Mapping[str, Mapping[str, EditCounts]],
    drawn: Sequence[str],
) -> tuple[float, float]:
    """Return (overall WER, worst-family WER) for one resample."""
    totals: dict[str, EditCounts] = defaultdict(_edit_counts_class())
    for speaker in drawn:
        for cell, counts in index[speaker].items():
            totals[cell] += counts
    families = {
        cell: counts.wer for cell, counts in totals.items() if cell != "__total__"
    }
    _, worst = worst_group(families)
    return totals["__total__"].wer, worst


def paired_speaker_bootstrap(
    control: Sequence[UtteranceResult],
    treatment: Sequence[UtteranceResult],
    *,
    resamples: int = 10_000,
    seed: int = 2026,
) -> dict[str, object]:
    """Paired speaker-clustered bootstrap of treatment-minus-control deltas.

    One speaker multiset is drawn per resample and applied identically to both
    arms, so speaker difficulty cancels in the paired delta.
    """
    if resamples <= 0:
        raise SpireContractError("resamples must be positive")
    control_uids = {result.uid for result in control}
    if control_uids != {result.uid for result in treatment}:
        raise SpireContractError("arms must be scored on identical utterances")

    control_index = _speaker_index(control)
    treatment_index = _speaker_index(treatment)
    speakers = sorted(control_index)
    if speakers != sorted(treatment_index):
        raise SpireContractError("arms must cover identical speakers")

    generator = random.Random(seed)
    overall_deltas: list[float] = []
    worst_deltas: list[float] = []
    for _ in range(resamples):
        drawn = [generator.choice(speakers) for _ in speakers]
        control_overall, control_worst = _resampled_metrics(control_index, drawn)
        treated_overall, treated_worst = _resampled_metrics(treatment_index, drawn)
        overall_deltas.append(treated_overall - control_overall)
        worst_deltas.append(treated_worst - control_worst)

    return {
        "resamples": resamples,
        "seed": seed,
        "clusters": len(speakers),
        "cluster_unit": "corpus_speaker",
        "overall_wer_delta": _interval(overall_deltas),
        "worst_family_wer_delta": _interval(worst_deltas),
    }


def _interval(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "low": _percentile(ordered, 2.5),
        "high": _percentile(ordered, 97.5),
    }


def _percentile(ordered: Sequence[float], percent: float) -> float:
    if not ordered:
        raise SpireContractError("cannot take a percentile of nothing")
    if len(ordered) == 1:
        return ordered[0]
    position = (percent / 100.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def interval_includes_harm(interval: Mapping[str, float]) -> bool:
    """True when a 'lower is better' delta interval still permits harm."""
    return interval["high"] > 0.0
