"""Deterministic speaker-balanced cross-validation manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass


@dataclass(frozen=True, order=True, slots=True)
class SpeakerProfile:
    speaker_id: str
    primary_language: str
    family: str
    gender: str
    age_group: str
    reference_word_count: int

    def __post_init__(self) -> None:
        if not all(
            (self.speaker_id, self.primary_language, self.family, self.gender, self.age_group)
        ):
            raise ValueError("speaker profile categorical fields cannot be empty")
        if self.reference_word_count <= 0:
            raise ValueError("reference word count must be positive")


@dataclass(frozen=True, slots=True)
class FoldManifest:
    fold: int
    seed: int
    train_speakers: tuple[str, ...]
    validation_speakers: tuple[str, ...]
    test_speakers: tuple[str, ...]
    speaker_profiles: tuple[SpeakerProfile, ...]

    def __post_init__(self) -> None:
        partitions = (
            set(self.train_speakers),
            set(self.validation_speakers),
            set(self.test_speakers),
        )
        if partitions[0] & partitions[1] or partitions[0] & partitions[2]:
            raise ValueError("speaker partitions overlap")
        if partitions[1] & partitions[2]:
            raise ValueError("speaker partitions overlap")
        profile_ids = {profile.speaker_id for profile in self.speaker_profiles}
        if set.union(*partitions) != profile_ids:
            raise ValueError("speaker partitions must cover every profile exactly once")

    def to_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "seed": self.seed,
            "train_speakers": list(self.train_speakers),
            "validation_speakers": list(self.validation_speakers),
            "test_speakers": list(self.test_speakers),
            "speaker_profiles": [asdict(profile) for profile in self.speaker_profiles],
        }

    @property
    def content_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _stable_fraction(seed: int, *parts: object) -> float:
    value = "|".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    digest = hashlib.sha256(value).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _profile_labels(profile: SpeakerProfile) -> tuple[tuple[str, str], ...]:
    return (
        ("language", profile.primary_language),
        ("family", profile.family),
        ("gender", profile.gender),
        ("age", profile.age_group),
    )


def assign_speaker_folds(
    profiles: Iterable[SpeakerProfile],
    *,
    seed: int,
    fold_count: int = 5,
) -> dict[str, int]:
    """Greedily balance speaker attributes and reference words across folds."""
    ordered_input = tuple(profiles)
    if fold_count < 2:
        raise ValueError("at least two folds are required")
    if len({profile.speaker_id for profile in ordered_input}) != len(ordered_input):
        raise ValueError("speaker IDs must be unique")
    if len(ordered_input) < fold_count:
        raise ValueError("speaker count cannot be smaller than fold count")

    label_totals: Counter[tuple[str, str]] = Counter()
    for profile in ordered_input:
        label_totals.update(_profile_labels(profile))
    total_words = sum(profile.reference_word_count for profile in ordered_input)

    def placement_priority(profile: SpeakerProfile) -> tuple[float, float]:
        rarity = sum(1.0 / label_totals[label] for label in _profile_labels(profile))
        return (-rarity, _stable_fraction(seed, profile.speaker_id, "order"))

    ordered = sorted(ordered_input, key=placement_priority)
    fold_sizes = [0] * fold_count
    fold_words = [0] * fold_count
    fold_labels = [Counter() for _ in range(fold_count)]
    assignment: dict[str, int] = {}
    target_size = len(ordered) / fold_count
    target_words = total_words / fold_count

    for profile in ordered:
        labels = _profile_labels(profile)

        def score(
            fold: int,
            *,
            current_profile: SpeakerProfile = profile,
            current_labels: tuple[tuple[str, str], ...] = labels,
        ) -> tuple[float, float]:
            size_error = ((fold_sizes[fold] + 1 - target_size) / target_size) ** 2
            word_error = (
                (fold_words[fold] + current_profile.reference_word_count - target_words)
                / target_words
            ) ** 2
            label_error = 0.0
            for label in current_labels:
                target = label_totals[label] / fold_count
                label_error += ((fold_labels[fold][label] + 1 - target) / target) ** 2
            return (
                2.0 * size_error + word_error + label_error,
                _stable_fraction(seed, current_profile.speaker_id, fold),
            )

        chosen = min(range(fold_count), key=score)
        assignment[profile.speaker_id] = chosen
        fold_sizes[chosen] += 1
        fold_words[chosen] += profile.reference_word_count
        fold_labels[chosen].update(labels)

    return assignment


def build_outer_fold_manifests(
    profiles: Iterable[SpeakerProfile],
    *,
    seed: int,
    expected_speakers: int = 117,
) -> tuple[FoldManifest, ...]:
    """Build five manifests where validation is the fold after test."""
    canonical_profiles = tuple(sorted(profiles, key=lambda profile: profile.speaker_id))
    if len(canonical_profiles) != expected_speakers:
        raise ValueError(
            f"Svarah must resolve to exactly {expected_speakers} speakers; "
            f"received {len(canonical_profiles)}"
        )
    assignments = assign_speaker_folds(canonical_profiles, seed=seed, fold_count=5)
    manifests = []
    for fold in range(5):
        validation_fold = (fold + 1) % 5
        test = tuple(sorted(speaker for speaker, value in assignments.items() if value == fold))
        validation = tuple(
            sorted(speaker for speaker, value in assignments.items() if value == validation_fold)
        )
        train = tuple(
            sorted(
                speaker
                for speaker, value in assignments.items()
                if value not in (fold, validation_fold)
            )
        )
        manifests.append(
            FoldManifest(
                fold=fold,
                seed=seed,
                train_speakers=train,
                validation_speakers=validation,
                test_speakers=test,
                speaker_profiles=canonical_profiles,
            )
        )
    return tuple(manifests)
