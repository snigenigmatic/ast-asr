"""Immutable rollout evidence shared by generation and policy optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch


class AcousticCondition(StrEnum):
    CLEAN = "clean"
    WHITE_10DB = "white_10db"
    WHITE_TRAIN = "white_train"
    MUSAN_BABBLE_10DB = "musan_babble_10db"


@dataclass(frozen=True, slots=True)
class CandidateRollout:
    hypothesis: str
    token_ids: tuple[int, ...]
    token_mask: tuple[bool, ...]
    old_token_log_probs: tuple[float, ...]
    old_sequence_log_probability: float
    wer: float

    def __post_init__(self) -> None:
        size = len(self.token_ids)
        if size == 0:
            raise ValueError("a rollout candidate must contain at least one token")
        if len(self.token_mask) != size or len(self.old_token_log_probs) != size:
            raise ValueError("token ids, mask, and old log-probabilities must align")
        selected = [
            value
            for value, keep in zip(self.old_token_log_probs, self.token_mask, strict=True)
            if keep
        ]
        if not selected:
            raise ValueError("a rollout candidate must score at least one token")
        expected = sum(selected) / len(selected)
        if not math.isclose(
            expected,
            self.old_sequence_log_probability,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise ValueError("old sequence log-probability must equal masked token mean")
        if not math.isfinite(self.wer) or self.wer < 0:
            raise ValueError("candidate WER must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "token_ids": list(self.token_ids),
            "token_mask": list(self.token_mask),
            "old_token_log_probs": list(self.old_token_log_probs),
            "old_sequence_log_probability": self.old_sequence_log_probability,
            "wer": self.wer,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CandidateRollout:
        return cls(
            hypothesis=str(value["hypothesis"]),
            token_ids=tuple(int(item) for item in value["token_ids"]),
            token_mask=tuple(bool(item) for item in value["token_mask"]),
            old_token_log_probs=tuple(float(item) for item in value["old_token_log_probs"]),
            old_sequence_log_probability=float(value["old_sequence_log_probability"]),
            wer=float(value["wer"]),
        )


@dataclass(frozen=True, slots=True)
class UtteranceRollout:
    utterance_id: str
    speaker_id: str
    primary_language: str
    family: str
    condition: AcousticCondition
    reference: str
    candidates: tuple[CandidateRollout, ...]

    def __post_init__(self) -> None:
        if not self.utterance_id or not self.speaker_id:
            raise ValueError("utterance and speaker IDs are required")
        if not self.family or not self.candidates:
            raise ValueError("family and rollout candidates are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "speaker_id": self.speaker_id,
            "primary_language": self.primary_language,
            "family": self.family,
            "condition": self.condition.value,
            "reference": self.reference,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> UtteranceRollout:
        return cls(
            utterance_id=str(value["utterance_id"]),
            speaker_id=str(value["speaker_id"]),
            primary_language=str(value["primary_language"]),
            family=str(value["family"]),
            condition=AcousticCondition(str(value["condition"])),
            reference=str(value["reference"]),
            candidates=tuple(
                CandidateRollout.from_dict(candidate)
                for candidate in value["candidates"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ObjectiveTensors:
    old_token_log_probs: torch.Tensor
    old_sequence_log_probs: torch.Tensor
    token_mask: torch.Tensor
    candidate_wers: torch.Tensor


@dataclass(frozen=True, slots=True)
class FrozenRolloutBatch:
    """One rollout batch whose old-policy evidence stays fixed for inner updates."""

    model_revision: str
    utterances: tuple[UtteranceRollout, ...]

    def __post_init__(self) -> None:
        if not self.model_revision or not self.utterances:
            raise ValueError("model revision and utterances are required")
        candidate_counts = {len(item.candidates) for item in self.utterances}
        if len(candidate_counts) != 1:
            raise ValueError("every utterance must contain the same candidate count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_revision": self.model_revision,
            "utterances": [utterance.to_dict() for utterance in self.utterances],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FrozenRolloutBatch:
        return cls(
            model_revision=str(value["model_revision"]),
            utterances=tuple(
                UtteranceRollout.from_dict(utterance)
                for utterance in value["utterances"]
            ),
        )

    def objective_tensors(self, device: torch.device | str = "cpu") -> ObjectiveTensors:
        batch_size = len(self.utterances)
        candidate_count = len(self.utterances[0].candidates)
        token_count = max(
            len(candidate.token_ids)
            for utterance in self.utterances
            for candidate in utterance.candidates
        )
        old_token = torch.zeros(
            (batch_size, candidate_count, token_count),
            dtype=torch.float32,
            device=device,
        )
        token_mask = torch.zeros_like(old_token, dtype=torch.bool)
        old_sequence = torch.empty(
            (batch_size, candidate_count),
            dtype=torch.float32,
            device=device,
        )
        wers = torch.empty_like(old_sequence)
        for utterance_index, utterance in enumerate(self.utterances):
            for candidate_index, candidate in enumerate(utterance.candidates):
                size = len(candidate.token_ids)
                old_token[utterance_index, candidate_index, :size] = torch.tensor(
                    candidate.old_token_log_probs,
                    dtype=torch.float32,
                    device=device,
                )
                token_mask[utterance_index, candidate_index, :size] = torch.tensor(
                    candidate.token_mask,
                    dtype=torch.bool,
                    device=device,
                )
                old_sequence[utterance_index, candidate_index] = (
                    candidate.old_sequence_log_probability
                )
                wers[utterance_index, candidate_index] = candidate.wer
        return ObjectiveTensors(
            old_token_log_probs=old_token,
            old_sequence_log_probs=old_sequence,
            token_mask=token_mask,
            candidate_wers=wers,
        )
