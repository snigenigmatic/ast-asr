"""Pure, fail-closed primitives for H7 historical sentinel-KL measurement."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from .rollouts import AcousticCondition, FrozenRolloutBatch

H7_CYCLES = tuple(range(28))
H7_FAMILIES = frozenset({"Dravidian", "Indo-Aryan", "Sino-Tibetan"})
H7_CANDIDATES = 4
H7_UTTERANCES_PER_BANK = 6
H7_K3_LIMIT = 0.1
H7_FROZEN_H6_INPUT_MANIFEST_SHA256 = (
    "92c5b7c7fb3457ca41da462fd8475d8a29694541f9f7ae2137fe989f10118c7e"
)


class K3Threshold(StrEnum):
    BELOW_LIMIT = "<0.1"
    AT_OR_ABOVE_LIMIT = ">=0.1"


@dataclass(frozen=True, slots=True)
class HistoricalBank:
    cycle: int
    frozen: FrozenRolloutBatch


@dataclass(frozen=True, slots=True)
class HistoricalBankSet:
    banks: tuple[HistoricalBank, ...]


@dataclass(frozen=True, slots=True)
class K3Summary:
    k3_per_token: torch.Tensor
    valid_token_count: int
    threshold: K3Threshold | None = None


@dataclass(frozen=True, slots=True)
class K3Decomposition:
    token_terms: torch.Tensor
    candidates: tuple[K3Summary, ...]
    utterances: tuple[K3Summary, ...]
    groups: dict[tuple[str, str], K3Summary]
    bank: K3Summary


@dataclass(frozen=True, slots=True)
class ContentFile:
    path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ContentManifest:
    files: tuple[ContentFile, ...]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "files": [
                {
                    "path": item.path,
                    "byte_count": item.byte_count,
                    "sha256": item.sha256,
                }
                for item in self.files
            ],
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class LockedInputFile:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class LockedH6InputManifest:
    """H6-compatible input-lock payload for the frozen H7 source artifacts."""

    arm: str
    modal_volume_path: str
    files: tuple[LockedInputFile, ...]
    manifest_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "modal_volume_path": self.modal_volume_path,
            "files": [
                {
                    "path": item.path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.files
            ],
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.canonical_payload()
        return {
            **payload,
            "diagnostic_file_count": sum(
                item.path.startswith("diagnostics/") for item in self.files
            ),
            "rollout_file_count": sum(
                item.path.startswith("rollouts/") for item in self.files
            ),
            "manifest_sha256": self.manifest_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_content_manifest(
    paths: tuple[Path, ...] | list[Path], *, root: Path
) -> ContentManifest:
    """Create a deterministic, content-addressed manifest without reading extras."""
    resolved_root = root.resolve()
    entries: list[ContentFile] = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise ValueError(f"manifest path is outside root: {path}") from error
        if not resolved.is_file():
            raise ValueError(f"manifest input is not a file: {path}")
        entries.append(
            ContentFile(
                path=relative,
                byte_count=resolved.stat().st_size,
                sha256=_sha256(resolved),
            )
        )
    files = tuple(sorted(entries, key=lambda entry: entry.path))
    if len({entry.path for entry in files}) != len(files):
        raise ValueError("manifest paths must be unique")
    payload = [
        {"path": item.path, "byte_count": item.byte_count, "sha256": item.sha256}
        for item in files
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return ContentManifest(files=files, sha256=hashlib.sha256(encoded).hexdigest())


def build_locked_h6_input_manifest(
    artifacts_root: Path,
    *,
    arm: str,
    modal_volume_path: str,
    arm_directory: Path | None = None,
    expected_cycles: tuple[int, ...] = H7_CYCLES,
) -> LockedH6InputManifest:
    """Reproduce H6's exact canonical diagnostic/rollout manifest digest.

    The digest is intentionally over only ``{arm, modal_volume_path, files}``
    with H6's field names and ordering. Counts and the digest are appended only
    after hashing, just as ``analyze_kl_failure.py`` did at ``ec40b77``.
    """
    if not arm or not modal_volume_path.startswith("/artifacts/"):
        raise ValueError("locked H6 manifest requires an arm and Modal artifact path")
    arm_root = arm_directory if arm_directory is not None else artifacts_root / arm
    if arm_directory is not None and not arm_root.is_relative_to(artifacts_root):
        raise ValueError("locked H6 arm directory must remain under artifacts root")
    expected = {f"cycle-{cycle:03d}.json" for cycle in expected_cycles}
    files: list[LockedInputFile] = []
    for kind in ("diagnostics", "rollouts"):
        directory = arm_root / kind
        observed = {path.name for path in directory.glob("cycle-*.json")}
        if observed != expected:
            raise ValueError(
                f"locked H6 {kind} files differ; missing={sorted(expected - observed)}, "
                f"unexpected={sorted(observed - expected)}"
            )
        for path in sorted(directory.glob("cycle-*.json")):
            files.append(
                LockedInputFile(
                    path=path.relative_to(arm_root).as_posix(),
                    sha256=_sha256(path),
                    size_bytes=path.stat().st_size,
                )
            )
    provisional = LockedH6InputManifest(
        arm=arm,
        modal_volume_path=modal_volume_path,
        files=tuple(files),
        manifest_sha256="",
    )
    encoded = json.dumps(
        provisional.canonical_payload(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return LockedH6InputManifest(
        arm=arm,
        modal_volume_path=modal_volume_path,
        files=tuple(files),
        manifest_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def assert_locked_manifest_digest(
    manifest: LockedH6InputManifest,
    expected_sha256: str = H7_FROZEN_H6_INPUT_MANIFEST_SHA256,
) -> LockedH6InputManifest:
    """Fail closed unless a locked H6-compatible digest is exactly reproduced."""
    if manifest.manifest_sha256 != expected_sha256:
        raise ValueError(
            "locked input manifest SHA-256 mismatch: "
            f"{manifest.manifest_sha256} != {expected_sha256}"
        )
    return manifest


def _require_exact_keys(
    value: dict[str, Any], *, expected: set[str], context: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context} keys differ; missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )


def _require_string(value: object, *, context: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{context} must be a non-empty JSON string")
    return value


def _require_float(value: object, *, context: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{context} must be a JSON float")
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    return value


def _validate_raw_bank(value: dict[str, Any], *, path: Path) -> None:
    """Reject type coercions and schema drift before rollout dataclasses parse it."""
    _require_exact_keys(
        value, expected={"model_revision", "utterances"}, context=path.name
    )
    _require_string(value["model_revision"], context=f"{path.name}.model_revision")
    utterances = value["utterances"]
    if type(utterances) is not list:
        raise TypeError(f"{path.name}.utterances must be a JSON list")
    utterance_keys = {
        "utterance_id",
        "speaker_id",
        "primary_language",
        "family",
        "condition",
        "reference",
        "candidates",
    }
    candidate_keys = {
        "hypothesis",
        "token_ids",
        "token_mask",
        "old_token_log_probs",
        "old_sequence_log_probability",
        "wer",
    }
    for utterance_index, utterance in enumerate(utterances):
        context = f"{path.name}.utterances[{utterance_index}]"
        if type(utterance) is not dict:
            raise TypeError(f"{context} must be a JSON object")
        _require_exact_keys(utterance, expected=utterance_keys, context=context)
        for key in utterance_keys - {"candidates"}:
            _require_string(utterance[key], context=f"{context}.{key}")
        if utterance["condition"] not in {"clean", "white_train"}:
            raise ValueError(f"{context}.condition is not an H7 training condition")
        candidates = utterance["candidates"]
        if type(candidates) is not list:
            raise TypeError(f"{context}.candidates must be a JSON list")
        for candidate_index, candidate in enumerate(candidates):
            candidate_context = f"{context}.candidates[{candidate_index}]"
            if type(candidate) is not dict:
                raise TypeError(f"{candidate_context} must be a JSON object")
            _require_exact_keys(
                candidate, expected=candidate_keys, context=candidate_context
            )
            _require_string(
                candidate["hypothesis"], context=f"{candidate_context}.hypothesis"
            )
            token_ids = candidate["token_ids"]
            token_mask = candidate["token_mask"]
            old_logs = candidate["old_token_log_probs"]
            if (
                type(token_ids) is not list
                or type(token_mask) is not list
                or type(old_logs) is not list
            ):
                raise TypeError(
                    f"{candidate_context} token evidence must use JSON lists"
                )
            if (
                not token_ids
                or len(token_ids) != len(token_mask)
                or len(token_ids) != len(old_logs)
            ):
                raise ValueError(
                    f"{candidate_context} token evidence lengths must align and be nonzero"
                )
            for token_index, token_id in enumerate(token_ids):
                if type(token_id) is not int or token_id < 0:
                    raise TypeError(
                        f"{candidate_context}.token_ids[{token_index}] must be a nonnegative integer"
                    )
            for token_index, keep in enumerate(token_mask):
                if type(keep) is not bool:
                    raise TypeError(
                        f"{candidate_context}.token_mask[{token_index}] must be boolean"
                    )
            for token_index, log_probability in enumerate(old_logs):
                probability = _require_float(
                    log_probability,
                    context=f"{candidate_context}.old_token_log_probs[{token_index}]",
                )
                round_trip = float(
                    torch.tensor(probability, dtype=torch.float32).item()
                )
                if round_trip != probability:
                    raise ValueError(
                        f"{candidate_context}.old_token_log_probs[{token_index}] must be exact FP32"
                    )
            _require_float(
                candidate["old_sequence_log_probability"],
                context=f"{candidate_context}.old_sequence_log_probability",
            )
            _require_float(candidate["wer"], context=f"{candidate_context}.wer")


def _validate_bank_shape(bank: HistoricalBank, *, path: Path) -> None:
    utterances = bank.frozen.utterances
    if len(utterances) != H7_UTTERANCES_PER_BANK:
        raise ValueError(f"{path.name} must contain exactly six utterances")
    if any(len(utterance.candidates) != H7_CANDIDATES for utterance in utterances):
        raise ValueError(
            f"{path.name} must contain exactly four candidates per utterance"
        )
    for utterance in utterances:
        for candidate in utterance.candidates:
            if not all(candidate.token_mask):
                raise ValueError(
                    f"{path.name} candidates must retain unpadded saved target tokens"
                )
    utterance_ids = [utterance.utterance_id for utterance in utterances]
    if len(set(utterance_ids)) != len(utterance_ids):
        raise ValueError(f"{path.name} contains duplicate utterance IDs")
    observed_families = {utterance.family for utterance in utterances}
    if observed_families != H7_FAMILIES:
        raise ValueError(f"{path.name} must contain exactly the three H7 families")
    for index in range(0, H7_UTTERANCES_PER_BANK, 2):
        clean, white = utterances[index : index + 2]
        if (
            clean.condition is not AcousticCondition.CLEAN
            or white.condition is not AcousticCondition.WHITE_TRAIN
        ):
            raise ValueError(
                f"{path.name} must preserve paired clean/white_train ordering"
            )
        if (
            clean.family != white.family
            or clean.speaker_id != white.speaker_id
            or clean.primary_language != white.primary_language
            or clean.reference != white.reference
        ):
            raise ValueError(f"{path.name} clean/white pair identity mismatch")
        white_id = re.fullmatch(
            rf"{re.escape(clean.utterance_id)}@white-([0-9]{{2}}\.[0-9]{{4}})db",
            white.utterance_id,
        )
        if white_id is None:
            raise ValueError(f"{path.name} white utterance ID has invalid SNR grammar")
        snr_db = float(white_id.group(1))
        if not 10.0 <= snr_db <= 20.0:
            raise ValueError(
                f"{path.name} white utterance SNR is outside the frozen bounds"
            )


def load_historical_banks(root: Path) -> HistoricalBankSet:
    """Load precisely cycles 000--027 and reject malformed historical evidence."""
    if not root.is_dir():
        raise ValueError(f"historical bank directory is missing: {root}")
    expected = {f"cycle-{cycle:03d}.json" for cycle in H7_CYCLES}
    observed = {path.name for path in root.glob("cycle-*.json")}
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"historical bank cycles differ; missing={missing}, unexpected={unexpected}"
        )
    banks = []
    for cycle in H7_CYCLES:
        path = root / f"cycle-{cycle:03d}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"{path.name} must be a JSON object")
        _validate_raw_bank(raw, path=path)
        bank = HistoricalBank(cycle=cycle, frozen=FrozenRolloutBatch.from_dict(raw))
        _validate_bank_shape(bank, path=path)
        banks.append(bank)
    return HistoricalBankSet(banks=tuple(banks))


def _masked_summary(
    terms: torch.Tensor, mask: torch.Tensor, *, threshold: bool = False
) -> K3Summary:
    selected = terms[mask]
    count = int(selected.numel())
    if count == 0:
        raise ValueError("K3 summary requires at least one valid token")
    value = selected.sum() / count
    return K3Summary(
        k3_per_token=value,
        valid_token_count=count,
        threshold=classify_k3(value) if threshold else None,
    )


def classify_k3(value: torch.Tensor) -> K3Threshold:
    """Classify a finite scalar under the registered inclusive 0.1 boundary."""
    if value.numel() != 1 or not bool(torch.isfinite(value).all()):
        raise ValueError("K3 threshold classification requires one finite scalar")
    return (
        K3Threshold.AT_OR_ABOVE_LIMIT
        if value >= H7_K3_LIMIT
        else K3Threshold.BELOW_LIMIT
    )


def decompose_sampled_k3(
    *,
    current_token_log_probs: torch.Tensor,
    reference_token_log_probs: torch.Tensor,
    token_mask: torch.Tensor,
    families: tuple[str, ...],
    conditions: tuple[str, ...],
) -> K3Decomposition:
    """Compute protocol K3 and token-weighted summaries, failing closed on bad terms."""
    if current_token_log_probs.shape != reference_token_log_probs.shape:
        raise ValueError("current and reference token log-probabilities must align")
    if (
        current_token_log_probs.ndim != 3
        or token_mask.shape != current_token_log_probs.shape
    ):
        raise ValueError(
            "K3 tensors must align with shape [utterance, candidate, token]"
        )
    if token_mask.dtype != torch.bool:
        raise ValueError("K3 token mask must use torch.bool")
    if current_token_log_probs.dtype != torch.float32:
        raise ValueError("current token log-probabilities must use FP32")
    if reference_token_log_probs.dtype != torch.float32:
        raise ValueError("reference token log-probabilities must use FP32")
    if (
        len(families) != current_token_log_probs.shape[0]
        or len(conditions) != current_token_log_probs.shape[0]
    ):
        raise ValueError("family and condition labels must align with utterances")
    if bool((token_mask.sum(dim=-1) == 0).any()):
        raise ValueError("every candidate requires at least one valid token")

    difference = (
        reference_token_log_probs.detach().float() - current_token_log_probs.float()
    )
    selected_difference = difference[token_mask]
    if not bool(torch.isfinite(selected_difference).all()):
        raise FloatingPointError("K3 received non-finite selected log-probabilities")
    if bool((selected_difference.abs() > 20).any()):
        raise FloatingPointError(
            "K3 selected log-ratio exceeded the numerical safety bound"
        )
    token_terms = torch.where(
        token_mask, torch.expm1(difference) - difference, torch.zeros_like(difference)
    )
    if not bool(torch.isfinite(token_terms[token_mask]).all()):
        raise FloatingPointError("K3 produced non-finite selected terms")
    candidates = tuple(
        _masked_summary(token_terms[index, candidate], token_mask[index, candidate])
        for index in range(token_terms.shape[0])
        for candidate in range(token_terms.shape[1])
    )
    utterances = tuple(
        _masked_summary(token_terms[index], token_mask[index])
        for index in range(token_terms.shape[0])
    )
    groups: dict[tuple[str, str], K3Summary] = {}
    for group in sorted(set(zip(families, conditions, strict=True))):
        indices = [
            index
            for index, value in enumerate(zip(families, conditions, strict=True))
            if value == group
        ]
        groups[group] = _masked_summary(token_terms[indices], token_mask[indices])
    return K3Decomposition(
        token_terms=token_terms,
        candidates=candidates,
        utterances=utterances,
        groups=groups,
        bank=_masked_summary(token_terms, token_mask, threshold=True),
    )
