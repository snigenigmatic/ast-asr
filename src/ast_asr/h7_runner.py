"""Fail-closed, measurement-only H7 sentinel-KL execution support.

This module deliberately has no optimizer, decoder, or Modal dependency.  It
contains the replay and output invariants used by the CUDA-only command runner.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import write_immutable_json
from .corruption import NoisePair, paired_white_noise
from .historical_kl import (
    H7_FROZEN_H6_INPUT_MANIFEST_SHA256,
    HistoricalBank,
    assert_locked_manifest_digest,
    build_locked_h6_input_manifest,
    decompose_sampled_k3,
    load_historical_banks,
)
from .modeling import (
    directory_content_hash,
    load_saved_processor,
    trainable_parameter_hash,
)
from .rollouts import AcousticCondition
from .sft import _load_audio
from .whisper_policy import score_saved_target_tokens

H7_SEED = 2028
H7_RESERVED_PROFILE_NAME = "profile-h7-fixed-policy-sentinel-kl-r1-s2028-20260812"
H7_RESERVED_OUTPUT_NAME = "h7-fixed-policy-sentinel-kl-r1"
H7_BETA0_CONFIG_SHA256 = (
    "3673abefc4322f4951ee067c8b6ed2c2fef93008b3f85c2cf66afd5abd406ae5"
)
H7_PREPARED_MANIFEST_SHA256 = (
    "65bdd8cf87f5db0f815e742739be815d2306ddd2b9977ee5687774feb1a18b56"
)
H7_FOLD_MANIFEST_SHA256 = (
    "22e9ab64006fe8a33bac37f5f2b98887df6aed061e158252778c29c6d928a1f0"
)
H7_H6_CANONICAL_ARM = "h6_s2028_beta0"
H7_H6_ARM_REMOTE_PATH = (
    "/artifacts/profile-h6-refkl-beta0-s2028-20260812/h6-beta0-fr-cispo"
)
H7_RESOLVED_CONFIG_REMOTE_PATH = (
    "/artifacts/profile-h6-refkl-beta0-s2028-20260812/"
    "resolved-policy-configs/h6-beta0-fr-cispo.json"
)
H7_PREPARED_REMOTE_PATH = "/data/fr_cispo_profile/prepared/dataset_manifest.json"
H7_FOLD_REMOTE_PATH = "/data/fr_cispo_profile/prepared/folds/fold-0.json"
H7_POLICY_REMOTE_PATH = (
    "/artifacts/profile-h6-refkl-beta0-s2028-20260812/"
    "h6-beta0-fr-cispo/checkpoint-last-safe"
)
H7_POLICY_DIRECTORY_HASH = (
    "a95530fd914b7fea9f3008a5c6451f3fedef2281443fce6b9dc0df5ba6a8d400"
)
H7_REFERENCE_REMOTE_PATH = (
    "/artifacts/profile-dev-full-sft-20260810/"
    "profile-sft-development/checkpoint-epoch-1"
)
H7_REFERENCE_DIRECTORY_HASH = (
    "d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e"
)


@dataclass(frozen=True, slots=True)
class ReplayedWhitePair:
    clean: torch.Tensor
    noisy: torch.Tensor
    snr_db: float
    seed: int

    @property
    def utterance_suffix(self) -> str:
        return f"@white-{self.snr_db:.4f}db"


def h7_white_seed(*, cycle: int, pair_index: int) -> int:
    if not 0 <= cycle <= 27:
        raise ValueError("H7 cycle must be in [0, 27]")
    if not 0 <= pair_index < 3:
        raise ValueError("H7 balanced pair index must be in [0, 3)")
    return H7_SEED * 1_000_003 + cycle * 101 + pair_index


def reconstruct_white_pair(
    clean: torch.Tensor,
    *,
    cycle: int,
    pair_index: int,
    expected_utterance_id: str | None = None,
) -> ReplayedWhitePair:
    """Recreate one registered H6 white-noise companion and validate its ID."""
    seed = h7_white_seed(cycle=cycle, pair_index=pair_index)
    replayed: NoisePair = paired_white_noise(clean.cpu(), seed=seed)
    result = ReplayedWhitePair(
        clean=replayed.clean,
        noisy=replayed.noisy,
        snr_db=replayed.snr_db,
        seed=seed,
    )
    if expected_utterance_id is not None and not expected_utterance_id.endswith(
        result.utterance_suffix
    ):
        raise ValueError(
            "saved white utterance SNR suffix does not match deterministic replay: "
            f"expected {result.utterance_suffix}, received {expected_utterance_id}"
        )
    return result


def reconstruct_bank_audio(
    bank: HistoricalBank,
    *,
    audio_by_clean_id: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Reconstruct a saved clean/white bank in its immutable six-row order.

    The caller supplies waveforms keyed only by clean utterance ID.  This keeps
    archive I/O outside the deterministic replay primitive and prevents noisy
    IDs from being treated as a second source artifact.
    """
    rows = bank.frozen.utterances
    if len(rows) % 2:
        raise ValueError("H7 bank must contain ordered clean/white pairs")
    reconstructed: list[torch.Tensor] = []
    for pair_index in range(0, len(rows), 2):
        clean, white = rows[pair_index : pair_index + 2]
        if (
            clean.condition is not AcousticCondition.CLEAN
            or white.condition is not AcousticCondition.WHITE_TRAIN
        ):
            raise ValueError("H7 bank must preserve clean/white pair ordering")
        try:
            waveform = audio_by_clean_id[clean.utterance_id]
        except KeyError as error:
            raise KeyError(f"missing clean audio for {clean.utterance_id}") from error
        replayed = reconstruct_white_pair(
            waveform,
            cycle=bank.cycle,
            pair_index=pair_index // 2,
            expected_utterance_id=white.utterance_id,
        )
        reconstructed.extend((replayed.clean, replayed.noisy))
    return tuple(reconstructed)


@dataclass(frozen=True, slots=True)
class H7ExecutionPlan:
    """Registered score ordering: Phase A twice, then each remaining bank once."""

    phase_a_cycle: int
    phase_b_cycles: tuple[int, ...]
    phase_b_reuses_phase_a: bool
    model_score_cycles: tuple[int, ...]

    @classmethod
    def build(cls) -> H7ExecutionPlan:
        return cls(
            phase_a_cycle=27,
            phase_b_cycles=tuple(range(27)),
            phase_b_reuses_phase_a=True,
            model_score_cycles=(27, 27, *range(27)),
        )


def require_phase_a_before_phase_b(identity_paths: Mapping[str, bool]) -> None:
    required = ("saved_old_path", "direct_policy_path")
    failed = [name for name in required if identity_paths.get(name) is not True]
    if failed:
        raise RuntimeError(f"Phase A identity paths did not pass: {', '.join(failed)}")


def classify_terminal_banks(
    classifications: list[str], *, measurement_error: str | None = None
) -> str:
    if measurement_error is not None:
        return "measurement_failed"
    allowed = {"<0.1", ">=0.1"}
    if not classifications or set(classifications) - allowed:
        raise ValueError(
            "terminal classification requires only registered nonempty bank labels"
        )
    labels = set(classifications)
    if len(labels) == 2:
        return "bank_dependent"
    return "all_below" if labels == {"<0.1"} else "all_at_or_above"


def write_h7_output(path: Path, payload: Mapping[str, object]) -> None:
    """Write a canonical H7 artifact only when its non-publication labels hold."""
    if payload.get("publication_valid") is not False:
        raise ValueError("H7 artifacts must set publication_valid to false")
    if payload.get("profile_cluster_count") != 115:
        raise ValueError("H7 artifacts must record exactly 115 profile clusters")
    write_immutable_json(path, dict(payload))


def write_h7_failure(
    path: Path,
    *,
    phase: str,
    error: BaseException,
    command: tuple[str, ...],
    provenance: Mapping[str, object] | None = None,
) -> None:
    """Persist terminal evidence for any consumed H7 attempt."""
    write_h7_output(
        path,
        {
            "artifact_kind": "h7_measurement_failure",
            "publication_valid": False,
            "profile_cluster_count": 115,
            "decision": "measurement_failed",
            "measurement_status": "non_evaluable",
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "command": list(command),
            "provenance": dict(provenance or {}),
        },
    )


def persist_h7_failure(
    path: Path,
    *,
    phase: str,
    error: BaseException,
    command: tuple[str, ...],
    provenance: Mapping[str, object] | None = None,
) -> bool:
    """Write terminal failure evidence once; return false when the runner did."""
    if path.exists():
        return False
    write_h7_failure(
        path, phase=phase, error=error, command=command, provenance=provenance
    )
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_files(paths: list[Path]) -> list[dict[str, object]]:
    resolved = sorted({path.resolve() for path in paths}, key=lambda item: str(item))
    return [
        {"path": str(path), "sha256": _sha256(path), "byte_count": path.stat().st_size}
        for path in resolved
        if path.is_file()
    ]


def _directory_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"H7 source directory is missing: {directory}")
    return sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda item: item.as_posix(),
    )


def tensor_identity(tensor: torch.Tensor) -> dict[str, object]:
    """Hash tensor values together with the dtype and shape that define them."""
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())
    return {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "sha256": digest.hexdigest(),
    }


def _require_sha256(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _require_identity(
    value: object, *, context: str, expected_sha256: str | None = None
) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{context} must contain exactly path and sha256")
    if not isinstance(value["path"], str) or not value["path"]:
        raise ValueError(f"{context}.path must be non-empty")
    actual = _require_sha256(value["sha256"], context=f"{context}.sha256")
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"{context} does not match its frozen SHA-256")


def _require_tensor_identity(value: object, *, context: str) -> None:
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "sha256"}:
        raise ValueError(f"{context} must contain dtype, shape, and sha256")
    if not isinstance(value["dtype"], str) or not isinstance(value["shape"], list):
        raise TypeError(f"{context} has invalid tensor dtype or shape")
    if not value["shape"] or any(
        type(item) is not int or item <= 0 for item in value["shape"]
    ):
        raise ValueError(f"{context}.shape must be positive integers")
    _require_sha256(value["sha256"], context=f"{context}.sha256")


def validate_h7_input_lock_payload(payload: object) -> dict[str, object]:
    """Validate one in-memory full H7 lock payload without writing a file."""
    if not isinstance(payload, dict):
        raise TypeError("H7 input lock must be a JSON object")
    expected = {
        "schema_version",
        "publication_valid",
        "profile_cluster_count",
        "resolved_config",
        "prepared_manifest",
        "fold_manifest",
        "policy_checkpoint",
        "reference_checkpoint",
        "h6_input_manifest_sha256",
        "banks",
    }
    if set(payload) != expected:
        raise ValueError("H7 input lock keys do not match the committed schema")
    if (
        payload["schema_version"] != 1
        or payload["publication_valid"] is not False
        or payload["profile_cluster_count"] != 115
    ):
        raise ValueError("H7 input lock has invalid fixed classification metadata")
    _require_identity(
        payload["resolved_config"],
        context="resolved_config",
        expected_sha256=H7_BETA0_CONFIG_SHA256,
    )
    _require_identity(
        payload["prepared_manifest"],
        context="prepared_manifest",
        expected_sha256=H7_PREPARED_MANIFEST_SHA256,
    )
    _require_identity(
        payload["fold_manifest"],
        context="fold_manifest",
        expected_sha256=H7_FOLD_MANIFEST_SHA256,
    )
    _require_identity(payload["policy_checkpoint"], context="policy_checkpoint")
    _require_identity(payload["reference_checkpoint"], context="reference_checkpoint")
    if payload.get("h6_input_manifest_sha256") != H7_FROZEN_H6_INPUT_MANIFEST_SHA256:
        raise ValueError("H7 input lock does not match the frozen H6 manifest")
    banks = payload["banks"]
    if not isinstance(banks, list) or [
        item.get("cycle") if isinstance(item, dict) else None for item in banks
    ] != list(range(28)):
        raise ValueError("H7 input lock must contain cycles 0 through 27 in order")
    for bank in banks:
        if not isinstance(bank, dict) or set(bank) != {"cycle", "pairs", "features"}:
            raise ValueError(
                "every H7 lock bank must contain cycle, pairs, and features"
            )
        pairs = bank["pairs"]
        if not isinstance(pairs, list) or [
            item.get("pair_index") if isinstance(item, dict) else None for item in pairs
        ] != [0, 1, 2]:
            raise ValueError(
                f"H7 input lock cycle {bank['cycle']} must contain three ordered pairs"
            )
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) != {
                "pair_index",
                "clean",
                "white",
            }:
                raise ValueError("H7 lock pair keys differ")
            clean, white = pair["clean"], pair["white"]
            clean_expected = {
                "utterance_id",
                "speaker_id",
                "primary_language",
                "family",
                "condition",
                "reference",
                "audio_path",
                "source_audio_sha256",
                "waveform_sha256",
            }
            white_expected = {
                "utterance_id",
                "speaker_id",
                "primary_language",
                "family",
                "condition",
                "reference",
                "seed",
                "snr_db",
                "waveform_sha256",
            }
            if (
                not isinstance(clean, dict)
                or set(clean) != clean_expected
                or clean.get("condition") != "clean"
            ):
                raise ValueError("H7 lock clean pair metadata differs")
            if (
                not isinstance(white, dict)
                or set(white) != white_expected
                or white.get("condition") != "white_train"
            ):
                raise ValueError("H7 lock white pair metadata differs")
            for key in ("speaker_id", "primary_language", "family", "reference"):
                if (
                    not isinstance(clean[key], str)
                    or not isinstance(white[key], str)
                    or clean[key] != white[key]
                ):
                    raise ValueError("H7 lock pair identity differs")
            if not isinstance(clean["audio_path"], str) or not clean["audio_path"]:
                raise ValueError("H7 lock clean audio path is invalid")
            _require_sha256(clean["source_audio_sha256"], context="clean source audio")
            _require_sha256(clean["waveform_sha256"], context="clean waveform")
            _require_sha256(white["waveform_sha256"], context="white waveform")
            if white["seed"] != h7_white_seed(
                cycle=int(bank["cycle"]), pair_index=int(pair["pair_index"])
            ):
                raise ValueError("H7 lock white seed differs from the registered rule")
            if (
                type(white["snr_db"]) is not float
                or not 10.0 <= white["snr_db"] <= 20.0
            ):
                raise ValueError("H7 lock white SNR is invalid")
            if (
                white["utterance_id"]
                != f"{clean['utterance_id']}@white-{white['snr_db']:.4f}db"
            ):
                raise ValueError("H7 lock white ID does not match four-decimal SNR")
        features = bank["features"]
        if not isinstance(features, dict) or set(features) != {
            "input_features",
            "attention_mask",
        }:
            raise ValueError("H7 lock bank features differ")
        _require_tensor_identity(features["input_features"], context="input_features")
        _require_tensor_identity(features["attention_mask"], context="attention_mask")
    return payload


def validate_h7_input_lock(path: Path) -> dict[str, object]:
    """Load and validate the committed full 28-bank H7 input lock."""
    if not path.is_file():
        raise FileNotFoundError(f"H7 input lock is missing: {path}")
    return validate_h7_input_lock_payload(json.loads(path.read_text(encoding="utf-8")))


def _load_prepared_audio(
    prepared_manifest: Path,
    archive_root: Path,
    *,
    required_clean_ids: set[str],
) -> dict[str, tuple[torch.Tensor, Path, str]]:
    raw = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    if (
        raw.get("identity_count") != 115
        or raw.get("identity_mode") != "demographic_profile"
    ):
        raise ValueError("H7 requires the frozen 115-profile prepared manifest")
    audio: dict[str, tuple[torch.Tensor, Path, str]] = {}
    for record in raw.get("utterances", []):
        utterance_id = str(record["utterance_id"])
        if utterance_id not in required_clean_ids:
            continue
        if utterance_id in audio:
            raise ValueError(
                f"prepared manifest has duplicate required utterance ID: {utterance_id}"
            )
        path = archive_root / str(record["audio_path"])
        if not path.is_file():
            raise FileNotFoundError(f"prepared source audio is missing: {path}")
        audio[utterance_id] = (_load_audio(path), path, str(record["audio_path"]))
    missing = sorted(required_clean_ids - set(audio))
    if missing:
        raise ValueError(
            f"prepared manifest is missing required H7 clean utterance IDs: {missing}"
        )
    return audio


def extract_h7_features(
    processor: Any, waveforms: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact feature-extractor call shape used by ``generate_frozen_rollout``."""
    acoustic = processor.feature_extractor(
        [item.cpu().numpy() for item in waveforms],
        sampling_rate=16_000,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )
    return acoustic.input_features.to(device), acoustic.attention_mask.to(device)


def validate_locked_bank_replay(
    bank: HistoricalBank,
    *,
    lock_bank: Mapping[str, object],
    audio_by_clean_id: Mapping[str, torch.Tensor],
    source_paths: Mapping[str, Path],
    prepared_audio_paths: Mapping[str, str],
    processor: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute and compare every locked pair/audio/feature identity before scoring."""
    if lock_bank.get("cycle") != bank.cycle:
        raise ValueError("H7 input-lock bank cycle differs from the loaded rollout")
    pairs = lock_bank.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 3:
        raise ValueError("H7 input-lock bank pair count differs")
    waveforms = reconstruct_bank_audio(bank, audio_by_clean_id=audio_by_clean_id)
    for pair_index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise TypeError("H7 input-lock pair is invalid")
        clean_lock, white_lock = pair["clean"], pair["white"]
        if not isinstance(clean_lock, dict) or not isinstance(white_lock, dict):
            raise TypeError("H7 input-lock pair metadata is invalid")
        clean, white = bank.frozen.utterances[pair_index * 2 : pair_index * 2 + 2]
        if (
            clean_lock["utterance_id"] != clean.utterance_id
            or white_lock["utterance_id"] != white.utterance_id
        ):
            raise ValueError("H7 input-lock utterance identity differs from rollout")
        for key in ("speaker_id", "primary_language", "family", "reference"):
            if clean_lock[key] != getattr(clean, key) or white_lock[key] != getattr(
                white, key
            ):
                raise ValueError("H7 input-lock group metadata differs from rollout")
        source = source_paths.get(clean.utterance_id)
        if source is None or _sha256(source) != clean_lock["source_audio_sha256"]:
            raise ValueError("H7 input-lock source audio SHA-256 mismatch")
        if prepared_audio_paths.get(clean.utterance_id) != clean_lock["audio_path"]:
            raise ValueError("H7 input-lock prepared relative audio path mismatch")
        replayed = reconstruct_white_pair(
            waveforms[pair_index * 2],
            cycle=bank.cycle,
            pair_index=pair_index,
            expected_utterance_id=white.utterance_id,
        )
        if (
            replayed.seed != white_lock["seed"]
            or replayed.snr_db != white_lock["snr_db"]
        ):
            raise ValueError("H7 input-lock white replay seed or SNR mismatch")
        if (
            tensor_identity(waveforms[pair_index * 2])["sha256"]
            != clean_lock["waveform_sha256"]
        ):
            raise ValueError("H7 input-lock clean waveform hash mismatch")
        if (
            tensor_identity(waveforms[pair_index * 2 + 1])["sha256"]
            != white_lock["waveform_sha256"]
        ):
            raise ValueError("H7 input-lock noisy waveform hash mismatch")
    features, attention = extract_h7_features(processor, waveforms, torch.device("cpu"))
    identities = lock_bank.get("features")
    if not isinstance(identities, dict) or tensor_identity(features) != identities.get(
        "input_features"
    ):
        raise ValueError("H7 input-lock extracted feature tensor mismatch")
    if tensor_identity(attention) != identities.get("attention_mask"):
        raise ValueError("H7 input-lock attention tensor mismatch")
    return features, attention


def prevalidate_h7_bank_inputs(
    banks: tuple[HistoricalBank, ...],
    *,
    locked_banks: object,
    audio_by_clean_id: Mapping[str, torch.Tensor],
    source_paths: Mapping[str, Path],
    prepared_audio_paths: Mapping[str, str],
    processor: Any,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Validate every historical acoustic bank before allowing one model forward."""
    if not isinstance(locked_banks, list) or len(locked_banks) != 28:
        raise ValueError("H7 requires 28 locked banks before Phase A")
    cached: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for bank in banks:
        locked_bank = locked_banks[bank.cycle]
        if not isinstance(locked_bank, dict):
            raise TypeError("H7 locked bank record must be an object")
        cached[bank.cycle] = validate_locked_bank_replay(
            bank,
            lock_bank=locked_bank,
            audio_by_clean_id=audio_by_clean_id,
            source_paths=source_paths,
            prepared_audio_paths=prepared_audio_paths,
            processor=processor,
        )
    return cached


def build_h7_input_lock_candidate(
    *,
    banks: tuple[HistoricalBank, ...],
    audio_by_clean_id: Mapping[str, torch.Tensor],
    source_paths: Mapping[str, Path],
    prepared_records: Mapping[str, Mapping[str, object]],
    processor: Any,
    identities: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    """Build a deterministic review candidate; this helper does not authorize H7.

    The caller must independently verify and commit the returned object before
    it can be consumed by ``run_h7_cuda``.
    """
    if tuple(bank.cycle for bank in banks) != tuple(range(28)):
        raise ValueError("H7 input-lock candidates require all 28 ordered banks")
    required_identities = {
        "resolved_config",
        "prepared_manifest",
        "fold_manifest",
        "policy_checkpoint",
        "reference_checkpoint",
    }
    if set(identities) != required_identities:
        raise ValueError("H7 input-lock candidate identities differ")
    locked_banks: list[dict[str, object]] = []
    for bank in banks:
        waveforms = reconstruct_bank_audio(bank, audio_by_clean_id=audio_by_clean_id)
        pairs = []
        for pair_index in range(3):
            clean, white = bank.frozen.utterances[pair_index * 2 : pair_index * 2 + 2]
            record = prepared_records[clean.utterance_id]
            source = source_paths[clean.utterance_id]
            replayed = reconstruct_white_pair(
                waveforms[pair_index * 2],
                cycle=bank.cycle,
                pair_index=pair_index,
                expected_utterance_id=white.utterance_id,
            )
            pairs.append(
                {
                    "pair_index": pair_index,
                    "clean": {
                        "utterance_id": clean.utterance_id,
                        "speaker_id": clean.speaker_id,
                        "primary_language": clean.primary_language,
                        "family": clean.family,
                        "condition": clean.condition.value,
                        "reference": clean.reference,
                        "audio_path": str(record["audio_path"]),
                        "source_audio_sha256": _sha256(source),
                        "waveform_sha256": tensor_identity(replayed.clean)["sha256"],
                    },
                    "white": {
                        "utterance_id": white.utterance_id,
                        "speaker_id": white.speaker_id,
                        "primary_language": white.primary_language,
                        "family": white.family,
                        "condition": white.condition.value,
                        "reference": white.reference,
                        "seed": replayed.seed,
                        "snr_db": replayed.snr_db,
                        "waveform_sha256": tensor_identity(replayed.noisy)["sha256"],
                    },
                }
            )
        features, attention = extract_h7_features(
            processor, waveforms, torch.device("cpu")
        )
        locked_banks.append(
            {
                "cycle": bank.cycle,
                "pairs": pairs,
                "features": {
                    "input_features": tensor_identity(features),
                    "attention_mask": tensor_identity(attention),
                },
            }
        )
    candidate: dict[str, object] = {
        "schema_version": 1,
        "publication_valid": False,
        "profile_cluster_count": 115,
        **{key: dict(value) for key, value in identities.items()},
        "h6_input_manifest_sha256": H7_FROZEN_H6_INPUT_MANIFEST_SHA256,
        "banks": locked_banks,
    }
    return candidate


def _saved_token_tensors(
    bank: HistoricalBank, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = bank.frozen.objective_tensors(device=device)
    max_tokens = tensors.old_token_log_probs.shape[-1]
    ids = torch.zeros_like(tensors.old_token_log_probs, dtype=torch.long)
    for row_index, utterance in enumerate(bank.frozen.utterances):
        for candidate_index, candidate in enumerate(utterance.candidates):
            ids[row_index, candidate_index, : len(candidate.token_ids)] = torch.tensor(
                candidate.token_ids, dtype=torch.long, device=device
            )
    if ids.shape[-1] != max_tokens:
        raise RuntimeError("saved target tensor shape drift")
    return ids, tensors.token_mask


def _rows_for_bank(
    bank: HistoricalBank,
    *,
    current: torch.Tensor,
    reference: torch.Tensor,
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if current.dtype != torch.float32 or reference.dtype != torch.float32:
        raise RuntimeError("H7 emitted token log probabilities must be FP32")
    labels = tuple(item.family for item in bank.frozen.utterances)
    conditions = tuple(item.condition.value for item in bank.frozen.utterances)
    decomposed = decompose_sampled_k3(
        current_token_log_probs=current,
        reference_token_log_probs=reference,
        token_mask=token_mask,
        families=labels,
        conditions=conditions,
    )
    rows: list[dict[str, object]] = []
    for utterance_index, utterance in enumerate(bank.frozen.utterances):
        for candidate_index, candidate in enumerate(utterance.candidates):
            for token_index, valid in enumerate(
                token_mask[utterance_index, candidate_index].tolist()
            ):
                if valid:
                    rows.append(
                        {
                            "bank": bank.cycle,
                            "utterance_id": utterance.utterance_id,
                            "speaker_id": utterance.speaker_id,
                            "family": utterance.family,
                            "condition": utterance.condition.value,
                            "candidate_index": candidate_index,
                            "token_position": token_index,
                            "token_id": int(
                                token_ids[utterance_index, candidate_index, token_index]
                            ),
                            "valid_mask": True,
                            "policy_log_probability_fp32": float(
                                current[utterance_index, candidate_index, token_index]
                            ),
                            "reference_log_probability_fp32": float(
                                reference[utterance_index, candidate_index, token_index]
                            ),
                            "log_ratio": float(
                                reference[utterance_index, candidate_index, token_index]
                                - current[utterance_index, candidate_index, token_index]
                            ),
                            "k3_contribution": float(
                                decomposed.token_terms[
                                    utterance_index, candidate_index, token_index
                                ]
                            ),
                        }
                    )
    summary = {
        "bank": bank.cycle,
        "k3_per_token": float(decomposed.bank.k3_per_token),
        "valid_token_count": decomposed.bank.valid_token_count,
        "threshold": decomposed.bank.threshold.value
        if decomposed.bank.threshold
        else None,
        "family_condition": {
            f"{family}::{condition}": {
                "k3_per_token": float(value.k3_per_token),
                "valid_token_count": value.valid_token_count,
            }
            for (family, condition), value in decomposed.groups.items()
        },
        "candidates": [
            {
                "utterance_id": utterance.utterance_id,
                "family": utterance.family,
                "condition": utterance.condition.value,
                "candidate_index": candidate_index,
                "k3_per_token": float(
                    decomposed.candidates[
                        utterance_index * 4 + candidate_index
                    ].k3_per_token
                ),
                "valid_token_count": decomposed.candidates[
                    utterance_index * 4 + candidate_index
                ].valid_token_count,
            }
            for utterance_index, utterance in enumerate(bank.frozen.utterances)
            for candidate_index in range(4)
        ],
        "length": [
            {
                "candidate_valid_token_count": item.valid_token_count,
                "k3_per_token": float(item.k3_per_token),
            }
            for item in decomposed.candidates
        ],
        "utterances": [
            {
                "utterance_id": utterance.utterance_id,
                "family": utterance.family,
                "condition": utterance.condition.value,
                "k3_per_token": float(decomposed.utterances[index].k3_per_token),
                "valid_token_count": decomposed.utterances[index].valid_token_count,
            }
            for index, utterance in enumerate(bank.frozen.utterances)
        ],
    }
    return rows, summary


def _run_h7_cuda(
    *,
    config_path: Path,
    bank_root: Path,
    archive_root: Path,
    policy_checkpoint: Path,
    reference_checkpoint: Path,
    output_dir: Path,
    input_lock: Path,
    expected_policy_revision: str,
    expected_reference_revision: str,
    expected_cycle27_model_revision: str,
    expected_config_sha256: str,
) -> dict[str, object]:
    """Execute the registered H7 score-only measurement on one CUDA process.

    The caller must preflight the immutable input lock and reserved output path.
    This routine neither generates nor decodes text and contains no optimizer.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("H7 requires CUDA FP16 scoring")
    if not os.environ.get("H7_SOURCE_COMMIT"):
        raise RuntimeError("H7 requires an explicit local source-commit identity")
    if not os.environ.get("MODAL_IMAGE_ID"):
        raise RuntimeError("H7 requires a Modal image identity")
    if (
        expected_config_sha256 != H7_BETA0_CONFIG_SHA256
        or _sha256(config_path) != expected_config_sha256
    ):
        raise ValueError("frozen beta-zero resolved config hash mismatch")
    if directory_content_hash(policy_checkpoint) != expected_policy_revision:
        raise ValueError("policy checkpoint directory hash mismatch")
    if directory_content_hash(reference_checkpoint) != expected_reference_revision:
        raise ValueError("reference checkpoint directory hash mismatch")
    from .config import ExperimentConfig
    from .modeling import build_lora_whisper

    locked_input = validate_h7_input_lock(input_lock)
    h6_arm_root = bank_root.parent
    assert_locked_manifest_digest(
        build_locked_h6_input_manifest(
            h6_arm_root.parent,
            arm=H7_H6_CANONICAL_ARM,
            modal_volume_path=str(h6_arm_root),
            arm_directory=h6_arm_root,
        )
    )
    config = ExperimentConfig.from_json(config_path)
    if (
        config.model.model_id != "openai/whisper-tiny"
        or config.model.revision != "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
        or config.dataset.dataset_id != "ai4bharat/Svarah"
        or config.dataset.revision != "ebbf7777fe771490696a3f7b007097606fa8c924"
        or config.policy.training_snr_db != (10.0, 20.0)
        or config.model.lora_dropout != 0.0
    ):
        raise ValueError(
            "H7 resolved config differs from the frozen model/data/noise contract"
        )
    fold_manifest = config.dataset.fold_directory / "fold-0.json"
    for actual_path, key in (
        (config_path, "resolved_config"),
        (config.dataset.prepared_manifest, "prepared_manifest"),
        (fold_manifest, "fold_manifest"),
    ):
        identity = locked_input[key]
        assert isinstance(identity, dict)
        if identity["path"] != str(actual_path) or identity["sha256"] != _sha256(
            actual_path
        ):
            raise ValueError(f"H7 input-lock {key} differs from the mounted input")
    for actual_path, key in (
        (policy_checkpoint, "policy_checkpoint"),
        (reference_checkpoint, "reference_checkpoint"),
    ):
        identity = locked_input[key]
        assert isinstance(identity, dict)
        if identity["path"] != str(actual_path) or identity[
            "sha256"
        ] != directory_content_hash(actual_path):
            raise ValueError(f"H7 input-lock {key} differs from the mounted input")
    banks = load_historical_banks(bank_root)
    device = torch.device("cuda")
    policy = build_lora_whisper(
        config.model,
        adapter_checkpoint=policy_checkpoint,
        trainable=True,
        device=device,
    )
    if trainable_parameter_hash(policy) != expected_cycle27_model_revision:
        raise ValueError("cycle-027 trainable parameter hash mismatch")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    reference = build_lora_whisper(
        config.model,
        adapter_checkpoint=reference_checkpoint,
        trainable=False,
        device=device,
    )
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    reference.eval()
    if (
        policy.training
        or reference.training
        or any(parameter.requires_grad for parameter in policy.parameters())
        or any(parameter.requires_grad for parameter in reference.parameters())
    ):
        raise RuntimeError(
            "H7 policy/reference must remain frozen and in evaluation mode"
        )
    if (
        next(policy.parameters()).dtype != torch.float16
        or next(reference.parameters()).dtype != torch.float16
    ):
        raise RuntimeError("H7 requires FP16 policy and reference compute dtypes")
    processor = load_saved_processor(reference_checkpoint)
    prepared_audio = _load_prepared_audio(
        config.dataset.prepared_manifest,
        archive_root,
        required_clean_ids={
            utterance.utterance_id
            for bank in banks.banks
            for utterance in bank.frozen.utterances
            if utterance.condition is AcousticCondition.CLEAN
        },
    )
    audio_by_id = {
        utterance_id: value[0] for utterance_id, value in prepared_audio.items()
    }
    audio_paths = {
        utterance_id: value[1] for utterance_id, value in prepared_audio.items()
    }
    prepared_audio_paths = {
        utterance_id: value[2] for utterance_id, value in prepared_audio.items()
    }
    cached_acoustics = prevalidate_h7_bank_inputs(
        banks.banks,
        locked_banks=locked_input["banks"],
        audio_by_clean_id=audio_by_id,
        source_paths=audio_paths,
        prepared_audio_paths=prepared_audio_paths,
        processor=processor,
    )
    diagnostic_root = bank_root.parent / "diagnostics"
    source_paths = [
        config_path,
        input_lock,
        config.dataset.prepared_manifest,
        fold_manifest,
        Path(__file__),
    ]
    source_paths.extend(bank_root / f"cycle-{cycle:03d}.json" for cycle in range(28))
    source_paths.extend(
        diagnostic_root / f"cycle-{cycle:03d}.json" for cycle in range(28)
    )
    source_paths.extend(audio_paths.values())
    source_paths.extend(_directory_files(policy_checkpoint))
    source_paths.extend(_directory_files(reference_checkpoint))
    project_root = Path(__file__).resolve().parents[2]
    source_paths.extend(_directory_files(Path(__file__).resolve().parent))
    source_paths.extend((project_root / "uv.lock", project_root / "pyproject.toml"))
    write_h7_output(
        output_dir / "source_manifest.json",
        {
            "publication_valid": False,
            "profile_cluster_count": 115,
            "decision": "not_yet_evaluated",
            "files": _manifest_files(source_paths),
            "policy_checkpoint_revision": expected_policy_revision,
            "reference_checkpoint_revision": expected_reference_revision,
            "source_commit": os.environ.get("H7_SOURCE_COMMIT"),
            "image_identity": os.environ.get("MODAL_IMAGE_ID"),
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "command": sys.argv,
            "model": {"id": config.model.model_id, "revision": config.model.revision},
            "dataset": {
                "id": config.dataset.dataset_id,
                "revision": config.dataset.revision,
            },
            "training_snr_db": list(config.policy.training_snr_db),
            "policy_eval": not policy.training,
            "reference_eval": not reference.training,
            "policy_parameters_frozen": not any(
                parameter.requires_grad for parameter in policy.parameters()
            ),
            "reference_parameters_frozen": not any(
                parameter.requires_grad for parameter in reference.parameters()
            ),
        },
    )
    prefix = (
        int(policy.config.decoder_start_token_id),
        *tuple(
            token
            for _, token in sorted(
                processor.get_decoder_prompt_ids(language="english", task="transcribe")
            )
        ),
    )
    pad = int(processor.tokenizer.pad_token_id)

    def score(
        bank: HistoricalBank,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, attention = cached_acoustics[bank.cycle]
        features, attention = features.to(device), attention.to(device)
        ids, mask = _saved_token_tensors(bank, device)
        policy_scores = score_saved_target_tokens(
            policy,
            input_features=features,
            attention_mask=attention,
            saved_token_ids=ids,
            saved_token_mask=mask,
            prefix_token_ids=prefix,
            pad_token_id=pad,
        )
        reference_scores = score_saved_target_tokens(
            reference,
            input_features=features,
            attention_mask=attention,
            saved_token_ids=ids,
            saved_token_mask=mask,
            prefix_token_ids=prefix,
            pad_token_id=pad,
        )
        if not torch.equal(policy_scores.token_ids, ids) or not torch.equal(
            policy_scores.token_mask, mask
        ):
            raise RuntimeError("saved policy target token/mask evidence changed")
        if not torch.equal(reference_scores.token_ids, ids) or not torch.equal(
            reference_scores.token_mask, mask
        ):
            raise RuntimeError("saved reference target token/mask evidence changed")
        if (
            policy_scores.token_log_probs.dtype != torch.float32
            or reference_scores.token_log_probs.dtype != torch.float32
        ):
            raise RuntimeError("H7 scorer did not emit FP32 token log probabilities")
        return (
            policy_scores.token_log_probs,
            reference_scores.token_log_probs,
            ids,
            mask,
        )

    plan = H7ExecutionPlan.build()
    if tuple(bank.cycle for bank in banks.banks) != tuple(
        range(28)
    ) or plan.model_score_cycles != (27, 27, *range(27)):
        raise RuntimeError(
            "H7 execution plan does not match the loaded historical bank set"
        )
    cycle27 = banks.banks[plan.phase_a_cycle]
    policy27, reference27, ids27, mask27 = score(cycle27)
    policy27_repeat, reference27_repeat, _, _ = score(cycle27)
    if not torch.allclose(
        policy27, policy27_repeat, atol=1e-7, rtol=0
    ) or not torch.allclose(reference27, reference27_repeat, atol=1e-7, rtol=0):
        raise RuntimeError("Phase A repeated full-bank score was nondeterministic")
    old27 = cycle27.frozen.objective_tensors(device=device).old_token_log_probs
    _old_rows, old_summary = _rows_for_bank(
        cycle27,
        current=old27,
        reference=reference27,
        token_ids=ids27,
        token_mask=mask27,
    )
    direct_rows, direct_summary = _rows_for_bank(
        cycle27,
        current=policy27,
        reference=reference27,
        token_ids=ids27,
        token_mask=mask27,
    )
    old_error = float((policy27[mask27] - old27[mask27]).abs().max())
    if old_error > 1e-5:
        raise RuntimeError(f"Phase A direct policy path mismatch: {old_error}")
    if abs(float(old_summary["k3_per_token"]) - 0.10983546078205109) > 1e-7:
        raise RuntimeError("Phase A saved-old path did not reproduce cycle-027 K3")
    if (
        abs(float(old_summary["k3_per_token"]) - float(direct_summary["k3_per_token"]))
        > 1e-7
    ):
        raise RuntimeError("Phase A K3 paths disagree")
    require_phase_a_before_phase_b({"saved_old_path": True, "direct_policy_path": True})
    token_rows, summaries = list(direct_rows), [direct_summary]
    for cycle in plan.phase_b_cycles:
        bank = banks.banks[cycle]
        current, reference_scores, ids, mask = score(bank)
        rows, summary = _rows_for_bank(
            bank,
            current=current,
            reference=reference_scores,
            token_ids=ids,
            token_mask=mask,
        )
        token_rows.extend(rows)
        summaries.append(summary)
    token_rows.sort(
        key=lambda item: (
            item["bank"],
            item["utterance_id"],
            item["candidate_index"],
            item["token_position"],
        )
    )
    summaries.sort(key=lambda item: int(item["bank"]))
    labels = [str(item["threshold"]) for item in summaries]
    decision = classify_terminal_banks(labels)
    output = {
        "publication_valid": False,
        "profile_cluster_count": 115,
        "decision": decision,
        "measurement_status": "evaluable",
        "phase_a": {
            "saved_old": old_summary,
            "direct_policy": direct_summary,
            "max_old_log_probability_error": old_error,
            "repeated_score_equal": True,
        },
        "bank_summaries": summaries,
        "token_rows": token_rows,
        "provenance": {
            "config_sha256": _sha256(config_path),
            "prepared_manifest_sha256": _sha256(config.dataset.prepared_manifest),
            "policy_checkpoint_revision": expected_policy_revision,
            "reference_checkpoint_revision": expected_reference_revision,
            "cycle27_trainable_parameter_hash": expected_cycle27_model_revision,
            "h6_input_manifest_sha256": H7_FROZEN_H6_INPUT_MANIFEST_SHA256,
            "input_lock_sha256": _sha256(input_lock),
            "locked_bank_count": len(locked_input["banks"]),
            "policy_compute_dtype": "float16",
            "reference_compute_dtype": "float16",
            "emitted_log_probability_dtype": "float32",
        },
    }
    write_h7_output(
        output_dir / "token_rows.json",
        {
            "publication_valid": False,
            "profile_cluster_count": 115,
            "decision": decision,
            "token_rows": token_rows,
        },
    )
    write_h7_output(
        output_dir / "bank_summaries.json",
        {
            "publication_valid": False,
            "profile_cluster_count": 115,
            "decision": decision,
            "bank_summaries": summaries,
        },
    )
    write_h7_output(output_dir / "terminal_decision.json", output)
    produced = tuple(
        output_dir / name
        for name in (
            "source_manifest.json",
            "token_rows.json",
            "bank_summaries.json",
            "terminal_decision.json",
        )
    )
    write_h7_output(
        output_dir / "final_manifest.json",
        {
            "publication_valid": False,
            "profile_cluster_count": 115,
            "decision": decision,
            "files": [
                {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "byte_count": path.stat().st_size,
                }
                for path in produced
            ],
        },
    )
    return output


def _failure_file_provenance(path: object) -> dict[str, object]:
    if not isinstance(path, Path):
        return {"path": None, "sha256": None, "error": "not_a_path"}
    if not path.is_file():
        return {"path": str(path), "sha256": None, "error": "unreadable_or_missing"}
    try:
        return {"path": str(path), "sha256": _sha256(path), "error": None}
    except OSError as error:
        return {
            "path": str(path),
            "sha256": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _failure_provenance(
    kwargs: Mapping[str, object], output_dir: Path
) -> dict[str, object]:
    return {
        "source_commit": os.environ.get("H7_SOURCE_COMMIT"),
        "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
        "config": _failure_file_provenance(kwargs.get("config_path")),
        "input_lock": _failure_file_provenance(kwargs.get("input_lock")),
        "bank_root": str(kwargs["bank_root"])
        if isinstance(kwargs.get("bank_root"), Path)
        else None,
        "archive_root": str(kwargs["archive_root"])
        if isinstance(kwargs.get("archive_root"), Path)
        else None,
        "output_dir": str(output_dir),
        "expected_policy_revision": kwargs.get("expected_policy_revision"),
        "expected_reference_revision": kwargs.get("expected_reference_revision"),
        "expected_cycle27_model_revision": kwargs.get(
            "expected_cycle27_model_revision"
        ),
        "expected_config_sha256": kwargs.get("expected_config_sha256"),
    }


def run_h7_cuda(**kwargs: Any) -> dict[str, object]:
    """Claim the reserved run root and persist any runner failure exactly once."""
    output_dir = kwargs.get("output_dir")
    if not isinstance(output_dir, Path):
        raise TypeError("H7 output_dir must be a pathlib.Path")
    run_root = output_dir.parent
    if run_root.exists():
        raise FileExistsError(f"reserved H7 run root already exists: {run_root}")
    try:
        return _run_h7_cuda(**kwargs)
    except BaseException as error:
        persist_h7_failure(
            output_dir / "failure.json",
            phase="runner",
            error=error,
            command=tuple(sys.argv),
            provenance=_failure_provenance(kwargs, output_dir),
        )
        raise
