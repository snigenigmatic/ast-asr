"""Offline H7 input-lock candidate construction; it never authorizes a launch."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .artifacts import write_immutable_json
from .h7_runner import (
    H7_BETA0_CONFIG_SHA256,
    H7_FOLD_MANIFEST_SHA256,
    H7_FOLD_REMOTE_PATH,
    H7_H6_ARM_REMOTE_PATH,
    H7_H6_CANONICAL_ARM,
    H7_POLICY_DIRECTORY_HASH,
    H7_POLICY_REMOTE_PATH,
    H7_PREPARED_MANIFEST_SHA256,
    H7_PREPARED_REMOTE_PATH,
    H7_REFERENCE_DIRECTORY_HASH,
    H7_REFERENCE_REMOTE_PATH,
    H7_RESOLVED_CONFIG_REMOTE_PATH,
    _load_prepared_audio,
    _sha256,
    build_h7_input_lock_candidate,
    validate_h7_input_lock,
    validate_h7_input_lock_payload,
)
from .historical_kl import (
    assert_locked_manifest_digest,
    build_locked_h6_input_manifest,
    load_historical_banks,
)
from .modeling import directory_content_hash, load_saved_processor
from .rollouts import AcousticCondition


def build_h7_input_lock_from_mirror(
    *,
    resolved_config: Path,
    resolved_config_remote_path: str,
    bank_root: Path,
    h6_arm_remote_path: str,
    archive_root: Path,
    prepared_manifest: Path,
    prepared_remote_path: str,
    fold_manifest: Path,
    fold_remote_path: str,
    policy_remote_path: str,
    policy_directory_hash: str,
    reference_remote_path: str,
    reference_directory_hash: str,
    reference_processor_checkpoint: Path,
    output: Path,
) -> dict[str, object]:
    """Build a deterministic candidate from a local mirror after frozen checks.

    Paths in the candidate are deliberately the frozen Linux paths supplied by
    the caller; hashes are always computed from the local mirror bytes.
    """
    frozen_paths = {
        "resolved config": (
            resolved_config_remote_path,
            H7_RESOLVED_CONFIG_REMOTE_PATH,
        ),
        "H6 arm": (h6_arm_remote_path, H7_H6_ARM_REMOTE_PATH),
        "prepared manifest": (prepared_remote_path, H7_PREPARED_REMOTE_PATH),
        "fold manifest": (fold_remote_path, H7_FOLD_REMOTE_PATH),
        "policy checkpoint": (policy_remote_path, H7_POLICY_REMOTE_PATH),
        "reference checkpoint": (reference_remote_path, H7_REFERENCE_REMOTE_PATH),
    }
    for label, (provided, frozen) in frozen_paths.items():
        if provided != frozen:
            raise ValueError(f"{label} remote path differs from the frozen H7 contract")
    if policy_directory_hash != H7_POLICY_DIRECTORY_HASH:
        raise ValueError("policy checkpoint hash differs from the frozen H7 contract")
    if reference_directory_hash != H7_REFERENCE_DIRECTORY_HASH:
        raise ValueError(
            "reference checkpoint hash differs from the frozen H7 contract"
        )
    if (
        directory_content_hash(reference_processor_checkpoint)
        != H7_REFERENCE_DIRECTORY_HASH
    ):
        raise ValueError(
            "reference processor checkpoint mirror hash differs from frozen reference"
        )
    expected = (
        (resolved_config, H7_BETA0_CONFIG_SHA256, "resolved config"),
        (prepared_manifest, H7_PREPARED_MANIFEST_SHA256, "prepared manifest"),
        (fold_manifest, H7_FOLD_MANIFEST_SHA256, "fold manifest"),
    )
    for path, digest, label in expected:
        if _sha256(path) != digest:
            raise ValueError(
                f"{label} local mirror hash differs from the frozen contract"
            )
    banks = load_historical_banks(bank_root)
    arm_root = bank_root.parent
    assert_locked_manifest_digest(
        build_locked_h6_input_manifest(
            arm_root.parent,
            arm=H7_H6_CANONICAL_ARM,
            modal_volume_path=h6_arm_remote_path,
            arm_directory=arm_root,
        )
    )
    required_clean_ids = {
        utterance.utterance_id
        for bank in banks.banks
        for utterance in bank.frozen.utterances
        if utterance.condition is AcousticCondition.CLEAN
    }
    prepared_raw = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    records = {
        str(record["utterance_id"]): record
        for record in prepared_raw["utterances"]
        if str(record["utterance_id"]) in required_clean_ids
    }
    prepared_audio = _load_prepared_audio(
        prepared_manifest, archive_root, required_clean_ids=required_clean_ids
    )
    processor = load_saved_processor(reference_processor_checkpoint)
    candidate = build_h7_input_lock_candidate(
        banks=banks.banks,
        audio_by_clean_id={
            identifier: value[0] for identifier, value in prepared_audio.items()
        },
        source_paths={
            identifier: value[1] for identifier, value in prepared_audio.items()
        },
        prepared_records=records,
        processor=processor,
        identities={
            "resolved_config": {
                "path": resolved_config_remote_path,
                "sha256": _sha256(resolved_config),
            },
            "prepared_manifest": {
                "path": prepared_remote_path,
                "sha256": _sha256(prepared_manifest),
            },
            "fold_manifest": {
                "path": fold_remote_path,
                "sha256": _sha256(fold_manifest),
            },
            "policy_checkpoint": {
                "path": policy_remote_path,
                "sha256": policy_directory_hash,
            },
            "reference_checkpoint": {
                "path": reference_remote_path,
                "sha256": reference_directory_hash,
            },
        },
    )
    validate_h7_input_lock_payload(candidate)
    write_immutable_json(output, candidate)
    validate_h7_input_lock(output)
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--resolved-config-remote-path", required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--h6-arm-remote-path", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-remote-path", required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--fold-remote-path", required=True)
    parser.add_argument("--policy-remote-path", required=True)
    parser.add_argument("--policy-directory-hash", required=True)
    parser.add_argument("--reference-remote-path", required=True)
    parser.add_argument("--reference-directory-hash", required=True)
    parser.add_argument("--reference-processor-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    candidate = build_h7_input_lock_from_mirror(
        resolved_config=args.resolved_config,
        resolved_config_remote_path=args.resolved_config_remote_path,
        bank_root=args.bank_root,
        h6_arm_remote_path=args.h6_arm_remote_path,
        archive_root=args.archive_root,
        prepared_manifest=args.prepared_manifest,
        prepared_remote_path=args.prepared_remote_path,
        fold_manifest=args.fold_manifest,
        fold_remote_path=args.fold_remote_path,
        policy_remote_path=args.policy_remote_path,
        policy_directory_hash=args.policy_directory_hash,
        reference_remote_path=args.reference_remote_path,
        reference_directory_hash=args.reference_directory_hash,
        reference_processor_checkpoint=args.reference_processor_checkpoint,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "candidate_written": str(args.output),
                "publication_valid": candidate["publication_valid"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
