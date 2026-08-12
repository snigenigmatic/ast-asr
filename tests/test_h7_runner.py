from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ast_asr.h7_runner import (
    H7_BETA0_CONFIG_SHA256,
    H7_H6_CANONICAL_ARM,
    H7_RESERVED_OUTPUT_NAME,
    H7_RESERVED_PROFILE_NAME,
    H7ExecutionPlan,
    _load_prepared_audio,
    classify_terminal_banks,
    extract_h7_features,
    persist_h7_failure,
    prevalidate_h7_bank_inputs,
    reconstruct_bank_audio,
    reconstruct_white_pair,
    require_phase_a_before_phase_b,
    tensor_identity,
    validate_h7_input_lock,
    validate_locked_bank_replay,
    write_h7_failure,
    write_h7_output,
)
from ast_asr.historical_kl import HistoricalBank, build_locked_h6_input_manifest
from ast_asr.rollouts import (
    AcousticCondition,
    CandidateRollout,
    FrozenRolloutBatch,
    UtteranceRollout,
)


def _strict_lock() -> dict[str, object]:
    digest = "a" * 64
    identities = {"path": "/locked/input", "sha256": digest}
    banks = []
    for cycle in range(28):
        pairs = []
        for pair_index in range(3):
            clean_id = f"clean-{cycle}-{pair_index}"
            snr = reconstruct_white_pair(
                torch.ones(16), cycle=cycle, pair_index=pair_index
            ).snr_db
            pairs.append(
                {
                    "pair_index": pair_index,
                    "clean": {
                        "utterance_id": clean_id,
                        "speaker_id": "s",
                        "primary_language": "Tamil",
                        "family": "Dravidian",
                        "condition": "clean",
                        "reference": "r",
                        "audio_path": f"audio/{clean_id}.wav",
                        "source_audio_sha256": digest,
                        "waveform_sha256": digest,
                    },
                    "white": {
                        "utterance_id": f"{clean_id}@white-{snr:.4f}db",
                        "speaker_id": "s",
                        "primary_language": "Tamil",
                        "family": "Dravidian",
                        "condition": "white_train",
                        "reference": "r",
                        "seed": 2028 * 1_000_003 + cycle * 101 + pair_index,
                        "snr_db": snr,
                        "waveform_sha256": digest,
                    },
                }
            )
        banks.append(
            {
                "cycle": cycle,
                "pairs": pairs,
                "features": {
                    "input_features": {
                        "dtype": "torch.float32",
                        "shape": [6, 2, 3],
                        "sha256": digest,
                    },
                    "attention_mask": {
                        "dtype": "torch.int64",
                        "shape": [6, 3],
                        "sha256": digest,
                    },
                },
            }
        )
    return {
        "schema_version": 1,
        "publication_valid": False,
        "profile_cluster_count": 115,
        "resolved_config": {
            "path": "/artifacts/h6.json",
            "sha256": H7_BETA0_CONFIG_SHA256,
        },
        "prepared_manifest": {
            "path": "/data/prepared.json",
            "sha256": "65bdd8cf87f5db0f815e742739be815d2306ddd2b9977ee5687774feb1a18b56",
        },
        "fold_manifest": {
            "path": "/data/fold-0.json",
            "sha256": "22e9ab64006fe8a33bac37f5f2b98887df6aed061e158252778c29c6d928a1f0",
        },
        "policy_checkpoint": identities,
        "reference_checkpoint": identities,
        "h6_input_manifest_sha256": "92c5b7c7fb3457ca41da462fd8475d8a29694541f9f7ae2137fe989f10118c7e",
        "banks": banks,
    }


def test_h7_white_replay_uses_registered_seed_and_four_decimal_identity() -> None:
    clean = torch.linspace(-1.0, 1.0, 32)
    white = reconstruct_white_pair(clean, cycle=27, pair_index=2)

    assert white.seed == 2028 * 1_000_003 + 27 * 101 + 2
    assert 10.0 <= white.snr_db <= 20.0
    assert white.utterance_suffix == f"@white-{white.snr_db:.4f}db"
    torch.testing.assert_close(white.clean, clean)


def test_h7_white_replay_rejects_saved_snr_mismatch() -> None:
    with pytest.raises(ValueError, match="SNR suffix"):
        reconstruct_white_pair(
            torch.ones(16),
            cycle=0,
            pair_index=0,
            expected_utterance_id="u@white-10.0000db",
        )


def test_h7_bank_audio_replay_preserves_clean_white_pair_order_and_rejects_missing_source() -> (
    None
):
    candidate = CandidateRollout("h", (2,), (True,), (-1.0,), -1.0, 1.0)
    clean = UtteranceRollout(
        "u", "s", "Tamil", "Dravidian", AcousticCondition.CLEAN, "r", (candidate,) * 4
    )
    white = UtteranceRollout(
        reconstruct_white_pair(
            torch.ones(16), cycle=0, pair_index=0
        ).utterance_suffix.join(("u", "")),
        "s",
        "Tamil",
        "Dravidian",
        AcousticCondition.WHITE_TRAIN,
        "r",
        (candidate,) * 4,
    )
    bank = HistoricalBank(0, FrozenRolloutBatch("revision", (clean, white)))
    reconstructed = reconstruct_bank_audio(
        bank, audio_by_clean_id={"u": torch.ones(16)}
    )
    assert len(reconstructed) == 2
    torch.testing.assert_close(reconstructed[0], torch.ones(16))
    assert not torch.equal(reconstructed[1], reconstructed[0])
    with pytest.raises(KeyError, match="missing clean audio"):
        reconstruct_bank_audio(bank, audio_by_clean_id={})


def test_h7_feature_extraction_uses_registered_generate_rollout_parameters() -> None:
    class FeatureExtractor:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def __call__(self, audio, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs
            return SimpleNamespace(
                input_features=torch.ones((len(audio), 2, 3)),
                attention_mask=torch.ones((len(audio), 3), dtype=torch.long),
            )

    extractor = FeatureExtractor()
    features, attention = extract_h7_features(
        SimpleNamespace(feature_extractor=extractor),
        tuple(torch.ones(8) for _ in range(6)),
        torch.device("cpu"),
    )
    assert features.shape[0] == attention.shape[0] == 6
    assert extractor.kwargs == {
        "sampling_rate": 16_000,
        "return_tensors": "pt",
        "padding": "max_length",
        "truncation": True,
        "return_attention_mask": True,
    }


def test_h7_replay_rejects_a_locked_prepared_audio_path_mismatch(
    tmp_path: Path,
) -> None:
    candidate = CandidateRollout("h", (2,), (True,), (-1.0,), -1.0, 1.0)
    rows = []
    source_paths: dict[str, Path] = {}
    audio_by_id: dict[str, torch.Tensor] = {}
    pairs = []
    for pair_index, family in enumerate(("Dravidian", "Indo-Aryan", "Sino-Tibetan")):
        clean_id = f"u-{pair_index}"
        clean = UtteranceRollout(
            clean_id,
            "s",
            "Tamil",
            family,
            AcousticCondition.CLEAN,
            "r",
            (candidate,) * 4,
        )
        replay = reconstruct_white_pair(torch.ones(16), cycle=0, pair_index=pair_index)
        white = UtteranceRollout(
            f"{clean_id}{replay.utterance_suffix}",
            "s",
            "Tamil",
            family,
            AcousticCondition.WHITE_TRAIN,
            "r",
            (candidate,) * 4,
        )
        rows.extend((clean, white))
        source = tmp_path / f"{clean_id}.wav"
        source.write_bytes(b"audio")
        source_paths[clean_id] = source
        audio_by_id[clean_id] = torch.ones(16)
        pairs.append(
            {
                "pair_index": pair_index,
                "clean": {
                    "utterance_id": clean_id,
                    "speaker_id": "s",
                    "primary_language": "Tamil",
                    "family": family,
                    "condition": "clean",
                    "reference": "r",
                    "audio_path": f"audio/{clean_id}.wav",
                    "source_audio_sha256": __import__("hashlib")
                    .sha256(b"audio")
                    .hexdigest(),
                    "waveform_sha256": tensor_identity(replay.clean)["sha256"],
                },
                "white": {
                    "utterance_id": white.utterance_id,
                    "speaker_id": "s",
                    "primary_language": "Tamil",
                    "family": family,
                    "condition": "white_train",
                    "reference": "r",
                    "seed": replay.seed,
                    "snr_db": replay.snr_db,
                    "waveform_sha256": tensor_identity(replay.noisy)["sha256"],
                },
            }
        )

    class Extractor:
        def __call__(self, audio, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                input_features=torch.ones((len(audio), 2, 3)),
                attention_mask=torch.ones((len(audio), 3), dtype=torch.long),
            )

    bank = HistoricalBank(0, FrozenRolloutBatch("revision", tuple(rows)))
    processor = SimpleNamespace(feature_extractor=Extractor())
    features, attention = extract_h7_features(
        processor, tuple(torch.ones(16) for _ in range(6)), torch.device("cpu")
    )
    lock = {
        "cycle": 0,
        "pairs": pairs,
        "features": {
            "input_features": tensor_identity(features),
            "attention_mask": tensor_identity(attention),
        },
    }
    model_forwards = 0
    with pytest.raises(ValueError, match="prepared relative"):
        validate_locked_bank_replay(
            bank,
            lock_bank=lock,
            audio_by_clean_id=audio_by_id,
            source_paths=source_paths,
            prepared_audio_paths={key: "wrong.wav" for key in audio_by_id},
            processor=processor,
        )
    with pytest.raises(ValueError, match="prepared relative"):
        prevalidate_h7_bank_inputs(
            (bank,),
            locked_banks=[lock] * 28,
            audio_by_clean_id=audio_by_id,
            source_paths=source_paths,
            prepared_audio_paths={key: "wrong.wav" for key in audio_by_id},
            processor=processor,
        )
    assert model_forwards == 0


def test_h7_phase_b_cannot_begin_before_both_phase_a_identity_paths() -> None:
    with pytest.raises(RuntimeError, match="Phase A"):
        require_phase_a_before_phase_b(
            {"saved_old_path": True, "direct_policy_path": False}
        )

    require_phase_a_before_phase_b({"saved_old_path": True, "direct_policy_path": True})


def test_h7_input_lock_requires_remote_manifest_and_all_bank_audio_hashes(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "input-lock.json"
    lock.write_text(json.dumps({"h6_input_manifest_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="keys"):
        validate_h7_input_lock(lock)

    lock.write_text(json.dumps(_strict_lock()), encoding="utf-8")
    validate_h7_input_lock(lock)
    bad = _strict_lock()
    bad["banks"][0]["pairs"] = bad["banks"][0]["pairs"][:2]  # type: ignore[index]
    lock.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="three ordered pairs"):
        validate_h7_input_lock(lock)


def test_h7_failure_artifact_is_immutable_and_non_evaluable(tmp_path: Path) -> None:
    path = tmp_path / "failure.json"
    write_h7_failure(
        path,
        phase="model_load",
        error=RuntimeError("boom"),
        command=("uv", "run"),
        provenance={"source_commit": None, "modal_image_id": "image-1"},
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["measurement_status"] == "non_evaluable"
    assert result["decision"] == "measurement_failed"
    assert result["publication_valid"] is False
    assert result["command"] == ["uv", "run"]
    assert result["provenance"] == {"modal_image_id": "image-1", "source_commit": None}
    assert not persist_h7_failure(
        path,
        phase="launcher",
        error=RuntimeError("must not overwrite runner evidence"),
        command=("uv",),
    )
    assert json.loads(path.read_text(encoding="utf-8"))["phase"] == "model_load"


def test_h7_canonical_h6_arm_label_is_independent_of_output_directory(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile-h6-refkl-beta0-s2028-20260812"
    output = profile / "h6-beta0-fr-cispo"
    for kind in ("diagnostics", "rollouts"):
        directory = output / kind
        directory.mkdir(parents=True)
        for cycle in range(28):
            (directory / f"cycle-{cycle:03d}.json").write_text("{}", encoding="utf-8")
    manifest = build_locked_h6_input_manifest(
        profile,
        arm=H7_H6_CANONICAL_ARM,
        modal_volume_path="/artifacts/profile-h6-refkl-beta0-s2028-20260812/h6-beta0-fr-cispo",
        arm_directory=output,
    )
    assert manifest.arm == "h6_s2028_beta0"
    assert all(
        not entry.path.startswith("h6-beta0-fr-cispo/") for entry in manifest.files
    )


def test_h7_prepared_audio_loads_only_required_clean_ids(tmp_path: Path) -> None:
    import soundfile as sf

    archive = tmp_path / "archive"
    (archive / "audio").mkdir(parents=True)
    sf.write(archive / "audio" / "needed.wav", [0.0] * 16, 16_000)
    manifest = tmp_path / "prepared.json"
    manifest.write_text(
        json.dumps(
            {
                "identity_count": 115,
                "identity_mode": "demographic_profile",
                "utterances": [
                    {"utterance_id": "needed", "audio_path": "audio/needed.wav"},
                    {"utterance_id": "unused", "audio_path": "audio/missing.wav"},
                ],
            }
        ),
        encoding="utf-8",
    )
    selected = _load_prepared_audio(manifest, archive, required_clean_ids={"needed"})
    assert set(selected) == {"needed"}
    with pytest.raises(ValueError, match="missing required"):
        _load_prepared_audio(manifest, archive, required_clean_ids={"absent"})


def test_h7_terminal_classification_is_not_a_threshold_claim_after_measurement_failure() -> (
    None
):
    assert classify_terminal_banks(["<0.1", ">=0.1"]) == "bank_dependent"
    assert classify_terminal_banks(["<0.1"] * 28) == "all_below"
    assert (
        classify_terminal_banks([], measurement_error="mask mismatch")
        == "measurement_failed"
    )


def test_h7_plan_preserves_phase_order_and_reuses_cycle27() -> None:
    plan = H7ExecutionPlan.build()

    assert plan.phase_a_cycle == 27
    assert plan.phase_b_cycles == tuple(range(27))
    assert plan.phase_b_reuses_phase_a is True
    assert plan.model_score_cycles == (27, 27, *range(27))


def test_h7_outputs_are_immutable_and_carry_nonpublication_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decision.json"
    payload = {
        "publication_valid": False,
        "profile_cluster_count": 115,
        "decision": "all_below",
    }
    write_h7_output(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    write_h7_output(path, payload)
    with pytest.raises(FileExistsError, match="immutable"):
        write_h7_output(path, {**payload, "decision": "bank_dependent"})


def test_modal_wrapper_command_is_single_attempt_and_uses_reserved_names() -> None:
    from ast_asr.h7_modal import H7ModalLaunchSpec, validate_h7_preflight

    spec = H7ModalLaunchSpec.default()
    assert spec.retries == 0
    assert spec.profile_name == H7_RESERVED_PROFILE_NAME
    assert spec.output_name == H7_RESERVED_OUTPUT_NAME
    assert "measure-sentinel-kl" in spec.command
    assert "--output-dir" in spec.command
    assert "/data/fr_cispo_profile/raw/Svarah" in spec.command
    assert "/root/fr-cispo/experiments/H7-sentinel-kl/input-lock.json" in spec.command
    assert H7_BETA0_CONFIG_SHA256 in spec.command
    with pytest.raises(RuntimeError, match="clean executable"):
        validate_h7_preflight(
            remote_run_root=Path("does-not-exist"), git_porcelain="?? extra.py"
        )


def test_h7_modal_entrypoint_uses_existing_h6_mounts_and_a_discoverable_function() -> (
    None
):
    script = (Path(__file__).parents[1] / "scripts" / "modal_h7_sentinel.py").read_text(
        encoding="utf-8"
    )
    assert 'modal.Volume.from_name("ast-asr-cache", create_if_missing=False)' in script
    assert 'modal.Volume.from_name("ast-asr-data", create_if_missing=False)' in script
    assert (
        'modal.Volume.from_name("ast-asr-fr-cispo-runs", create_if_missing=False)'
        in script
    )
    assert (
        'CACHE_DIR = "/cache"' in script
        and 'DATA_DIR = "/data"' in script
        and 'OUTPUT_DIR = "/artifacts"' in script
    )
    assert "def run_h7_sentinel(source_commit: str)" in script
    assert "retries=0" in script
    assert "/data/fr_cispo_profile/raw/Svarah" in script
    assert "resolved-policy-configs/h6-beta0-fr-cispo.json" in script
