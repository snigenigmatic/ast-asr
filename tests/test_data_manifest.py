from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

from ast_asr.data import (
    SpeakerKeyMode,
    audit_authoritative_speaker_metadata,
    prepare_svarah_archive,
)
from ast_asr.taxonomy import SVARAH_LANGUAGE_FAMILIES


def _write_silent_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * 160)


def test_prepare_data_uses_official_speaker_ids_and_content_hashes(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_silent_wav(audio_dir / "opaque-file-1.wav")
    _write_silent_wav(audio_dir / "opaque-file-2.wav")
    rows = [
        {
            "audio_filepath": "audio/opaque-file-1.wav",
            "duration": 0.01,
            "text": "one short test",
        },
        {
            "audio_filepath": "audio/opaque-file-2.wav",
            "duration": 0.01,
            "text": "another test",
        },
    ]
    (tmp_path / "svarah_manifest.json").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with (tmp_path / "meta_speaker_stats.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "speaker_id",
                "duration",
                "text",
                "gender",
                "age-group",
                "primary_language",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "speaker_id": "authoritative-A",
                "duration": 0.01,
                "text": "one short test",
                "gender": "Female",
                "age-group": "18-30",
                "primary_language": "Tamil",
            }
        )
        writer.writerow(
            {
                "speaker_id": "authoritative-B",
                "duration": 0.01,
                "text": "another test",
                "gender": "Male",
                "age-group": "30-45",
                "primary_language": "Hindi",
            }
        )

    prepared = prepare_svarah_archive(
        tmp_path,
        dataset_revision="ebbf7777fe771490696a3f7b007097606fa8c924",
        family_mapping=SVARAH_LANGUAGE_FAMILIES,
        expected_speakers=2,
    )

    assert {record.speaker_id for record in prepared.utterances} == {
        "authoritative-A",
        "authoritative-B",
    }
    assert len({record.content_hash for record in prepared.utterances}) == 2
    assert prepared.speaker_profiles[0].reference_word_count > 0
    assert prepared.content_hash == prepared.content_hash
    assert prepared.source_hashes["meta_speaker_stats.csv"]
    assert prepared.to_dict()["source_hashes"]["svarah_manifest.json"]


def test_prepare_data_can_opt_in_to_demographic_profile_clusters(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    for index in range(3):
        path = audio_dir / f"opaque-{index}.wav"
        _write_silent_wav(path)
        rows.append(
            {
                "audio_filepath": f"audio/{path.name}",
                "duration": 0.01,
                "text": f"test words {index}",
            }
        )
    (tmp_path / "svarah_manifest.json").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    fieldnames = [
        "audio_filepath",
        "duration",
        "text",
        "gender",
        "age-group",
        "primary_language",
        "native_place_state",
        "native_place_district",
        "highest_qualification",
        "job_category",
        "occupation_domain",
    ]
    with (tmp_path / "meta_speaker_stats.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    **row,
                    "gender": "Female" if index < 2 else "Male",
                    "age-group": "18-30" if index < 2 else "30-45",
                    "primary_language": "Tamil" if index < 2 else "Hindi",
                    "native_place_state": "Tamil Nadu" if index < 2 else "Delhi",
                    "native_place_district": "Chennai" if index < 2 else "New Delhi",
                    "highest_qualification": "Graduate",
                    "job_category": "Full Time",
                    "occupation_domain": "Technology &amp; Services",
                }
            )

    prepared = prepare_svarah_archive(
        tmp_path,
        dataset_revision="ebbf7777fe771490696a3f7b007097606fa8c924",
        family_mapping=SVARAH_LANGUAGE_FAMILIES,
        expected_speakers=2,
        speaker_key_mode=SpeakerKeyMode.DEMOGRAPHIC_PROFILE,
    )

    assert prepared.identity_mode == "demographic_profile"
    assert len(prepared.speaker_profiles) == 2
    assert len({record.speaker_id for record in prepared.utterances}) == 2
    assert all(record.speaker_id.startswith("profile-") for record in prepared.utterances)
    assert "not authoritative speakers" in prepared.identity_warning
    assert prepared.to_dict()["identity_count"] == 2


def test_authoritative_speaker_metadata_audit_requires_real_ids(tmp_path: Path) -> None:
    metadata = tmp_path / "meta_speaker_stats.csv"
    fieldnames = [
        "speaker_id",
        "duration",
        "text",
        "gender",
        "age-group",
        "primary_language",
    ]
    with metadata.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "speaker_id": "real-1",
                    "duration": 1.0,
                    "text": "one",
                    "gender": "Female",
                    "age-group": "18-30",
                    "primary_language": "Tamil",
                },
                {
                    "speaker_id": "real-1",
                    "duration": 1.0,
                    "text": "two",
                    "gender": "Female",
                    "age-group": "18-30",
                    "primary_language": "Tamil",
                },
                {
                    "speaker_id": "real-2",
                    "duration": 1.0,
                    "text": "three",
                    "gender": "Male",
                    "age-group": "30-45",
                    "primary_language": "Hindi",
                },
            ]
        )

    audit = audit_authoritative_speaker_metadata(metadata, expected_speakers=2)

    assert audit["structurally_valid_for_speaker_folds"] is True
    assert audit["row_count"] == 3
    assert audit["distinct_speaker_count"] == 2
    assert audit["sha256"]


def test_authoritative_speaker_metadata_audit_rejects_profile_only_table(tmp_path: Path) -> None:
    metadata = tmp_path / "meta_speaker_stats.csv"
    metadata.write_text(
        "duration,text,gender,age-group,primary_language\n"
        "1.0,one,Female,18-30,Tamil\n",
        encoding="utf-8",
    )

    audit = audit_authoritative_speaker_metadata(metadata, expected_speakers=117)

    assert audit["structurally_valid_for_speaker_folds"] is False
    assert audit["missing_required_columns"] == ["speaker_id"]
