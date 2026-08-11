"""Authoritative Svarah archive preparation and immutable data manifests."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from .folds import SpeakerProfile
from .taxonomy import TAXONOMY_RECONCILIATION


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


class SpeakerKeyMode(StrEnum):
    AUTHORITATIVE = "authoritative"
    DEMOGRAPHIC_PROFILE = "demographic_profile"


DEMOGRAPHIC_PROFILE_FIELDS = (
    "gender",
    "age-group",
    "primary_language",
    "native_place_state",
    "native_place_district",
    "highest_qualification",
    "job_category",
    "occupation_domain",
)

# These are the fields needed to establish a usable speaker-level identity
# table.  The upstream README describes additional demographic fields, but
# preparation deliberately requires only fields it actually consumes.  This
# makes a failed audit about identity evidence rather than an unrelated schema
# extension.
AUTHORITATIVE_SPEAKER_REQUIRED_FIELDS = (
    "speaker_id",
    "duration",
    "text",
    "gender",
    "age-group",
    "primary_language",
)


def _normalize_category(value: str) -> str:
    return _normalize_text(html.unescape(value)).strip()


def _profile_cluster_id(metadata: Mapping[str, str]) -> str:
    values = [_normalize_category(metadata[field]) for field in DEMOGRAPHIC_PROFILE_FIELDS]
    if any(not value for value in values):
        raise ValueError("demographic profile fields cannot be empty")
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"profile-{hashlib.sha256(payload).hexdigest()[:20]}"


@dataclass(frozen=True, slots=True)
class SvarahUtterance:
    utterance_id: str
    speaker_id: str
    audio_path: str
    duration_seconds: float
    reference: str
    primary_language: str
    family: str
    gender: str
    age_group: str
    content_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreparedSvarahDataset:
    dataset_revision: str
    taxonomy_reconciliation: str
    identity_mode: str
    identity_warning: str
    source_hashes: Mapping[str, str]
    utterances: tuple[SvarahUtterance, ...]
    speaker_profiles: tuple[SpeakerProfile, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_revision": self.dataset_revision,
            "taxonomy_reconciliation": self.taxonomy_reconciliation,
            "identity_mode": self.identity_mode,
            "identity_warning": self.identity_warning,
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "identity_count": len(self.speaker_profiles),
            "utterances": [utterance.to_dict() for utterance in self.utterances],
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


def _read_json_lines(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if raw_line.strip():
            try:
                rows.append(json.loads(raw_line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def _content_hash(audio_path: Path, reference: str) -> str:
    digest = hashlib.sha256()
    with audio_path.open("rb") as audio:
        for chunk in iter(lambda: audio.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\0")
    digest.update(reference.encode("utf-8"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_authoritative_speaker_metadata(
    path: Path,
    *,
    expected_speakers: int = 117,
) -> dict[str, object]:
    """Return a structural audit for a candidate Svarah speaker table.

    This deliberately does *not* attest that a file is an official Svarah
    release; provenance must be recorded separately with the source revision
    and file hash.  It does prevent a filename-only storage check from being
    mistaken for evidence that speaker-disjoint folds are possible.
    """
    result: dict[str, object] = {
        "path": str(path),
        "expected_speakers": expected_speakers,
        "exists": path.is_file(),
        "sha256": None,
        "row_count": 0,
        "columns": [],
        "missing_required_columns": list(AUTHORITATIVE_SPEAKER_REQUIRED_FIELDS),
        "empty_speaker_id_rows": 0,
        "distinct_speaker_count": 0,
        "structurally_valid_for_speaker_folds": False,
        "blocked_reasons": [],
    }
    if not path.is_file():
        result["blocked_reasons"] = ["candidate metadata file does not exist"]
        return result

    result["sha256"] = _file_sha256(path)

    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = tuple(reader.fieldnames or ())
        result["columns"] = list(columns)
        missing = sorted(set(AUTHORITATIVE_SPEAKER_REQUIRED_FIELDS) - set(columns))
        result["missing_required_columns"] = missing
        if missing:
            result["blocked_reasons"] = [
                f"missing required columns: {', '.join(missing)}"
            ]
            return result

        speaker_ids: set[str] = set()
        empty_speaker_ids = 0
        row_count = 0
        for row in reader:
            row_count += 1
            speaker_id = (row.get("speaker_id") or "").strip()
            if not speaker_id:
                empty_speaker_ids += 1
            else:
                speaker_ids.add(speaker_id)

    result["row_count"] = row_count
    result["empty_speaker_id_rows"] = empty_speaker_ids
    result["distinct_speaker_count"] = len(speaker_ids)
    blocked_reasons: list[str] = []
    if row_count == 0:
        blocked_reasons.append("metadata table has no rows")
    if empty_speaker_ids:
        blocked_reasons.append(f"{empty_speaker_ids} rows have an empty speaker_id")
    if len(speaker_ids) != expected_speakers:
        blocked_reasons.append(
            f"expected {expected_speakers} distinct speaker_id values; found {len(speaker_ids)}"
        )
    result["blocked_reasons"] = blocked_reasons
    result["structurally_valid_for_speaker_folds"] = not blocked_reasons
    return result


def prepare_svarah_archive(
    archive_root: Path,
    *,
    dataset_revision: str,
    family_mapping: Mapping[str, str],
    expected_speakers: int = 117,
    speaker_key_mode: SpeakerKeyMode | str = SpeakerKeyMode.AUTHORITATIVE,
) -> PreparedSvarahDataset:
    """Prepare Svarah only from its official manifest and speaker metadata.

    The two source files are aligned row by row and their transcript/duration
    fields are cross-checked. No speaker identifier is inferred from filenames.
    """
    mode = SpeakerKeyMode(speaker_key_mode)
    if not dataset_revision or dataset_revision.lower() in {"main", "latest"}:
        raise ValueError("dataset_revision must be an immutable resolved revision")
    root = archive_root.resolve()
    manifest_path = root / "svarah_manifest.json"
    metadata_path = root / "meta_speaker_stats.csv"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "official svarah_manifest.json and meta_speaker_stats.csv are required"
        )

    manifest_rows = _read_json_lines(manifest_path)
    metadata_audit = audit_authoritative_speaker_metadata(
        metadata_path,
        expected_speakers=expected_speakers,
    )
    if mode is SpeakerKeyMode.AUTHORITATIVE and not metadata_audit[
        "structurally_valid_for_speaker_folds"
    ]:
        reasons = "; ".join(str(value) for value in metadata_audit["blocked_reasons"])
        raise ValueError(f"authoritative speaker metadata audit failed: {reasons}")
    with metadata_path.open(newline="", encoding="utf-8-sig") as metadata_file:
        metadata_rows = list(csv.DictReader(metadata_file))
    if len(manifest_rows) != len(metadata_rows):
        raise ValueError("Svarah manifest and speaker metadata row counts differ")

    if mode is SpeakerKeyMode.DEMOGRAPHIC_PROFILE:
        metadata_by_audio: dict[str, dict[str, str]] = {}
        for row in metadata_rows:
            audio_key = Path(row.get("audio_filepath", "")).as_posix()
            if not audio_key:
                raise ValueError("profile metadata requires audio_filepath")
            if audio_key in metadata_by_audio:
                raise ValueError(f"duplicate profile metadata path: {audio_key}")
            metadata_by_audio[audio_key] = row
        try:
            aligned_metadata = [
                metadata_by_audio[Path(str(row.get("audio_filepath", ""))).as_posix()]
                for row in manifest_rows
            ]
        except KeyError as error:
            raise ValueError(f"manifest audio is missing profile metadata: {error.args[0]}") from error
    else:
        aligned_metadata = metadata_rows

    utterances: list[SvarahUtterance] = []
    speaker_attributes: dict[str, tuple[str, str, str, str]] = {}
    speaker_words: dict[str, int] = {}
    for index, (manifest, metadata) in enumerate(
        zip(manifest_rows, aligned_metadata, strict=True)
    ):
        required_metadata = {
            "duration",
            "text",
            "gender",
            "age-group",
            "primary_language",
        }
        if mode is SpeakerKeyMode.AUTHORITATIVE:
            required_metadata.add("speaker_id")
        else:
            required_metadata.update(("audio_filepath", *DEMOGRAPHIC_PROFILE_FIELDS))
        missing = required_metadata - metadata.keys()
        if missing:
            raise ValueError(f"speaker metadata is missing columns: {sorted(missing)}")

        reference = _normalize_text(str(manifest.get("text", "")))
        metadata_text = _normalize_text(metadata["text"])
        if not reference or reference != metadata_text:
            raise ValueError(f"manifest/metadata transcript mismatch at row {index}")
        duration = float(manifest.get("duration", math.nan))
        metadata_duration = float(metadata["duration"])
        if not math.isclose(duration, metadata_duration, rel_tol=1e-4, abs_tol=1e-3):
            raise ValueError(f"manifest/metadata duration mismatch at row {index}")

        speaker_id = (
            metadata["speaker_id"].strip()
            if mode is SpeakerKeyMode.AUTHORITATIVE
            else _profile_cluster_id(metadata)
        )
        language = metadata["primary_language"].strip().title()
        if not speaker_id:
            raise ValueError(f"empty authoritative speaker ID at row {index}")
        if language not in family_mapping:
            raise ValueError(f"primary language has no reconciled family: {language!r}")
        family = family_mapping[language]
        gender = _normalize_category(metadata["gender"]) or "Unknown"
        age_group = _normalize_category(metadata["age-group"]) or "Unknown"

        relative_audio = Path(str(manifest.get("audio_filepath", "")))
        audio_path = (root / relative_audio).resolve()
        try:
            stable_relative_audio = audio_path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("audio path escapes the Svarah archive root") from error
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        digest = _content_hash(audio_path, reference)
        utterances.append(
            SvarahUtterance(
                utterance_id=f"svarah-{digest[:20]}",
                speaker_id=speaker_id,
                audio_path=stable_relative_audio,
                duration_seconds=duration,
                reference=reference,
                primary_language=language,
                family=family,
                gender=gender,
                age_group=age_group,
                content_hash=digest,
            )
        )
        attributes = (language, family, gender, age_group)
        previous = speaker_attributes.setdefault(speaker_id, attributes)
        if previous != attributes:
            raise ValueError(f"inconsistent metadata for speaker {speaker_id!r}")
        speaker_words[speaker_id] = speaker_words.get(speaker_id, 0) + len(reference.split())

    if len({item.utterance_id for item in utterances}) != len(utterances):
        raise ValueError("duplicate utterance content detected")
    if len(speaker_attributes) != expected_speakers:
        raise ValueError(
            f"Svarah must resolve to exactly {expected_speakers} speakers; "
            f"received {len(speaker_attributes)}"
        )

    profiles = tuple(
        SpeakerProfile(
            speaker_id=speaker_id,
            primary_language=attributes[0],
            family=attributes[1],
            gender=attributes[2],
            age_group=attributes[3],
            reference_word_count=speaker_words[speaker_id],
        )
        for speaker_id, attributes in sorted(speaker_attributes.items())
    )
    return PreparedSvarahDataset(
        dataset_revision=dataset_revision,
        taxonomy_reconciliation=TAXONOMY_RECONCILIATION,
        identity_mode=mode.value,
        identity_warning=(
            ""
            if mode is SpeakerKeyMode.AUTHORITATIVE
            else (
                "Demographic profile clusters are conservative development groups, "
                "not authoritative speakers."
            )
        ),
        source_hashes={
            "svarah_manifest.json": _file_sha256(manifest_path),
            "meta_speaker_stats.csv": str(metadata_audit["sha256"]),
        },
        utterances=tuple(sorted(utterances, key=lambda item: item.utterance_id)),
        speaker_profiles=profiles,
    )
