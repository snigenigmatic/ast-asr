"""Materialize the SPIRE-SIES validation split on Modal.

Runs entirely on Modal into a dedicated Modal volume. The corpus is ~29 GB and
must never be downloaded to a local machine.

Registered contract: experiments/SPIRE-crosscorpus/protocol.md

Audit first (cheap, metadata only, writes no audio):

    $env:PYTHONIOENCODING='utf-8'
    uvx modal run scripts/modal_spire_prepare.py::audit_spire_val \
        --run-name spire-val-audit-20260812

Then materialize:

    uvx modal run --detach scripts/modal_spire_prepare.py::prepare_spire_val \
        --run-name spire-val-20260812
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("ast-asr-spire-crosscorpus")

# A dedicated volume. The frozen Svarah/H7 inputs live in ast-asr-data and are
# deliberately not touched by this workstream.
spire_volume = modal.Volume.from_name("ast-asr-spire", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

SPIRE_DIR = "/spire"
MODULE_DIR = "/root/spire"
REPO_ID = "VectorSigma389/spire-sies"
SPLITS_FILE = "splits.json"

# The 17 per-language shards this corpus ships.
LANGUAGES: tuple[str, ...] = (
    "bengali",
    "dogri",
    "gujarati",
    "hindi",
    "kannada",
    "kashmiri",
    "konkani",
    "maithili",
    "malayalam",
    "marathi",
    "nepali",
    "odia",
    "punjabi",
    "sindhi",
    "tamil",
    "telugu",
    "urdu",
)

prep_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .pip_install(
        "pyarrow==21.0.0",
        "numpy>=2.0.0",
        "soundfile>=0.13.1",
        "soxr>=0.5.0",
        "huggingface_hub>=0.35",
    )
    .add_local_file(
        "scripts/spire_crosscorpus.py",
        remote_path=f"{MODULE_DIR}/spire_crosscorpus.py",
        copy=True,
    )
)


def _write_once(path: Path, value: object) -> None:
    """Never silently replace an evidence artifact."""
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"refusing to overwrite Modal artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _load_contract():
    """Import the shared pure logic inside the container."""
    import sys

    if MODULE_DIR not in sys.path:
        sys.path.insert(0, MODULE_DIR)
    import spire_crosscorpus

    return spire_crosscorpus


def _validated_val_speakers(token: str, contract):
    """Download and validate the corpus split file."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            REPO_ID,
            SPLITS_FILE,
            repo_type="dataset",
            token=token,
        )
    )
    splits = json.loads(path.read_text(encoding="utf-8"))
    return contract.validate_splits(splits), path


def _scan(
    *,
    audit_only: bool,
    run_name: str,
) -> dict[str, object]:
    """Scan every shard, accept validation rows, and optionally write audio."""
    import hashlib
    import io
    import os
    import shutil
    import tempfile
    from collections import defaultdict

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import soxr
    from huggingface_hub import hf_hub_download

    contract = _load_contract()
    token = os.environ["HF_TOKEN"]
    val_speakers, splits_path = _validated_val_speakers(token, contract)

    root = Path(SPIRE_DIR) / "val"
    audio_root = root / "wav"
    if not audit_only:
        audio_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    per_language: dict[str, int] = defaultdict(int)
    per_family_seconds: dict[str, float] = defaultdict(float)
    observed_speakers: set[str] = set()
    scanned_rows = 0
    skipped_rows = 0

    for language in LANGUAGES:
        staging = Path(tempfile.mkdtemp(prefix=f"spire-{language}-"))
        try:
            shard = Path(
                hf_hub_download(
                    REPO_ID,
                    f"data/{language}.parquet",
                    repo_type="dataset",
                    token=token,
                    local_dir=str(staging),
                )
            )
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    scanned_rows += 1
                    utterance = contract.accept_row(row, val_speakers)
                    if utterance is None:
                        skipped_rows += 1
                        continue

                    relative = f"wav/{utterance.uid}.wav"
                    if not audit_only:
                        waveform, sample_rate = sf.read(
                            io.BytesIO(row["audio_bytes"]),
                            dtype="float32",
                            always_2d=False,
                        )
                        waveform = np.asarray(waveform, dtype=np.float32)
                        if waveform.ndim > 1:
                            waveform = waveform.mean(axis=1, dtype=np.float32)
                        if sample_rate != contract.TARGET_SAMPLE_RATE:
                            waveform = soxr.resample(
                                waveform,
                                sample_rate,
                                contract.TARGET_SAMPLE_RATE,
                                quality="HQ",
                            )
                        destination = root / relative
                        if destination.exists():
                            raise FileExistsError(
                                f"refusing to overwrite audio: {destination}"
                            )
                        sf.write(
                            destination,
                            np.asarray(waveform, dtype=np.float32),
                            contract.TARGET_SAMPLE_RATE,
                            subtype="PCM_16",
                        )

                    manifest.append(
                        {
                            "uid": utterance.uid,
                            "speaker_id": utterance.speaker_id,
                            "accent": utterance.language,
                            "language_family": utterance.family,
                            "gender": utterance.gender,
                            "duration": f"{utterance.duration:.6f}",
                            "source_sample_rate": int(row["src_sr"]),
                            "reference": str(row["reference"]).strip(),
                            "path": relative,
                        }
                    )
                    per_language[utterance.language] += 1
                    per_family_seconds[utterance.family] += utterance.duration
                    observed_speakers.add(utterance.speaker_id)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    if not manifest:
        raise RuntimeError("no validation utterances were accepted")
    stray = observed_speakers - set(val_speakers)
    if stray:
        raise RuntimeError(
            f"accepted rows from non-validation speakers: {sorted(stray)}"
        )

    manifest.sort(key=lambda record: str(record["uid"]))
    report: dict[str, object] = {
        "artifact_kind": "spire_sies_validation_preparation",
        "mode": "audit_only" if audit_only else "materialized",
        "repo_id": REPO_ID,
        "splits_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        "speaker_identity_mode": "corpus_provided_speaker",
        "speaker_disjoint_split": True,
        "identity_note": (
            "SPIRE-SIES ships real speaker identifiers and a speaker-disjoint "
            "split, unlike the Svarah folds which use demographic profile "
            "clusters. Grouping here is a genuine speaker."
        ),
        "declared_val_speakers": len(val_speakers),
        "observed_val_speakers": len(observed_speakers),
        "scanned_rows": scanned_rows,
        "skipped_rows": skipped_rows,
        "accepted_utterances": len(manifest),
        "utterances_by_language": dict(sorted(per_language.items())),
        "hours_by_family": {
            family: round(seconds / 3600.0, 4)
            for family, seconds in sorted(per_family_seconds.items())
        },
        "total_hours": round(sum(per_family_seconds.values()) / 3600.0, 4),
        "families": sorted(per_family_seconds),
        "target_sample_rate": contract.TARGET_SAMPLE_RATE,
        "duration_filter_seconds": [
            contract.MINIMUM_DURATION_SECONDS,
            contract.MAXIMUM_DURATION_SECONDS,
        ],
        "corpus_role": "evaluation_only_never_trained_on",
    }

    if not audit_only:
        import csv

        manifest_path = root / "manifest.csv"
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite manifest: {manifest_path}")
        fields = list(manifest[0])
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manifest)
        report["manifest_path"] = str(manifest_path)
        report["manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        _write_once(root / "preparation-report.json", report)

    _write_once(Path(SPIRE_DIR) / "runs" / run_name / "preparation.json", report)
    spire_volume.commit()
    return report


@app.function(
    image=prep_image,
    cpu=4.0,
    memory=16384,
    timeout=2 * 60 * 60,
    retries=0,
    volumes={SPIRE_DIR: spire_volume},
    secrets=[hf_secret],
)
def audit_spire_val(run_name: str) -> dict[str, object]:
    """Verify the split, taxonomy, and counts without writing any audio."""
    return _scan(audit_only=True, run_name=run_name)


@app.function(
    image=prep_image,
    cpu=8.0,
    memory=32768,
    timeout=8 * 60 * 60,
    retries=0,
    volumes={SPIRE_DIR: spire_volume},
    secrets=[hf_secret],
)
def prepare_spire_val(run_name: str) -> dict[str, object]:
    """Materialize the 198-speaker validation split as 16 kHz mono WAV."""
    return _scan(audit_only=False, run_name=run_name)
