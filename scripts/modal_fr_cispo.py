"""Isolated Modal launcher for cost-bounded FR-CISPO validation.

Run both the storage audit and L4 objective smoke:

    uvx modal run scripts/modal_fr_cispo.py --run-name smoke-20260810

Artifacts are written to the separate ``ast-asr-fr-cispo-runs`` volume.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal

# Do not import ast_asr here. Modal loads this module before the remote image
# has installed the project package; the equivalent pure helpers are exercised
# by tests in src/ast_asr/modal_policy_config.py.

app = modal.App("ast-asr-fr-cispo")
cache_volume = modal.Volume.from_name("ast-asr-cache", create_if_missing=False)
data_volume = modal.Volume.from_name("ast-asr-data", create_if_missing=False)
output_volume = modal.Volume.from_name("ast-asr-fr-cispo-runs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

PROJECT_ROOT = "/root/fr-cispo"
CACHE_DIR = "/cache"
DATA_DIR = "/data"
OUTPUT_DIR = "/artifacts"
SVARAH_REVISION = "ebbf7777fe771490696a3f7b007097606fa8c924"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
MUSAN_MD5 = "0c472d4fc0c5141eca47ad1ffeb2a7df"
PROFILE_SFT_EPOCH1_REVISION = (
    "d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e"
)

CACHE_ENV = {
    "HF_HOME": f"{CACHE_DIR}/hf",
    "HF_HUB_CACHE": f"{CACHE_DIR}/hf/hub",
    "HUGGINGFACE_HUB_CACHE": f"{CACHE_DIR}/hf/hub",
    "HF_DATASETS_CACHE": f"{CACHE_DIR}/ai4bharat___svarah",
    "TRANSFORMERS_CACHE": f"{CACHE_DIR}/hf/hub",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "UV_LINK_MODE": "copy",
}

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("uv>=0.8,<0.9")
    .env(CACHE_ENV)
    .add_local_file(
        "pyproject.toml",
        remote_path=f"{PROJECT_ROOT}/pyproject.toml",
        copy=True,
    )
    .add_local_file("uv.lock", remote_path=f"{PROJECT_ROOT}/uv.lock", copy=True)
    .run_commands(
        f"cd {PROJECT_ROOT} && uv sync --frozen --no-dev --no-install-project"
    )
    .add_local_dir("src", remote_path=f"{PROJECT_ROOT}/src", copy=True)
    .add_local_dir("configs", remote_path=f"{PROJECT_ROOT}/configs", copy=True)
    .run_commands(
        f"cd {PROJECT_ROOT} && uv sync --frozen --no-dev --offline --no-build-isolation"
    )
)

parquet_audit_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "duckdb==1.4.1",
    "pyarrow==21.0.0",
)
data_prep_image = image.pip_install("pyarrow==21.0.0")

VOLUMES = {
    CACHE_DIR: cache_volume,
    DATA_DIR: data_volume,
    OUTPUT_DIR: output_volume,
}


def _write_once(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"refusing to overwrite Modal artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _write_text_once(path: Path, value: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != value:
        raise FileExistsError(f"refusing to overwrite Modal artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_content_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _modal_runtime_identity() -> dict[str, object]:
    """Record the concrete runtime plus the reproducible image declaration."""
    import torch

    return {
        "modal_app": "ast-asr-fr-cispo",
        "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None
        ),
        "image_declaration": {
            "base": "nvidia/cuda:12.8.0-devel-ubuntu22.04",
            "python": "3.12",
            "gpu_request": "L4",
            "uv_requirement": "uv>=0.8,<0.9",
            "uv_lock_sha256": _file_sha256(Path(PROJECT_ROOT) / "uv.lock"),
            "source_directory_sha256": _directory_content_hash(
                Path(PROJECT_ROOT) / "src"
            ),
        },
    }


def _resolve_evaluation_checkpoint_in_project(
    arm: str,
    *,
    checkpoint_run_name: str,
    checkpoint_output_name: str,
    checkpoint_name: str,
) -> str:
    """Reuse package validation through ``uv run`` rather than Modal's importer.

    The Modal wrapper is imported before its remote function executes and that
    interpreter does not have the project package installed. The resolver runs
    inside the already-built project environment, so it keeps the exact arm
    semantics from ``ast_asr.modal_evaluation`` without duplicating them here.
    """
    command = [
        "uv",
        "run",
        "--frozen",
        "python",
        "-m",
        "ast_asr.modal_evaluation",
        "resolve-checkpoint",
        "--arm",
        arm,
        "--output-root",
        OUTPUT_DIR,
        "--checkpoint-run-name",
        checkpoint_run_name,
        "--checkpoint-output-name",
        checkpoint_output_name,
        "--checkpoint-name",
        checkpoint_name,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    resolved = completed.stdout.splitlines()
    if len(resolved) != 1 or not resolved[0]:
        raise RuntimeError(
            "checkpoint resolver must emit exactly one non-empty checkpoint path"
        )
    return resolved[0]


def _derive_policy_config_in_project(
    *,
    immutable_config: Path,
    run_artifact_root: Path,
    output_name: str,
    reference_kl_beta: float,
) -> Path:
    """Create a run config through the tested package helper under ``uv run``."""
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "python",
            "-m",
            "ast_asr.modal_policy_config",
            "derive",
            "--immutable-config",
            str(immutable_config),
            "--run-artifact-root",
            str(run_artifact_root),
            "--output-name",
            output_name,
            "--reference-kl-beta",
            str(reference_kl_beta),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    lines = completed.stdout.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise RuntimeError("policy config helper must emit exactly one path")
    return Path(lines[0])


@app.function(
    image=data_prep_image,
    cpu=4.0,
    memory=16384,
    timeout=60 * 60,
    volumes=VOLUMES,
)
def prepare_profile_cluster_data(run_name: str) -> dict[str, object]:
    """Materialize Svarah and immutable 115-profile development folds."""
    import csv
    import hashlib

    import pyarrow.parquet as pq

    source_csv = Path(DATA_DIR) / "fr_cispo" / "source" / "meta_speaker_stats.csv"
    expected_csv_hash = "e2daa48863581eb41befd1826b7b14cd80e05f3a1bab9b72d08e7814248f1f94"
    if not source_csv.is_file():
        raise FileNotFoundError(f"upload the supplied metadata CSV to {source_csv}")
    csv_bytes = source_csv.read_bytes()
    csv_hash = hashlib.sha256(csv_bytes).hexdigest()
    if csv_hash != expected_csv_hash:
        raise RuntimeError(f"unexpected metadata CSV hash: {csv_hash}")
    with source_csv.open(newline="", encoding="utf-8-sig") as source:
        metadata_rows = list(csv.DictReader(source))
    metadata_paths = {Path(row["audio_filepath"]).name for row in metadata_rows}
    if len(metadata_rows) != 6656 or len(metadata_paths) != 6656:
        raise RuntimeError("metadata CSV must contain 6,656 unique audio paths")

    snapshot = (
        Path(CACHE_DIR)
        / "hf"
        / "hub"
        / "datasets--ai4bharat--Svarah"
        / "snapshots"
        / SVARAH_REVISION
        / "data"
    )
    parquet_files = sorted(snapshot.glob("*.parquet"))
    if len(parquet_files) != 3:
        raise RuntimeError("expected all three pinned Svarah Parquet shards")

    profile_root = Path(DATA_DIR) / "fr_cispo_profile"
    archive_root = profile_root / "raw" / "Svarah"
    audio_root = archive_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    extracted_paths = set()
    for parquet_path in parquet_files:
        parquet = pq.ParquetFile(parquet_path)
        for row_group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(
                row_group,
                columns=["audio_filepath", "duration", "text"],
            )
            for row in table.to_pylist():
                audio = row["audio_filepath"]
                name = Path(audio["path"]).name
                if name not in metadata_paths:
                    raise RuntimeError(f"Parquet audio path is absent from metadata: {name}")
                destination = audio_root / name
                payload = audio["bytes"]
                if destination.exists():
                    if destination.stat().st_size != len(payload):
                        raise RuntimeError(f"partial audio extraction detected: {destination}")
                else:
                    destination.write_bytes(payload)
                extracted_paths.add(name)
                manifest_rows.append(
                    {
                        "audio_filepath": f"audio/{name}",
                        "duration": float(row["duration"]),
                        "text": row["text"],
                    }
                )
    if extracted_paths != metadata_paths:
        raise RuntimeError("Parquet and metadata audio path sets differ")

    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in manifest_rows
    )
    _write_text_once(archive_root / "svarah_manifest.json", manifest_text)
    metadata_destination = archive_root / "meta_speaker_stats.csv"
    if metadata_destination.exists() and metadata_destination.read_bytes() != csv_bytes:
        raise FileExistsError("refusing to replace profile-cluster metadata source")
    metadata_destination.write_bytes(csv_bytes)

    config = json.loads(
        (Path(PROJECT_ROOT) / "configs" / "fr_cispo_tiny.json").read_text(
            encoding="utf-8"
        )
    )
    prepared_root = profile_root / "prepared"
    config["dataset"].update(
        {
            "prepared_manifest": str(prepared_root / "dataset_manifest.json"),
            "fold_directory": str(prepared_root / "folds"),
            "archive_root": str(archive_root),
            "musan_root": str(profile_root / "raw" / "musan"),
        }
    )
    modal_config = profile_root / "configs" / "fr_cispo_tiny.json"
    _write_once(modal_config, config)
    subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "ast-asr",
            "prepare-data",
            "--archive-root",
            str(archive_root),
            "--dataset-revision",
            SVARAH_REVISION,
            "--output-dir",
            str(prepared_root),
            "--fold-seed",
            "2026",
            "--expected-speakers",
            "115",
            "--speaker-key-mode",
            "demographic_profile",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        env=os.environ.copy(),
    )
    manifest_path = prepared_root / "dataset_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    prepared = json.loads(manifest_bytes)
    report: dict[str, object] = {
        "artifact_kind": "svarah_profile_cluster_preparation",
        "research_valid": False,
        "research_invalid_reason": (
            "The 115 identities are demographic profile clusters, not authoritative speakers."
        ),
        "dataset_revision": SVARAH_REVISION,
        "source_csv_sha256": csv_hash,
        "utterances": len(prepared["utterances"]),
        "identity_mode": prepared["identity_mode"],
        "identity_count": prepared["identity_count"],
        "manifest_content_hash": hashlib.sha256(manifest_bytes).hexdigest(),
        "config": str(modal_config),
    }
    _write_once(Path(OUTPUT_DIR) / run_name / "profile-data-preparation.json", report)
    data_volume.commit()
    output_volume.commit()
    return report


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=4 * 60 * 60,
    volumes=VOLUMES,
)
def prepare_musan_speech(run_name: str) -> dict[str, object]:
    """Download verified OpenSLR MUSAN and extract only its speech partition."""
    import hashlib
    import os
    import tarfile
    import urllib.request

    archive = Path(CACHE_DIR) / "musan" / "musan.tar.gz"
    partial = Path(f"{archive}.partial")
    final_root = Path(DATA_DIR) / "fr_cispo_profile" / "raw" / "musan"
    marker = final_root / ".fr_cispo_musan.json"

    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("archive_md5") != MUSAN_MD5:
            raise RuntimeError("existing MUSAN marker does not match the pinned checksum")
        return recorded
    if final_root.exists():
        raise FileExistsError(f"unverified MUSAN target already exists: {final_root}")

    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        if partial.exists():
            raise FileExistsError(f"partial MUSAN download requires inspection: {partial}")
        digest = hashlib.md5(usedforsecurity=False)
        with urllib.request.urlopen(MUSAN_URL, timeout=120) as response:
            expected_size = int(response.headers.get("Content-Length", "0"))
            with partial.open("xb") as destination:
                while chunk := response.read(8 * 1024 * 1024):
                    destination.write(chunk)
                    digest.update(chunk)
        if expected_size and partial.stat().st_size != expected_size:
            partial.unlink()
            raise RuntimeError("MUSAN download size did not match Content-Length")
        if digest.hexdigest() != MUSAN_MD5:
            partial.unlink()
            raise RuntimeError("MUSAN archive checksum mismatch")
        os.replace(partial, archive)
        cache_volume.commit()

    digest = hashlib.md5(usedforsecurity=False)
    with archive.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != MUSAN_MD5:
        raise RuntimeError("cached MUSAN archive checksum mismatch")

    staging = final_root.with_name(f"musan-staging-{MUSAN_MD5[:12]}")
    if staging.exists():
        raise FileExistsError(f"MUSAN staging directory requires inspection: {staging}")
    staging.mkdir(parents=True)
    extracted_files = 0
    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle:
            normalized = member.name.replace("\\", "/").lstrip("./")
            if not normalized.startswith("musan/speech/"):
                continue
            if member.issym() or member.islnk():
                raise RuntimeError("MUSAN speech archive unexpectedly contains links")
            if member.isfile():
                extracted_files += 1
            bundle.extract(member, path=staging, filter="data")
    speech_root = staging / "musan" / "speech"
    wave_files = sum(1 for _ in speech_root.rglob("*.wav"))
    if extracted_files == 0 or wave_files == 0:
        raise RuntimeError("MUSAN speech extraction produced no audio")

    report: dict[str, object] = {
        "artifact_kind": "musan_speech_preparation",
        "archive_md5": MUSAN_MD5,
        "archive_url": MUSAN_URL,
        "extracted_members": extracted_files,
        "speech_wave_files": wave_files,
    }
    _write_once(staging / ".fr_cispo_musan.json", report)
    os.replace(staging, final_root)
    _write_once(Path(OUTPUT_DIR) / run_name / "musan-preparation.json", report)
    data_volume.commit()
    output_volume.commit()
    return report


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=4 * 60 * 60,
    volumes=VOLUMES,
    secrets=[hf_secret],
)
def run_profile_sft_smoke(run_name: str, seed: int = 2026) -> dict[str, object]:
    """Run two real Svarah LoRA optimizer steps and a checkpoint reload."""
    output = Path(OUTPUT_DIR) / run_name / "profile-sft-smoke"
    subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "ast-asr",
            "train-sft",
            "--config",
            f"{DATA_DIR}/fr_cispo_profile/configs/fr_cispo_tiny.json",
            "--fold",
            "0",
            "--seed",
            str(seed),
            "--output-dir",
            str(output),
            "--maximum-epochs",
            "1",
            "--max-train-examples",
            "48",
            "--max-validation-examples",
            "12",
            "--max-optimizer-steps",
            "2",
            "--maximum-new-tokens",
            "32",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        env=os.environ.copy(),
    )
    output_volume.commit()
    return json.loads((output / "run.json").read_text(encoding="utf-8"))


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=8 * 60 * 60,
    volumes=VOLUMES,
    secrets=[hf_secret],
)
def run_profile_sft_development(
    run_name: str,
    seed: int = 2026,
) -> dict[str, object]:
    """Train the complete five-epoch SFT development fold."""
    output = Path(OUTPUT_DIR) / run_name / "profile-sft-development"
    subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "ast-asr",
            "train-sft",
            "--config",
            f"{DATA_DIR}/fr_cispo_profile/configs/fr_cispo_tiny.json",
            "--fold",
            "0",
            "--seed",
            str(seed),
            "--output-dir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        env=os.environ.copy(),
    )
    output_volume.commit()
    return json.loads((output / "run.json").read_text(encoding="utf-8"))


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=[hf_secret],
)
def run_profile_fr_cispo_smoke(
    run_name: str,
    seed: int = 2026,
    sft_run_name: str = "",
    sft_output_name: str = "profile-sft-smoke",
    output_name: str = "profile-fr-cispo-smoke",
    learning_rate: float = 0.00003,
    rollout_cycles: int = 2,
    probe_examples: int = 6,
    maximum_new_tokens: int = 32,
    reference_kl_beta: float = 0.0,
) -> dict[str, object]:
    """Run bounded FR-CISPO cycles over real profile-clustered Svarah.

    H5 must use distinct output names for the paired beta=0 and beta=.04
    commands (for example ``profile-h5-beta0`` and ``profile-h5-beta04``).
    """
    source_run = sft_run_name or run_name
    for artifact_component in (run_name, source_run, sft_output_name, output_name):
        if (
            not artifact_component
            or artifact_component in {".", ".."}
            or "/" in artifact_component
            or "\\" in artifact_component
            or Path(artifact_component).name != artifact_component
        ):
            raise ValueError("artifact output names must be single path components")
    if not isinstance(reference_kl_beta, (int, float)) or not math.isfinite(
        float(reference_kl_beta)
    ):
        raise ValueError("reference_kl_beta must be finite")
    if reference_kl_beta < 0:
        raise ValueError("reference_kl_beta must be nonnegative")
    sft = Path(OUTPUT_DIR) / source_run / sft_output_name / "checkpoint-epoch-1"
    if not sft.is_dir():
        raise FileNotFoundError(f"profile SFT smoke checkpoint is missing: {sft}")
    source_sft_revision = _directory_content_hash(sft)
    if source_sft_revision != PROFILE_SFT_EPOCH1_REVISION:
        raise RuntimeError(
            "profile SFT checkpoint revision differs from the frozen H5 source: "
            f"expected {PROFILE_SFT_EPOCH1_REVISION}, found {source_sft_revision}"
        )
    output = Path(OUTPUT_DIR) / run_name / output_name
    immutable_source_config = (
        Path(DATA_DIR) / "fr_cispo_profile" / "configs" / "fr_cispo_tiny.json"
    )
    resolved_config = _derive_policy_config_in_project(
        immutable_config=immutable_source_config,
        run_artifact_root=Path(OUTPUT_DIR) / run_name,
        output_name=output_name,
        reference_kl_beta=float(reference_kl_beta),
    )
    resolved_raw = json.loads(resolved_config.read_text(encoding="utf-8"))
    prepared_manifest = Path(resolved_raw["dataset"]["prepared_manifest"])
    fold_manifest = Path(resolved_raw["dataset"]["fold_directory"]) / "fold-0.json"
    prepared = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    if (
        prepared.get("identity_mode") != "demographic_profile"
        or prepared.get("identity_count") != 115
    ):
        raise RuntimeError(
            "H5 requires the frozen 115 demographic-profile clusters; "
            "authoritative speaker folds are a separate future protocol"
        )
    command = [
        "uv",
        "run",
        "--frozen",
        "ast-asr",
        "train-policy",
        "--config",
        str(resolved_config),
        "--fold",
        "0",
        "--seed",
        str(seed),
        "--arm",
        "fr-cispo",
        "--sft-checkpoint",
        str(sft),
        "--learning-rate",
        str(learning_rate),
        "--output-dir",
        str(output),
        "--rollout-cycles",
        str(rollout_cycles),
        "--probe-examples",
        str(probe_examples),
        "--maximum-new-tokens",
        str(maximum_new_tokens),
    ]
    launcher = {
        "artifact_kind": "modal_policy_launcher",
        "reference_kl_beta": float(reference_kl_beta),
        "immutable_source_config": str(immutable_source_config),
        "resolved_config": str(resolved_config),
        "immutable_source_config_sha256": _file_sha256(immutable_source_config),
        "resolved_config_sha256": _file_sha256(resolved_config),
        "prepared_manifest_sha256": _file_sha256(prepared_manifest),
        "fold_manifest_sha256": _file_sha256(fold_manifest),
        "source_sft_revision": source_sft_revision,
        "expected_source_sft_revision": PROFILE_SFT_EPOCH1_REVISION,
        "identity_mode": prepared.get("identity_mode"),
        "identity_count": prepared.get("identity_count"),
        "identity_warning": prepared.get("identity_warning"),
        "authoritative_svarah_speakers_expected": 117,
        "publication_valid": False,
        "randomness": {
            "root_seed": seed,
            "rollout_seed_rule": "root_seed * 1000003 + cycle",
            "corruption_seed_rule": (
                "root_seed * 1000003 + cycle * 101 + balanced_batch_index"
            ),
            "realizations": "stored per utterance in rollouts/cycle-*.json",
        },
        "modal_runtime": _modal_runtime_identity(),
        "command": command,
    }
    _write_once(
        Path(OUTPUT_DIR) / run_name / "policy-launches" / f"{output_name}.json",
        launcher,
    )
    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            env=os.environ.copy(),
        )
    except subprocess.CalledProcessError as error:
        # The CLI writes failure.json before raising. Commit it before Modal
        # turns the subprocess failure into a remote exception.
        output_volume.commit()
        failure = output / "failure.json"
        if failure.is_file():
            raise RuntimeError(
                f"policy run stopped by a safety gate; inspect {failure}"
            ) from error
        raise
    output_volume.commit()
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    run["movement"] = json.loads(
        (output / "movement.json").read_text(encoding="utf-8")
    )
    run["reference_kl_beta"] = float(reference_kl_beta)
    run["resolved_config"] = str(resolved_config)
    run["launcher_artifact"] = str(
        Path(OUTPUT_DIR) / run_name / "policy-launches" / f"{output_name}.json"
    )
    return run


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=4 * 60 * 60,
    volumes=VOLUMES,
    secrets=[hf_secret],
)
def run_profile_evaluation(
    run_name: str,
    arm: str,
    checkpoint_run_name: str = "",
    checkpoint_output_name: str = "",
    checkpoint_name: str = "",
) -> dict[str, object]:
    """Evaluate one base or adapter checkpoint on all protocol conditions."""
    checkpoint = _resolve_evaluation_checkpoint_in_project(
        arm,
        checkpoint_run_name=checkpoint_run_name,
        checkpoint_output_name=checkpoint_output_name,
        checkpoint_name=checkpoint_name,
    )
    if checkpoint != "base" and not Path(checkpoint).is_dir():
        raise FileNotFoundError(f"evaluation checkpoint is missing: {checkpoint}")

    output_name = f"evaluation-{arm}"
    output = Path(OUTPUT_DIR) / run_name / output_name
    subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "ast-asr",
            "evaluate-fold",
            "--config",
            f"{DATA_DIR}/fr_cispo_profile/configs/fr_cispo_tiny.json",
            "--fold",
            "0",
            "--arm",
            arm,
            "--checkpoint",
            checkpoint,
            "--output-dir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        env=os.environ.copy(),
    )
    output_volume.commit()
    return {
        "run": json.loads((output / "run.json").read_text(encoding="utf-8")),
        "metrics": json.loads((output / "metrics.json").read_text(encoding="utf-8")),
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=[hf_secret],
)
def diagnose_profile_batch_invariance(run_name: str) -> dict[str, object]:
    """Reproduce SFT solo/batched decoding divergence on the eight-item probe."""
    checkpoint = (
        Path(OUTPUT_DIR)
        / "profile-dev-full-sft-20260810"
        / "profile-sft-development"
        / "checkpoint-epoch-1"
    )
    output = Path(OUTPUT_DIR) / run_name / "batch-invariance.json"
    subprocess.run(
        [
            "uv",
            "run",
            "--frozen",
            "ast-asr",
            "diagnose-invariance",
            "--config",
            f"{DATA_DIR}/fr_cispo_profile/configs/fr_cispo_tiny.json",
            "--checkpoint",
            str(checkpoint),
            "--fold",
            "0",
            "--output",
            str(output),
            "--probe-examples",
            "8",
            "--batch-size",
            "8",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        env=os.environ.copy(),
    )
    output_volume.commit()
    return json.loads(output.read_text(encoding="utf-8"))


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=10 * 60,
    volumes=VOLUMES,
    secrets=[hf_secret],
)
def inspect_svarah_storage(run_name: str) -> dict[str, object]:
    """Record whether cached official files can support speaker-disjoint folds."""
    from ast_asr.data import audit_authoritative_speaker_metadata

    roots = (
        Path(CACHE_DIR) / "hf" / "hub" / "datasets--ai4bharat--Svarah",
        Path(CACHE_DIR) / "ai4bharat___svarah",
        Path(DATA_DIR),
    )
    visible_files: list[str] = []
    metadata_candidates: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rendered = str(path)
            visible_files.append(rendered)
            lowered = path.name.lower()
            if any(token in lowered for token in ("speaker", "meta", "manifest", ".csv")):
                metadata_candidates.append(rendered)

    dataset_info = next(
        (
            path
            for path in Path(CACHE_DIR).rglob("dataset_info.json")
            if "svarah" in str(path).lower()
        ),
        None,
    )
    feature_names: list[str] = []
    if dataset_info is not None:
        raw = json.loads(dataset_info.read_text(encoding="utf-8"))
        features = raw.get("features", {})
        feature_names = sorted(features if isinstance(features, dict) else [])

    official_speaker_csv = [
        path for path in metadata_candidates if Path(path).name == "meta_speaker_stats.csv"
    ]
    speaker_metadata_audits = [
        audit_authoritative_speaker_metadata(Path(path), expected_speakers=117)
        for path in sorted(official_speaker_csv)
    ]
    structurally_valid_metadata = [
        audit
        for audit in speaker_metadata_audits
        if audit["structurally_valid_for_speaker_folds"]
    ]
    report: dict[str, object] = {
        "artifact_kind": "svarah_storage_audit",
        "dataset_revision": SVARAH_REVISION,
        "visible_file_count": len(visible_files),
        "metadata_candidates": sorted(metadata_candidates),
        "dataset_feature_names": feature_names,
        "official_meta_speaker_stats_csv": official_speaker_csv,
        "speaker_metadata_audits": speaker_metadata_audits,
        "speaker_fold_ready": bool(structurally_valid_metadata),
        "required_speaker_count": 117,
        "blocked_reason": None
        if structurally_valid_metadata
        else (
            "No candidate metadata file has the required identity fields and exactly "
            "117 non-empty distinct speaker_id values; filename-derived pseudo-speakers "
            "are forbidden. A structurally valid candidate still requires recorded "
            "official provenance and prepare-data's manifest alignment check."
        ),
    }
    _write_once(Path(OUTPUT_DIR) / run_name / "storage-audit.json", report)
    output_volume.commit()
    return report


@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    timeout=10 * 60,
    volumes=VOLUMES,
    secrets=[hf_secret],
)
def inspect_svarah_history(run_name: str) -> dict[str, object]:
    """Search every accessible official repository revision for speaker metadata."""
    code = """
import json
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
commits = api.list_repo_commits("ai4bharat/Svarah", repo_type="dataset")
rows = []
for commit in commits:
    files = api.list_repo_files(
        "ai4bharat/Svarah",
        repo_type="dataset",
        revision=commit.commit_id,
    )
    candidates = [
        name for name in files
        if any(token in name.lower() for token in ("speaker", "meta", "manifest", ".csv"))
    ]
    rows.append({
        "commit_id": commit.commit_id,
        "title": commit.title,
        "created_at": commit.created_at.isoformat(),
        "metadata_candidates": candidates,
    })
print(json.dumps(rows, sort_keys=True))
"""
    completed = subprocess.run(
        ["uv", "run", "--frozen", "python", "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    revisions = json.loads(completed.stdout)
    official_csv_revisions = [
        row["commit_id"]
        for row in revisions
        if any(
            Path(name).name == "meta_speaker_stats.csv"
            for name in row["metadata_candidates"]
        )
    ]
    report: dict[str, object] = {
        "artifact_kind": "svarah_repository_history_audit",
        "repository": "ai4bharat/Svarah",
        "revision_count": len(revisions),
        "revisions": revisions,
        "official_meta_speaker_stats_revisions": official_csv_revisions,
    }
    _write_once(Path(OUTPUT_DIR) / run_name / "repository-history-audit.json", report)
    output_volume.commit()
    return report


@app.function(
    image=parquet_audit_image,
    cpu=4.0,
    memory=8192,
    timeout=20 * 60,
    volumes=VOLUMES,
)
def inspect_svarah_parquet(run_name: str) -> dict[str, object]:
    """Decompress the cached Parquet metadata and non-audio row fields."""
    import re
    from pathlib import PurePosixPath

    import duckdb
    import pyarrow.parquet as pq

    snapshot = (
        Path(CACHE_DIR)
        / "hf"
        / "hub"
        / "datasets--ai4bharat--Svarah"
        / "snapshots"
        / SVARAH_REVISION
        / "data"
    )
    parquet_files = sorted(snapshot.glob("*.parquet"))
    if len(parquet_files) != 3:
        raise RuntimeError(f"expected three Svarah Parquet shards, found {parquet_files}")

    file_reports = []
    metadata_search_hits: dict[str, list[str]] = {}
    for path in parquet_files:
        parquet = pq.ParquetFile(path)
        metadata = parquet.metadata
        key_values = {
            key.decode("utf-8", errors="replace"): value.decode(
                "utf-8", errors="replace"
            )
            for key, value in (metadata.metadata or {}).items()
        }
        searchable = "\n".join((str(parquet.schema), *key_values.values()))
        hits = sorted(
            set(re.findall(r"(?i).{0,40}(?:speaker|spk|speaker_id).{0,80}", searchable))
        )
        metadata_search_hits[path.name] = hits
        file_reports.append(
            {
                "file": path.name,
                "rows": metadata.num_rows,
                "row_groups": metadata.num_row_groups,
                "serialized_size": metadata.serialized_size,
                "arrow_schema": str(parquet.schema_arrow),
                "physical_schema": str(parquet.schema),
                "metadata_keys": sorted(key_values),
            }
        )

    paths_sql = ", ".join(
        "'" + str(path).replace("'", "''") + "'" for path in parquet_files
    )
    connection = duckdb.connect()
    connection.execute(
        f"CREATE VIEW svarah AS SELECT * FROM read_parquet([{paths_sql}], union_by_name=true)"
    )
    described = [
        {"name": row[0], "type": row[1], "nullable": row[2]}
        for row in connection.execute("DESCRIBE svarah").fetchall()
    ]
    row_count = connection.execute("SELECT count(*) FROM svarah").fetchone()[0]
    audio_paths = [
        row[0]
        for row in connection.execute(
            "SELECT audio_filepath.path FROM svarah ORDER BY audio_filepath.path"
        ).fetchall()
    ]
    if any(path is None for path in audio_paths):
        raise RuntimeError("Svarah Parquet contains a null audio path")

    demographic_columns = (
        "gender",
        '"age-group"',
        "primary_language",
        "native_place_state",
        "native_place_district",
        "highest_qualification",
        "job_category",
        "occupation_domain",
    )
    demographic_projection = ", ".join(demographic_columns)
    demographic_group_count = connection.execute(
        f"SELECT count(*) FROM (SELECT DISTINCT {demographic_projection} FROM svarah)"
    ).fetchone()[0]
    demographic_group_sizes = [
        row[0]
        for row in connection.execute(
            f"SELECT count(*) AS n FROM svarah GROUP BY {demographic_projection} ORDER BY n"
        ).fetchall()
    ]
    null_counts = {
        name: connection.execute(
            f'SELECT count(*) FROM svarah WHERE "{name}" IS NULL'
        ).fetchone()[0]
        for name in (
            "gender",
            "age-group",
            "primary_language",
            "native_place_state",
            "native_place_district",
            "highest_qualification",
            "job_category",
            "occupation_domain",
        )
    }

    names = [PurePosixPath(path).name for path in audio_paths]
    stems = [PurePosixPath(path).stem for path in audio_paths]
    parents = [str(PurePosixPath(path).parent) for path in audio_paths]
    token_group_counts: dict[str, int] = {}
    exact_117_groupings: dict[str, list[str]] = {}
    for separator in ("_", "-"):
        for width in range(1, 6):
            label = f"first_{width}_tokens_by_{separator}"
            groups = [separator.join(stem.split(separator)[:width]) for stem in stems]
            count = len(set(groups))
            token_group_counts[label] = count
            if count == 117:
                exact_117_groupings[label] = sorted(set(groups))

    path_examples_by_language = {
        language: [row[1] for row in rows[:5]]
        for language, rows in (
            (
                language,
                connection.execute(
                    "SELECT primary_language, audio_filepath.path FROM svarah "
                    "WHERE primary_language = ? ORDER BY audio_filepath.path LIMIT 5",
                    [language],
                ).fetchall(),
            )
            for (language,) in connection.execute(
                "SELECT DISTINCT primary_language FROM svarah ORDER BY primary_language"
            ).fetchall()
        )
    }
    report: dict[str, object] = {
        "artifact_kind": "svarah_parquet_decompression_audit",
        "dataset_revision": SVARAH_REVISION,
        "files": file_reports,
        "logical_schema": described,
        "row_count": row_count,
        "distinct_audio_paths": len(set(audio_paths)),
        "distinct_parent_paths": len(set(parents)),
        "distinct_file_names": len(set(names)),
        "distinct_stems": len(set(stems)),
        "demographic_signature_columns": [
            value.replace('"', "") for value in demographic_columns
        ],
        "distinct_demographic_signatures": demographic_group_count,
        "demographic_group_size": {
            "minimum": min(demographic_group_sizes),
            "maximum": max(demographic_group_sizes),
            "median": demographic_group_sizes[len(demographic_group_sizes) // 2],
        },
        "demographic_null_counts": null_counts,
        "metadata_speaker_search_hits": metadata_search_hits,
        "filename_token_group_counts": token_group_counts,
        "filename_groupings_with_exactly_117_groups": exact_117_groupings,
        "path_examples_by_language": path_examples_by_language,
        "speaker_id_column_present": any(row["name"] == "speaker_id" for row in described),
    }
    _write_once(Path(OUTPUT_DIR) / run_name / "parquet-decompression-audit.json", report)
    output_volume.commit()
    return report


@app.function(
    image=parquet_audit_image,
    cpu=4.0,
    memory=8192,
    timeout=20 * 60,
    volumes=VOLUMES,
)
def inspect_svarah_row_structure(run_name: str) -> dict[str, object]:
    """Test whether preserved row order or filename components recover 117 groups."""
    import hashlib
    from collections import defaultdict
    from pathlib import PurePosixPath

    import duckdb

    snapshot = (
        Path(CACHE_DIR)
        / "hf"
        / "hub"
        / "datasets--ai4bharat--Svarah"
        / "snapshots"
        / SVARAH_REVISION
        / "data"
    )
    parquet_files = sorted(snapshot.glob("*.parquet"))
    connection = duckdb.connect()
    demographic_columns = (
        "gender",
        '"age-group"',
        "primary_language",
        "native_place_state",
        "native_place_district",
        "highest_qualification",
        "job_category",
        "occupation_domain",
    )
    projection = ", ".join(demographic_columns)
    ordered_rows = []
    shard_boundaries = []
    for path in parquet_files:
        rows = connection.execute(
            f"SELECT audio_filepath.path, {projection} FROM read_parquet(?)",
            [str(path)],
        ).fetchall()
        ordered_rows.extend(rows)
        shard_boundaries.append(len(ordered_rows))

    def signature_id(signature: tuple[object, ...]) -> str:
        encoded = "\x1f".join(str(value) for value in signature).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    runs = []
    current_signature = None
    for index, row in enumerate(ordered_rows):
        signature = tuple(row[1:])
        if signature != current_signature:
            runs.append(
                {
                    "start_row": index,
                    "end_row": index,
                    "size": 1,
                    "signature_id": signature_id(signature),
                    "primary_language": signature[2],
                }
            )
            current_signature = signature
        else:
            runs[-1]["end_row"] = index
            runs[-1]["size"] += 1

    runs_by_signature: dict[str, list[dict[str, object]]] = defaultdict(list)
    signature_paths: dict[str, list[str]] = defaultdict(list)
    recording_signatures: dict[str, set[str]] = defaultdict(set)
    field_signatures: dict[str, set[str]] = defaultdict(set)
    for row in ordered_rows:
        path = row[0]
        signature = signature_id(tuple(row[1:]))
        signature_paths[signature].append(path)
        tokens = PurePosixPath(path).stem.split("_")
        recording_signatures[tokens[0]].add(signature)
        if len(tokens) > 1:
            field_signatures[tokens[1]].add(signature)
    for run in runs:
        runs_by_signature[run["signature_id"]].append(run)

    run_counts = {
        signature: len(values) for signature, values in runs_by_signature.items()
    }
    most_fragmented = sorted(
        run_counts,
        key=lambda signature: run_counts[signature],
        reverse=True,
    )[:10]
    report: dict[str, object] = {
        "artifact_kind": "svarah_parquet_row_structure_audit",
        "dataset_revision": SVARAH_REVISION,
        "row_count": len(ordered_rows),
        "shard_boundaries_after_rows": shard_boundaries,
        "distinct_demographic_signatures": len(signature_paths),
        "contiguous_demographic_runs": len(runs),
        "row_order_recovers_exactly_117_groups": len(runs) == 117,
        "signatures_split_across_multiple_runs": sum(
            count > 1 for count in run_counts.values()
        ),
        "demographic_run_count": {
            "minimum": min(run_counts.values()),
            "maximum": max(run_counts.values()),
        },
        "run_size": {
            "minimum": min(run["size"] for run in runs),
            "maximum": max(run["size"] for run in runs),
        },
        "distinct_first_filename_components": len(recording_signatures),
        "first_components_linking_multiple_demographic_signatures": sum(
            len(values) > 1 for values in recording_signatures.values()
        ),
        "distinct_second_filename_components": len(field_signatures),
        "second_components_linking_multiple_demographic_signatures": sum(
            len(values) > 1 for values in field_signatures.values()
        ),
        "sample_paths_for_repeated_signatures": {
            signature: paths[:10]
            for signature, paths in signature_paths.items()
            if signature in most_fragmented
        },
    }
    _write_once(Path(OUTPUT_DIR) / run_name / "parquet-row-structure-audit.json", report)
    output_volume.commit()
    return report


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=[hf_secret],
)
def run_objective_smoke(run_name: str, seed: int = 2026) -> dict[str, object]:
    run_dir = Path(OUTPUT_DIR) / run_name / "cuda-smoke"
    command = [
        "uv",
        "run",
        "--frozen",
        "python",
        "-m",
        "ast_asr.modal_smoke",
        "--config",
        f"{PROJECT_ROOT}/configs/fr_cispo_tiny.json",
        "--output-dir",
        str(run_dir),
        "--seed",
        str(seed),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=os.environ.copy())
    output_volume.commit()
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


@app.local_entrypoint()
def main(run_name: str = "", seed: int = 2026) -> None:
    resolved_name = run_name or datetime.now(UTC).strftime("smoke-%Y%m%dT%H%M%SZ")
    storage = inspect_svarah_storage.remote(resolved_name)
    smoke = run_objective_smoke.remote(resolved_name, seed)
    print(
        json.dumps(
            {
                "run_name": resolved_name,
                "speaker_fold_ready": storage["speaker_fold_ready"],
                "gpu": smoke["gpu"],
                "arms": [item["arm"] for item in smoke["arms"]],
                "round_trip_predictions_equal": smoke[
                    "round_trip_predictions_equal"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
