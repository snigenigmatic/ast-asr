"""Discoverable, one-attempt Modal entrypoint for locked H7 measurement.

This file defines remote work but never invokes it implicitly.  The local
entrypoint performs the executable-tree guard before the sole paid attempt.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

app = modal.App("ast-asr-h7-sentinel-kl-r1")
cache_volume = modal.Volume.from_name("ast-asr-cache", create_if_missing=False)
data_volume = modal.Volume.from_name("ast-asr-data", create_if_missing=False)
output_volume = modal.Volume.from_name("ast-asr-fr-cispo-runs", create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

PROJECT_ROOT = "/root/fr-cispo"
CACHE_DIR = "/cache"
DATA_DIR = "/data"
OUTPUT_DIR = "/artifacts"
H7_PROFILE = "profile-h7-fixed-policy-sentinel-kl-r1-s2028-20260812"
H7_OUTPUT = "h7-fixed-policy-sentinel-kl-r1"

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
        "nvidia/cuda@sha256:09d8951b943dee03cf8fc841b6ea1f201ad33f82f76567171394853c0f494054",
        add_python="3.12",
    )
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("uv>=0.8,<0.9")
    .env(CACHE_ENV)
    .add_local_file(
        "pyproject.toml", remote_path=f"{PROJECT_ROOT}/pyproject.toml", copy=True
    )
    .add_local_file("uv.lock", remote_path=f"{PROJECT_ROOT}/uv.lock", copy=True)
    .run_commands(
        f"cd {PROJECT_ROOT} && uv sync --frozen --no-dev --no-install-project"
    )
    .add_local_dir("src", remote_path=f"{PROJECT_ROOT}/src", copy=True)
    .add_local_dir("configs", remote_path=f"{PROJECT_ROOT}/configs", copy=True)
    .add_local_file(
        "experiments/H7-sentinel-kl/input-lock.json",
        remote_path=f"{PROJECT_ROOT}/experiments/H7-sentinel-kl/input-lock.json",
        copy=True,
    )
    .run_commands(
        f"cd {PROJECT_ROOT} && uv sync --frozen --no-dev --offline --no-build-isolation"
    )
)

VOLUMES = {CACHE_DIR: cache_volume, DATA_DIR: data_volume, OUTPUT_DIR: output_volume}


def _write_once(path: Path, value: object) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"refusing to overwrite H7 evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _local_source_commit(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def assert_clean_h7_executable_tree(project_root: Path) -> str:
    """Local guard: remote images intentionally contain no .git directory."""
    scoped = (
        "scripts/modal_h7_sentinel.py",
        "src",
        "configs",
        "uv.lock",
        "pyproject.toml",
        "experiments/H7-sentinel-kl/input-lock.json",
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "--", *scoped], cwd=project_root, check=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *scoped],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError(
            "H7 executable inputs are dirty or untracked; refuse the sole attempt"
        )
    return _local_source_commit(project_root)


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=60 * 60,
    retries=0,
    volumes=VOLUMES,
    secrets=[hf_secret],
)
def run_h7_sentinel(source_commit: str) -> dict[str, object]:
    """One FP16 rescore attempt; every post-claim exception is committed evidence."""
    output = Path(OUTPUT_DIR) / H7_PROFILE / H7_OUTPUT
    run_root = output.parent
    if run_root.exists():
        raise FileExistsError(f"reserved H7 run root already exists: {run_root}")
    command = [
        "uv",
        "run",
        "--frozen",
        "ast-asr",
        "measure-sentinel-kl",
        "--config",
        "/artifacts/profile-h6-refkl-beta0-s2028-20260812/resolved-policy-configs/h6-beta0-fr-cispo.json",
        "--bank-root",
        "/artifacts/profile-h6-refkl-beta0-s2028-20260812/h6-beta0-fr-cispo/rollouts",
        "--archive-root",
        "/data/fr_cispo_profile/raw/Svarah",
        "--policy-checkpoint",
        "/artifacts/profile-h6-refkl-beta0-s2028-20260812/h6-beta0-fr-cispo/checkpoint-last-safe",
        "--reference-checkpoint",
        "/artifacts/profile-dev-full-sft-20260810/profile-sft-development/checkpoint-epoch-1",
        "--input-lock",
        f"{PROJECT_ROOT}/experiments/H7-sentinel-kl/input-lock.json",
        "--output-dir",
        str(output),
        "--expected-policy-revision",
        "a95530fd914b7fea9f3008a5c6451f3fedef2281443fce6b9dc0df5ba6a8d400",
        "--expected-reference-revision",
        "d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e",
        "--expected-cycle27-model-revision",
        "5292f4896a06ce2d1c7abf9dd589af01fb5b702e5bbf6fff8f4ea2fcb66c8ea9",
        "--expected-config-sha256",
        "3673abefc4322f4951ee067c8b6ed2c2fef93008b3f85c2cf66afd5abd406ae5",
    ]
    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            env={**os.environ, "H7_SOURCE_COMMIT": source_commit},
        )
    except BaseException as error:
        failure = output / "failure.json"
        if not failure.exists():
            _write_once(
                failure,
                {
                    "artifact_kind": "h7_modal_launcher_failure",
                    "publication_valid": False,
                    "profile_cluster_count": 115,
                    "decision": "measurement_failed",
                    "measurement_status": "non_evaluable",
                    "phase": "launcher_or_process_boundary",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "source_commit": source_commit,
                    "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
                    "command": command,
                },
            )
        output_volume.commit()
        raise
    output_volume.commit()
    return json.loads((output / "terminal_decision.json").read_text(encoding="utf-8"))


@app.local_entrypoint()
def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_commit = assert_clean_h7_executable_tree(root)
    print(json.dumps(run_h7_sentinel.remote(source_commit), sort_keys=True))
