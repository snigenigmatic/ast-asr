"""Declarative H7 Modal launch specification; this module never launches work."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .h7_runner import (
    H7_BETA0_CONFIG_SHA256,
    H7_RESERVED_OUTPUT_NAME,
    H7_RESERVED_PROFILE_NAME,
)

H7_PROJECT_ROOT = "/root/fr-cispo"
H7_CACHE_MOUNT = "/cache"
H7_DATA_MOUNT = "/data"
H7_ARTIFACT_MOUNT = "/artifacts"
H7_RESOLVED_CONFIG = (
    "/artifacts/profile-h6-refkl-beta0-s2028-20260812/"
    "resolved-policy-configs/h6-beta0-fr-cispo.json"
)
H7_ARCHIVE_ROOT = "/data/fr_cispo_profile/raw/Svarah"


@dataclass(frozen=True, slots=True)
class H7ModalLaunchSpec:
    profile_name: str
    output_name: str
    retries: int
    command: tuple[str, ...]

    @classmethod
    def default(cls) -> H7ModalLaunchSpec:
        return cls(
            profile_name=H7_RESERVED_PROFILE_NAME,
            output_name=H7_RESERVED_OUTPUT_NAME,
            retries=0,
            command=(
                "uv",
                "run",
                "--frozen",
                "ast-asr",
                "measure-sentinel-kl",
                "--config",
                H7_RESOLVED_CONFIG,
                "--bank-root",
                "/artifacts/profile-h6-refkl-beta0-s2028-20260812/h6-beta0-fr-cispo/rollouts",
                "--archive-root",
                H7_ARCHIVE_ROOT,
                "--policy-checkpoint",
                "/artifacts/profile-h6-refkl-beta0-s2028-20260812/h6-beta0-fr-cispo/checkpoint-last-safe",
                "--reference-checkpoint",
                "/artifacts/profile-dev-full-sft-20260810/profile-sft-development/checkpoint-epoch-1",
                "--input-lock",
                f"{H7_PROJECT_ROOT}/experiments/H7-sentinel-kl/input-lock.json",
                "--output-dir",
                f"/artifacts/{H7_RESERVED_PROFILE_NAME}/{H7_RESERVED_OUTPUT_NAME}",
                "--expected-policy-revision",
                "a95530fd914b7fea9f3008a5c6451f3fedef2281443fce6b9dc0df5ba6a8d400",
                "--expected-reference-revision",
                "d204df40dfcd694733a171998ad5d97fdb43eecbc5dc19846d98bce012cd4c1e",
                "--expected-cycle27-model-revision",
                "5292f4896a06ce2d1c7abf9dd589af01fb5b702e5bbf6fff8f4ea2fcb66c8ea9",
                "--expected-config-sha256",
                H7_BETA0_CONFIG_SHA256,
            ),
        )


def validate_h7_preflight(*, remote_run_root: Path, git_porcelain: str) -> None:
    """Fail before the sole paid attempt when names or executable inputs drift."""
    if remote_run_root.exists():
        raise FileExistsError(f"reserved H7 run root already exists: {remote_run_root}")
    if git_porcelain.strip():
        raise RuntimeError(
            "H7 requires a clean executable worktree before its sole attempt"
        )
