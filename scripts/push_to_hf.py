"""
push_to_hf.py
Build Parquet shards (one per language) from the extracted SPIRE-SIES raw
corpus and upload them as a HuggingFace dataset.

Each shard carries the normalized transcripts, the precomputed speaker
split assignment, and the audio as raw bytes (so future runs can
load the dataset without reaching back to the original tarballs).

Usage:
    python scripts/push_to_hf.py --repo VectorSigma389/spire-sies --private
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ast-asr"))
from spire_loader import (  # noqa: E402
    apply_split,
    build_manifest,
    make_speaker_split,
    DEFAULT_RAW_ROOT,
    DEFAULT_SPLIT_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        token = token.strip().strip("'").strip('"')
        return token or None

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "HF_TOKEN":
            token = value.strip().strip("'").strip('"')
            return token or None
    return None


def _split_repo_id(repo_id: str) -> tuple[str | None, str]:
    if "/" not in repo_id:
        return None, repo_id
    namespace, name = repo_id.split("/", 1)
    return namespace, name


def _org_names(whoami: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for org in whoami.get("orgs", []) or []:
        if isinstance(org, str):
            names.add(org)
        elif isinstance(org, dict):
            for key in ("name", "id", "username"):
                value = org.get(key)
                if isinstance(value, str) and value:
                    names.add(value)
                    break
    return names


def _build_hf_api() -> Any:
    from huggingface_hub import HfApi

    token = _resolve_hf_token()
    if not token:
        raise RuntimeError(
            "HF_TOKEN not found. Set it in the environment or in .env before uploading."
        )
    return HfApi(token=token)


def _preflight_upload(api: Any, repo_id: str, private: bool) -> str:
    """
    Validate auth/namespace permissions and create (or verify) the destination
    dataset repo *before* expensive shard construction.
    Returns the fully-qualified repo id.
    """
    try:
        whoami = api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "Unable to authenticate to Hugging Face with HF_TOKEN. "
            "Use a valid token with write permission."
        ) from exc

    username = whoami.get("name") if isinstance(whoami, dict) else None
    namespace, repo_name = _split_repo_id(repo_id)
    resolved_repo_id = repo_id

    if username and namespace is None:
        resolved_repo_id = f"{username}/{repo_name}"
        namespace = username
        logger.info("No namespace in --repo; resolved target to %s", resolved_repo_id)

    if namespace and username:
        allowed_namespaces = {username, *_org_names(whoami)}
        if namespace not in allowed_namespaces:
            allowed = ", ".join(sorted(allowed_namespaces))
            raise PermissionError(
                f"Authenticated as '{username}', but cannot create datasets under "
                f"namespace '{namespace}'. Allowed namespaces: {allowed}. "
                f"Try --repo {username}/{repo_name} or use a token/account with "
                f"write access to '{namespace}'."
            )

    try:
        api.create_repo(
            resolved_repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
    except Exception as exc:
        msg = str(exc)
        if "403" in msg:
            ns = namespace or "(unknown namespace)"
            owner_hint = username or "<your-username>"
            raise PermissionError(
                f"403 Forbidden while creating dataset repo '{resolved_repo_id}'. "
                f"Token lacks permission for namespace '{ns}'. "
                f"Try --repo {owner_hint}/{repo_name} or use a token with write "
                f"permissions for '{ns}'."
            ) from exc
        raise

    logger.info("Target HF repo: %s (private=%s)", resolved_repo_id, private)
    return resolved_repo_id


def _load_audio_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _build_shards(manifest: pd.DataFrame, split_map: dict, out_dir: Path) -> list[Path]:
    """
    Write one Parquet shard per language. Columns:
        uid, speaker_id, accent, language_family, gender, reference,
        duration, src_sr, split, audio_bytes
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    train_spks = set(split_map["train"])
    val_spks = set(split_map["val"])

    def _split_for(spk: str) -> str:
        if spk in train_spks:
            return "train"
        if spk in val_spks:
            return "val"
        return "unassigned"

    manifest = manifest.copy()
    manifest["split"] = manifest["speaker_id"].map(_split_for)

    shards: list[Path] = []
    for lang, sub in manifest.groupby("accent"):
        logger.info("Loading audio bytes for %s (%d utterances)", lang, len(sub))
        sub = sub.copy()
        sub["audio_bytes"] = sub["path"].apply(_load_audio_bytes)
        sub = sub.drop(columns=["path"])

        shard_path = out_dir / f"{lang.lower()}.parquet"
        sub.to_parquet(shard_path, index=False)
        logger.info("Wrote %s (%.1f MB)", shard_path, shard_path.stat().st_size / 1e6)
        shards.append(shard_path)
    return shards


def _push(shards: list[Path], repo_id: str, api: Any) -> None:

    for shard in shards:
        logger.info("Uploading %s ...", shard.name)
        api.upload_file(
            path_or_fileobj=str(shard),
            path_in_repo=f"data/{shard.name}",
            repo_id=repo_id,
            repo_type="dataset",
        )

    # Also push the split manifest so downstream users can reproduce train/val.
    split_path = Path(DEFAULT_SPLIT_PATH)
    if split_path.exists():
        api.upload_file(
            path_or_fileobj=str(split_path),
            path_in_repo="splits.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        logger.info("Uploaded splits.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="VectorSigma389/spire-sies")
    ap.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    ap.add_argument("--out-dir", default="data/spire-sies/parquet")
    ap.add_argument("--private", action="store_true", help="Create repo as private")
    ap.add_argument("--skip-upload", action="store_true", help="Build shards but don't push")
    args = ap.parse_args()

    api: Any | None = None
    target_repo_id = args.repo
    if not args.skip_upload:
        api = _build_hf_api()
        target_repo_id = _preflight_upload(api, args.repo, private=args.private)

    manifest = build_manifest(raw_root=args.raw_root)
    split_map = make_speaker_split(
        manifest, split_path=DEFAULT_SPLIT_PATH
    )

    shards = _build_shards(manifest, split_map, Path(args.out_dir))
    logger.info("Built %d shards", len(shards))

    if args.skip_upload:
        logger.info("--skip-upload set, not pushing to HF Hub")
        return
    _push(shards, target_repo_id, api)
    logger.info("Done.")


if __name__ == "__main__":
    main()
