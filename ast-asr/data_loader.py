"""
data_loader.py
Loads Svarah from HuggingFace and maps each utterance to:
  - language_family  (Dravidian / Indo-Aryan / Sino-Tibetan)
  - accent           (raw Svarah label, e.g. "Tamil", "Hindi")
  - gender, age      (from Svarah metadata where available)
  - audio array      (resampled to 16 kHz)
  - reference text
"""

import io
import logging
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)

# ── Svarah accent → language family ──────────────────────────────────────────
# Source: Javed et al. 2023 + Figure 1 of our position paper
ACCENT_TO_FAMILY = {
    # Dravidian
    "Tamil":     "Dravidian",
    "Telugu":    "Dravidian",
    "Kannada":   "Dravidian",
    "Malayalam": "Dravidian",
    # Indo-Aryan
    "Hindi":     "Indo-Aryan",
    "Bengali":   "Indo-Aryan",
    "Marathi":   "Indo-Aryan",
    "Gujarati":  "Indo-Aryan",
    "Punjabi":   "Indo-Aryan",
    "Odia":      "Indo-Aryan",
    "Assamese":  "Indo-Aryan",
    "Urdu":      "Indo-Aryan",
    "Nepali":    "Indo-Aryan",
    "Maithili":  "Indo-Aryan",
    "Kashmiri":  "Indo-Aryan",
    "Konkani":   "Indo-Aryan",
    "Dogri":     "Indo-Aryan",
    "Sindhi":    "Indo-Aryan",
    # Sino-Tibetan
    "Mizo":      "Sino-Tibetan",
    "Manipuri":  "Sino-Tibetan",
    "Bodo":      "Sino-Tibetan",
}

TARGET_SR = 16_000  # Whisper / Wav2Vec 2.0 / HuBERT all expect 16 kHz


def _read_hf_token_from_env_file(env_path: Path) -> str | None:
    """Best-effort parser for HF_TOKEN from a local .env file."""
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        if key.strip() != "HF_TOKEN":
            continue

        token = value.strip()
        if token.startswith("'") and token.endswith("'") and len(token) >= 2:
            token = token[1:-1]
        elif token.startswith('"') and token.endswith('"') and len(token) >= 2:
            token = token[1:-1]

        return token or None

    return None


def _resolve_hf_token() -> str | None:
    """Resolve Hugging Face token from process env first, then project .env."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()

    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    token = _read_hf_token_from_env_file(env_path)
    if token:
        os.environ.setdefault("HF_TOKEN", token)
    return token


def _decode_audio_payload(audio_payload) -> tuple[np.ndarray, int]:
    """
    Decode an audio payload returned by HuggingFace datasets with decode=False.

    Supports payloads containing either `bytes` or a local cached `path`.
    Falls back gracefully when payload is already decoded as {array, sampling_rate}.
    """
    if isinstance(audio_payload, dict) and "array" in audio_payload:
        waveform = np.asarray(audio_payload["array"], dtype=np.float32)
        sr = int(audio_payload["sampling_rate"])
    elif isinstance(audio_payload, dict):
        audio_bytes = audio_payload.get("bytes")
        audio_path = audio_payload.get("path")

        if audio_bytes is not None:
            waveform, sr = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32",
                always_2d=False,
            )
        elif audio_path:
            waveform, sr = sf.read(
                audio_path,
                dtype="float32",
                always_2d=False,
            )
        else:
            raise ValueError(
                "Audio payload did not include 'bytes', 'path', or decoded 'array'."
            )
    else:
        raise TypeError(f"Unexpected audio payload type: {type(audio_payload)!r}")

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        # Convert multi-channel audio to mono for ASR models expecting 1 channel.
        waveform = np.mean(waveform, axis=1, dtype=np.float32)

    return waveform, int(sr)


def load_svarah(
    split: str = "test",
    max_samples: int | None = None,
    cache_dir: str | None = None,
    svarah_split: str = "all",
    split_manifest_dir: str = "data/svarah_split",
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        uid, accent, language_family, gender, age,
        audio_array (np.ndarray @ 16kHz), reference

    svarah_split: "train", "eval", or "all" (default). When non-"all",
        filters to UIDs listed in split_manifest_dir/{train,eval}_uids.txt.
    """
    try:
        from datasets import Audio, load_dataset
    except ImportError:
        raise ImportError("Install HuggingFace `datasets`: pip install datasets")

    hf_token = _resolve_hf_token()
    if hf_token:
        logger.info("HF_TOKEN detected; attempting authenticated dataset access.")
    else:
        logger.info("HF_TOKEN not found; attempting public dataset access.")

    logger.info("Loading Svarah (%s split) from HuggingFace …", split)
    load_kwargs = {
        "split": split,
        "cache_dir": cache_dir,
    }
    if hf_token:
        load_kwargs["token"] = hf_token

    try:
        ds = load_dataset("ai4bharat/svarah", **load_kwargs)
    except TypeError:
        # Backward compatibility with datasets versions that use `use_auth_token`.
        if not hf_token:
            raise

        legacy_kwargs = {
            "split": split,
            "cache_dir": cache_dir,
            "use_auth_token": hf_token,
        }
        ds = load_dataset("ai4bharat/svarah", **legacy_kwargs)

    # Detect the audio column name (Svarah uses "audio_filepath", older dumps use "audio").
    audio_col = next(
        (c for c in ds.column_names if isinstance(ds.features[c], Audio)),
        None,
    )
    # Force non-torchcodec audio payloads; we decode with soundfile below.
    if audio_col is not None:
        ds = ds.cast_column(audio_col, Audio(decode=False))

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    records = []
    for i, sample in enumerate(ds):
        accent = sample.get("accent", sample.get("primary_language", sample.get("language", "Unknown")))
        # Normalise capitalisation so the mapping works
        accent = accent.strip().title()

        family = ACCENT_TO_FAMILY.get(accent, "Unknown")
        if family == "Unknown":
            logger.debug("Unrecognised accent '%s' at index %d", accent, i)

        # Audio: read bytes/path payload directly to avoid torchcodec + FFmpeg runtime.
        audio_info = sample[audio_col]
        waveform, sr = _decode_audio_payload(audio_info)

        if sr != TARGET_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

        records.append(
            {
                "uid":             i,
                "accent":          accent,
                "language_family": family,
                "gender":          sample.get("gender", "Unknown"),
                "age":             sample.get("age-group", sample.get("age", "Unknown")),
                "audio_array":     waveform,
                "reference":       sample.get("text", sample.get("transcript", "")),
            }
        )

    df = pd.DataFrame(records)

    # Optional train/eval split filtering
    if svarah_split in ("train", "eval"):
        manifest = Path(split_manifest_dir) / f"{svarah_split}_uids.txt"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Split manifest not found: {manifest}. "
                "Run: python scripts/make_svarah_split.py"
            )
        keep_uids = set(int(u) for u in manifest.read_text().strip().split("\n"))
        df = df[df["uid"].isin(keep_uids)].reset_index(drop=True)
        logger.info("Filtered to svarah_split='%s': %d utterances", svarah_split, len(df))

    logger.info(
        "Loaded %d utterances across %d accent groups (%d language families)",
        len(df),
        df["accent"].nunique(),
        df["language_family"].nunique(),
    )
    _log_family_distribution(df)
    return df


def _log_family_distribution(df: pd.DataFrame) -> None:
    dist = df["language_family"].value_counts()
    for fam, n in dist.items():
        logger.info("  %-15s  %d utterances", fam, n)