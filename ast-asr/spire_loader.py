
"""
spire_loader.py
Loads the IISc SPIRE-SIES corpus from extracted tarballs into either:
  - a lightweight manifest DataFrame (metadata + file path, no audio in memory)
  - a fully-materialised DataFrame with decoded 16 kHz mono float32 audio
    (compatible with data_loader.load_svarah() and the existing eval pipeline)

Expected layout under `raw_root`:
    raw_root/
        IISc_SPIRE_SIES_Transcription.csv
        IISc_SPIRE_SIES_<Language>/<speaker_id>/<filename>.wav

Speaker-level 85/15 train/val split stratified by language family is saved
to `<raw_root>/../splits.json` so downstream runs are reproducible.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from data_loader import ACCENT_TO_FAMILY, TARGET_SR

logger = logging.getLogger(__name__)

DEFAULT_RAW_ROOT = Path("data/spire-sies/raw")
DEFAULT_SPLIT_PATH = Path("data/spire-sies/splits.json")

# Languages we intentionally skip (no audio / not useful for family training).
SKIPPED_DIRS = {
    "IISc_SPIRE_SIES_Images",
    "IISc_SPIRE_SIES_Image_Captions",
}

# Default: train the adversary on languages we can actually map to a family.
# "Other" is kept out of the default training set because we can't anchor it
# to a language family; callers can opt in explicitly via `languages=[..., "Other"]`.
DEFAULT_LANGUAGES = tuple(ACCENT_TO_FAMILY.keys())

# Noise / disfluency tags produced by the SIES annotators.
# We strip these before training so the CTC targets are plain English text.
_TAG_RE = re.compile(r"<[^>]+>")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_NON_SPEECH_RE = re.compile(r"[^A-Z' ]")
_WS_RE = re.compile(r"\s+")


# ── Transcript normalization ─────────────────────────────────────────────────

def normalize_transcript(text: str) -> str:
    """
    Clean a SPIRE-SIES transcript down to the Wav2Vec2-base vocabulary:
    uppercase letters, space, and apostrophe. Removes annotator tags
    (<SSIL>, <noise>, <PAUSE>, ...), bracketed markers ([unintelligible]),
    and any remaining punctuation/digits.
    """
    if text is None:
        return ""
    t = str(text)
    t = _TAG_RE.sub(" ", t)
    t = _BRACKET_RE.sub(" ", t)
    t = t.upper()
    t = _NON_SPEECH_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


# ── Filename parsing ─────────────────────────────────────────────────────────

def _parse_speaker_id(stem: str) -> str:
    """
    SPIRE-SIES filenames look like:
        IISc_SPIRE_SIES_<speaker>_<image_id>_<timestamp>[_<utt_idx>]
    The speaker is always the 4th underscore-delimited field.
    """
    parts = stem.split("_")
    if len(parts) < 4:
        return "UNKNOWN"
    return parts[3]


def _gender_from_speaker(speaker_id: str) -> str:
    if not speaker_id:
        return "Unknown"
    prefix = speaker_id[0]
    return {"F": "Female", "M": "Male", "O": "Other"}.get(prefix, "Unknown")


# ── Transcription CSV loader ─────────────────────────────────────────────────

def _load_transcription_index(raw_root: Path) -> dict[str, str]:
    csv_path = raw_root / "IISc_SPIRE_SIES_Transcription.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Transcription CSV not found at {csv_path}. "
            "Extract IISc_SPIRE_SIES_Transcription.tar.gz into raw_root first."
        )
    df = pd.read_csv(csv_path)
    expected = {"File_Name", "Transcript"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"Transcription CSV schema unexpected: got {list(df.columns)}"
        )
    logger.info("Loaded %d transcripts from %s", len(df), csv_path.name)
    return dict(zip(df["File_Name"], df["Transcript"]))


# ── Manifest construction ────────────────────────────────────────────────────

def _language_dir(raw_root: Path, language: str) -> Path:
    return raw_root / f"IISc_SPIRE_SIES_{language}"


def build_manifest(
    raw_root: Path | str = DEFAULT_RAW_ROOT,
    languages: Iterable[str] | None = None,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    min_words: int = 3,
) -> pd.DataFrame:
    """
    Walk extracted audio directories, join against the transcription CSV,
    and return a manifest DataFrame with columns:
        uid, path, accent, language_family, speaker_id, gender, age,
        reference, duration

    Audio is NOT decoded here — use `materialize_audio()` to load bytes
    for a specific subset.
    """
    raw_root = Path(raw_root)
    langs = list(languages) if languages is not None else list(DEFAULT_LANGUAGES)
    transcripts = _load_transcription_index(raw_root)

    records: list[dict] = []
    dropped_no_transcript = 0
    dropped_empty_text = 0
    dropped_bad_duration = 0

    for lang in langs:
        lang_dir = _language_dir(raw_root, lang)
        if not lang_dir.is_dir():
            logger.debug("Language dir missing, skipping: %s", lang_dir)
            continue

        family = ACCENT_TO_FAMILY.get(lang, "Unknown")

        for wav_path in lang_dir.rglob("*.wav"):
            stem = wav_path.stem
            raw_text = transcripts.get(stem)
            if raw_text is None:
                dropped_no_transcript += 1
                continue

            cleaned = normalize_transcript(raw_text)
            if len(cleaned.split()) < min_words:
                dropped_empty_text += 1
                continue

            try:
                info = sf.info(wav_path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("sf.info failed for %s: %s", wav_path, exc)
                continue

            dur = info.duration
            if not (min_duration <= dur <= max_duration):
                dropped_bad_duration += 1
                continue

            speaker_id = _parse_speaker_id(stem)
            records.append(
                {
                    "uid":             stem,
                    "path":            str(wav_path),
                    "accent":          lang,
                    "language_family": family,
                    "speaker_id":      speaker_id,
                    "gender":          _gender_from_speaker(speaker_id),
                    "age":             "Unknown",
                    "reference":       cleaned,
                    "duration":        float(dur),
                    "src_sr":          int(info.samplerate),
                }
            )

    df = pd.DataFrame.from_records(records)
    logger.info(
        "Manifest built: %d utterances (dropped: %d no-transcript, %d empty-text, %d bad-duration)",
        len(df), dropped_no_transcript, dropped_empty_text, dropped_bad_duration,
    )
    if not df.empty:
        _log_family_distribution(df)
    return df


# ── Speaker-level split ──────────────────────────────────────────────────────

def make_speaker_split(
    manifest: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
    split_path: Path | str | None = DEFAULT_SPLIT_PATH,
) -> dict[str, list[str]]:
    """
    Assigns each speaker to train or val, stratified by language family, so
    every family is represented in both splits. The decision is cached to
    `split_path` (JSON) on first call and reused thereafter for reproducibility.
    """
    if split_path is not None:
        split_path = Path(split_path)
        if split_path.exists():
            with split_path.open() as f:
                cached = json.load(f)
            logger.info("Reusing speaker split from %s", split_path)
            return cached

    rng = np.random.default_rng(seed)

    # Group speakers by family (speakers rarely span families, but if they do
    # we take the majority family for the assignment decision).
    spk_family = (
        manifest.groupby("speaker_id")["language_family"]
        .agg(lambda s: s.value_counts().idxmax())
    )

    train_spk: list[str] = []
    val_spk: list[str] = []
    for group in spk_family.groupby(spk_family).groups.values():
        speakers = sorted(group.tolist())
        rng.shuffle(speakers)
        n_val = max(1, int(round(len(speakers) * val_ratio)))
        val_spk.extend(speakers[:n_val])
        train_spk.extend(speakers[n_val:])

    split = {"train": sorted(train_spk), "val": sorted(val_spk)}
    logger.info(
        "Speaker split: %d train / %d val (from %d total)",
        len(split["train"]), len(split["val"]), len(spk_family),
    )

    if split_path is not None:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("w") as f:
            json.dump(split, f, indent=2)
        logger.info("Saved split to %s", split_path)
    return split


def apply_split(manifest: pd.DataFrame, split: dict[str, list[str]], which: str) -> pd.DataFrame:
    if which not in split:
        raise ValueError(f"split must be one of {list(split.keys())}, got {which!r}")
    speakers = set(split[which])
    return manifest[manifest["speaker_id"].isin(speakers)].reset_index(drop=True)


# ── Audio materialisation (for eval-style subsets) ───────────────────────────

def _decode_wav(path: str) -> tuple[np.ndarray, int]:
    waveform, sr = sf.read(path, dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1, dtype=np.float32)
    return waveform, int(sr)


def materialize_audio(manifest: pd.DataFrame, max_samples: int | None = None) -> pd.DataFrame:
    """
    Decode audio for every row in `manifest` (or the first `max_samples`)
    and return a DataFrame compatible with the existing pipeline (adds
    `audio_array` column at 16 kHz, drops `path` + `src_sr`).
    """
    if max_samples is not None:
        manifest = manifest.head(max_samples).copy()
    else:
        manifest = manifest.copy()

    audio_arrays: list[np.ndarray] = []
    for path in manifest["path"]:
        waveform, sr = _decode_wav(path)
        if sr != TARGET_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
        audio_arrays.append(waveform)

    manifest["audio_array"] = audio_arrays
    return manifest.drop(columns=["path", "src_sr"])


# ── High-level API mirroring load_svarah() ───────────────────────────────────

def load_spire_sies(
    split: str = "train",
    max_samples: int | None = None,
    raw_root: Path | str = DEFAULT_RAW_ROOT,
    languages: Iterable[str] | None = None,
    val_ratio: float = 0.15,
    seed: int = 42,
    split_path: Path | str | None = DEFAULT_SPLIT_PATH,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns compatible with load_svarah():
        uid, accent, language_family, gender, age,
        audio_array (np.ndarray @ 16 kHz), reference, speaker_id, duration

    By default builds a manifest from `raw_root`, applies the speaker-level
    split, and materialises audio into memory. Intended for smaller eval runs;
    for training on the full corpus use `build_manifest()` + a streaming
    Dataset wrapper (see `to_hf_dataset` below).
    """
    manifest = build_manifest(raw_root=raw_root, languages=languages)
    split_map = make_speaker_split(
        manifest, val_ratio=val_ratio, seed=seed, split_path=split_path
    )
    part = apply_split(manifest, split_map, split)
    logger.info("SPIRE-SIES[%s] manifest: %d utterances", split, len(part))
    return materialize_audio(part, max_samples=max_samples)


def _log_family_distribution(df: pd.DataFrame) -> None:
    if "duration" in df.columns:
        hours = df.groupby("language_family")["duration"].sum() / 3600.0
    else:
        hours = None
    for fam, n in df["language_family"].value_counts().items():
        if hours is not None:
            logger.info("  %-15s  %6d utterances  (%.1f h)", fam, n, hours[fam])
        else:
            logger.info("  %-15s  %6d utterances", fam, n)


# ── Optional: HuggingFace Dataset wrapper for training ───────────────────────

def to_hf_dataset(manifest: pd.DataFrame):
    """
    Convert a manifest DataFrame into a HuggingFace `datasets.Dataset` with
    lazy-decoded audio. Used by the training loop so the full corpus does
    not need to fit in RAM.
    """
    try:
        from datasets import Audio, Dataset
    except ImportError as exc:
        raise ImportError("Install `datasets` to use to_hf_dataset()") from exc

    cols_to_keep = [c for c in manifest.columns if c != "src_sr"]
    ds = Dataset.from_pandas(manifest[cols_to_keep], preserve_index=False)
    ds = ds.cast_column("path", Audio(sampling_rate=TARGET_SR))
    ds = ds.rename_column("path", "audio")
    return ds
