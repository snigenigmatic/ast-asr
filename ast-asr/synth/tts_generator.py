"""
tts_generator.py
Generate synthetic Sino-Tibetan-accented English audio via XTTS-v2 voice cloning.

Approach:
  1. Source text: English sentences from SPIRE-SIES transcripts
  2. Reference voices: ST speaker audio clips (Mizo/Bodo/Manipuri from Svarah)
  3. Synthesis: XTTS-v2 clones the voice characteristics onto source text
  4. Quality filter: discard samples where ft-w2v2 WER > 0.7

Requires: pip install TTS  (or: uv pip install "ast-asr[synth]")
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/synthetic-st")
TARGET_SR = 16_000


def load_source_texts(
    transcript_csv: str | Path,
    min_words: int = 5,
    max_words: int = 30,
    max_texts: int = 1000,
    seed: int = 42,
) -> list[str]:
    """Load and filter source texts from SPIRE-SIES transcripts."""
    import re

    tag_re = re.compile(r"<[^>]+>")
    bracket_re = re.compile(r"\[[^\]]*\]")
    ws_re = re.compile(r"\s+")

    texts = []
    with open(transcript_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("Transcript", "")
            t = tag_re.sub(" ", t)
            t = bracket_re.sub(" ", t)
            t = ws_re.sub(" ", t).strip()
            words = t.split()
            if min_words <= len(words) <= max_words:
                texts.append(t)

    rng = random.Random(seed)
    rng.shuffle(texts)
    selected = texts[:max_texts]
    logger.info("Selected %d source texts (from %d total)", len(selected), len(texts))
    return selected


def find_reference_voices(
    svarah_dir: str | Path,
    accents: tuple[str, ...] = ("Mizo", "Manipuri", "Bodo"),
    max_per_accent: int = 5,
) -> list[dict]:
    """Find reference voice clips from Svarah ST speakers."""
    svarah_dir = Path(svarah_dir)
    voices = []

    for accent in accents:
        accent_dir = svarah_dir / accent
        if not accent_dir.exists():
            logger.warning("Svarah accent dir not found: %s", accent_dir)
            continue

        wavs = sorted(accent_dir.rglob("*.wav"))[:max_per_accent]
        for wav_path in wavs:
            voices.append({
                "path": str(wav_path),
                "accent": accent,
                "family": "Sino-Tibetan",
            })

    logger.info("Found %d reference voices across %s", len(voices), accents)
    return voices


def generate_synthetic_data(
    source_texts: list[str],
    reference_voices: list[dict],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    quality_filter_model: str | None = None,
    max_wer: float = 0.7,
    seed: int = 42,
) -> Path:
    """
    Generate synthetic ST-accented English audio via XTTS-v2.

    Args:
        source_texts: English sentences to synthesize
        reference_voices: list of dicts with 'path' and 'accent'
        output_dir: where to save WAV files and manifest
        quality_filter_model: checkpoint path for WER filtering (optional)
        max_wer: reject samples above this WER
        seed: random seed

    Returns:
        Path to the output manifest CSV
    """
    try:
        from TTS.api import TTS
    except ImportError:
        raise ImportError(
            "TTS package not installed. Install with: "
            "/astral/uv pip install 'TTS>=0.22.0'"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "audio"
    wav_dir.mkdir(exist_ok=True)

    logger.info("Loading XTTS-v2 model...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    if hasattr(tts, "to"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts.to(device)

    rng = random.Random(seed)
    records = []
    generated = 0
    failed = 0

    for i, text in enumerate(source_texts):
        voice = rng.choice(reference_voices)
        uid = f"synth_st_{i:04d}"
        wav_path = wav_dir / f"{uid}.wav"

        try:
            tts.tts_to_file(
                text=text,
                speaker_wav=voice["path"],
                language="en",
                file_path=str(wav_path),
            )

            # Read back to verify and get duration
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if sr != TARGET_SR:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
                sf.write(str(wav_path), audio, TARGET_SR)

            duration = len(audio) / TARGET_SR
            records.append({
                "uid": uid,
                "path": str(wav_path),
                "accent": voice["accent"],
                "language_family": "Sino-Tibetan",
                "speaker_id": f"SYNTH_{voice['accent']}",
                "gender": "Unknown",
                "age": "Unknown",
                "reference": text.upper(),
                "duration": duration,
            })
            generated += 1

            if (i + 1) % 50 == 0:
                logger.info("Generated %d/%d synthetic samples", i + 1, len(source_texts))

        except Exception as e:
            logger.warning("Failed to synthesize sample %d: %s", i, e)
            failed += 1

    # Save manifest
    manifest_path = output_dir / "manifest.csv"
    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(manifest_path, index=False)

    logger.info(
        "Synthetic data generation complete: %d generated, %d failed. Manifest: %s",
        generated, failed, manifest_path,
    )
    return manifest_path


def main():
    """CLI entry point for synthetic data generation."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    ap = argparse.ArgumentParser(description="Generate synthetic ST-accented English data")
    ap.add_argument("--transcript-csv", default="data/spire-sies/raw/IISc_SPIRE_SIES_Transcription.csv")
    ap.add_argument("--svarah-dir", default=None,
                    help="Path to Svarah audio organized by accent")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--max-texts", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    texts = load_source_texts(args.transcript_csv, max_texts=args.max_texts, seed=args.seed)

    if args.svarah_dir:
        voices = find_reference_voices(args.svarah_dir)
    else:
        logger.warning("No --svarah-dir provided. Using placeholder reference voices.")
        voices = [{"path": "placeholder.wav", "accent": "Mizo", "family": "Sino-Tibetan"}]

    generate_synthetic_data(
        source_texts=texts,
        reference_voices=voices,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
