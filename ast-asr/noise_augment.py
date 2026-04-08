"""
noise_augment.py
Adds synthetic noise to audio arrays at a specified SNR (dB).
Used to produce the noisy split needed for Δg_noise (Metric 3).

Noise types supported:
    "white"   — additive white Gaussian noise
    "pink"    — 1/f noise (closer to real-world ambience)
    "babble"  — sum of N random utterances from the dataset (in-corpus babble)

For SPIRE-SIES noise conditions (when access is granted), swap
in the recorded noise files via add_recorded_noise().
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _snr_scale(signal: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Scale noise so that SNR(signal, scaled_noise) == target_snr_db."""
    sig_power   = np.mean(signal ** 2) + 1e-10
    noise_power = np.mean(noise  ** 2) + 1e-10
    snr_linear  = 10 ** (target_snr_db / 10)
    scale        = np.sqrt(sig_power / (snr_linear * noise_power))
    return noise * scale


def _white_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(length).astype(np.float32)


def _pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate pink noise via 1/f spectrum shaping."""
    white = rng.standard_normal(length)
    freqs = np.fft.rfftfreq(length)
    freqs[0] = 1e-6  # avoid division by zero at DC
    spectrum = np.fft.rfft(white)
    pink_spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(pink_spectrum, n=length).astype(np.float32)
    return pink


def add_noise(
    df: pd.DataFrame,
    snr_db: float = 0.0,
    noise_type: str = "white",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns a copy of df with 'audio_array' replaced by noise-corrupted audio.
    SNR is applied per-utterance so each signal has exactly snr_db dB.

    Parameters
    ----------
    df         : DataFrame from data_loader (must have 'audio_array' column)
    snr_db     : Target SNR in dB. 0 dB is the paper's evaluation condition.
    noise_type : "white" | "pink" | "babble"
    seed       : RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    if noise_type == "babble":
        all_audio = df["audio_array"].tolist()

    noisy_arrays = []
    for i, row in df.iterrows():
        sig = row["audio_array"]
        n   = len(sig)

        if noise_type == "white":
            noise = _white_noise(n, rng)
        elif noise_type == "pink":
            noise = _pink_noise(n, rng)
        elif noise_type == "babble":
            # Mix 5 random utterances (excluding self)
            idxs  = rng.choice(len(all_audio), size=5, replace=False)
            parts = []
            for idx in idxs:
                a = all_audio[idx]
                if len(a) < n:
                    a = np.pad(a, (0, n - len(a)))
                parts.append(a[:n])
            noise = np.mean(parts, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unknown noise_type '{noise_type}'")

        scaled_noise = _snr_scale(sig, noise, snr_db)
        noisy = np.clip(sig + scaled_noise, -1.0, 1.0)
        noisy_arrays.append(noisy)

    out["audio_array"] = noisy_arrays
    logger.info(
        "Added %s noise at %d dB SNR to %d utterances", noise_type, snr_db, len(df)
    )
    return out