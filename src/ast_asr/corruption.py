"""Deterministic acoustic corruptions with measured SNR."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class NoisePair:
    clean: torch.Tensor
    noisy: torch.Tensor
    snr_db: float


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, *, snr_db: float) -> torch.Tensor:
    """Scale ``noise`` to a measured power SNR and add it to ``clean``."""
    if clean.ndim != 1 or noise.shape != clean.shape:
        raise ValueError("clean and noise must be aligned mono waveforms")
    clean_power = clean.float().square().mean()
    noise = noise.float() - noise.float().mean()
    noise_power = noise.square().mean()
    if clean_power <= 0 or noise_power <= 0:
        raise ValueError("clean and noise waveforms must have non-zero power")
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scaled_noise = noise * torch.sqrt(target_noise_power / noise_power)
    return clean + scaled_noise.to(dtype=clean.dtype)


def paired_white_noise(
    audio: torch.Tensor,
    *,
    seed: int,
    minimum_snr_db: float = 10.0,
    maximum_snr_db: float = 20.0,
) -> NoisePair:
    """Return a clean/noisy pair with SNR sampled uniformly in dB."""
    if audio.ndim != 1:
        raise ValueError("audio must be a mono waveform")
    if minimum_snr_db > maximum_snr_db:
        raise ValueError("minimum SNR cannot exceed maximum SNR")
    generator = torch.Generator(device=audio.device)
    generator.manual_seed(seed)
    fraction = torch.rand((), generator=generator, device=audio.device).item()
    snr_db = minimum_snr_db + fraction * (maximum_snr_db - minimum_snr_db)
    white = torch.randn(
        audio.shape,
        generator=generator,
        device=audio.device,
        dtype=torch.float32,
    )
    return NoisePair(
        clean=audio.clone(),
        noisy=mix_at_snr(audio, white, snr_db=snr_db),
        snr_db=snr_db,
    )


def musan_babble(
    clean: torch.Tensor,
    speech_noise_sources: Sequence[torch.Tensor],
    *,
    seed: int,
    snr_db: float = 10.0,
) -> torch.Tensor:
    """Mix independently offset MUSAN speech tracks into babble at fixed SNR."""
    if not speech_noise_sources:
        raise ValueError("at least one MUSAN speech source is required")
    generator = torch.Generator(device=clean.device)
    generator.manual_seed(seed)
    aligned = []
    for source in speech_noise_sources:
        source = source.to(device=clean.device, dtype=torch.float32).flatten()
        if source.numel() == 0:
            raise ValueError("MUSAN speech sources cannot be empty")
        repeats = math.ceil(clean.numel() / source.numel()) + 1
        repeated = source.repeat(repeats)
        maximum_offset = repeated.numel() - clean.numel()
        offset = int(
            torch.randint(
                0,
                maximum_offset + 1,
                (),
                generator=generator,
                device=clean.device,
            ).item()
        )
        aligned.append(repeated[offset : offset + clean.numel()])
    babble = torch.stack(aligned).mean(dim=0)
    return mix_at_snr(clean, babble, snr_db=snr_db)
