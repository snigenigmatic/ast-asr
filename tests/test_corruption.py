from __future__ import annotations

import math

import pytest
import torch

from ast_asr.corruption import paired_white_noise


def test_paired_white_noise_is_seeded_and_within_requested_snr() -> None:
    audio = torch.sin(torch.linspace(0, 40 * math.pi, 16_000))

    first = paired_white_noise(audio, seed=17, minimum_snr_db=10.0, maximum_snr_db=20.0)
    second = paired_white_noise(audio, seed=17, minimum_snr_db=10.0, maximum_snr_db=20.0)
    noise = first.noisy - first.clean
    measured_snr = 10 * torch.log10(audio.square().mean() / noise.square().mean())

    torch.testing.assert_close(first.clean, audio)
    torch.testing.assert_close(first.noisy, second.noisy)
    assert measured_snr.item() == pytest.approx(first.snr_db, abs=1e-4)
    assert 10.0 <= first.snr_db <= 20.0
