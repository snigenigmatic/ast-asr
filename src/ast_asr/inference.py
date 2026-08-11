"""Deterministic Whisper inference with explicit acoustic attention masks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .modeling import model_input_dtype, whisper_runtime_model


def greedy_transcribe(
    model: Any,
    processor: Any,
    audios: Sequence[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device | str = "cpu",
    sampling_rate: int = 16_000,
    max_new_tokens: int = 225,
) -> list[str]:
    """Transcribe audios greedily while preserving processor padding masks."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if hasattr(model, "eval"):
        model.eval()
    runtime_model = whisper_runtime_model(model)
    predictions: list[str] = []
    for start in range(0, len(audios), batch_size):
        chunk = audios[start : start + batch_size]
        processor_audio = [audio.detach().float().cpu().numpy() for audio in chunk]
        encoded = processor(
            processor_audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        input_features = encoded.input_features.to(
            device=device,
            dtype=model_input_dtype(model),
        )
        attention_mask = encoded.attention_mask.to(device)
        with torch.no_grad():
            generated = runtime_model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                language="en",
                task="transcribe",
                max_new_tokens=max_new_tokens,
            )
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        predictions.extend(
            text.strip()
            for text in processor.batch_decode(sequences, skip_special_tokens=True)
        )
    return predictions
