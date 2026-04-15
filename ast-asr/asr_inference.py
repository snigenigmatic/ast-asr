"""
asr_inference.py
Runs ASR inference for any registered model on a DataFrame produced by data_loader.
Adds a 'hypothesis' column with the decoded transcript.

Supported model keys (pass as model_name arg):
  "whisper-tiny"     openai/whisper-tiny
  "whisper-base"     openai/whisper-base
  "whisper-small"    openai/whisper-small
  "whisper-medium"   openai/whisper-medium
  "wav2vec2-base"    facebook/wav2vec2-base-960h
  "wav2vec2-large"   facebook/wav2vec2-large-960h-lv60-self
  "hubert-large"     facebook/hubert-large-ls960-ft
"""

import logging
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Load HF_TOKEN from project .env if present (for authenticated HF API calls)
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from data_loader import _resolve_hf_token
    _resolve_hf_token()
except ImportError:
    pass

MODEL_REGISTRY = {
    "whisper-tiny":    ("openai/whisper-tiny",              "whisper"),
    "whisper-base":    ("openai/whisper-base",              "whisper"),
    "whisper-small":   ("openai/whisper-small",             "whisper"),
    "whisper-medium":  ("openai/whisper-medium",            "whisper"),
    "wav2vec2-base":   ("facebook/wav2vec2-base-960h",      "ctc"),
    "wav2vec2-large":  ("facebook/wav2vec2-large-960h-lv60-self", "ctc"),
    "hubert-large":    ("facebook/hubert-large-ls960-ft",   "ctc"),
    # Adapter checkpoints saved by train.py live under the configured output
    # dir. The first tuple entry is the default checkpoint dir; callers can
    # override via the `model_path` kwarg in run_inference().
    "hybrid-w2v2-grl": ("outputs/checkpoints/hybrid_w2v2_lora_grl/final", "hybrid"),
    "rl-grpo":         ("outputs/checkpoints/rl-grpo/final",             "hybrid"),
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        dev = torch.device("cpu")
        logger.warning("CUDA not available – running on CPU (will be slow)")
    return dev


# ── Whisper inference ─────────────────────────────────────────────────────────

def _load_whisper(hf_id: str, device: torch.device):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    logger.info("Loading Whisper model: %s", hf_id)
    processor = WhisperProcessor.from_pretrained(hf_id)
    model     = WhisperForConditionalGeneration.from_pretrained(hf_id).to(device)
    model.eval()
    return processor, model


def _infer_whisper(
    audio_arrays: list[np.ndarray],
    processor,
    model,
    device: torch.device,
    batch_size: int = 8,
) -> list[str]:
    hypotheses = []
    for i in range(0, len(audio_arrays), batch_size):
        batch = audio_arrays[i : i + batch_size]
        inputs = processor(
            batch,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features, language="en", task="transcribe")
        decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        hypotheses.extend(decoded)
    return hypotheses


# ── CTC inference (Wav2Vec2 / HuBERT) ────────────────────────────────────────

def _load_ctc(hf_id: str, device: torch.device):
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    logger.info("Loading CTC model: %s", hf_id)
    processor = Wav2Vec2Processor.from_pretrained(hf_id)
    model     = Wav2Vec2ForCTC.from_pretrained(hf_id).to(device)
    model.eval()
    return processor, model


def _infer_ctc(
    audio_arrays: list[np.ndarray],
    processor,
    model,
    device: torch.device,
    batch_size: int = 8,
) -> list[str]:
    hypotheses = []
    for i in range(0, len(audio_arrays), batch_size):
        batch = audio_arrays[i : i + batch_size]
        # Pad to equal length manually
        max_len = max(a.shape[0] for a in batch)
        padded  = [np.pad(a, (0, max_len - a.shape[0])) for a in batch]
        inputs  = processor(
            padded,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded = processor.batch_decode(predicted_ids)
        hypotheses.extend(decoded)
    return hypotheses


# ── Hybrid (LoRA + GRL) inference ────────────────────────────────────────────

def _load_hybrid(ckpt_dir: str, device: torch.device):
    """
    Load a trained HybridAdversarialASR checkpoint. Returns a
    (processor, model) tuple that behaves like `_load_ctc` for the inference
    path — the adversary head is untouched at inference time.
    """
    import json
    import os
    from pathlib import Path

    import torch as _torch
    from transformers import Wav2Vec2Processor

    # Import from the models package at the top of the project.
    from models import HybridAdversarialASR, HybridConfig  # noqa: E402

    ckpt = Path(ckpt_dir)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Hybrid checkpoint not found at {ckpt}. Train with ast-asr/train.py first."
        )
    logger.info("Loading hybrid (Wav2Vec2 + LoRA + GRL) checkpoint: %s", ckpt)

    # Look for hybrid_config.json in the checkpoint dir first, then the
    # parent (train.py historically wrote it one level up).
    cfg_candidates = [ckpt / "hybrid_config.json", ckpt.parent / "hybrid_config.json"]
    cfg_dict: dict | None = None
    for cand in cfg_candidates:
        if cand.exists():
            with open(cand) as f:
                cfg_dict = json.load(f)
            break

    if cfg_dict is not None:
        cfg_dict["lora_target_modules"] = tuple(cfg_dict.get("lora_target_modules", ()))
        hybrid_cfg = HybridConfig(**cfg_dict)
    else:
        # Fall back: at minimum we need the right adversary size, so read
        # family_to_id.json to determine num_accent_classes.
        fam_path = ckpt / "family_to_id.json"
        if fam_path.exists():
            with open(fam_path) as f:
                fam_map = json.load(f)
            hybrid_cfg = HybridConfig(num_accent_classes=len(fam_map))
        else:
            hybrid_cfg = HybridConfig()

    model = HybridAdversarialASR(hybrid_cfg)
    model.load_adapter(str(ckpt))
    model.to(device).eval()

    # Prefer the processor saved alongside the checkpoint; otherwise fall
    # back to the base Wav2Vec2 processor.
    proc_dir = ckpt if (ckpt / "tokenizer_config.json").exists() else hybrid_cfg.wav2vec2_id
    processor = Wav2Vec2Processor.from_pretrained(proc_dir)
    return processor, model


def _infer_hybrid(
    audio_arrays: list[np.ndarray],
    processor,
    model,
    device: torch.device,
    batch_size: int = 8,
) -> list[str]:
    """CTC argmax decoding on the hybrid model's ASR branch."""
    hypotheses = []
    for i in range(0, len(audio_arrays), batch_size):
        batch = audio_arrays[i : i + batch_size]
        max_len = max(a.shape[0] for a in batch)
        padded = [np.pad(a, (0, max_len - a.shape[0])) for a in batch]
        inputs = processor(
            padded,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)
        attention_mask = torch.ones_like(input_values, dtype=torch.long)
        with torch.no_grad():
            logits = model.asr(
                input_values=input_values,
                attention_mask=attention_mask,
            ).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded = processor.batch_decode(predicted_ids)
        hypotheses.extend(decoded)
    return hypotheses


# ── Public API ────────────────────────────────────────────────────────────────

def run_inference(
    df: pd.DataFrame,
    model_name: str = "whisper-tiny",
    batch_size: int = 8,
    device: torch.device | None = None,
    model_path: str | None = None,
) -> pd.DataFrame:
    """
    Takes the DataFrame from data_loader and adds a 'hypothesis' column.
    Returns a copy of df with the new column.

    `model_path` overrides the default checkpoint directory for hybrid models.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    hf_id, model_type = MODEL_REGISTRY[model_name]
    device = device or get_device()

    if model_type == "whisper":
        processor, model = _load_whisper(hf_id, device)
        infer_fn = lambda arrays: _infer_whisper(
            arrays, processor, model, device, batch_size
        )
    elif model_type == "hybrid":
        ckpt_dir = model_path or hf_id
        processor, model = _load_hybrid(ckpt_dir, device)
        infer_fn = lambda arrays: _infer_hybrid(
            arrays, processor, model, device, batch_size
        )
    else:
        processor, model = _load_ctc(hf_id, device)
        infer_fn = lambda arrays: _infer_ctc(
            arrays, processor, model, device, batch_size
        )

    audio_arrays = df["audio_array"].tolist()

    logger.info(
        "Running inference with %s on %d utterances (batch_size=%d) …",
        model_name, len(audio_arrays), batch_size,
    )
    t0 = time.time()
    hypotheses = infer_fn(audio_arrays)
    elapsed = time.time() - t0
    logger.info("Inference done in %.1f s (%.2f s/utt)", elapsed, elapsed / len(audio_arrays))

    out = df.copy()
    out["hypothesis"] = hypotheses
    out["model"] = model_name
    return out