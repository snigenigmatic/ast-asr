"""Pinned Whisper-tiny and LoRA construction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig


def load_processor(config: ModelConfig) -> Any:
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(
        config.model_id,
        revision=config.revision,
    )
    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")
    return processor


def load_saved_processor(checkpoint: Path) -> Any:
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained(checkpoint / "processor")


def build_lora_whisper(
    config: ModelConfig,
    *,
    adapter_checkpoint: Path | None = None,
    trainable: bool = True,
    device: torch.device | str | None = None,
) -> Any:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import WhisperForConditionalGeneration

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = WhisperForConditionalGeneration.from_pretrained(
        config.model_id,
        revision=config.revision,
        torch_dtype=dtype,
    )
    base.config.use_cache = False
    for dropout in (
        module for module in base.modules() if isinstance(module, torch.nn.Dropout)
    ):
        dropout.p = 0.0
    base.generation_config.language = "en"
    base.generation_config.task = "transcribe"
    base.generation_config.forced_decoder_ids = None
    if adapter_checkpoint is None:
        lora = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(base, lora)
    else:
        model = PeftModel.from_pretrained(
            base,
            adapter_checkpoint,
            is_trainable=trainable,
        )
    if device is not None:
        model.to(device)
    return model


def trainable_parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def whisper_runtime_model(model: Any) -> Any:
    """Bypass PEFT's text-seq2seq argument adapter while retaining LoRA layers."""
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if get_base_model is not None else model


def model_input_dtype(model: Any) -> torch.dtype:
    """Return the floating-point dtype expected by Whisper's acoustic encoder."""
    runtime = whisper_runtime_model(model)
    try:
        return next(runtime.parameters()).dtype
    except StopIteration as error:
        raise ValueError("cannot infer input dtype from a parameterless model") from error


def directory_content_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    files = (
        (path.relative_to(directory).as_posix(), path)
        for path in directory.rglob("*")
        if path.is_file()
    )
    for relative_key, path in sorted(files, key=lambda item: item[0]):
        digest.update(relative_key.encode("utf-8"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
