"""Fold-specific LoRA supervised fine-tuning for Whisper-tiny."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .artifacts import write_immutable_json
from .config import ExperimentConfig
from .gates import require_development_gate
from .inference import greedy_transcribe
from .metrics import EditCounts, word_edit_counts
from .modeling import (
    build_lora_whisper,
    directory_content_hash,
    load_processor,
    load_saved_processor,
    model_input_dtype,
    whisper_runtime_model,
)


@dataclass(frozen=True, slots=True)
class AudioExample:
    utterance_id: str
    speaker_id: str
    family: str
    primary_language: str
    audio_path: Path
    reference: str


class _AudioDataset(Dataset):
    def __init__(self, examples: Sequence[AudioExample]) -> None:
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> AudioExample:
        return self.examples[index]


def _load_audio(path: Path) -> torch.Tensor:
    import numpy as np
    import soundfile as sf

    waveform, sampling_rate = sf.read(path, dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sampling_rate != 16_000:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=16_000)
    return torch.from_numpy(np.asarray(waveform, dtype=np.float32))


class WhisperSftCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, examples: Sequence[AudioExample]) -> dict[str, torch.Tensor]:
        audios = [_load_audio(example.audio_path).numpy() for example in examples]
        acoustic = self.processor.feature_extractor(
            audios,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        tokenized = self.processor.tokenizer(
            [example.reference for example in examples],
            return_tensors="pt",
            padding=True,
        )
        labels = tokenized.input_ids.masked_fill(tokenized.attention_mask.ne(1), -100)
        decoder_start = self.processor.tokenizer.bos_token_id
        if labels.shape[1] > 0 and torch.all(labels[:, 0] == decoder_start):
            labels = labels[:, 1:]
        return {
            "input_features": acoustic.input_features,
            "attention_mask": acoustic.attention_mask,
            "labels": labels,
        }


def _load_experiment_records(
    config: ExperimentConfig,
    fold: int,
) -> tuple[list[AudioExample], list[AudioExample]]:
    prepared = json.loads(config.dataset.prepared_manifest.read_text(encoding="utf-8"))
    if prepared["dataset_revision"] != config.dataset.revision:
        raise ValueError("prepared data revision differs from experiment configuration")
    fold_manifest = json.loads(
        (config.dataset.fold_directory / f"fold-{fold}.json").read_text(encoding="utf-8")
    )
    train_speakers = set(fold_manifest["train_speakers"])
    validation_speakers = set(fold_manifest["validation_speakers"])
    examples = [
        AudioExample(
            utterance_id=row["utterance_id"],
            speaker_id=row["speaker_id"],
            family=row["family"],
            primary_language=row["primary_language"],
            audio_path=config.dataset.archive_root / row["audio_path"],
            reference=row["reference"],
        )
        for row in prepared["utterances"]
    ]
    train = [example for example in examples if example.speaker_id in train_speakers]
    validation = [
        example for example in examples if example.speaker_id in validation_speakers
    ]
    if not train or not validation:
        raise ValueError("fold produced an empty SFT train or validation partition")
    return train, validation


def _macro_family_wer(
    examples: Sequence[AudioExample],
    predictions: Sequence[str],
) -> tuple[float, dict[str, float]]:
    counts: dict[str, EditCounts] = defaultdict(EditCounts)
    for example, prediction in zip(examples, predictions, strict=True):
        counts[example.family] = counts[example.family] + word_edit_counts(
            example.reference,
            prediction,
        )
    family_wers = {family: value.wer for family, value in sorted(counts.items())}
    return sum(family_wers.values()) / len(family_wers), family_wers


def _validation_predictions(
    model: Any,
    processor: Any,
    examples: Sequence[AudioExample],
    *,
    batch_size: int,
    device: torch.device,
    max_new_tokens: int = 225,
) -> list[str]:
    audios = [_load_audio(example.audio_path) for example in examples]
    return greedy_transcribe(
        model,
        processor,
        audios,
        batch_size=batch_size,
        device=device,
        max_new_tokens=max_new_tokens,
    )


def _balanced_limit(
    examples: Sequence[AudioExample],
    maximum: int | None,
    *,
    seed: int,
) -> list[AudioExample]:
    if maximum is None or maximum >= len(examples):
        return list(examples)
    families = sorted({example.family for example in examples})
    if maximum < len(families):
        raise ValueError("bounded samples must retain at least one example per family")
    grouped: dict[str, list[AudioExample]] = defaultdict(list)
    for example in examples:
        grouped[example.family].append(example)
    generator = random.Random(seed)
    for values in grouped.values():
        generator.shuffle(values)
    selected = []
    while len(selected) < maximum:
        made_progress = False
        for family in families:
            if grouped[family]:
                selected.append(grouped[family].pop())
                made_progress = True
                if len(selected) == maximum:
                    break
        if not made_progress:
            break
    return selected


def _seed_everything(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_sft(args: argparse.Namespace) -> None:
    require_development_gate(args.fold, args.development_gate)
    config = ExperimentConfig.from_json(args.config)
    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_examples, validation_examples = _load_experiment_records(config, args.fold)
    train_examples = _balanced_limit(
        train_examples,
        args.max_train_examples,
        seed=args.seed,
    )
    validation_examples = _balanced_limit(
        validation_examples,
        args.max_validation_examples,
        seed=args.seed + 1,
    )
    maximum_epochs = (
        config.sft.maximum_epochs if args.maximum_epochs is None else args.maximum_epochs
    )
    if not 1 <= maximum_epochs <= config.sft.maximum_epochs:
        raise ValueError("maximum epochs override must be within the configured protocol")
    if args.max_optimizer_steps is not None and args.max_optimizer_steps < 1:
        raise ValueError("max optimizer steps must be positive")
    maximum_new_tokens = (
        config.policy.maximum_new_tokens
        if args.maximum_new_tokens is None
        else args.maximum_new_tokens
    )
    if not 1 <= maximum_new_tokens <= config.policy.maximum_new_tokens:
        raise ValueError("maximum new tokens override exceeds the configured protocol")
    processor = load_processor(config.model)
    model = build_lora_whisper(config.model, device=device)
    runtime_model = whisper_runtime_model(model)
    model.print_trainable_parameters()
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        _AudioDataset(train_examples),
        batch_size=config.sft.train_batch_size,
        shuffle=True,
        generator=loader_generator,
        collate_fn=WhisperSftCollator(processor),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.sft.learning_rate,
    )
    total_updates = math.ceil(
        len(loader) / config.sft.gradient_accumulation_steps
    ) * maximum_epochs
    if args.max_optimizer_steps is not None:
        total_updates = min(total_updates, args.max_optimizer_steps)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.sft.warmup_ratio * total_updates),
        num_training_steps=total_updates,
    )
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite SFT run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_immutable_json(output / "resolved_config.json", config.to_dict())
    execution_overrides = {
        "maximum_epochs": maximum_epochs,
        "max_train_examples": args.max_train_examples,
        "max_validation_examples": args.max_validation_examples,
        "max_optimizer_steps": args.max_optimizer_steps,
        "maximum_new_tokens": maximum_new_tokens,
    }
    write_immutable_json(output / "execution_overrides.json", execution_overrides)

    history = []
    best_macro = float("inf")
    best_checkpoint: Path | None = None
    best_predictions: list[str] | None = None
    optimizer.zero_grad(set_to_none=True)
    optimizer_updates = 0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        running_loss = 0.0
        batches_seen = 0
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            batch["input_features"] = batch["input_features"].to(
                dtype=model_input_dtype(model)
            )
            result = runtime_model(**batch)
            loss = result.loss / config.sft.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite SFT loss at epoch {epoch}, step {step}")
            loss.backward()
            batches_seen += 1
            running_loss += float(loss.detach().cpu()) * config.sft.gradient_accumulation_steps
            should_step = (
                step % config.sft.gradient_accumulation_steps == 0 or step == len(loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                if (
                    args.max_optimizer_steps is not None
                    and optimizer_updates >= args.max_optimizer_steps
                ):
                    break

        predictions = _validation_predictions(
            model,
            processor,
            validation_examples,
            batch_size=config.sft.evaluation_batch_size,
            device=device,
            max_new_tokens=maximum_new_tokens,
        )
        macro_wer, family_wers = _macro_family_wer(validation_examples, predictions)
        checkpoint = output / f"checkpoint-epoch-{epoch}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        processor.save_pretrained(checkpoint / "processor")
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / batches_seen,
                "optimizer_updates": optimizer_updates,
                "validation_macro_family_wer": macro_wer,
                "validation_family_wer": family_wers,
                "checkpoint": checkpoint.name,
                "checkpoint_hash": directory_content_hash(checkpoint),
            }
        )
        if macro_wer < best_macro:
            best_macro = macro_wer
            best_checkpoint = checkpoint
            best_predictions = predictions
        if (
            args.max_optimizer_steps is not None
            and optimizer_updates >= args.max_optimizer_steps
        ):
            break

    assert best_checkpoint is not None and best_predictions is not None
    reloaded = build_lora_whisper(
        config.model,
        adapter_checkpoint=best_checkpoint,
        trainable=False,
        device=device,
    )
    reloaded_processor = load_saved_processor(best_checkpoint)
    reloaded_predictions = _validation_predictions(
        reloaded,
        reloaded_processor,
        validation_examples,
        batch_size=config.sft.evaluation_batch_size,
        device=device,
        max_new_tokens=maximum_new_tokens,
    )
    if reloaded_predictions != best_predictions:
        raise RuntimeError("checkpoint round-trip changed saved validation predictions")

    checkpoint_revision = directory_content_hash(best_checkpoint)
    write_immutable_json(
        output / "run.json",
        {
            "arm": "sft",
            "fold": args.fold,
            "seed": args.seed,
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "dataset_id": config.dataset.dataset_id,
            "dataset_revision": config.dataset.revision,
            "best_checkpoint": best_checkpoint.name,
            "best_checkpoint_revision": checkpoint_revision,
            "selection_metric": "validation_macro_family_wer",
            "selection_value": best_macro,
            "execution_mode": (
                "bounded_smoke"
                if any(
                    value is not None
                    for value in (
                        args.maximum_epochs,
                        args.max_train_examples,
                        args.max_validation_examples,
                        args.max_optimizer_steps,
                        args.maximum_new_tokens,
                    )
                )
                else "protocol"
            ),
            "execution_overrides": execution_overrides,
            "history": history,
            "round_trip_predictions_equal": True,
        },
    )
    write_immutable_json(
        output / "validation_predictions.json",
        [
            {
                "utterance_id": example.utterance_id,
                "speaker_id": example.speaker_id,
                "family": example.family,
                "primary_language": example.primary_language,
                "reference": example.reference,
                "hypothesis": prediction,
                "checkpoint_revision": checkpoint_revision,
            }
            for example, prediction in zip(
                validation_examples,
                best_predictions,
                strict=True,
            )
        ],
    )
