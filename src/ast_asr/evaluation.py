"""Clean, seen-noise, and unseen-MUSAN fold evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from .analysis import summarize_prediction_records
from .artifacts import write_immutable_json, write_immutable_text
from .config import ExperimentConfig
from .corruption import musan_babble, paired_white_noise
from .inference import greedy_transcribe
from .metrics import PredictionRecord, word_edit_counts
from .modeling import build_lora_whisper, directory_content_hash, load_processor
from .sft import AudioExample, _load_audio


def _prepare_evaluation_model(model):
    """Use FP32 evaluation to make greedy decoding invariant to batch shape."""
    model.eval()
    model.float()
    return model


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _load_test_examples(config: ExperimentConfig, fold: int) -> list[AudioExample]:
    prepared = json.loads(config.dataset.prepared_manifest.read_text(encoding="utf-8"))
    fold_manifest = json.loads(
        (config.dataset.fold_directory / f"fold-{fold}.json").read_text(encoding="utf-8")
    )
    test_speakers = set(fold_manifest["test_speakers"])
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
        if row["speaker_id"] in test_speakers
    ]
    observed = {example.speaker_id for example in examples}
    if observed != test_speakers:
        raise ValueError("test examples do not exactly cover the fold's speakers")
    return examples


def _musan_speech_files(root: Path) -> tuple[Path, ...]:
    candidates = (root / "speech", root / "musan" / "speech")
    speech_root = next((path for path in candidates if path.is_dir()), None)
    if speech_root is None:
        raise FileNotFoundError("MUSAN speech directory not found")
    files = tuple(
        sorted(
            path
            for path in speech_root.rglob("*")
            if path.suffix.lower() in {".wav", ".flac"}
        )
    )
    if len(files) < 3:
        raise ValueError("MUSAN babble evaluation requires at least three speech files")
    return files


def _noise_segment(path: Path, length: int, seed: int) -> torch.Tensor:
    import numpy as np
    import soundfile as sf

    info = sf.info(path)
    generator = np.random.default_rng(seed)
    if info.samplerate == 16_000 and info.frames >= length:
        maximum_start = max(0, info.frames - length)
        start = int(generator.integers(0, maximum_start + 1))
        waveform, _ = sf.read(
            path,
            start=start,
            frames=length,
            dtype="float32",
            always_2d=False,
        )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return torch.from_numpy(waveform)
    waveform = _load_audio(path)
    if waveform.numel() < length:
        waveform = waveform.repeat((length + waveform.numel() - 1) // waveform.numel())
    maximum_start = waveform.numel() - length
    start = int(generator.integers(0, maximum_start + 1))
    return waveform[start : start + length]


def _condition_audio(
    example: AudioExample,
    condition: str,
    *,
    musan_files: Sequence[Path],
) -> torch.Tensor:
    import numpy as np

    clean = _load_audio(example.audio_path)
    seed = _stable_seed(example.utterance_id, condition)
    if condition == "clean":
        return clean
    if condition == "white_10db":
        return paired_white_noise(
            clean,
            seed=seed,
            minimum_snr_db=10.0,
            maximum_snr_db=10.0,
        ).noisy
    if condition == "musan_babble_10db":
        generator = np.random.default_rng(seed)
        selected_indices = generator.choice(len(musan_files), size=3, replace=False)
        sources = [
            _noise_segment(musan_files[int(index)], clean.numel(), seed + offset)
            for offset, index in enumerate(selected_indices)
        ]
        return musan_babble(clean, sources, seed=seed, snr_db=10.0)
    raise ValueError(f"unknown evaluation condition: {condition}")


def evaluate_fold(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_json(args.config)
    write_immutable_json(args.output_dir / "resolved_config.json", config.to_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = load_processor(config.model)
    if str(args.checkpoint).lower() == "base":
        model = build_lora_whisper(config.model, trainable=False, device=device)
        checkpoint_revision = f"base@{config.model.revision}"
    else:
        model = build_lora_whisper(
            config.model,
            adapter_checkpoint=args.checkpoint,
            trainable=False,
            device=device,
        )
        checkpoint_revision = directory_content_hash(args.checkpoint)
    model = _prepare_evaluation_model(model)
    examples = _load_test_examples(config, args.fold)
    musan_files = (
        _musan_speech_files(config.dataset.musan_root)
        if "musan_babble_10db" in config.evaluation.conditions
        else ()
    )
    records = []
    batch_size = config.sft.evaluation_batch_size
    for condition in config.evaluation.conditions:
        predictions = []
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            audios = [
                _condition_audio(example, condition, musan_files=musan_files)
                for example in chunk
            ]
            predictions.extend(
                greedy_transcribe(
                    model,
                    processor,
                    audios,
                    batch_size=batch_size,
                    device=device,
                )
            )
        records.extend(
            PredictionRecord(
                fold=args.fold,
                utterance_id=example.utterance_id,
                speaker_id=example.speaker_id,
                primary_language=example.primary_language,
                family=example.family,
                condition=condition,
                reference=example.reference,
                hypothesis=prediction,
                checkpoint_revision=checkpoint_revision,
                arm=args.arm,
            )
            for example, prediction in zip(examples, predictions, strict=True)
        )

    clean_by_utterance = {
        record.utterance_id: record.hypothesis
        for record in records
        if record.condition == "clean"
    }
    invariance_probe = examples[: min(8, len(examples))]
    solo_predictions = greedy_transcribe(
        model,
        processor,
        [_load_audio(example.audio_path) for example in invariance_probe],
        batch_size=1,
        device=device,
    )
    expected_probe = [clean_by_utterance[example.utterance_id] for example in invariance_probe]
    if solo_predictions != expected_probe:
        raise RuntimeError("solo and batched greedy predictions differ")

    output = args.output_dir
    lines = "".join(
        json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
        for record in records
    )
    write_immutable_text(output / "predictions.jsonl", lines)
    write_immutable_json(
        output / "edit_counts.json",
        [
            {
                "utterance_id": record.utterance_id,
                "speaker_id": record.speaker_id,
                "condition": record.condition,
                **asdict(word_edit_counts(record.reference, record.hypothesis)),
            }
            for record in records
        ],
    )
    write_immutable_json(output / "metrics.json", summarize_prediction_records(records))
    write_immutable_json(
        output / "run.json",
        {
            "fold": args.fold,
            "arm": args.arm,
            "checkpoint": str(args.checkpoint),
            "checkpoint_revision": checkpoint_revision,
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "dataset_id": config.dataset.dataset_id,
            "dataset_revision": config.dataset.revision,
            "musan_revision": config.dataset.musan_revision,
            "conditions": list(config.evaluation.conditions),
            "prediction_count": len(records),
            "solo_batched_predictions_equal": True,
        },
    )
