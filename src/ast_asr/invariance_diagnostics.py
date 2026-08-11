"""Focused inference-invariance diagnostics for saved Whisper adapters."""

from __future__ import annotations

import argparse

import torch

from .artifacts import write_immutable_json
from .config import ExperimentConfig
from .evaluation import _load_test_examples
from .inference import greedy_transcribe
from .modeling import build_lora_whisper, directory_content_hash, load_processor
from .sft import _load_audio


def diagnose_batch_invariance(args: argparse.Namespace) -> None:
    config = ExperimentConfig.from_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = load_processor(config.model)
    model = build_lora_whisper(
        config.model,
        adapter_checkpoint=args.checkpoint,
        trainable=False,
        device=device,
    )
    examples = _load_test_examples(config, args.fold)[: args.probe_examples]
    audios = [_load_audio(example.audio_path) for example in examples]

    solo = greedy_transcribe(model, processor, audios, batch_size=1, device=device)
    batch_first = greedy_transcribe(
        model,
        processor,
        audios,
        batch_size=args.batch_size,
        device=device,
    )
    batch_second = greedy_transcribe(
        model,
        processor,
        audios,
        batch_size=args.batch_size,
        device=device,
    )

    processor_audio = [audio.detach().float().cpu().numpy() for audio in audios]
    batch_encoded = processor(
        processor_audio,
        sampling_rate=16_000,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )
    feature_max_abs_differences = []
    attention_masks_equal = []
    for index, audio in enumerate(processor_audio):
        solo_encoded = processor(
            [audio],
            sampling_rate=16_000,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        feature_max_abs_differences.append(
            float(
                (
                    solo_encoded.input_features[0]
                    - batch_encoded.input_features[index]
                )
                .abs()
                .max()
            )
        )
        attention_masks_equal.append(
            bool(
                torch.equal(
                    solo_encoded.attention_mask[0],
                    batch_encoded.attention_mask[index],
                )
            )
        )

    model.float()
    fp32_solo = greedy_transcribe(model, processor, audios, batch_size=1, device=device)
    fp32_batch = greedy_transcribe(
        model,
        processor,
        audios,
        batch_size=args.batch_size,
        device=device,
    )
    rows = [
        {
            "utterance_id": example.utterance_id,
            "reference": example.reference,
            "solo": solo_prediction,
            "batch_first": first_prediction,
            "batch_second": second_prediction,
            "fp32_solo": fp32_solo_prediction,
            "fp32_batch": fp32_batch_prediction,
        }
        for (
            example,
            solo_prediction,
            first_prediction,
            second_prediction,
            fp32_solo_prediction,
            fp32_batch_prediction,
        ) in zip(
            examples,
            solo,
            batch_first,
            batch_second,
            fp32_solo,
            fp32_batch,
            strict=True,
        )
    ]
    report = {
        "artifact_kind": "sft_batch_invariance_reproduction",
        "checkpoint_revision": directory_content_hash(args.checkpoint),
        "model_training": model.training,
        "solo_equals_batch": solo == batch_first,
        "batch_repeat_equal": batch_first == batch_second,
        "fp32_solo_equals_batch": fp32_solo == fp32_batch,
        "feature_max_abs_differences": feature_max_abs_differences,
        "attention_masks_equal": attention_masks_equal,
        "divergent_utterances": sum(
            first != second for first, second in zip(solo, batch_first, strict=True)
        ),
        "rows": rows,
    }
    write_immutable_json(args.output, report)
