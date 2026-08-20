"""Cross-corpus SPIRE evaluation driver, run inside the project virtualenv.

Invoked by scripts/modal_spire_eval.py via `uv run`, so that ast_asr and torch
resolve from the project environment. Decoding is greedy and FP32, matching
src/ast_asr/evaluation.py, so numbers are comparable with Svarah results.

Registered contract: experiments/SPIRE-crosscorpus/protocol.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spire_crosscorpus as contract

from ast_asr.config import ExperimentConfig
from ast_asr.inference import greedy_transcribe
from ast_asr.modeling import (
    build_lora_whisper,
    directory_content_hash,
    load_processor,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _utterance_from_manifest(record: dict[str, str]) -> contract.SpireUtterance:
    """Rebuild the accepted utterance, re-verifying the family mapping."""
    return contract.SpireUtterance(
        uid=record["uid"],
        speaker_id=record["speaker_id"],
        language=record["accent"],
        family=contract.resolve_family(record["accent"], record["language_family"]),
        gender=record["gender"] or "Unknown",
        duration=float(record["duration"]),
    )


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite evaluation run: {args.output_dir}"
        )

    config = ExperimentConfig.from_json(args.config)
    records = _load_manifest(args.manifest)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("manifest contained no rows")

    utterances = [_utterance_from_manifest(record) for record in records]
    audios: list[torch.Tensor] = []
    for record in records:
        waveform, sample_rate = sf.read(
            args.audio_root / record["path"],
            dtype="float32",
            always_2d=False,
        )
        if sample_rate != contract.TARGET_SAMPLE_RATE:
            raise RuntimeError(
                f"expected {contract.TARGET_SAMPLE_RATE} Hz audio, got {sample_rate}"
            )
        audios.append(torch.from_numpy(waveform))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = load_processor(config.model)
    model = build_lora_whisper(
        config.model,
        adapter_checkpoint=args.adapter_checkpoint,
        trainable=False,
        device=device,
    )
    # Mirrors evaluation._prepare_evaluation_model: FP32 makes greedy decoding
    # invariant to batch shape, which an earlier FP16 audit proved it is not.
    model.eval()
    model.float()

    batch_size = args.batch_size or config.sft.evaluation_batch_size
    hypotheses = greedy_transcribe(
        model,
        processor,
        audios,
        batch_size=batch_size,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    if len(hypotheses) != len(records):
        raise RuntimeError("transcription returned the wrong number of hypotheses")

    results = [
        contract.score_utterance(utterance, record["reference"], hypothesis)
        for utterance, record, hypothesis in zip(
            utterances, records, hypotheses, strict=True
        )
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for record, result, hypothesis in zip(
            records, results, hypotheses, strict=True
        ):
            handle.write(
                json.dumps(
                    {
                        "uid": result.uid,
                        "speaker_id": result.speaker_id,
                        "accent": record["accent"],
                        "family": result.family,
                        "gender": result.gender,
                        "reference": record["reference"],
                        "hypothesis": hypothesis,
                        "substitutions": result.counts.substitutions,
                        "deletions": result.counts.deletions,
                        "insertions": result.counts.insertions,
                        "reference_words": result.counts.reference_words,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    metrics: dict[str, object] = {
        "artifact_kind": "spire_crosscorpus_evaluation",
        "arm": args.arm,
        "corpus": "spire-sies",
        "corpus_role": "evaluation_only_never_trained_on",
        "split": "val",
        "cluster_unit": "corpus_speaker",
        "condition": "clean",
        "decoding": "greedy_fp32",
        "max_new_tokens": args.max_new_tokens,
        "batch_size": batch_size,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "adapter_checkpoint": (
            str(args.adapter_checkpoint) if args.adapter_checkpoint else None
        ),
        "adapter_checkpoint_revision": (
            directory_content_hash(args.adapter_checkpoint)
            if args.adapter_checkpoint
            else None
        ),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "endpoints": contract.summarize_arm(results),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics["endpoints"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
