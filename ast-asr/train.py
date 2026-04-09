"""
train.py
Train a HybridAdversarialASR (Wav2Vec2 + LoRA + GRL adversary) on the
IISc SPIRE-SIES corpus. Evaluates each epoch on a held-out speaker split
and reports CTC loss, adversary loss, and per-family validation WER.

Usage:
    python ast-asr/train.py --config configs/train_hybrid.yaml
    python ast-asr/train.py --config configs/train_hybrid.yaml --max-train-samples 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import HybridAdversarialASR, HybridConfig  # noqa: E402
from spire_loader import (  # noqa: E402
    apply_split,
    build_manifest,
    make_speaker_split,
)

logger = logging.getLogger(__name__)

TARGET_SR = 16_000


# ── Dataset ──────────────────────────────────────────────────────────────────

class SpireAsrDataset(Dataset):
    """
    Yields pre-processed training samples from a manifest DataFrame.
    Audio is decoded lazily (on __getitem__) via soundfile + librosa.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        processor,  # Wav2Vec2Processor
        family_to_id: dict[str, int],
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.processor = processor
        self.family_to_id = family_to_id

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import librosa
        import soundfile as sf

        row = self.manifest.iloc[idx]
        waveform, sr = sf.read(row["path"], dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1, dtype=np.float32)
        if sr != TARGET_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

        audio_in = self.processor(
            waveform, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_values[0]

        labels = self.processor.tokenizer(
            row["reference"], return_tensors="pt"
        ).input_ids[0]

        return {
            "input_values": audio_in,
            "labels": labels,
            "accent_label": int(self.family_to_id[row["language_family"]]),
        }


# ── Collator ────────────────────────────────────────────────────────────────

class HybridDataCollator:
    """Pad `input_values` and `labels` (with -100) within a batch."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]
        accent_labels = torch.tensor(
            [f["accent_label"] for f in features], dtype=torch.long
        )

        max_in = max(x.size(0) for x in input_values)
        padded_in = torch.zeros(len(input_values), max_in, dtype=input_values[0].dtype)
        attention_mask = torch.zeros(len(input_values), max_in, dtype=torch.long)
        for i, x in enumerate(input_values):
            padded_in[i, : x.size(0)] = x
            attention_mask[i, : x.size(0)] = 1

        max_lab = max(x.size(0) for x in labels)
        padded_lab = torch.full(
            (len(labels), max_lab), fill_value=-100, dtype=labels[0].dtype
        )
        for i, x in enumerate(labels):
            padded_lab[i, : x.size(0)] = x

        return {
            "input_values": padded_in,
            "attention_mask": attention_mask,
            "labels": padded_lab,
            "accent_labels": accent_labels,
        }


# ── Lambda scheduling callback ───────────────────────────────────────────────

def _make_lambda_callback(target_lambda: float, warmup_fraction: float):
    from transformers import TrainerCallback

    class LambdaScheduleCallback(TrainerCallback):
        def on_train_begin(self, _args, _state, _control, model=None, **_kwargs):
            if model is not None:
                model.lambda_adv = 0.0

        def on_step_begin(self, _args, state, _control, model=None, **_kwargs):
            if model is None or state.max_steps <= 0:
                return
            warmup_steps = max(1, int(state.max_steps * warmup_fraction))
            if state.global_step >= warmup_steps:
                model.lambda_adv = target_lambda
            else:
                model.lambda_adv = target_lambda * (state.global_step / warmup_steps)

    return LambdaScheduleCallback()


# ── Trainer subclass ─────────────────────────────────────────────────────────

def _make_trainer_cls():
    from transformers import Trainer

    class HybridTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):  # type: ignore[override]
            outputs = model(**inputs)
            loss = outputs["loss"]
            if self.state.global_step % max(1, self.args.logging_steps) == 0:
                self.log({
                    "ctc_loss": float(outputs["ctc_loss"].detach().cpu()),
                    "adv_loss": float(outputs["adv_loss"].detach().cpu()),
                    "lambda_adv": float(model.lambda_adv),
                })
            return (loss, outputs) if return_outputs else loss

    return HybridTrainer


# ── Main ─────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_manifest_and_split(
    data_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = build_manifest(
        raw_root=data_cfg["raw_root"],
        languages=data_cfg.get("languages"),
        min_duration=data_cfg.get("min_duration", 1.0),
        max_duration=data_cfg.get("max_duration", 20.0),
        min_words=data_cfg.get("min_words", 3),
    )
    split = make_speaker_split(
        manifest,
        val_ratio=data_cfg.get("val_ratio", 0.15),
        seed=data_cfg.get("seed", 42),
        split_path=data_cfg.get("split_path"),
    )
    train_df = apply_split(manifest, split, "train")
    val_df = apply_split(manifest, split, "val")
    return manifest, train_df, val_df


def _derive_family_map(manifest: pd.DataFrame) -> dict[str, int]:
    families = sorted(manifest["language_family"].unique().tolist())
    return {fam: i for i, fam in enumerate(families)}


def train(config_path: str, max_train_samples: int | None, max_eval_samples: int | None) -> None:
    cfg = _load_config(config_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Data -----------------------------------------------------------------
    manifest, train_df, val_df = _build_manifest_and_split(cfg["data"])
    # Derive family map from the FULL manifest so tiny debugging subsets
    # still produce the same label space as a real run.
    family_to_id = _derive_family_map(manifest)
    if max_train_samples:
        # Take a balanced slice across families so both adversary classes
        # appear in the tiny debug set.
        parts = []
        per_fam = max(1, max_train_samples // max(1, len(family_to_id)))
        for fam in family_to_id:
            parts.append(train_df[train_df["language_family"] == fam].head(per_fam))
        train_df = pd.concat(parts).reset_index(drop=True)
    if max_eval_samples:
        parts = []
        per_fam = max(1, max_eval_samples // max(1, len(family_to_id)))
        for fam in family_to_id:
            parts.append(val_df[val_df["language_family"] == fam].head(per_fam))
        val_df = pd.concat(parts).reset_index(drop=True)
    logger.info("family_to_id = %s", family_to_id)
    logger.info(
        "Train family distribution: %s",
        train_df["language_family"].value_counts().to_dict(),
    )
    logger.info(
        "Val family distribution: %s",
        val_df["language_family"].value_counts().to_dict(),
    )

    # Processor + model ----------------------------------------------------
    from transformers import Wav2Vec2Processor

    m_cfg = cfg["model"]
    wav2vec2_id = m_cfg["wav2vec2_id"]
    processor = Wav2Vec2Processor.from_pretrained(wav2vec2_id)

    hybrid_cfg = HybridConfig(
        wav2vec2_id=wav2vec2_id,
        num_accent_classes=len(family_to_id),
        lambda_adv=0.0,  # callback will schedule
        adversary_hidden=cfg["adversary"]["hidden"],
        adversary_dropout=cfg["adversary"]["dropout"],
        lora_r=m_cfg["lora_r"],
        lora_alpha=m_cfg["lora_alpha"],
        lora_dropout=m_cfg["lora_dropout"],
        lora_target_modules=tuple(m_cfg["lora_target_modules"]),
        freeze_feature_extractor=m_cfg.get("freeze_feature_extractor", True),
    )
    model = HybridAdversarialASR(hybrid_cfg)

    train_ds = SpireAsrDataset(train_df, processor, family_to_id)
    eval_ds = SpireAsrDataset(val_df, processor, family_to_id)
    collator = HybridDataCollator(processor)

    # Training arguments ---------------------------------------------------
    from transformers import TrainingArguments

    t_cfg = cfg["training"]
    output_dir = Path(t_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the family map alongside the checkpoint for eval time.
    with open(output_dir / "family_to_id.json", "w") as f:
        json.dump(family_to_id, f, indent=2)
    with open(output_dir / "hybrid_config.json", "w") as f:
        json.dump(asdict(hybrid_cfg), f, indent=2)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=t_cfg["learning_rate"],
        warmup_ratio=t_cfg.get("warmup_ratio", 0.1),
        weight_decay=t_cfg.get("weight_decay", 0.0),
        bf16=t_cfg.get("bf16", False),
        fp16=t_cfg.get("fp16", False),
        logging_steps=t_cfg.get("logging_steps", 50),
        eval_strategy="steps",
        eval_steps=t_cfg.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=t_cfg.get("save_steps", 500),
        save_total_limit=t_cfg.get("save_total_limit", 5),
        load_best_model_at_end=t_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=t_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=t_cfg.get("greater_is_better", False),
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 4),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", False),
        report_to=t_cfg.get("report_to", []),
        remove_unused_columns=False,
        label_names=["labels", "accent_labels"],
    )

    # Separate LR for the adversary (via param groups inside optimizer) ----
    TrainerCls = _make_trainer_cls()
    lambda_cb = _make_lambda_callback(
        target_lambda=cfg["grl"]["target_lambda"],
        warmup_fraction=cfg["grl"].get("warmup_steps_fraction", 0.2),
    )

    # Build optimizer with two param groups (main LR, adversary LR).
    from torch.optim import AdamW

    main_params, adv_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "adversary" in name:
            adv_params.append(p)
        else:
            main_params.append(p)
    optimizer = AdamW(
        [
            {"params": main_params, "lr": t_cfg["learning_rate"]},
            {"params": adv_params, "lr": t_cfg.get("adversary_learning_rate", t_cfg["learning_rate"])},
        ],
        weight_decay=t_cfg.get("weight_decay", 0.0),
    )

    trainer = TrainerCls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[lambda_cb],
        optimizers=(optimizer, None),
    )

    logger.info(
        "Starting training: |train|=%d |val|=%d families=%s",
        len(train_ds), len(eval_ds), list(family_to_id.keys()),
    )
    trainer.train()

    # Final save (adapter + adversary only) --------------------------------
    save_dir = output_dir / "final"
    model.save_adapter(str(save_dir))
    # Also copy the processor for inference-time tokenisation.
    processor.save_pretrained(str(save_dir))
    with open(save_dir / "family_to_id.json", "w") as f:
        json.dump(family_to_id, f, indent=2)
    # Persist the hybrid config next to the adapter so _load_hybrid() can
    # reconstruct the architecture (num_accent_classes, LoRA rank, etc.).
    with open(save_dir / "hybrid_config.json", "w") as f:
        json.dump(asdict(hybrid_cfg), f, indent=2)
    logger.info("Training complete. Final checkpoint at %s", save_dir)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_hybrid.yaml")
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-eval-samples", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    train(args.config, args.max_train_samples, args.max_eval_samples)


if __name__ == "__main__":
    main()
