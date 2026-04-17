"""
train_rl.py
RL post-training via GRPO on a fine-tuned Wav2Vec2+LoRA checkpoint.

Custom training loop (not HF Trainer) since GRPO requires alternating
beam search rollout generation and policy gradient optimization.

Usage:
    python ast-asr/train_rl.py --config configs/train_rl.yaml
    python ast-asr/train_rl.py --config configs/train_rl.yaml --max-steps 50  # smoke test
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import _resolve_hf_token as _rht
_rht()

from models import HybridAdversarialASR, HybridConfig
from rl.beam_search import build_ctc_decoder, generate_rollouts
from rl.curriculum import (
    CurriculumScheduler,
    FamilyBalancedSampler,
    StageConfig,
    add_noise,
)
from rl.grpo import grpo_step
from spire_loader import apply_split, build_manifest, make_speaker_split

logger = logging.getLogger(__name__)

TARGET_SR = 16_000


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_policy_and_ref(ckpt_dir: str, device: torch.device):
    """Load ft-w2v2 as policy model and clone a frozen reference."""
    from transformers import Wav2Vec2Processor

    ckpt = Path(ckpt_dir)
    with open(ckpt / "hybrid_config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["lora_target_modules"] = tuple(cfg_dict.get("lora_target_modules", ()))
    hybrid_cfg = HybridConfig(**cfg_dict)

    # Policy model (trainable)
    policy = HybridAdversarialASR(hybrid_cfg)
    policy.load_adapter(str(ckpt))
    policy.to(device)
    policy.train()

    # Frozen reference (for KL penalty)
    ref = copy.deepcopy(policy)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    processor = Wav2Vec2Processor.from_pretrained(str(ckpt))
    return policy, ref, processor, hybrid_cfg


def prepare_batch(
    manifest_slice: pd.DataFrame,
    processor,
    device: torch.device,
    noise_prob: float = 0.0,
    noise_snr_range: tuple[float, float] = (10.0, 20.0),
):
    """Load audio, optionally add noise, and prepare tensors for a batch."""
    import librosa
    import soundfile as sf

    arrays = []
    references = []
    families = []
    rng = np.random.default_rng()

    for _, row in manifest_slice.iterrows():
        wav, sr = sf.read(row["path"], dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1, dtype=np.float32)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

        # Optional noise augmentation
        if noise_prob > 0 and rng.random() < noise_prob:
            snr = rng.uniform(*noise_snr_range)
            wav = add_noise(wav, snr_db=snr, rng=rng)

        arrays.append(wav)
        references.append(row["reference"])
        families.append(row["language_family"])

    # Pad and process
    max_len = max(len(a) for a in arrays)
    padded = [np.pad(a, (0, max_len - len(a))) for a in arrays]

    inputs = processor(
        padded, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    )
    input_values = inputs.input_values.to(device)
    attention_mask = torch.ones_like(input_values, dtype=torch.long)

    return input_values, attention_mask, references, families


def parse_stages(cfg_stages: list[dict]) -> list[StageConfig]:
    """Parse stage configs from YAML."""
    stages = []
    for s in cfg_stages:
        snr = s.get("noise_snr_range", [10.0, 20.0])
        stages.append(StageConfig(
            name=s["name"],
            families=s["families"],
            max_duration=s.get("max_duration", 10.0),
            family_weights=s.get("family_weights", {}),
            noise_prob=s.get("noise_prob", 0.0),
            noise_snr_range=tuple(snr),
            min_step=s.get("min_step", 0),
            max_step=s.get("max_step", 500),
        ))
    return stages


def train_rl(config_path: str, max_steps: int | None = None, output_dir: str | None = None):
    cfg = load_config(config_path)
    if output_dir:
        cfg["output"]["dir"] = output_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load models ──────────────────────────────────────────────────────────
    ckpt_path = cfg["checkpoint"]["path"]
    logger.info("Loading policy + reference from %s", ckpt_path)
    policy, ref, processor, hybrid_cfg = load_policy_and_ref(ckpt_path, device)

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    logger.info("Policy: %d trainable / %d total params", trainable, total)

    # ── Build beam decoder ───────────────────────────────────────────────────
    vocab_path = Path(ckpt_path) / "vocab.json"
    rl_cfg = cfg["rl"]
    beam_width = rl_cfg.get("beam_width", 8)
    decoder = build_ctc_decoder(str(vocab_path), beam_width=beam_width)

    # ── Build data manifest ──────────────────────────────────────────────────
    data_cfg = cfg["data"]
    manifest = build_manifest(
        raw_root=data_cfg["raw_root"],
        languages=data_cfg.get("languages"),
        min_duration=data_cfg.get("min_duration", 1.0),
        max_duration=data_cfg.get("max_duration", 20.0),
        min_words=data_cfg.get("min_words", 3),
    )
    split = make_speaker_split(
        manifest,
        val_ratio=0.15,
        seed=data_cfg.get("seed", 42),
        split_path=data_cfg.get("split_path"),
    )
    train_df = apply_split(manifest, split, "train")
    logger.info(
        "Training manifest: %d samples, families=%s",
        len(train_df),
        train_df["language_family"].value_counts().to_dict(),
    )

    # ── Curriculum ───────────────────────────────────────────────────────────
    stages = parse_stages(cfg["curriculum"]["stages"])
    scheduler = CurriculumScheduler(stages=stages, manifest=train_df)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=rl_cfg.get("lr", 5e-6),
        weight_decay=0.01,
    )

    # ── Output directory ─────────────────────────────────────────────────────
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "hybrid_config.json", "w") as f:
        json.dump(asdict(hybrid_cfg), f, indent=2)
    with open(out_dir / "rl_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── Training loop ────────────────────────────────────────────────────────
    total_steps = max_steps or rl_cfg.get("max_steps", 3000)
    batch_size = rl_cfg.get("batch_size", 4)
    grad_accum = rl_cfg.get("gradient_accumulation", 4)
    temperature = rl_cfg.get("temperature", 1.2)
    beta_kl = rl_cfg.get("beta_kl", 0.1)
    grad_clip = rl_cfg.get("grad_clip", 1.0)
    log_every = cfg["output"].get("log_every", 10)
    eval_every = cfg["eval"].get("eval_every", 200)
    save_every = cfg["eval"].get("save_every", 500)

    reward_cfg = cfg["reward"]

    logger.info(
        "Starting RL training: %d steps, batch=%d, grad_accum=%d, beam_width=%d",
        total_steps, batch_size, grad_accum, beam_width,
    )
    logger.info("Reward: cer_w=%.1f, wer_w=%.1f, alpha_fair=%.1f, threshold=%.2f",
                reward_cfg["cer_weight"], reward_cfg["wer_weight"],
                reward_cfg["alpha_fairness"], reward_cfg["fairness_threshold"])

    metrics_log = []
    t0 = time.time()
    sampler = None
    current_stage_name = None

    for step in range(total_steps):
        # Check curriculum transitions
        step_metrics = metrics_log[-1] if metrics_log else {}
        transitioned = scheduler.maybe_advance(step, step_metrics)

        # Rebuild sampler on stage change
        if transitioned or sampler is None or current_stage_name != scheduler.stage_name:
            current_stage_name = scheduler.stage_name
            stage = scheduler.current_stage
            stage_df = scheduler.filter_manifest()

            if len(stage_df) == 0:
                logger.warning(
                    "Stage '%s' has 0 samples — skipping sampler rebuild. "
                    "Check if families %s exist in the manifest.",
                    stage.name, stage.families,
                )
                # Fall back to all available data with equal weights
                stage_df = train_df
                weights = {f: 1.0 for f in train_df["language_family"].unique()}
            else:
                weights = stage.family_weights

            # Only include families that have data
            available_families = set(stage_df["language_family"].unique())
            weights = {f: w for f, w in weights.items() if f in available_families}
            if not weights:
                weights = {f: 1.0 for f in available_families}

            sampler = FamilyBalancedSampler(
                manifest=stage_df,
                batch_size=batch_size,
                family_weights=weights,
                seed=42 + step,
            )
            sampler_iter = iter(sampler)
            logger.info(
                "Stage '%s': %d samples, families=%s, weights=%s",
                stage.name, len(stage_df), list(available_families), weights,
            )

        # Get batch indices
        try:
            batch_indices = next(sampler_iter)
        except StopIteration:
            sampler_iter = iter(sampler)
            batch_indices = next(sampler_iter)

        batch_df = stage_df.iloc[batch_indices]

        # Load audio and prepare tensors
        stage = scheduler.current_stage
        input_values, attention_mask, references, families = prepare_batch(
            batch_df, processor, device,
            noise_prob=stage.noise_prob,
            noise_snr_range=stage.noise_snr_range,
        )

        # Generate rollouts (eval mode, no grad)
        policy.eval()
        rollouts = generate_rollouts(
            model=policy,
            input_values=input_values,
            attention_mask=attention_mask,
            references=references,
            families=families,
            decoder=decoder,
            processor=processor,
            beam_width=beam_width,
            temperature=temperature,
        )
        policy.train()

        # GRPO step
        step_result = grpo_step(
            policy_model=policy,
            ref_model=ref,
            rollouts=rollouts,
            processor=processor,
            optimizer=optimizer,
            beta_kl=beta_kl,
            grad_clip=grad_clip,
            cer_weight=reward_cfg["cer_weight"],
            wer_weight=reward_cfg["wer_weight"],
            alpha_fairness=reward_cfg["alpha_fairness"],
            fairness_threshold=reward_cfg["fairness_threshold"],
        )

        step_result["step"] = step
        step_result["stage"] = scheduler.stage_name
        step_result["elapsed"] = time.time() - t0
        metrics_log.append(step_result)

        # Logging
        if step % log_every == 0:
            elapsed = step_result["elapsed"]
            logger.info(
                "[%d/%d] stage=%s | loss=%.4f (pg=%.4f kl=%.4f) | "
                "reward=%.3f±%.3f | kl=%.4f | %.1fs",
                step, total_steps, scheduler.stage_name,
                step_result["loss"], step_result["pg_loss"], step_result["kl_loss"],
                step_result["reward_mean"], step_result["reward_std"],
                step_result["kl_mean"], elapsed,
            )

        # Save checkpoint
        if step > 0 and step % save_every == 0:
            save_path = out_dir / f"step-{step}"
            policy.save_adapter(str(save_path))
            processor.save_pretrained(str(save_path))
            with open(save_path / "hybrid_config.json", "w") as f:
                json.dump(asdict(hybrid_cfg), f, indent=2)
            logger.info("Saved checkpoint: %s", save_path)

    # ── Final save ───────────────────────────────────────────────────────────
    final_dir = out_dir / "final"
    policy.save_adapter(str(final_dir))
    processor.save_pretrained(str(final_dir))
    with open(final_dir / "hybrid_config.json", "w") as f:
        json.dump(asdict(hybrid_cfg), f, indent=2)

    # Save metrics log
    metrics_df = pd.DataFrame(metrics_log)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    elapsed_total = time.time() - t0
    logger.info(
        "RL training complete: %d steps in %.1f min. Final checkpoint: %s",
        total_steps, elapsed_total / 60, final_dir,
    )

    # Print summary
    if len(metrics_log) > 0:
        last = metrics_log[-1]
        first = metrics_log[0]
        logger.info(
            "Summary: reward %.3f → %.3f | loss %.4f → %.4f | kl %.4f → %.4f",
            first["reward_mean"], last["reward_mean"],
            first["loss"], last["loss"],
            first["kl_mean"], last["kl_mean"],
        )


def main():
    ap = argparse.ArgumentParser(description="RL post-training via GRPO")
    ap.add_argument("--config", default="configs/train_rl.yaml")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override max_steps from config")
    ap.add_argument("--output-dir", default=None,
                    help="Override output directory from config")
    args = ap.parse_args()
    train_rl(args.config, args.max_steps, args.output_dir)


if __name__ == "__main__":
    main()
