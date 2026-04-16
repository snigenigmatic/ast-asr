"""
train_rl_whisper.py
RL post-training via GRPO on Whisper-small with LoRA.

Uses Svarah train split for training, Svarah eval split for periodic evaluation.

Usage:
    python ast-asr/train_rl_whisper.py --config configs/train_rl_whisper.yaml
    python ast-asr/train_rl_whisper.py --config configs/train_rl_whisper.yaml --max-steps 50
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_svarah

from rl.whisper_grpo import (
    generate_whisper_rollouts,
    whisper_grpo_step,
)

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_whisper_policy_and_ref(model_id: str, lora_cfg: dict, device: torch.device):
    """Load Whisper-small as policy (with LoRA) and clone a frozen reference."""
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    logger.info("Loading base Whisper model: %s", model_id)
    processor = WhisperProcessor.from_pretrained(model_id)
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Frozen reference (before LoRA)
    ref = copy.deepcopy(base_model)
    ref.to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    # Apply LoRA to policy
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
    )
    policy = get_peft_model(base_model, lora_config)
    policy.to(device)
    policy.train()

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    logger.info("Policy: %d trainable / %d total params (%.2f%%)", trainable, total, 100 * trainable / total)

    return policy, ref, processor


class SvarahBatchSampler:
    """Family-balanced sampler for Svarah DataFrame."""

    def __init__(self, df: pd.DataFrame, batch_size: int, seed: int = 42):
        self.df = df
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.families = sorted(df["language_family"].unique())
        self.family_indices = {
            fam: df[df["language_family"] == fam].index.tolist()
            for fam in self.families
        }

    def sample_batch(self) -> pd.DataFrame:
        """Sample a batch ensuring at least one sample from each family."""
        indices = []
        # One per family first
        for fam in self.families:
            idx = self.rng.choice(self.family_indices[fam])
            indices.append(idx)

        # Fill remaining slots randomly (weighted toward smaller families)
        remaining = self.batch_size - len(indices)
        if remaining > 0:
            all_idx = self.df.index.tolist()
            extra = self.rng.choice(all_idx, size=remaining, replace=True)
            indices.extend(extra.tolist())

        return self.df.loc[indices].reset_index(drop=True)


def train_rl_whisper(config_path: str, max_steps: int | None = None, alpha_fairness: float | None = None):
    cfg = load_config(config_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Override alpha_fairness from CLI if provided
    reward_cfg = cfg["reward"]
    if alpha_fairness is not None:
        reward_cfg["alpha_fairness"] = alpha_fairness

    # ── Load models ──────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    policy, ref, processor = load_whisper_policy_and_ref(
        model_cfg["id"], model_cfg["lora"], device
    )

    # ── Load Svarah train split ──────────────────────────────────────────────
    logger.info("Loading Svarah train split...")
    train_df = load_svarah(max_samples=None, cache_dir="cache", svarah_split="train")
    logger.info("Train set: %d utterances", len(train_df))
    for fam, count in train_df["language_family"].value_counts().items():
        logger.info("  %s: %d", fam, count)

    # ── Sampler ──────────────────────────────────────────────────────────────
    rl_cfg = cfg["rl"]
    batch_size = rl_cfg.get("batch_size", 2)
    sampler = SvarahBatchSampler(train_df, batch_size=batch_size, seed=42)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=rl_cfg.get("lr", 1e-6),
        weight_decay=0.01,
    )

    # ── Output directory ─────────────────────────────────────────────────────
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rl_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── Training loop ────────────────────────────────────────────────────────
    total_steps = max_steps or rl_cfg.get("max_steps", 1500)
    K = rl_cfg.get("K", 4)
    temperature = rl_cfg.get("temperature", 1.2)
    beta_kl = rl_cfg.get("beta_kl", 0.1)
    grad_clip = rl_cfg.get("grad_clip", 1.0)
    grad_accum = rl_cfg.get("gradient_accumulation", 4)
    eval_interval = rl_cfg.get("eval_interval", 100)
    save_interval = rl_cfg.get("save_interval", 500)

    logger.info(
        "Starting Whisper RL training: %d steps, batch=%d, K=%d, temp=%.1f, beta_kl=%.2f",
        total_steps, batch_size, K, temperature, beta_kl,
    )
    logger.info(
        "Reward: cer_w=%.1f, wer_w=%.1f, alpha_fair=%.1f, threshold=%.2f",
        reward_cfg["cer_weight"], reward_cfg["wer_weight"],
        reward_cfg["alpha_fairness"], reward_cfg["fairness_threshold"],
    )

    metrics_log = []
    t0 = time.time()
    accum_count = 0

    for step in range(total_steps):
        # Sample a batch
        batch_df = sampler.sample_batch()
        audio_arrays = batch_df["audio_array"].tolist()
        references = batch_df["reference"].tolist()
        families = batch_df["language_family"].tolist()

        # Generate rollouts
        policy.eval()
        rollouts = generate_whisper_rollouts(
            model=policy,
            processor=processor,
            audio_arrays=audio_arrays,
            references=references,
            families=families,
            device=device,
            K=K,
            temperature=temperature,
        )
        policy.train()

        # GRPO step
        step_result = whisper_grpo_step(
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
        step_result["elapsed"] = time.time() - t0
        metrics_log.append(step_result)

        # Logging
        if step % 10 == 0:
            logger.info(
                "[%d/%d] loss=%.4f (pg=%.4f kl=%.4f) | "
                "reward=%.3f±%.3f | kl=%.4f | %.1fs",
                step, total_steps,
                step_result["loss"], step_result["pg_loss"], step_result["kl_loss"],
                step_result["reward_mean"], step_result["reward_std"],
                step_result["kl_mean"], step_result["elapsed"],
            )

        # Save checkpoint
        if step > 0 and step % save_interval == 0:
            save_path = out_dir / f"step-{step}"
            save_path.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(str(save_path))
            logger.info("Saved checkpoint: %s", save_path)

    # ── Final save ───────────────────────────────────────────────────────────
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(final_dir))

    # Save metrics
    metrics_df = pd.DataFrame(metrics_log)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    elapsed_total = time.time() - t0
    logger.info(
        "Training complete: %d steps in %.1f min. Checkpoint: %s",
        total_steps, elapsed_total / 60, final_dir,
    )

    if metrics_log:
        first, last = metrics_log[0], metrics_log[-1]
        logger.info(
            "Summary: reward %.3f → %.3f | loss %.4f → %.4f | kl %.4f → %.4f",
            first["reward_mean"], last["reward_mean"],
            first["loss"], last["loss"],
            first["kl_mean"], last["kl_mean"],
        )


def main():
    ap = argparse.ArgumentParser(description="Whisper RL post-training via GRPO")
    ap.add_argument("--config", default="configs/train_rl_whisper.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--alpha-fairness", type=float, default=None,
                    help="Override alpha_fairness from config")
    args = ap.parse_args()
    train_rl_whisper(args.config, args.max_steps, args.alpha_fairness)


if __name__ == "__main__":
    main()
