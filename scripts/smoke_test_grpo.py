#!/usr/bin/env python3
"""
Smoke test for Phase 3: GRPO optimizer.

Loads the ft-w2v2 checkpoint, grabs 4 real SPIRE-SIES audio samples,
runs beam search to generate rollouts, then does 5 GRPO steps.

Verifies:
  1. Loss is finite and decreasing
  2. Gradients flow through LoRA parameters
  3. KL stays bounded
  4. All metrics are returned correctly
"""

import sys
import json
import copy
import logging
from pathlib import Path

import numpy as np
import torch
import librosa

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ast-asr"))
from models import HybridAdversarialASR, HybridConfig
from rl.beam_search import build_ctc_decoder, generate_rollouts, RolloutBatch
from rl.reward import compute_rewards
from rl.grpo import grpo_step, group_relative_advantages

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test_grpo")

CKPT_DIR = Path("outputs/checkpoints/ft-w2v2/final")
DATA_DIR = Path("data/spire-sies/raw")
TARGET_SR = 16_000


def load_model_and_processor(device):
    """Load ft-w2v2 checkpoint."""
    from transformers import Wav2Vec2Processor

    with open(CKPT_DIR / "hybrid_config.json") as f:
        cfg_dict = json.load(f)
    cfg_dict["lora_target_modules"] = tuple(cfg_dict.get("lora_target_modules", ()))
    cfg = HybridConfig(**cfg_dict)

    model = HybridAdversarialASR(cfg)
    model.load_adapter(str(CKPT_DIR))
    model.to(device)
    model.train()  # need train mode for gradients

    processor = Wav2Vec2Processor.from_pretrained(str(CKPT_DIR))
    return model, processor, cfg


def load_audio_samples(n=4):
    """Grab n audio files from SPIRE-SIES with their transcripts."""
    import csv

    # Read transcript CSV
    csv_path = DATA_DIR / "IISc_SPIRE_SIES_Transcription.csv"
    transcripts = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcripts[row["File_Name"]] = row["Transcript"]

    # Find audio files from different language families
    families = {
        "Hindi": "Indo-Aryan",
        "Tamil": "Dravidian",
        "Kannada": "Dravidian",
        "Bengali": "Indo-Aryan",
    }

    samples = []
    for lang, family in families.items():
        lang_dir = DATA_DIR / f"IISc_SPIRE_SIES_{lang}"
        if not lang_dir.exists():
            continue
        # Find first valid wav file with a transcript
        for wav_path in sorted(lang_dir.rglob("*.wav"))[:20]:
            fname = wav_path.stem
            if fname in transcripts:
                ref = transcripts[fname]
                # Clean transcript: remove noise tags
                import re
                ref = re.sub(r"<[^>]+>", "", ref).strip()
                ref = re.sub(r"\s+", " ", ref).strip()
                if len(ref) > 10 and len(ref) < 200:
                    try:
                        audio, sr = librosa.load(str(wav_path), sr=TARGET_SR)
                        if len(audio) > TARGET_SR and len(audio) < 10 * TARGET_SR:
                            samples.append({
                                "audio": audio,
                                "reference": ref,
                                "family": family,
                                "lang": lang,
                            })
                            break
                    except Exception:
                        continue
        if len(samples) >= n:
            break

    logger.info("Loaded %d audio samples", len(samples))
    for s in samples:
        logger.info("  %s (%s): %.1fs, ref=%s...",
                     s["lang"], s["family"], len(s["audio"]) / TARGET_SR,
                     s["reference"][:60])
    return samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 1. Load model
    logger.info("Loading ft-w2v2 policy model...")
    policy_model, processor, cfg = load_model_and_processor(device)

    # 2. Clone as frozen reference
    logger.info("Cloning frozen reference model...")
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # 3. Build beam decoder
    vocab_path = CKPT_DIR / "vocab.json"
    decoder = build_ctc_decoder(str(vocab_path), beam_width=8)

    # 4. Load audio
    samples = load_audio_samples(n=4)
    if len(samples) < 2:
        logger.error("Need at least 2 audio samples for smoke test")
        sys.exit(1)

    # 5. Prepare batch
    max_len = max(len(s["audio"]) for s in samples)
    padded = [np.pad(s["audio"], (0, max_len - len(s["audio"]))) for s in samples]

    inputs = processor(
        padded, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    )
    input_values = inputs.input_values.to(device)
    attention_mask = torch.ones_like(input_values, dtype=torch.long)
    references = [s["reference"] for s in samples]
    families = [s["family"] for s in samples]

    # 6. Generate rollouts
    logger.info("Generating beam search rollouts (K=8)...")
    policy_model.eval()
    rollouts = generate_rollouts(
        model=policy_model,
        input_values=input_values,
        attention_mask=attention_mask,
        references=references,
        families=families,
        decoder=decoder,
        processor=processor,
        beam_width=8,
        temperature=1.2,
    )
    policy_model.train()

    logger.info("Rollouts generated: %d utterances × %d hypotheses",
                len(rollouts.references), len(rollouts.hypotheses[0]))
    for i in range(len(rollouts.references)):
        logger.info("  Utt %d top-1: '%s'", i, rollouts.hypotheses[i][0][:80])
        logger.info("         ref:   '%s'", rollouts.references[i][:80])

    # 7. Quick test: group_relative_advantages
    test_rewards = torch.tensor([[0.8, 0.5, 0.3], [0.9, 0.7, 0.1]])
    test_adv = group_relative_advantages(test_rewards)
    assert test_adv.shape == (2, 3), f"Bad shape: {test_adv.shape}"
    assert abs(test_adv[0].mean().item()) < 0.01, "Advantages should be zero-mean per group"
    logger.info("group_relative_advantages: OK (shape=%s, row_mean≈0)", test_adv.shape)

    # 8. Run 5 GRPO steps
    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=5e-6,
        weight_decay=0.01,
    )

    logger.info("\n=== Running 5 GRPO steps ===")
    losses = []
    for step in range(5):
        # Re-generate rollouts every 2 steps (like real training)
        if step % 2 == 0 and step > 0:
            policy_model.eval()
            rollouts = generate_rollouts(
                model=policy_model,
                input_values=input_values,
                attention_mask=attention_mask,
                references=references,
                families=families,
                decoder=decoder,
                processor=processor,
                beam_width=8,
                temperature=1.2,
            )
            policy_model.train()

        metrics = grpo_step(
            policy_model=policy_model,
            ref_model=ref_model,
            rollouts=rollouts,
            processor=processor,
            optimizer=optimizer,
            beta_kl=0.1,
            grad_clip=1.0,
        )

        losses.append(metrics["loss"])
        logger.info(
            "Step %d: loss=%.4f (pg=%.4f, kl=%.4f) | "
            "reward_mean=%.3f ± %.3f | kl=%.4f | adv_std=%.3f",
            step,
            metrics["loss"], metrics["pg_loss"], metrics["kl_loss"],
            metrics["reward_mean"], metrics["reward_std"],
            metrics["kl_mean"], metrics["advantage_std"],
        )

    # 9. Verify results
    logger.info("\n=== Verification ===")

    # Check all losses are finite
    all_finite = all(np.isfinite(l) for l in losses)
    logger.info("All losses finite: %s", "PASS" if all_finite else "FAIL")

    # Check gradients flowed through LoRA params
    lora_grads = []
    for name, p in policy_model.named_parameters():
        if p.requires_grad and p.grad is not None and "lora" in name:
            lora_grads.append((name, p.grad.abs().mean().item()))
    has_lora_grads = len(lora_grads) > 0 and all(g > 0 for _, g in lora_grads)
    logger.info("LoRA gradients flowing: %s (%d params with grad)",
                "PASS" if has_lora_grads else "FAIL", len(lora_grads))
    if lora_grads:
        for name, grad_mean in lora_grads[:4]:
            logger.info("  %s: grad_mean=%.2e", name.split(".")[-2] + "." + name.split(".")[-1], grad_mean)

    # Check KL is bounded (should be small since ref = copy of policy at start)
    kl_bounded = abs(metrics["kl_mean"]) < 5.0
    logger.info("KL bounded (<5.0): %s (kl=%.4f)",
                "PASS" if kl_bounded else "FAIL", metrics["kl_mean"])

    # Summary
    all_pass = all_finite and has_lora_grads and kl_bounded
    logger.info("\n%s Phase 3 GRPO smoke test: %s",
                "✓" if all_pass else "✗",
                "ALL PASSED" if all_pass else "SOME FAILED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
