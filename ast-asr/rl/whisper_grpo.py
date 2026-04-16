"""
whisper_grpo.py
Autoregressive GRPO for Whisper ASR.

Adapts the CTC-based GRPO to Whisper's encoder-decoder architecture:
- Rollouts via model.generate() with sampling (not beam search)
- Log-probs via forward pass with labels (cross-entropy per token)
- Same reward function and K3 KL estimator as rl/grpo.py
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .beam_search import RolloutBatch
from .reward import compute_rewards

logger = logging.getLogger(__name__)


def generate_whisper_rollouts(
    model,
    processor,
    audio_arrays: list,
    references: list[str],
    families: list[str],
    device: torch.device,
    K: int = 4,
    temperature: float = 1.2,
    max_new_tokens: int = 200,
) -> RolloutBatch:
    """
    Generate K sampled hypotheses per utterance from Whisper.

    Returns a RolloutBatch with the same interface as CTC beam search,
    so reward.py works unchanged.
    """
    # Prepare encoder inputs
    inputs = processor(
        audio_arrays,
        sampling_rate=16_000,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    input_features = inputs.input_features.to(device)
    B = input_features.shape[0]

    # Generate K hypotheses per utterance via sampling
    with torch.no_grad():
        gen_out = model.generate(
            input_features.repeat_interleave(K, dim=0),  # [B*K, ...]
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            language="en",
            task="transcribe",
            return_dict_in_generate=True,
        )

    # Decode all B*K sequences
    all_texts = processor.batch_decode(gen_out.sequences, skip_special_tokens=True)

    # Reshape into [B, K]
    hypotheses = []
    for i in range(B):
        hyps = all_texts[i * K : (i + 1) * K]
        hypotheses.append(hyps)

    # Compute log-probs for each hypothesis (not needed for RolloutBatch,
    # but stored for reference; actual differentiable log-probs computed
    # in the GRPO step)
    log_probs = [[0.0] * K for _ in range(B)]

    return RolloutBatch(
        hypotheses=hypotheses,
        log_probs=log_probs,
        references=references,
        families=families,
        input_values=input_features,  # [B, n_mels, T_enc]
        attention_mask=torch.ones(B, 1, device=device),  # placeholder
    )


def compute_whisper_log_probs(
    model,
    processor,
    input_features: torch.Tensor,
    hypotheses: list[list[str]],
) -> torch.Tensor:
    """
    Compute differentiable per-sequence log-probabilities under Whisper.

    For each (utterance, hypothesis) pair, runs a forward pass with the
    hypothesis as decoder labels and sums the per-token log-probs.

    Returns: [B, K] tensor of log-probs (differentiable w.r.t. model params).
    """
    device = input_features.device
    B = len(hypotheses)
    K = len(hypotheses[0])

    all_log_probs = []

    for i in range(B):
        features_i = input_features[i : i + 1]  # [1, n_mels, T]

        for k in range(K):
            hyp_text = hypotheses[i][k]
            if not hyp_text.strip():
                all_log_probs.append(torch.tensor(-100.0, device=device))
                continue

            # Tokenize hypothesis as decoder labels
            labels = processor.tokenizer(
                hyp_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(device)

            # Prepend decoder_start_token_id
            start_token = torch.tensor(
                [[model.config.decoder_start_token_id]], device=device
            )
            # For Whisper, the forced decoder IDs handle language/task tokens
            # but for log-prob computation we just need the sequence
            decoder_input_ids = torch.cat([start_token, labels], dim=1)

            # Forward pass
            outputs = model(
                input_features=features_i,
                decoder_input_ids=decoder_input_ids[:, :-1],
                labels=None,  # we compute loss manually
            )

            # logits: [1, seq_len, vocab_size]
            logits = outputs.logits
            log_probs_dist = F.log_softmax(logits, dim=-1)

            # Gather log-probs of actual tokens (labels = decoder_input_ids[:, 1:])
            target_ids = decoder_input_ids[:, 1:]  # [1, seq_len]
            seq_len = min(log_probs_dist.shape[1], target_ids.shape[1])
            target_ids = target_ids[:, :seq_len]
            log_probs_dist = log_probs_dist[:, :seq_len, :]

            token_log_probs = log_probs_dist.gather(
                2, target_ids.unsqueeze(-1)
            ).squeeze(-1)  # [1, seq_len]

            # Sum over sequence → total log-prob
            seq_log_prob = token_log_probs.sum()
            all_log_probs.append(seq_log_prob)

    # Stack into [B, K]
    result = torch.stack(all_log_probs).view(B, K)
    return result


def group_relative_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """GRPO advantages: normalize rewards within each utterance's K hypotheses."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


def whisper_grpo_step(
    policy_model,
    ref_model,
    rollouts: RolloutBatch,
    processor,
    optimizer: torch.optim.Optimizer,
    beta_kl: float = 0.1,
    grad_clip: float = 1.0,
    cer_weight: float = 0.6,
    wer_weight: float = 0.4,
    alpha_fairness: float = 2.0,
    fairness_threshold: float = 0.05,
) -> dict[str, float]:
    """
    One GRPO step for Whisper.

    1. Compute rewards from rollouts
    2. Compute group-relative advantages
    3. Forward pass through policy for differentiable log-probs
    4. Forward pass through frozen reference for KL penalty (K3 estimator)
    5. Compute loss and backprop
    """
    device = rollouts.input_values.device
    B = len(rollouts.references)
    K = len(rollouts.hypotheses[0])

    # 1. Rewards [B, K]
    rewards = compute_rewards(
        rollouts,
        cer_weight=cer_weight,
        wer_weight=wer_weight,
        alpha_fairness=alpha_fairness,
        fairness_threshold=fairness_threshold,
        normalize=True,
    ).to(device)

    # 2. Advantages [B, K]
    advantages = group_relative_advantages(rewards)

    # 3. Policy log-probs [B, K] — differentiable
    policy_log_probs = compute_whisper_log_probs(
        policy_model, processor, rollouts.input_values, rollouts.hypotheses
    )

    # 4. Reference log-probs [B, K] — detached
    with torch.no_grad():
        ref_log_probs = compute_whisper_log_probs(
            ref_model, processor, rollouts.input_values, rollouts.hypotheses
        )

    # 5. Loss: policy gradient + K3 KL penalty
    log_ratio = (ref_log_probs - policy_log_probs).clamp(-20.0, 20.0)
    kl = torch.exp(log_ratio) - log_ratio - 1.0  # K3: always >= 0

    pg_loss = -(advantages * policy_log_probs).mean()
    kl_loss = beta_kl * kl.mean()
    total_loss = pg_loss + kl_loss

    # 6. Backward + step
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in policy_model.parameters() if p.requires_grad],
        max_norm=grad_clip,
    )
    optimizer.step()

    # Metrics (raw, unnormalized rewards)
    with torch.no_grad():
        raw_rewards = compute_rewards(
            rollouts,
            cer_weight=cer_weight,
            wer_weight=wer_weight,
            alpha_fairness=alpha_fairness,
            fairness_threshold=fairness_threshold,
            normalize=False,
        )

    return {
        "loss": float(total_loss.detach().cpu()),
        "pg_loss": float(pg_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "reward_mean": float(raw_rewards.mean()),
        "reward_std": float(raw_rewards.std()),
        "kl_mean": float(kl.mean().detach().cpu()),
        "advantage_std": float(advantages.std().detach().cpu()),
    }
