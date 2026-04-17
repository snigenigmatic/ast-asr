"""
beam_search.py
CTC beam search rollout generation for GRPO-based RL post-training.

Generates K diverse hypotheses per utterance with their CTC log-probabilities,
suitable for group-relative policy optimization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class RolloutBatch:
    """Container for beam search rollouts over a batch of utterances."""

    hypotheses: list[list[str]]       # [B, K] decoded text per hypothesis
    log_probs: list[list[float]]      # [B, K] CTC log-prob under the generating policy
    references: list[str]             # [B] ground-truth transcripts
    families: list[str]               # [B] language family per utterance
    input_values: torch.Tensor        # [B, T] raw audio for the gradient pass
    attention_mask: torch.Tensor      # [B, T]


def build_ctc_decoder(
    vocab_path: str | Path,
    beam_width: int = 8,
) -> "pyctcdecode.BeamSearchDecoderCTC":
    """
    Build a pyctcdecode beam search decoder from a Wav2Vec2 vocab.json.

    The vocab maps characters to token IDs. pyctcdecode expects a list of
    labels indexed by token ID, with the blank token at index 0.
    """
    from pyctcdecode import BeamSearchDecoderCTC, Alphabet

    with open(vocab_path) as f:
        vocab: dict[str, int] = json.load(f)

    # Build labels list indexed by token ID.
    # pyctcdecode requires unique labels. Special tokens that the decoder
    # should never emit get unique sentinel strings (⁰, ¹, ², ³).
    n_tokens = max(vocab.values()) + 1
    labels = [""] * n_tokens
    for char, idx in vocab.items():
        if char == "<pad>":
            labels[idx] = ""   # CTC blank — must be empty string at idx 0
        elif char == "|":
            labels[idx] = " "  # word boundary → space
        elif char == "<s>":
            labels[idx] = "\u2071"   # unique placeholder (superscript i)
        elif char == "</s>":
            labels[idx] = "\u207F"   # unique placeholder (superscript n)
        elif char == "<unk>":
            labels[idx] = "\u2070"   # unique placeholder (superscript 0)
        else:
            labels[idx] = char

    alphabet = Alphabet.build_alphabet(labels)
    decoder = BeamSearchDecoderCTC(alphabet, language_model=None)
    logger.info(
        "Built CTC beam decoder: %d tokens, beam_width=%d", n_tokens, beam_width
    )
    return decoder


def generate_rollouts(
    model: torch.nn.Module,
    input_values: torch.Tensor,
    attention_mask: torch.Tensor,
    references: list[str],
    families: list[str],
    decoder: "pyctcdecode.BeamSearchDecoderCTC",
    processor,
    beam_width: int = 8,
    temperature: float = 1.0,
) -> RolloutBatch:
    """
    Generate K beam search hypotheses per utterance.

    Args:
        model: HybridAdversarialASR (or any model with .asr attribute)
        input_values: [B, T] raw audio tensor (on device)
        attention_mask: [B, T]
        references: ground-truth transcripts
        families: language family labels
        decoder: pyctcdecode BeamSearchDecoderCTC
        processor: Wav2Vec2Processor (for tokenizing hypotheses)
        beam_width: number of hypotheses per utterance (K)
        temperature: >1.0 increases beam diversity

    Returns:
        RolloutBatch with K hypotheses and their log-probs
    """
    device = input_values.device

    # Forward pass to get logits (detached from graph for rollout generation)
    with torch.no_grad():
        logits = model.asr(
            input_values=input_values,
            attention_mask=attention_mask,
        ).logits  # [B, T', V]

    # Apply temperature scaling for diversity
    if temperature != 1.0:
        logits = logits / temperature

    # Convert to log-probabilities on CPU for pyctcdecode
    log_probs_tensor = F.log_softmax(logits, dim=-1)
    log_probs_np = log_probs_tensor.cpu().float().numpy()  # [B, T', V]

    B = log_probs_np.shape[0]
    all_hypotheses: list[list[str]] = []
    all_log_probs: list[list[float]] = []

    for i in range(B):
        # pyctcdecode expects [T', V] numpy array of log-probs
        beams = decoder.decode_beams(
            log_probs_np[i],
            beam_width=beam_width,
        )

        # Each beam is (text, frames, indices, logit_score, lm_score)
        hyps: list[str] = []
        scores: list[float] = []
        for beam in beams[:beam_width]:
            text = beam[0].strip()
            logit_score = beam[3]  # CTC log-probability
            hyps.append(text if text else "<empty>")
            scores.append(float(logit_score))

        # Pad if fewer than beam_width hypotheses returned
        while len(hyps) < beam_width:
            hyps.append(hyps[-1] if hyps else "<empty>")
            scores.append(scores[-1] if scores else -1e6)

        all_hypotheses.append(hyps)
        all_log_probs.append(scores)

    return RolloutBatch(
        hypotheses=all_hypotheses,
        log_probs=all_log_probs,
        references=references,
        families=families,
        input_values=input_values,
        attention_mask=attention_mask,
    )


def compute_ctc_log_probs(
    model: torch.nn.Module,
    input_values: torch.Tensor,
    attention_mask: torch.Tensor,
    hypotheses: list[list[str]],
    processor,
) -> torch.Tensor:
    """
    Compute differentiable CTC log-probabilities for each hypothesis.

    This is the key function that enables policy gradients through CTC:
    log P(hypothesis | audio) = -CTC_loss(logits, encoded_hypothesis)

    Args:
        model: the policy model (gradients flow through this)
        input_values: [B, T] audio
        attention_mask: [B, T]
        hypotheses: [B, K] text hypotheses from beam search
        processor: Wav2Vec2Processor for tokenizing hypotheses

    Returns:
        log_probs: [B, K] differentiable CTC log-probabilities
    """
    # Forward pass WITH gradient tracking
    logits = model.asr(
        input_values=input_values,
        attention_mask=attention_mask,
    ).logits  # [B, T', V]

    log_probs = F.log_softmax(logits, dim=-1)  # [B, T', V]
    T_prime = log_probs.size(1)
    B = log_probs.size(0)
    K = len(hypotheses[0])

    # Compute input lengths (post-CNN downsampling)
    base = model.asr.base_model.model if hasattr(model.asr, "base_model") else model.asr
    input_lengths = base._get_feat_extract_output_lengths(
        attention_mask.sum(-1)
    ).long()  # [B]

    # For each hypothesis k, compute CTC log-prob
    all_log_probs = []
    for k in range(K):
        # Tokenize hypothesis k for each utterance in the batch
        hyp_texts = [hypotheses[i][k] for i in range(B)]

        # Encode hypotheses as token sequences
        targets_list = []
        target_lengths = []
        for text in hyp_texts:
            if text == "<empty>" or not text.strip():
                # Empty hypothesis: single blank token gets infinite CTC loss;
                # use a single space which maps to word delimiter
                tokens = torch.tensor([4], dtype=torch.long)  # "|" = space
            else:
                encoded = processor.tokenizer(
                    text.upper(),  # Wav2Vec2 vocab is uppercase
                    return_tensors="pt",
                ).input_ids[0]
                # Remove special tokens (BOS/EOS if present)
                tokens = encoded[encoded >= 4]  # keep only real tokens (>=4)
                if len(tokens) == 0:
                    tokens = torch.tensor([4], dtype=torch.long)
            targets_list.append(tokens)
            target_lengths.append(len(tokens))

        # Pad targets
        max_target_len = max(target_lengths)
        padded_targets = torch.zeros(B, max_target_len, dtype=torch.long)
        for i, tgt in enumerate(targets_list):
            padded_targets[i, : len(tgt)] = tgt

        target_lengths_t = torch.tensor(target_lengths, dtype=torch.long)

        # CTC loss with reduction='none' gives per-example negative log-prob
        ctc_loss = F.ctc_loss(
            log_probs.transpose(0, 1),  # [T', B, V]
            padded_targets.to(log_probs.device),
            input_lengths.clamp(max=T_prime),
            target_lengths_t.to(log_probs.device),
            reduction="none",
            zero_infinity=True,
        )  # [B]

        # log P(hyp | audio) = -CTC_loss
        all_log_probs.append(-ctc_loss)

    return torch.stack(all_log_probs, dim=1)  # [B, K]
