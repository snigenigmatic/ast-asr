"""Whisper rollout generation and correctly masked autoregressive scoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .metrics import word_edit_counts
from .modeling import model_input_dtype, whisper_runtime_model
from .optimization import deterministic_model_mode
from .rollouts import (
    AcousticCondition,
    CandidateRollout,
    FrozenRolloutBatch,
    UtteranceRollout,
)


@dataclass(frozen=True, slots=True)
class PackedDecoderBatch:
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    target_ids: torch.Tensor
    score_mask: torch.Tensor


def pack_decoder_targets(
    *,
    prefix_token_ids: Sequence[int],
    target_token_ids: Sequence[Sequence[int]],
    pad_token_id: int,
    device: torch.device | str = "cpu",
) -> PackedDecoderBatch:
    """Pack prefix plus targets and mark only hypothesis targets for scoring."""
    prefix = tuple(int(token) for token in prefix_token_ids)
    if not prefix:
        raise ValueError("decoder prefix cannot be empty")
    targets = tuple(tuple(int(token) for token in item) for item in target_token_ids)
    if not targets or any(not item for item in targets):
        raise ValueError("every hypothesis must contain at least one target token")
    maximum_full_length = max(len(prefix) + len(item) for item in targets)
    decoder_inputs = []
    decoder_masks = []
    target_rows = []
    scoring_masks = []
    for item in targets:
        full = prefix + item
        padding = maximum_full_length - len(full)
        padded = full + (pad_token_id,) * padding
        decoder_inputs.append(padded[:-1])
        target_rows.append(padded[1:])
        valid_decoder_length = len(full) - 1
        decoder_masks.append(
            (True,) * valid_decoder_length + (False,) * (maximum_full_length - 1 - valid_decoder_length)
        )
        prefix_targets = len(prefix) - 1
        scoring_masks.append(
            (False,) * prefix_targets
            + (True,) * len(item)
            + (False,) * padding
        )
    return PackedDecoderBatch(
        decoder_input_ids=torch.tensor(decoder_inputs, dtype=torch.long, device=device),
        decoder_attention_mask=torch.tensor(decoder_masks, dtype=torch.long, device=device),
        target_ids=torch.tensor(target_rows, dtype=torch.long, device=device),
        score_mask=torch.tensor(scoring_masks, dtype=torch.bool, device=device),
    )


@dataclass(frozen=True, slots=True)
class ScoredHypotheses:
    token_ids: torch.Tensor
    token_log_probs: torch.Tensor
    token_mask: torch.Tensor


def _decoder_prefix(model: Any, processor: Any) -> tuple[int, ...]:
    prompt = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    prompt_tokens = tuple(token for _, token in sorted(prompt))
    return (int(model.config.decoder_start_token_id), *prompt_tokens)


def _hypothesis_targets(processor: Any, hypotheses: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    eos = int(processor.tokenizer.eos_token_id)
    targets = []
    for hypothesis in hypotheses:
        tokenized = processor.tokenizer(hypothesis, add_special_tokens=False)
        ids = tokenized.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        targets.append((*map(int, ids), eos))
    return tuple(targets)


def score_hypotheses(
    model: Any,
    processor: Any,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
    hypotheses: Sequence[Sequence[str]],
) -> ScoredHypotheses:
    """Return per-generated-token log-probabilities with gradients intact."""
    input_features = input_features.to(dtype=model_input_dtype(model))
    batch_size = len(hypotheses)
    if batch_size != input_features.shape[0] or batch_size == 0:
        raise ValueError("hypotheses and acoustic batch must align")
    candidate_counts = {len(items) for items in hypotheses}
    if len(candidate_counts) != 1:
        raise ValueError("every utterance must have the same candidate count")
    candidate_count = len(hypotheses[0])
    flattened = [hypothesis for items in hypotheses for hypothesis in items]
    targets = _hypothesis_targets(processor, flattened)
    packed = pack_decoder_targets(
        prefix_token_ids=_decoder_prefix(model, processor),
        target_token_ids=targets,
        pad_token_id=int(processor.tokenizer.pad_token_id),
        device=input_features.device,
    )
    repeated_features = input_features.repeat_interleave(candidate_count, dim=0)
    repeated_attention = attention_mask.repeat_interleave(candidate_count, dim=0)
    with deterministic_model_mode(model):
        outputs = whisper_runtime_model(model)(
            input_features=repeated_features,
            attention_mask=repeated_attention,
            decoder_input_ids=packed.decoder_input_ids,
            decoder_attention_mask=packed.decoder_attention_mask,
            use_cache=False,
        )
    distributions = F.log_softmax(outputs.logits.float(), dim=-1)
    gathered = distributions.gather(2, packed.target_ids.unsqueeze(-1)).squeeze(-1)
    maximum_targets = max(len(item) for item in targets)
    token_ids = torch.zeros(
        (len(flattened), maximum_targets),
        dtype=torch.long,
        device=input_features.device,
    )
    log_probs = torch.zeros(
        (len(flattened), maximum_targets),
        dtype=torch.float32,
        device=input_features.device,
    )
    token_mask = torch.zeros_like(log_probs, dtype=torch.bool)
    for index in range(len(flattened)):
        selected_ids = packed.target_ids[index][packed.score_mask[index]]
        selected_log_probs = gathered[index][packed.score_mask[index]]
        size = selected_ids.numel()
        token_ids[index, :size] = selected_ids
        log_probs[index, :size] = selected_log_probs
        token_mask[index, :size] = True
    shape = (batch_size, candidate_count, maximum_targets)
    return ScoredHypotheses(
        token_ids=token_ids.view(shape),
        token_log_probs=log_probs.view(shape),
        token_mask=token_mask.view(shape),
    )


@dataclass(frozen=True, slots=True)
class RolloutInput:
    utterance_id: str
    speaker_id: str
    primary_language: str
    family: str
    condition: AcousticCondition
    reference: str
    audio: torch.Tensor


@dataclass(frozen=True, slots=True)
class GeneratedRollout:
    frozen: FrozenRolloutBatch
    input_features: torch.Tensor
    attention_mask: torch.Tensor


def generate_frozen_rollout(
    model: Any,
    processor: Any,
    inputs: Sequence[RolloutInput],
    *,
    candidates: int,
    temperature: float,
    maximum_new_tokens: int,
    model_revision: str,
    device: torch.device | str,
    seed: int,
) -> GeneratedRollout:
    if candidates < 2:
        raise ValueError("policy rollouts require at least two candidates")
    acoustic = processor.feature_extractor(
        [item.audio.cpu().numpy() for item in inputs],
        sampling_rate=16_000,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )
    input_features = acoustic.input_features.to(
        device=device,
        dtype=model_input_dtype(model),
    )
    attention_mask = acoustic.attention_mask.to(device)
    generation_kwargs = {
        "input_features": input_features,
        "attention_mask": attention_mask,
        "do_sample": True,
        "temperature": temperature,
        "top_p": 0.95,
        "num_return_sequences": candidates,
        "max_new_tokens": maximum_new_tokens,
        "language": "en",
        "task": "transcribe",
    }
    rng_devices = [input_features.device.index or 0] if input_features.is_cuda else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        with torch.no_grad(), deterministic_model_mode(model):
            sequences = whisper_runtime_model(model).generate(**generation_kwargs)
        flat_hypotheses = processor.batch_decode(sequences, skip_special_tokens=True)
    hypotheses = tuple(
        tuple(
            hypothesis.strip()
            for hypothesis in flat_hypotheses[index * candidates : (index + 1) * candidates]
        )
        for index in range(len(inputs))
    )
    with torch.no_grad():
        old_scores = score_hypotheses(
            model,
            processor,
            input_features,
            attention_mask,
            hypotheses,
        )
    utterances = []
    for utterance_index, item in enumerate(inputs):
        candidate_records = []
        for candidate_index, hypothesis in enumerate(hypotheses[utterance_index]):
            mask = old_scores.token_mask[utterance_index, candidate_index]
            ids = old_scores.token_ids[utterance_index, candidate_index][mask]
            log_probs = old_scores.token_log_probs[utterance_index, candidate_index][mask]
            candidate_records.append(
                CandidateRollout(
                    hypothesis=hypothesis,
                    token_ids=tuple(int(value) for value in ids.cpu()),
                    token_mask=(True,) * ids.numel(),
                    old_token_log_probs=tuple(float(value) for value in log_probs.float().cpu()),
                    old_sequence_log_probability=float(log_probs.float().mean().cpu()),
                    wer=word_edit_counts(item.reference, hypothesis).wer,
                )
            )
        utterances.append(
            UtteranceRollout(
                utterance_id=item.utterance_id,
                speaker_id=item.speaker_id,
                primary_language=item.primary_language,
                family=item.family,
                condition=item.condition,
                reference=item.reference,
                candidates=tuple(candidate_records),
            )
        )
    return GeneratedRollout(
        frozen=FrozenRolloutBatch(
            model_revision=model_revision,
            utterances=tuple(utterances),
        ),
        input_features=input_features,
        attention_mask=attention_mask,
    )
