"""Cost-bounded CUDA validation for the FR-CISPO training machinery.

This module deliberately uses synthetic audio.  It validates the runtime and
optimization path, but its outputs are not ASR experiment results.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import write_immutable_json
from .config import ExperimentConfig
from .corruption import paired_white_noise
from .group_weights import DualRiskWeights, RiskGroup
from .inference import greedy_transcribe
from .ladder import TRAINING_LADDER
from .metrics import word_edit_counts
from .modeling import (
    build_lora_whisper,
    directory_content_hash,
    load_processor,
    load_saved_processor,
    trainable_parameter_hash,
)
from .objectives import CorruptionPolicy, GroupWeighting
from .optimization import optimize_frozen_rollout
from .rollouts import AcousticCondition
from .sft import _seed_everything
from .whisper_policy import RolloutInput, generate_frozen_rollout, score_hypotheses


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    utterance_id: str
    speaker_id: str
    family: str
    condition: AcousticCondition
    reference: str
    hypotheses: tuple[str, ...]
    audio: torch.Tensor


_REFERENCES = (
    "the quick brown fox",
    "speech recognition should be robust",
    "small models can learn carefully",
)

_HYPOTHESES = (
    (
        "the quick brown fox",
        "the quick fox",
        "quick brown",
        "hello world",
    ),
    (
        "speech recognition should be robust",
        "speech recognition is robust",
        "recognition should be robust",
        "speech model",
    ),
    (
        "small models can learn carefully",
        "small models learn carefully",
        "models can learn",
        "large systems fail",
    ),
)


def build_synthetic_cases(
    *,
    seed: int,
    corruption: CorruptionPolicy,
) -> tuple[SyntheticCase, ...]:
    """Create one deterministic runtime probe per placeholder family."""
    generator = torch.Generator().manual_seed(seed)
    time = torch.arange(16_000, dtype=torch.float32) / 16_000
    cases: list[SyntheticCase] = []
    for index, (reference, hypotheses) in enumerate(
        zip(_REFERENCES, _HYPOTHESES, strict=True)
    ):
        family = f"smoke-family-{index + 1}"
        clean = (
            0.04 * torch.sin(2 * torch.pi * (180 + 70 * index) * time)
            + 0.002 * torch.randn(16_000, generator=generator)
        ).float()
        base = SyntheticCase(
            utterance_id=f"synthetic-{index + 1}",
            speaker_id=f"synthetic-speaker-{index + 1}",
            family=family,
            condition=AcousticCondition.CLEAN,
            reference=reference,
            hypotheses=hypotheses,
            audio=clean,
        )
        cases.append(base)
        if corruption is CorruptionPolicy.PAIRED_WHITE:
            noisy = paired_white_noise(
                clean,
                seed=seed * 101 + index,
                minimum_snr_db=10.0,
                maximum_snr_db=20.0,
            )
            cases.append(
                SyntheticCase(
                    utterance_id=f"{base.utterance_id}@white-{noisy.snr_db:.4f}db",
                    speaker_id=base.speaker_id,
                    family=base.family,
                    condition=AcousticCondition.WHITE_TRAIN,
                    reference=base.reference,
                    hypotheses=base.hypotheses,
                    audio=noisy.noisy,
                )
            )
    return tuple(cases)


def _features(processor: Any, cases: tuple[SyntheticCase, ...], device: torch.device):
    acoustic = processor.feature_extractor(
        [case.audio.numpy() for case in cases],
        sampling_rate=16_000,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )
    return acoustic.input_features.to(device), acoustic.attention_mask.to(device)


def _candidate_wers(cases: tuple[SyntheticCase, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [word_edit_counts(case.reference, hypothesis).wer for hypothesis in case.hypotheses]
            for case in cases
        ],
        dtype=torch.float32,
        device=device,
    )


def _dual_weights(
    cases: tuple[SyntheticCase, ...],
    candidate_wers: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    groups = tuple(RiskGroup(case.family, case.condition.value) for case in cases)
    dual = DualRiskWeights(
        groups,
        ema_decay=config.policy.risk_ema,
        dual_learning_rate=config.policy.dual_learning_rate,
        uniform_mix=config.policy.uniform_mix,
    )
    observed = {
        group: float(candidate_wers[index].clamp(max=2.0).mean().cpu())
        for index, group in enumerate(groups)
    }
    probabilities = dual.update(observed)
    weights = torch.tensor(dual.loss_weights(groups), dtype=torch.float32, device=device)
    return weights, {group.key: value for group, value in probabilities.items()}


def _parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            parameter.detach().float().cpu().flatten()
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
    )


def _run_arm(
    *,
    arm: str,
    config: ExperimentConfig,
    processor: Any,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], Any, tuple[SyntheticCase, ...]]:
    spec = TRAINING_LADDER[arm]
    _seed_everything(seed)
    model = build_lora_whisper(config.model, trainable=True, device=device)
    reference_model = copy.deepcopy(model).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    cases = build_synthetic_cases(seed=seed, corruption=spec.corruption)
    input_features, attention_mask = _features(processor, cases, device)
    hypotheses = tuple(case.hypotheses for case in cases)
    wers = _candidate_wers(cases, device)
    with torch.no_grad():
        first_old = score_hypotheses(
            model,
            processor,
            input_features,
            attention_mask,
            hypotheses,
        )
        repeated_old = score_hypotheses(
            model,
            processor,
            input_features,
            attention_mask,
            hypotheses,
        )
    torch.testing.assert_close(first_old.token_log_probs, repeated_old.token_log_probs)
    if first_old.token_log_probs.dtype != torch.float32:
        raise RuntimeError("smoke rollout log-probabilities were not FP32")
    with torch.no_grad():
        reference_scores = score_hypotheses(
            reference_model,
            processor,
            input_features,
            attention_mask,
            hypotheses,
        )
    if not torch.equal(reference_scores.token_mask, first_old.token_mask):
        raise RuntimeError("reference and rollout smoke token masks diverged")

    weights = None
    probabilities: dict[str, float] = {}
    if spec.group_weighting is GroupWeighting.DUAL:
        weights, probabilities = _dual_weights(cases, wers, config, device)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=3e-5,
    )
    initial = _parameter_vector(model)

    def score_current() -> torch.Tensor:
        current = score_hypotheses(
            model,
            processor,
            input_features,
            attention_mask,
            hypotheses,
        )
        if not torch.equal(current.token_mask, first_old.token_mask):
            raise RuntimeError("current and old smoke token masks diverged")
        return current.token_log_probs

    diagnostics = optimize_frozen_rollout(
        spec=spec,
        score_current=score_current,
        old_token_log_probs=first_old.token_log_probs,
        token_mask=first_old.token_mask,
        candidate_wers=wers,
        optimizer=optimizer,
        inner_updates=config.policy.inner_updates,
        utterance_weights=weights,
        max_gradient_norm=config.policy.gradient_clip,
        reference_token_log_probs=reference_scores.token_log_probs,
        reference_kl_beta=config.policy.reference_kl_beta,
    )
    first_ratios = diagnostics[0].ratios
    torch.testing.assert_close(first_ratios, torch.ones_like(first_ratios))
    moved_ratios = diagnostics[1].ratios
    movement = float((moved_ratios - 1.0).abs().max())
    if movement == 0.0:
        raise RuntimeError(f"{arm} ratio did not move after its first optimizer update")
    drift = float(torch.linalg.vector_norm(_parameter_vector(model) - initial))
    if drift == 0.0:
        raise RuntimeError(f"{arm} adapter did not move")

    result = {
        "arm": arm,
        "objective": {
            "advantage": spec.advantage.value,
            "ratio_unit": spec.ratio_unit.value,
            "clip_rule": spec.clip_rule.value,
            "group_weighting": spec.group_weighting.value,
            "corruption": spec.corruption.value,
        },
        "utterances": len(cases),
        "candidates": len(cases[0].hypotheses),
        "inner_updates": config.policy.inner_updates,
        "old_log_probs_fp32": True,
        "deterministic_rescore_equal": True,
        "initial_ratios_equal_one": True,
        "ratio_movement_after_first_update": movement,
        "ratio_p99": [item.ratio_p99 for item in diagnostics],
        "losses": [item.loss for item in diagnostics],
        "base_policy_losses": [item.base_policy_loss for item in diagnostics],
        "reference_kl_values": [item.reference_kl_value for item in diagnostics],
        "reference_kl_losses": [item.reference_kl_loss for item in diagnostics],
        "total_losses": [item.total_loss for item in diagnostics],
        "reference_kl_beta": config.policy.reference_kl_beta,
        "gradient_norms": [item.gradient_norm for item in diagnostics],
        "adapter_drift": drift,
        "group_probabilities": probabilities,
        "finite": all(
            torch.isfinite(torch.tensor(value)).all().item()
            for value in (
                [item.loss for item in diagnostics],
                [item.gradient_norm for item in diagnostics],
                [item.ratio_p99 for item in diagnostics],
            )
        ),
    }
    return result, model, cases


def run_cuda_smoke(*, config_path: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    """Run every policy arm against a real CUDA Whisper-tiny LoRA model."""
    if not torch.cuda.is_available():
        raise RuntimeError("Modal objective smoke requires a CUDA GPU")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite smoke run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig.from_json(config_path)
    device = torch.device("cuda")
    processor = load_processor(config.model)
    arm_results = []
    final_model = None
    final_cases: tuple[SyntheticCase, ...] | None = None
    for arm in TRAINING_LADDER:
        result, model, cases = _run_arm(
            arm=arm,
            config=config,
            processor=processor,
            device=device,
            seed=seed,
        )
        arm_results.append(result)
        if arm == "fr-cispo":
            final_model = model
            final_cases = cases
        else:
            del model
            torch.cuda.empty_cache()

    assert final_model is not None and final_cases is not None
    clean_cases = tuple(
        case for case in final_cases if case.condition is AcousticCondition.CLEAN
    )
    clean_audio = [case.audio for case in clean_cases]
    predictions_before = greedy_transcribe(
        final_model,
        processor,
        clean_audio,
        batch_size=len(clean_audio),
        device=device,
        max_new_tokens=16,
    )
    checkpoint = output_dir / "fr-cispo-checkpoint"
    final_model.save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(checkpoint / "processor")
    checkpoint_revision = directory_content_hash(checkpoint)
    reloaded = build_lora_whisper(
        config.model,
        adapter_checkpoint=checkpoint,
        trainable=False,
        device=device,
    )
    reloaded_processor = load_saved_processor(checkpoint)
    predictions_after = greedy_transcribe(
        reloaded,
        reloaded_processor,
        clean_audio,
        batch_size=len(clean_audio),
        device=device,
        max_new_tokens=16,
    )
    if predictions_after != predictions_before:
        raise RuntimeError("smoke checkpoint round trip changed greedy predictions")

    rollout_input = clean_cases[0]
    generated = generate_frozen_rollout(
        reloaded,
        reloaded_processor,
        [
            RolloutInput(
                utterance_id=rollout_input.utterance_id,
                speaker_id=rollout_input.speaker_id,
                primary_language="synthetic",
                family=rollout_input.family,
                condition=rollout_input.condition,
                reference=rollout_input.reference,
                audio=rollout_input.audio,
            )
        ],
        candidates=config.policy.candidates,
        temperature=config.policy.rollout_temperature,
        maximum_new_tokens=16,
        model_revision=trainable_parameter_hash(reloaded),
        device=device,
        seed=seed,
    )
    write_immutable_json(output_dir / "rollout-probe.json", generated.frozen.to_dict())

    summary = {
        "artifact_kind": "synthetic_cuda_runtime_smoke",
        "research_valid": False,
        "research_invalid_reason": (
            "Synthetic waveforms and references validate execution only; they are not Svarah evidence."
        ),
        "seed": seed,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "arms": arm_results,
        "generated_rollout_candidates": config.policy.candidates,
        "checkpoint_revision": checkpoint_revision,
        "round_trip_predictions_equal": True,
        "greedy_predictions": predictions_before,
    }
    write_immutable_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = run_cuda_smoke(
        config_path=args.config,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
