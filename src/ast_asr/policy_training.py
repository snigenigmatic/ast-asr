"""Live GRPO/CISPO/FR-CISPO post-training over frozen rollout batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch

from .artifacts import write_immutable_json
from .config import ExperimentConfig
from .corruption import paired_white_noise
from .gates import (
    MAX_KL_PER_TOKEN,
    MAX_RATIO_P99,
    MovementMetrics,
    require_development_gate,
)
from .group_weights import DualRiskWeights, RiskGroup
from .inference import greedy_transcribe
from .ladder import TRAINING_LADDER
from .modeling import (
    build_lora_whisper,
    directory_content_hash,
    load_processor,
    load_saved_processor,
    trainable_parameter_hash,
)
from .objectives import CorruptionPolicy, GroupWeighting, sampled_k3_reference_kl
from .optimization import (
    InnerUpdateDiagnostics,
    InnerUpdateSafetyStop,
    optimize_frozen_rollout,
)
from .rollouts import AcousticCondition
from .sft import AudioExample, _load_audio, _load_experiment_records, _seed_everything
from .whisper_policy import (
    RolloutInput,
    generate_frozen_rollout,
    score_hypotheses,
)

# A separately loaded adapter should reproduce the frozen rollout policy at
# inner update zero.  This is a numerical-invariance check, not a tunable KL
# coefficient.
MAX_INITIAL_REFERENCE_K3_PER_TOKEN = 1e-6


def _balanced_batch(
    examples_by_family: dict[str, list[AudioExample]],
    generator: random.Random,
) -> list[AudioExample]:
    return [
        generator.choice(examples_by_family[family])
        for family in sorted(examples_by_family)
    ]


def _balanced_probe(
    examples: Sequence[AudioExample],
    limit: int,
) -> tuple[AudioExample, ...]:
    """Select a deterministic round-robin probe across language families."""
    by_family: dict[str, list[AudioExample]] = defaultdict(list)
    for example in examples:
        by_family[example.family].append(example)
    for family_examples in by_family.values():
        family_examples.sort(key=lambda example: example.utterance_id)

    selected: list[AudioExample] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for family in sorted(by_family):
            family_examples = by_family[family]
            if offset < len(family_examples):
                selected.append(family_examples[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return tuple(selected)


def _rollout_inputs(
    examples: Sequence[AudioExample],
    *,
    corruption: CorruptionPolicy,
    cycle: int,
    seed: int,
    minimum_snr: float,
    maximum_snr: float,
) -> list[RolloutInput]:
    inputs = []
    for index, example in enumerate(examples):
        audio = _load_audio(example.audio_path)
        clean = RolloutInput(
            utterance_id=example.utterance_id,
            speaker_id=example.speaker_id,
            primary_language=example.primary_language,
            family=example.family,
            condition=AcousticCondition.CLEAN,
            reference=example.reference,
            audio=audio,
        )
        inputs.append(clean)
        if corruption is CorruptionPolicy.PAIRED_WHITE:
            pair = paired_white_noise(
                audio,
                seed=seed * 1_000_003 + cycle * 101 + index,
                minimum_snr_db=minimum_snr,
                maximum_snr_db=maximum_snr,
            )
            inputs.append(
                RolloutInput(
                    utterance_id=f"{example.utterance_id}@white-{pair.snr_db:.4f}db",
                    speaker_id=example.speaker_id,
                    primary_language=example.primary_language,
                    family=example.family,
                    condition=AcousticCondition.WHITE_TRAIN,
                    reference=example.reference,
                    audio=pair.noisy,
                )
            )
    return inputs


def _group_risks(generated) -> dict[RiskGroup, float]:
    risks: dict[RiskGroup, list[float]] = defaultdict(list)
    for utterance in generated.frozen.utterances:
        group = RiskGroup(utterance.family, utterance.condition.value)
        risks[group].append(
            sum(min(candidate.wer, 2.0) for candidate in utterance.candidates)
            / len(utterance.candidates)
        )
    return {group: sum(values) / len(values) for group, values in risks.items()}


def _candidate_texts(generated) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(candidate.hypothesis for candidate in utterance.candidates)
        for utterance in generated.frozen.utterances
    )


def _sampled_k3_kl(
    current: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    return float(
        sampled_k3_reference_kl(current, reference, mask).detach().cpu()
    )


def _freeze_sft_reference_model(model: torch.nn.Module) -> torch.nn.Module:
    """Make the SFT reference immutable while retaining policy compute dtype.

    ``score_hypotheses`` always returns FP32 log-probabilities.  Keeping model
    compute dtype equal to the policy avoids a precision-mismatch K3 baseline
    at inner update zero.
    """
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _validate_cycle_start_reference_kl(value: float, *, cycle: int) -> None:
    """Validate fixed-reference scoring without forbidding real policy drift.

    At cycle zero the policy and SFT reference are independently loaded copies
    of the same adapter, so their sampled K3 should be numerical zero.  From
    cycle one onward the rollout policy has already moved; a nonzero distance
    from SFT is expected and is governed by the ordinary per-cycle KL gate.
    """
    if not math.isfinite(value):
        raise RuntimeError(
            f"SFT reference K3 at cycle start was non-finite at cycle {cycle}"
        )
    if cycle == 0 and value > MAX_INITIAL_REFERENCE_K3_PER_TOKEN:
        raise RuntimeError(
            "SFT reference K3 at cycle zero exceeded the numerical "
            f"tolerance {MAX_INITIAL_REFERENCE_K3_PER_TOKEN:.6g}: {value:.6g}"
        )


def _loss_trajectory(
    diagnostics: Sequence[InnerUpdateDiagnostics],
) -> list[dict[str, float | int]]:
    """Render all loss components without changing the legacy total-loss field."""
    return [
        {
            "update": item.update,
            "base_policy_loss": item.base_policy_loss,
            "reference_kl_value": item.reference_kl_value,
            "reference_kl_loss": item.reference_kl_loss,
            "total_loss": item.total_loss,
            "reference_kl_evaluated": item.reference_kl_evaluated,
            "optimizer_step_applied": item.optimizer_step_applied,
        }
        for item in diagnostics
    ]


def _render_group_values(values: dict[RiskGroup, float]) -> dict[str, float]:
    """Make a stable, JSON-ready representation of group-indexed state."""
    return {group.key: values[group] for group in sorted(values)}


def _source_tree_content_hash(config_path: Path) -> str:
    """Hash executable source plus the exact invoked configuration, without Git."""
    source_root = Path(__file__).resolve().parent
    files = [
        (f"src/ast_asr/{path.relative_to(source_root).as_posix()}", path)
        for path in source_root.rglob("*.py")
    ]
    files.append((f"invoked-config/{config_path.name}", config_path.resolve()))
    digest = hashlib.sha256()
    for label, path in sorted(files):
        digest.update(label.encode("utf-8"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_provenance(
    config: ExperimentConfig,
    *,
    config_path: Path,
    fold: int,
    sft_checkpoint: Path,
) -> dict[str, object]:
    """Bind a policy run to its exact config, data split, and source adapter."""
    prepared_manifest = config.dataset.prepared_manifest
    fold_manifest = config.dataset.fold_directory / f"fold-{fold}.json"
    for label, path in (
        ("prepared dataset manifest", prepared_manifest),
        ("fold manifest", fold_manifest),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if not sft_checkpoint.is_dir():
        raise FileNotFoundError(f"SFT checkpoint is missing: {sft_checkpoint}")

    prepared = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    identity_mode = str(prepared.get("identity_mode", "unknown"))
    identity_count = int(prepared.get("identity_count", 0))
    publication_valid = identity_mode == "authoritative" and identity_count == 117
    return {
        "config_path": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "prepared_manifest_path": str(prepared_manifest),
        "prepared_manifest_sha256": _file_sha256(prepared_manifest),
        "prepared_source_hashes": prepared.get("source_hashes", {}),
        "fold_manifest_path": str(fold_manifest),
        "fold_manifest_sha256": _file_sha256(fold_manifest),
        "sft_checkpoint_path": str(sft_checkpoint),
        "sft_checkpoint_revision": directory_content_hash(sft_checkpoint),
        "identity_mode": identity_mode,
        "identity_count": identity_count,
        "identity_warning": prepared.get("identity_warning"),
        "authoritative_svarah_speakers_expected": 117,
        "publication_valid": publication_valid,
    }


def _execution_checkpoint_spec(
    *,
    rollout_cycles: int,
    configured_rollout_cycles: int,
    probe_examples: int,
    maximum_new_tokens: int,
    configured_maximum_new_tokens: int,
) -> tuple[str, str, str]:
    """Return execution mode, checkpoint directory, and non-ambiguous role."""
    is_protocol = (
        rollout_cycles == configured_rollout_cycles
        and probe_examples == 32
        and maximum_new_tokens == configured_maximum_new_tokens
    )
    if is_protocol:
        return "protocol", "checkpoint-final", "final"
    return "exploratory_bounded", "checkpoint-last-safe", "last_safe_bounded"


def _ratio_protocol_violation(
    diagnostics: Sequence[InnerUpdateDiagnostics],
) -> tuple[str, str] | None:
    """Enforce identity, movement, step, and stability gates for one cycle."""
    if not diagnostics:
        return "missing_ratio_diagnostics", "optimizer emitted no ratio diagnostics"
    if not torch.allclose(
        diagnostics[0].ratios,
        torch.ones_like(diagnostics[0].ratios),
    ):
        return (
            "update_zero_ratio_identity_violated",
            "rollout/current ratio was not one at inner update zero",
        )
    if len(diagnostics) < 2 or torch.allclose(
        diagnostics[1].ratios,
        torch.ones_like(diagnostics[1].ratios),
    ):
        return (
            "post_update_ratio_did_not_move",
            "rollout/current ratio did not move after the first optimizer update",
        )
    applied = sum(item.optimizer_step_applied for item in diagnostics)
    expected = diagnostics[-1].update
    if applied != expected:
        return (
            "optimizer_step_skipped",
            f"optimizer applied {applied} of {expected} expected inner updates",
        )
    return _ratio_stability_violation(diagnostics)


def _ratio_stability_violation(
    diagnostics: Sequence[InnerUpdateDiagnostics],
) -> tuple[str, str] | None:
    """Return the first ratio safety violation from one frozen-rollout cycle."""
    for item in diagnostics:
        values = (item.ratio_p01, item.ratio_median, item.ratio_p99, item.ratio_max)
        if not item.ratio_is_finite or not all(math.isfinite(value) for value in values):
            return (
                "non_finite_ratio",
                f"non-finite ratio diagnostic at inner update {item.update}",
            )
        if item.ratio_p99 >= MAX_RATIO_P99:
            return (
                "ratio_p99_limit_violated",
                (
                    "ratio p99 reached the preregistered limit "
                    f"{MAX_RATIO_P99:.6g} at inner update {item.update}: "
                    f"{item.ratio_p99:.6g}"
                ),
            )
    return None


def _reference_kl_violation(
    value: float,
    *,
    cycle: int,
) -> tuple[str, str] | None:
    """Fail closed when the post-cycle reference-KL diagnostic is unsafe."""
    if not math.isfinite(value):
        return (
            "non_finite_reference_kl",
            f"sampled K3 KL/token from the SFT checkpoint was non-finite at cycle {cycle}",
        )
    if value >= MAX_KL_PER_TOKEN:
        return (
            "kl_limit_violated",
            (
                "sampled K3 KL/token from the SFT checkpoint reached the preregistered "
                f"limit {MAX_KL_PER_TOKEN:.6g} at cycle {cycle}: {value:.6g}"
            ),
        )
    return None


def _clone_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Hold one adapter state in memory, avoiding per-cycle checkpoints."""
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise ValueError("policy model has no trainable parameters to snapshot")
    return state


def _restore_trainable_state(
    model: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> None:
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if current.keys() != snapshot.keys():
        raise RuntimeError("trainable parameter names changed since the safe snapshot")
    with torch.no_grad():
        for name, parameter in current.items():
            saved = snapshot[name]
            if parameter.shape != saved.shape:
                raise RuntimeError(f"trainable parameter shape changed for {name}")
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))


def _save_last_safe_adapter(
    output: Path,
    *,
    policy_model: torch.nn.Module,
    processor,
    snapshot: dict[str, torch.Tensor],
) -> dict[str, str]:
    """Restore and persist the state immediately before a failed later cycle."""
    checkpoint = output / "checkpoint-last-safe"
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite last-safe checkpoint: {checkpoint}")
    _restore_trainable_state(policy_model, snapshot)
    policy_model.save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(checkpoint / "processor")
    return {
        "path": checkpoint.name,
        "revision": directory_content_hash(checkpoint),
    }


def _raise_if_stability_limit_violated(
    output: Path,
    *,
    cycle: int,
    failure_reason: str,
    message: str,
    current_cycle_kl: float | None,
    running_max_kl: float,
    old_model_revision: str,
    last_safe_checkpoint: dict[str, str] | None,
    source_tree_content_hash: str,
) -> None:
    """Persist a safety-stop artifact before stopping an unsafe policy run."""
    write_immutable_json(
        output / "failure.json",
        {
            "artifact_kind": "policy_stability_failure",
            "status": "failed",
            "failure_reason": failure_reason,
            "message": message,
            "cycle": cycle,
            "old_model_revision": old_model_revision,
            "current_cycle_sampled_k3_kl_per_token_from_sft": current_cycle_kl,
            "running_max_sampled_k3_kl_per_token_from_sft": running_max_kl,
            "preregistered_kl_per_token_limit": MAX_KL_PER_TOKEN,
            "preregistered_ratio_p99_limit": MAX_RATIO_P99,
            "last_safe_adapter_checkpoint": last_safe_checkpoint,
            "source_tree_content_hash": source_tree_content_hash,
        },
    )
    raise RuntimeError(message)


def _fail_cycle_before_full_diagnostics(
    output: Path,
    *,
    cycle: int,
    failure_reason: str,
    message: str,
    generated,
    policy_model: torch.nn.Module,
    processor,
    safe_state_before_cycle: dict[str, torch.Tensor],
    running_max_kl: float,
    source_tree_content_hash: str,
) -> None:
    """Persist provenance when reference validation/optimization fails early."""
    write_immutable_json(
        output / "rollouts" / f"cycle-{cycle:03d}.json",
        generated.frozen.to_dict(),
    )
    write_immutable_json(
        output / "diagnostics" / f"cycle-{cycle:03d}.json",
        {
            "cycle": cycle,
            "status": "failed_before_full_diagnostics",
            "failure_reason": failure_reason,
            "message": message,
            "old_model_revision": generated.frozen.model_revision,
        },
    )
    last_safe_checkpoint = (
        _save_last_safe_adapter(
            output,
            policy_model=policy_model,
            processor=processor,
            snapshot=safe_state_before_cycle,
        )
        if cycle > 0
        else None
    )
    _raise_if_stability_limit_violated(
        output,
        cycle=cycle,
        failure_reason=failure_reason,
        message=message,
        current_cycle_kl=None,
        running_max_kl=running_max_kl,
        old_model_revision=generated.frozen.model_revision,
        last_safe_checkpoint=last_safe_checkpoint,
        source_tree_content_hash=source_tree_content_hash,
    )


def _parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    values = [
        parameter.detach().float().cpu().flatten()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return torch.cat(values)


def train_policy(args: argparse.Namespace) -> None:
    require_development_gate(args.fold, args.development_gate)
    config = ExperimentConfig.from_json(args.config)
    source_tree_content_hash = _source_tree_content_hash(args.config)
    if args.arm not in TRAINING_LADDER:
        raise ValueError(f"unknown policy arm {args.arm!r}; choose from {tuple(TRAINING_LADDER)}")
    if args.learning_rate not in config.policy.learning_rate_grid:
        raise ValueError("policy learning rate must come from the configured development grid")
    spec = TRAINING_LADDER[args.arm]
    if args.fold != 0 and args.arm not in {"cispo-mwer", "fair-cispo", "fr-cispo"}:
        raise ValueError("only the preregistered full-fold arms may run after development")
    _seed_everything(args.seed)
    rollout_cycles = (
        config.policy.rollout_cycles
        if args.rollout_cycles is None
        else args.rollout_cycles
    )
    if not 1 <= rollout_cycles <= config.policy.rollout_cycles:
        raise ValueError("rollout cycles override exceeds the configured protocol")
    probe_examples = 32 if args.probe_examples is None else args.probe_examples
    if probe_examples < 1:
        raise ValueError("probe examples must be positive")
    maximum_new_tokens = (
        config.policy.maximum_new_tokens
        if args.maximum_new_tokens is None
        else args.maximum_new_tokens
    )
    if not 1 <= maximum_new_tokens <= config.policy.maximum_new_tokens:
        raise ValueError("maximum new tokens override exceeds the configured protocol")
    execution_mode, checkpoint_name, checkpoint_role = _execution_checkpoint_spec(
        rollout_cycles=rollout_cycles,
        configured_rollout_cycles=config.policy.rollout_cycles,
        probe_examples=probe_examples,
        maximum_new_tokens=maximum_new_tokens,
        configured_maximum_new_tokens=config.policy.maximum_new_tokens,
    )
    input_provenance = _input_provenance(
        config,
        config_path=args.config,
        fold=args.fold,
        sft_checkpoint=args.sft_checkpoint,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_examples, validation_examples = _load_experiment_records(config, args.fold)
    examples_by_family: dict[str, list[AudioExample]] = defaultdict(list)
    for example in train_examples:
        examples_by_family[example.family].append(example)
    if any(not values for values in examples_by_family.values()):
        raise ValueError("every language family requires training utterances")

    processor = load_processor(config.model)
    policy_model = build_lora_whisper(
        config.model,
        adapter_checkpoint=args.sft_checkpoint,
        trainable=True,
        device=device,
    )
    reference_model = build_lora_whisper(
        config.model,
        adapter_checkpoint=args.sft_checkpoint,
        trainable=False,
        device=device,
    )
    _freeze_sft_reference_model(reference_model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy_model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite policy run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_immutable_json(output / "resolved_config.json", config.to_dict())
    execution_overrides = {
        "rollout_cycles": rollout_cycles,
        "probe_examples": probe_examples,
        "maximum_new_tokens": maximum_new_tokens,
        "source_tree_content_hash": source_tree_content_hash,
        "execution_mode": execution_mode,
        "checkpoint_role": checkpoint_role,
        "input_provenance": input_provenance,
        "randomness": {
            "root_seed": args.seed,
            "balanced_batch_rng": "python.Random(root_seed), consumed once per cycle",
            "rollout_seed_rule": "root_seed * 1000003 + cycle",
            "corruption_seed_rule": (
                "root_seed * 1000003 + cycle * 101 + balanced_batch_index"
            ),
            "corruption_realizations_recorded_in": "rollouts/cycle-*.json",
        },
    }
    write_immutable_json(output / "execution_overrides.json", execution_overrides)

    conditions = [AcousticCondition.CLEAN.value]
    if spec.corruption is CorruptionPolicy.PAIRED_WHITE:
        conditions.append(AcousticCondition.WHITE_TRAIN.value)
    risk_groups = tuple(
        RiskGroup(family, condition)
        for family in sorted(examples_by_family)
        for condition in conditions
    )
    dual = DualRiskWeights(
        risk_groups,
        ema_decay=config.policy.risk_ema,
        dual_learning_rate=config.policy.dual_learning_rate,
        uniform_mix=config.policy.uniform_mix,
    )
    batch_generator = random.Random(args.seed)
    initial_parameters = _parameter_vector(policy_model)
    probe = _balanced_probe(validation_examples, probe_examples)
    probe_audios = [_load_audio(example.audio_path) for example in probe]
    before_predictions = greedy_transcribe(
        policy_model,
        processor,
        probe_audios,
        batch_size=config.sft.evaluation_batch_size,
        device=device,
        max_new_tokens=maximum_new_tokens,
    )

    all_ratio_p99 = []
    running_max_kl = 0.0
    ratio_movement_cycles = 0
    optimizer_steps_applied = 0
    for cycle in range(rollout_cycles):
        safe_state_before_cycle = _clone_trainable_state(policy_model)
        selected = _balanced_batch(examples_by_family, batch_generator)
        inputs = _rollout_inputs(
            selected,
            corruption=spec.corruption,
            cycle=cycle,
            seed=args.seed,
            minimum_snr=config.policy.training_snr_db[0],
            maximum_snr=config.policy.training_snr_db[1],
        )
        generated = generate_frozen_rollout(
            policy_model,
            processor,
            inputs,
            candidates=config.policy.candidates,
            temperature=config.policy.rollout_temperature,
            maximum_new_tokens=maximum_new_tokens,
            model_revision=trainable_parameter_hash(policy_model),
            device=device,
            seed=args.seed * 1_000_003 + cycle,
        )
        tensors = generated.frozen.objective_tensors(device)
        hypotheses = _candidate_texts(generated)
        with torch.no_grad():
            reference_scores = score_hypotheses(
                reference_model,
                processor,
                generated.input_features,
                generated.attention_mask,
                hypotheses,
            )
        if not torch.equal(reference_scores.token_mask, tensors.token_mask):
            _fail_cycle_before_full_diagnostics(
                output,
                cycle=cycle,
                failure_reason="reference_rollout_token_mask_mismatch",
                message="reference and rollout token masks diverged",
                generated=generated,
                policy_model=policy_model,
                processor=processor,
                safe_state_before_cycle=safe_state_before_cycle,
                running_max_kl=running_max_kl,
                source_tree_content_hash=source_tree_content_hash,
            )
        reference_token_log_probs = reference_scores.token_log_probs.detach().clone()
        if reference_token_log_probs.dtype != torch.float32:
            _fail_cycle_before_full_diagnostics(
                output,
                cycle=cycle,
                failure_reason="reference_log_prob_dtype_invalid",
                message="reference token log-probabilities were not FP32",
                generated=generated,
                policy_model=policy_model,
                processor=processor,
                safe_state_before_cycle=safe_state_before_cycle,
                running_max_kl=running_max_kl,
                source_tree_content_hash=source_tree_content_hash,
            )
        try:
            update_zero_reference_kl = _sampled_k3_kl(
                tensors.old_token_log_probs,
                reference_token_log_probs,
                tensors.token_mask,
            )
            _validate_cycle_start_reference_kl(update_zero_reference_kl, cycle=cycle)
        except (FloatingPointError, RuntimeError) as error:
            _fail_cycle_before_full_diagnostics(
                output,
                cycle=cycle,
                failure_reason="cycle_start_reference_kl_invalid",
                message=str(error),
                generated=generated,
                policy_model=policy_model,
                processor=processor,
                safe_state_before_cycle=safe_state_before_cycle,
                running_max_kl=running_max_kl,
                source_tree_content_hash=source_tree_content_hash,
            )

        observed_risks = _group_risks(generated)
        probabilities_before_update = dual.probabilities
        ema_risks_before_update = dual.ema_risks
        probabilities_after_update = probabilities_before_update
        ema_risks_after_update = ema_risks_before_update
        utterance_weights = None
        utterance_weight_values: tuple[float, ...] | None = None
        if spec.group_weighting is GroupWeighting.DUAL:
            probabilities_after_update = dual.update(observed_risks)
            ema_risks_after_update = dual.ema_risks
            groups = tuple(
                RiskGroup(utterance.family, utterance.condition.value)
                for utterance in generated.frozen.utterances
            )
            utterance_weight_values = dual.loss_weights(groups)
            utterance_weights = torch.tensor(
                utterance_weight_values,
                dtype=torch.float32,
                device=device,
            )

        def score_current(
            generated_batch=generated,
            current_hypotheses=hypotheses,
            objective_tensors=tensors,
        ) -> torch.Tensor:
            scored = score_hypotheses(
                policy_model,
                processor,
                generated_batch.input_features,
                generated_batch.attention_mask,
                current_hypotheses,
            )
            if not torch.equal(scored.token_mask, objective_tensors.token_mask):
                raise RuntimeError("current and old token masks diverged")
            return scored.token_log_probs

        def ratio_safety_check(item: InnerUpdateDiagnostics) -> str | None:
            if item.update == 0 and not torch.allclose(
                item.ratios,
                torch.ones_like(item.ratios),
            ):
                return "rollout/current ratio was not one at inner update zero"
            if item.update == 1 and torch.allclose(
                item.ratios,
                torch.ones_like(item.ratios),
            ):
                return "rollout/current ratio did not move after the first optimizer update"
            violation = _ratio_stability_violation((item,))
            return None if violation is None else violation[1]

        try:
            diagnostics = optimize_frozen_rollout(
                spec=spec,
                score_current=score_current,
                old_token_log_probs=tensors.old_token_log_probs,
                token_mask=tensors.token_mask,
                candidate_wers=tensors.candidate_wers,
                optimizer=optimizer,
                inner_updates=config.policy.inner_updates,
                utterance_weights=utterance_weights,
                max_gradient_norm=config.policy.gradient_clip,
                update_safety_check=ratio_safety_check,
                reference_token_log_probs=reference_token_log_probs,
                reference_kl_beta=config.policy.reference_kl_beta,
            )
        except InnerUpdateSafetyStop as stop:
            diagnostics = stop.diagnostics
        except (FloatingPointError, RuntimeError) as error:
            _fail_cycle_before_full_diagnostics(
                output,
                cycle=cycle,
                failure_reason="policy_optimization_numerical_failure",
                message=str(error),
                generated=generated,
                policy_model=policy_model,
                processor=processor,
                safe_state_before_cycle=safe_state_before_cycle,
                running_max_kl=running_max_kl,
                source_tree_content_hash=source_tree_content_hash,
            )
        ratio_violation = _ratio_protocol_violation(diagnostics)
        cycle_optimizer_steps = sum(
            item.optimizer_step_applied for item in diagnostics
        )
        optimizer_steps_applied += cycle_optimizer_steps
        if ratio_violation is None:
            ratio_movement_cycles += 1
        all_ratio_p99.extend(item.ratio_p99 for item in diagnostics)
        current_cycle_kl: float | None = None
        if ratio_violation is None:
            with torch.no_grad():
                final_current = score_hypotheses(
                    policy_model,
                    processor,
                    generated.input_features,
                    generated.attention_mask,
                    hypotheses,
                ).token_log_probs
            current_cycle_kl = _sampled_k3_kl(
                final_current,
                reference_token_log_probs,
                tensors.token_mask,
            )
            if math.isfinite(current_cycle_kl):
                running_max_kl = max(running_max_kl, current_cycle_kl)
        current_cycle_kl_for_artifact = (
            current_cycle_kl
            if current_cycle_kl is None or math.isfinite(current_cycle_kl)
            else None
        )
        cycle_summary = {
            "cycle": cycle,
            "old_model_revision": generated.frozen.model_revision,
            "losses": [item.loss for item in diagnostics],
            "loss_trajectory": _loss_trajectory(diagnostics),
            "reference_kl": {
                "beta": config.policy.reference_kl_beta,
                "estimator": "sampled_k3_response_tokens",
                "reference_fixed_eval": True,
                "reference_token_log_probs_fp32": True,
                "update_zero_k3_kl_per_token": update_zero_reference_kl,
                "update_zero_k3_tolerance": MAX_INITIAL_REFERENCE_K3_PER_TOKEN,
            },
            "ratio_p99": [item.ratio_p99 for item in diagnostics],
            "gradient_norm": [item.gradient_norm for item in diagnostics],
            "optimizer_steps_applied": cycle_optimizer_steps,
            "optimizer_steps_expected": config.policy.inner_updates,
            "ratio_trajectory": [
                {
                    "update": item.update,
                    "finite": item.ratio_is_finite,
                    "p01": item.ratio_p01,
                    "median": item.ratio_median,
                    "p99": item.ratio_p99,
                    "maximum": item.ratio_max,
                }
                for item in diagnostics
            ],
            "group_weight_trajectory": {
                "observed_risks": _render_group_values(observed_risks),
                "ema_risks_before_update": _render_group_values(ema_risks_before_update),
                "ema_risks_after_update": _render_group_values(ema_risks_after_update),
                "probabilities_before_update": _render_group_values(
                    probabilities_before_update
                ),
                "probabilities_after_update": _render_group_values(
                    probabilities_after_update
                ),
                "utterance_loss_weights": (
                    list(utterance_weight_values)
                    if utterance_weight_values is not None
                    else None
                ),
            },
            "group_probabilities": _render_group_values(probabilities_after_update),
            "current_cycle_sampled_k3_kl_per_token_from_sft": current_cycle_kl_for_artifact,
            "current_cycle_sampled_k3_kl_measured": current_cycle_kl is not None,
            "current_cycle_sampled_k3_kl_is_finite": (
                current_cycle_kl is not None and math.isfinite(current_cycle_kl)
            ),
            "running_max_sampled_k3_kl_per_token_from_sft": running_max_kl,
            "sampled_k3_kl_from_sft": running_max_kl,
        }
        write_immutable_json(
            output / "rollouts" / f"cycle-{cycle:03d}.json",
            generated.frozen.to_dict(),
        )
        write_immutable_json(
            output / "diagnostics" / f"cycle-{cycle:03d}.json",
            cycle_summary,
        )
        if ratio_violation is not None:
            failure_reason, message = ratio_violation
            last_safe_checkpoint = (
                _save_last_safe_adapter(
                    output,
                    policy_model=policy_model,
                    processor=processor,
                    snapshot=safe_state_before_cycle,
                )
                if cycle > 0
                else None
            )
            _raise_if_stability_limit_violated(
                output,
                cycle=cycle,
                failure_reason=failure_reason,
                message=message,
                current_cycle_kl=None,
                running_max_kl=running_max_kl,
                old_model_revision=generated.frozen.model_revision,
                last_safe_checkpoint=last_safe_checkpoint,
                source_tree_content_hash=source_tree_content_hash,
            )
        assert current_cycle_kl is not None
        kl_violation = _reference_kl_violation(current_cycle_kl, cycle=cycle)
        if kl_violation is not None:
            failure_reason, message = kl_violation
            last_safe_checkpoint = (
                _save_last_safe_adapter(
                    output,
                    policy_model=policy_model,
                    processor=processor,
                    snapshot=safe_state_before_cycle,
                )
                if cycle > 0
                else None
            )
            _raise_if_stability_limit_violated(
                output,
                cycle=cycle,
                failure_reason=failure_reason,
                message=message,
                current_cycle_kl=current_cycle_kl_for_artifact,
                running_max_kl=running_max_kl,
                old_model_revision=generated.frozen.model_revision,
                last_safe_checkpoint=last_safe_checkpoint,
                source_tree_content_hash=source_tree_content_hash,
            )

    saved_checkpoint = output / checkpoint_name
    policy_model.save_pretrained(saved_checkpoint, safe_serialization=True)
    processor.save_pretrained(saved_checkpoint / "processor")
    checkpoint_revision = directory_content_hash(saved_checkpoint)
    after_predictions = greedy_transcribe(
        policy_model,
        processor,
        probe_audios,
        batch_size=config.sft.evaluation_batch_size,
        device=device,
        max_new_tokens=maximum_new_tokens,
    )
    final_parameters = _parameter_vector(policy_model)
    drift = float(torch.linalg.vector_norm(final_parameters - initial_parameters))
    has_non_finite_values = not bool(torch.isfinite(final_parameters).all())
    expected_optimizer_steps = rollout_cycles * config.policy.inner_updates
    skipped_steps = expected_optimizer_steps - optimizer_steps_applied
    movement = MovementMetrics(
        has_non_finite_values=has_non_finite_values,
        skipped_steps=skipped_steps,
        adapter_drift=drift,
        greedy_predictions_changed=after_predictions != before_predictions,
        ratio_p99=max(all_ratio_p99),
        kl_per_token=running_max_kl,
    )

    reloaded = build_lora_whisper(
        config.model,
        adapter_checkpoint=saved_checkpoint,
        trainable=False,
        device=device,
    )
    reloaded_processor = load_saved_processor(saved_checkpoint)
    round_trip_predictions = greedy_transcribe(
        reloaded,
        reloaded_processor,
        probe_audios,
        batch_size=config.sft.evaluation_batch_size,
        device=device,
        max_new_tokens=maximum_new_tokens,
    )
    if round_trip_predictions != after_predictions:
        raise RuntimeError("policy checkpoint round-trip changed predictions")

    write_immutable_json(
        output / "movement.json",
        {
            "passed": movement.passed,
            "has_non_finite_values": movement.has_non_finite_values,
            "skipped_steps": movement.skipped_steps,
            "adapter_drift": movement.adapter_drift,
            "greedy_predictions_changed": movement.greedy_predictions_changed,
            "ratio_p99": movement.ratio_p99,
            "running_max_sampled_k3_kl_per_token_from_sft": movement.kl_per_token,
            "preregistered_kl_per_token_limit": MAX_KL_PER_TOKEN,
            "ratio_movement_cycles": ratio_movement_cycles,
            "rollout_cycles": rollout_cycles,
            "optimizer_steps_applied": optimizer_steps_applied,
            "optimizer_steps_expected": expected_optimizer_steps,
        },
    )
    write_immutable_json(
        output / "run.json",
        {
            "arm": args.arm,
            "objective": {
                "advantage": spec.advantage.value,
                "ratio_unit": spec.ratio_unit.value,
                "clip_rule": spec.clip_rule.value,
                "group_weighting": spec.group_weighting.value,
                "corruption": spec.corruption.value,
            },
            "reference_kl": {
                "beta": config.policy.reference_kl_beta,
                "estimator": "sampled_k3_response_tokens",
                "reference_fixed_eval": True,
                "reference_token_log_probs_fp32": True,
                "update_zero_k3_tolerance": MAX_INITIAL_REFERENCE_K3_PER_TOKEN,
            },
            "fold": args.fold,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "model_id": config.model.model_id,
            "model_revision": config.model.revision,
            "dataset_id": config.dataset.dataset_id,
            "dataset_revision": config.dataset.revision,
            "sft_checkpoint": str(args.sft_checkpoint),
            "input_provenance": input_provenance,
            "publication_valid": input_provenance["publication_valid"],
            "checkpoint": {
                "path": saved_checkpoint.name,
                "revision": checkpoint_revision,
                "role": checkpoint_role,
            },
            "final_checkpoint": saved_checkpoint.name if checkpoint_role == "final" else None,
            "final_checkpoint_revision": (
                checkpoint_revision if checkpoint_role == "final" else None
            ),
            "last_safe_checkpoint": (
                saved_checkpoint.name if checkpoint_role != "final" else None
            ),
            "last_safe_checkpoint_revision": (
                checkpoint_revision if checkpoint_role != "final" else None
            ),
            "inner_updates": config.policy.inner_updates,
            "rollout_cycles": rollout_cycles,
            "execution_mode": execution_mode,
            "execution_overrides": execution_overrides,
            "source_tree_content_hash": source_tree_content_hash,
            "movement_gate_passed": movement.passed,
            "round_trip_predictions_equal": True,
        },
    )
    write_immutable_json(
        output / "movement_probe_predictions.json",
        [
            {
                "utterance_id": example.utterance_id,
                "speaker_id": example.speaker_id,
                "reference": example.reference,
                "before": before,
                "after": after,
            }
            for example, before, after in zip(
                probe,
                before_predictions,
                after_predictions,
                strict=True,
            )
        ],
    )
