import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ast_asr.gates import MAX_KL_PER_TOKEN, MAX_RATIO_P99
from ast_asr.optimization import InnerUpdateDiagnostics
from ast_asr.policy_training import (
    _balanced_probe,
    _clone_trainable_state,
    _execution_checkpoint_spec,
    _fail_cycle_before_full_diagnostics,
    _freeze_sft_reference_model,
    _input_provenance,
    _loss_trajectory,
    _raise_if_stability_limit_violated,
    _ratio_protocol_violation,
    _ratio_stability_violation,
    _reference_kl_violation,
    _save_last_safe_adapter,
    _source_tree_content_hash,
    _validate_cycle_start_reference_kl,
)
from ast_asr.sft import AudioExample


def _example(utterance_id: str, family: str) -> AudioExample:
    return AudioExample(
        utterance_id=utterance_id,
        speaker_id=f"speaker-{utterance_id}",
        family=family,
        primary_language="language",
        audio_path=Path(f"{utterance_id}.wav"),
        reference="reference",
    )


def test_balanced_probe_round_robins_families_deterministically() -> None:
    examples = [
        _example("z2", "Z-family"),
        _example("a2", "A-family"),
        _example("z1", "Z-family"),
        _example("a1", "A-family"),
        _example("a3", "A-family"),
    ]

    selected = _balanced_probe(examples, 4)

    assert [(example.family, example.utterance_id) for example in selected] == [
        ("A-family", "a1"),
        ("Z-family", "z1"),
        ("A-family", "a2"),
        ("Z-family", "z2"),
    ]


def _diagnostic(*, p99: float, finite: bool = True) -> InnerUpdateDiagnostics:
    return InnerUpdateDiagnostics(
        update=2,
        loss=0.0,
        ratios=torch.ones(1, 1),
        ratio_is_finite=finite,
        ratio_p01=1.0,
        ratio_median=1.0,
        ratio_p99=p99,
        ratio_max=p99,
        gradient_norm=0.1,
    )


class _FakeAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor([1.0]))

    def save_pretrained(self, path: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        path.mkdir(parents=True)
        torch.save(self.adapter.detach(), path / "adapter.pt")


class _FakeProcessor:
    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True)
        (path / "processor.json").write_text("{}\n", encoding="utf-8")


def test_stability_failure_writes_an_immutable_artifact_before_stopping(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="preregistered limit"):
        _raise_if_stability_limit_violated(
            tmp_path,
            cycle=7,
            failure_reason="kl_limit_violated",
            message="preregistered limit reached",
            current_cycle_kl=MAX_KL_PER_TOKEN,
            running_max_kl=MAX_KL_PER_TOKEN,
            old_model_revision="frozen-rollout-revision",
            last_safe_checkpoint=None,
            source_tree_content_hash="source-tree-hash",
        )

    failure = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    assert failure == {
        "artifact_kind": "policy_stability_failure",
        "cycle": 7,
        "current_cycle_sampled_k3_kl_per_token_from_sft": MAX_KL_PER_TOKEN,
        "failure_reason": "kl_limit_violated",
        "message": failure["message"],
        "old_model_revision": "frozen-rollout-revision",
        "preregistered_kl_per_token_limit": MAX_KL_PER_TOKEN,
        "preregistered_ratio_p99_limit": MAX_RATIO_P99,
        "running_max_sampled_k3_kl_per_token_from_sft": MAX_KL_PER_TOKEN,
        "last_safe_adapter_checkpoint": None,
        "source_tree_content_hash": "source-tree-hash",
        "status": "failed",
    }
    assert failure["message"] == "preregistered limit reached"


def test_ratio_safety_finds_non_finite_and_p99_violations() -> None:
    assert _ratio_stability_violation([_diagnostic(p99=float("nan"))])[0] == "non_finite_ratio"
    assert _ratio_stability_violation([_diagnostic(p99=1.0, finite=False)])[0] == "non_finite_ratio"
    reason, message = _ratio_stability_violation([_diagnostic(p99=MAX_RATIO_P99)])
    assert reason == "ratio_p99_limit_violated"
    assert "inner update 2" in message
    assert _ratio_stability_violation([_diagnostic(p99=MAX_RATIO_P99 - 1e-6)]) is None


def test_reference_kl_safety_fails_closed_on_nan_and_limit() -> None:
    assert _reference_kl_violation(float("nan"), cycle=3)[0] == "non_finite_reference_kl"
    assert _reference_kl_violation(MAX_KL_PER_TOKEN, cycle=3)[0] == "kl_limit_violated"
    assert _reference_kl_violation(MAX_KL_PER_TOKEN - 1e-6, cycle=3) is None


def test_frozen_reference_is_eval_gradient_free_and_retains_compute_dtype() -> None:
    reference = torch.nn.Linear(2, 1).half().train()

    frozen = _freeze_sft_reference_model(reference)

    assert frozen.training is False
    assert next(frozen.parameters()).dtype == torch.float16
    assert all(not parameter.requires_grad for parameter in frozen.parameters())


def test_reference_identity_tolerance_applies_only_at_cycle_zero() -> None:
    _validate_cycle_start_reference_kl(0.5, cycle=1)

    with pytest.raises(RuntimeError, match="cycle zero"):
        _validate_cycle_start_reference_kl(0.5, cycle=0)
    with pytest.raises(RuntimeError, match="non-finite"):
        _validate_cycle_start_reference_kl(float("nan"), cycle=1)


def test_early_reference_failure_persists_rollout_diagnostic_and_failure(
    tmp_path: Path,
) -> None:
    frozen = SimpleNamespace(
        model_revision="policy-before-cycle",
        to_dict=lambda: {"model_revision": "policy-before-cycle"},
    )
    generated = SimpleNamespace(frozen=frozen)

    with pytest.raises(RuntimeError, match="identity mismatch"):
        _fail_cycle_before_full_diagnostics(
            tmp_path,
            cycle=0,
            failure_reason="cycle_start_reference_kl_invalid",
            message="identity mismatch",
            generated=generated,
            policy_model=SimpleNamespace(),
            processor=SimpleNamespace(),
            safe_state_before_cycle={},
            running_max_kl=0.0,
            source_tree_content_hash="source-hash",
        )

    assert json.loads((tmp_path / "failure.json").read_text())["failure_reason"] == (
        "cycle_start_reference_kl_invalid"
    )
    assert (tmp_path / "rollouts" / "cycle-000.json").is_file()
    assert (tmp_path / "diagnostics" / "cycle-000.json").is_file()


def test_loss_trajectory_reports_all_reference_kl_components() -> None:
    diagnostic = InnerUpdateDiagnostics(
        update=2,
        loss=1.2,
        ratios=torch.ones(1, 1),
        ratio_is_finite=True,
        ratio_p01=1.0,
        ratio_median=1.0,
        ratio_p99=1.0,
        ratio_max=1.0,
        gradient_norm=0.1,
        base_policy_loss=1.0,
        reference_kl_value=0.4,
        reference_kl_loss=0.2,
        total_loss=1.2,
    )

    assert _loss_trajectory((diagnostic,)) == [
        {
            "update": 2,
            "base_policy_loss": 1.0,
            "reference_kl_value": 0.4,
            "reference_kl_loss": 0.2,
            "total_loss": 1.2,
                "reference_kl_evaluated": False,
                "optimizer_step_applied": False,
        }
    ]


def test_last_safe_adapter_restores_the_snapshot_before_persisting(tmp_path: Path) -> None:
    model = _FakeAdapter()
    snapshot = _clone_trainable_state(model)
    with torch.no_grad():
        model.adapter.fill_(9.0)

    checkpoint = _save_last_safe_adapter(
        tmp_path,
        policy_model=model,
        processor=_FakeProcessor(),
        snapshot=snapshot,
    )

    torch.testing.assert_close(model.adapter, torch.tensor([1.0]))
    torch.testing.assert_close(
        torch.load(tmp_path / checkpoint["path"] / "adapter.pt", weights_only=True),
        torch.tensor([1.0]),
    )
    assert len(checkpoint["revision"]) == 64


def test_source_tree_hash_is_stable_without_git_metadata() -> None:
    config = Path("configs/fr_cispo_tiny.json")
    assert _source_tree_content_hash(config) == _source_tree_content_hash(config)
    assert len(_source_tree_content_hash(config)) == 64


def test_input_provenance_records_profile_cluster_limitations_and_hashes(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    prepared_manifest = tmp_path / "dataset_manifest.json"
    prepared_manifest.write_text(
        json.dumps(
            {
                "identity_mode": "demographic_profile",
                "identity_count": 115,
                "identity_warning": "not authoritative speakers",
                "source_hashes": {"svarah_manifest.json": "abc"},
            }
        ),
        encoding="utf-8",
    )
    fold_directory = tmp_path / "folds"
    fold_directory.mkdir()
    (fold_directory / "fold-0.json").write_text("{}\n", encoding="utf-8")
    sft_checkpoint = tmp_path / "checkpoint-epoch-1"
    sft_checkpoint.mkdir()
    (sft_checkpoint / "adapter.bin").write_bytes(b"adapter")
    config = SimpleNamespace(
        dataset=SimpleNamespace(
            prepared_manifest=prepared_manifest,
            fold_directory=fold_directory,
        )
    )

    provenance = _input_provenance(
        config,
        config_path=config_path,
        fold=0,
        sft_checkpoint=sft_checkpoint,
    )

    assert provenance["identity_count"] == 115
    assert provenance["publication_valid"] is False
    assert provenance["authoritative_svarah_speakers_expected"] == 117
    assert len(provenance["config_sha256"]) == 64
    assert len(provenance["prepared_manifest_sha256"]) == 64
    assert len(provenance["fold_manifest_sha256"]) == 64
    assert len(provenance["sft_checkpoint_revision"]) == 64


def test_bounded_execution_cannot_be_labeled_as_a_final_checkpoint() -> None:
    protocol = _execution_checkpoint_spec(
        rollout_cycles=300,
        configured_rollout_cycles=300,
        probe_examples=32,
        maximum_new_tokens=225,
        configured_maximum_new_tokens=225,
    )
    bounded = _execution_checkpoint_spec(
        rollout_cycles=2,
        configured_rollout_cycles=300,
        probe_examples=6,
        maximum_new_tokens=32,
        configured_maximum_new_tokens=225,
    )

    assert protocol == ("protocol", "checkpoint-final", "final")
    assert bounded == (
        "exploratory_bounded",
        "checkpoint-last-safe",
        "last_safe_bounded",
    )


def test_ratio_protocol_requires_update_zero_identity_and_live_movement() -> None:
    identity = _diagnostic(p99=1.0)
    identity = replace(identity, update=0, optimizer_step_applied=True)
    still_one = replace(identity, update=1, optimizer_step_applied=False)

    violation = _ratio_protocol_violation((identity, still_one))

    assert violation is not None
    assert violation[0] == "post_update_ratio_did_not_move"
