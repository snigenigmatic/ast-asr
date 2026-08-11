import json
from pathlib import Path

import pytest
import torch

from ast_asr.gates import MAX_KL_PER_TOKEN, MAX_RATIO_P99
from ast_asr.optimization import InnerUpdateDiagnostics
from ast_asr.policy_training import (
    _balanced_probe,
    _clone_trainable_state,
    _raise_if_stability_limit_violated,
    _ratio_stability_violation,
    _save_last_safe_adapter,
    _source_tree_content_hash,
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
