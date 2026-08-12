from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ast_asr.historical_kl import (
    K3Threshold,
    assert_locked_manifest_digest,
    build_content_manifest,
    build_locked_h6_input_manifest,
    classify_k3,
    decompose_sampled_k3,
    load_historical_banks,
)
from ast_asr.objectives import sampled_k3_reference_kl
from ast_asr.whisper_policy import score_saved_target_tokens


class _SavedTokenModel(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.config = SimpleNamespace(decoder_start_token_id=1)
        self.dtype_anchor = torch.nn.Parameter(torch.zeros((), dtype=dtype))
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(
            {key: value.detach().clone() for key, value in kwargs.items() if isinstance(value, torch.Tensor)}
        )
        decoder = kwargs["decoder_input_ids"]
        logits = torch.arange(128, dtype=torch.float32).view(1, 1, -1)
        logits = logits.expand(*decoder.shape, -1).clone()
        return SimpleNamespace(logits=logits)


def test_saved_target_scorer_uses_exact_saved_ids_and_never_needs_tokenizer() -> None:
    model = _SavedTokenModel()
    saved_ids = torch.tensor([[[4, 5, 0], [7, 0, 0]]], dtype=torch.long)
    saved_mask = torch.tensor([[[True, True, False], [True, False, False]]])

    scored = score_saved_target_tokens(
        model,
        input_features=torch.ones((1, 2, 3)),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        saved_token_ids=saved_ids,
        saved_token_mask=saved_mask,
        prefix_token_ids=(1,),
        pad_token_id=0,
    )

    torch.testing.assert_close(scored.token_ids, saved_ids)
    torch.testing.assert_close(scored.token_mask, saved_mask)
    assert len(model.calls) == 1
    assert model.calls[0]["input_features"].shape[0] == 2
    assert scored.token_log_probs.dtype == torch.float32


def test_saved_target_scorer_packs_variable_targets_and_keeps_fp16_compute() -> None:
    model = _SavedTokenModel(dtype=torch.float16)
    saved_ids = torch.tensor([[[4, 5], [7, 0]]], dtype=torch.long)
    saved_mask = torch.tensor([[[True, True], [True, False]]])

    scored = score_saved_target_tokens(
        model,
        input_features=torch.ones((1, 2, 3), dtype=torch.float32),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        saved_token_ids=saved_ids,
        saved_token_mask=saved_mask,
        prefix_token_ids=(1, 9, 10),
        pad_token_id=0,
    )

    call = model.calls[0]
    torch.testing.assert_close(
        call["decoder_input_ids"],
        torch.tensor([[1, 9, 10, 4], [1, 9, 10, 7]]),
    )
    torch.testing.assert_close(
        call["decoder_attention_mask"], torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    )
    assert call["input_features"].dtype == torch.float16
    assert scored.token_log_probs.dtype == torch.float32
    expected_log_probs = torch.log_softmax(torch.arange(128, dtype=torch.float32), dim=0)
    torch.testing.assert_close(
        scored.token_log_probs,
        torch.tensor([[[expected_log_probs[4], expected_log_probs[5]], [expected_log_probs[7], 0.0]]]),
    )


def _candidate(index: int) -> dict[str, object]:
    return {
        "hypothesis": "same hypothesis",
        "token_ids": [index + 2, index + 3],
        "token_mask": [True, True],
        "old_token_log_probs": [-1.0, -2.0],
        "old_sequence_log_probability": -1.5,
        "wer": 0.5,
    }


def _bank_payload(cycle: int) -> dict[str, object]:
    utterances = []
    for family_index, family in enumerate(("Dravidian", "Indo-Aryan", "Sino-Tibetan")):
        clean_id = f"u-{cycle}-{family_index}"
        for condition, utterance_id in (("clean", clean_id), ("white_train", f"{clean_id}@white-12.0000db")):
            utterances.append(
                {
                    "utterance_id": utterance_id,
                    "speaker_id": f"speaker-{family_index}",
                    "primary_language": f"language-{family_index}",
                    "family": family,
                    "condition": condition,
                    "reference": "reference",
                    "candidates": [_candidate(candidate) for candidate in range(4)],
                }
            )
    return {"model_revision": f"revision-{cycle}", "utterances": utterances}


def _write_banks(root: Path) -> None:
    root.mkdir()
    for cycle in range(28):
        (root / f"cycle-{cycle:03d}.json").write_text(
            json.dumps(_bank_payload(cycle)), encoding="utf-8"
        )


def test_historical_bank_loader_requires_exact_cycle_pairs_and_allows_same_text(tmp_path: Path) -> None:
    root = tmp_path / "banks"
    _write_banks(root)

    banks = load_historical_banks(root)

    assert tuple(bank.cycle for bank in banks.banks) == tuple(range(28))
    assert banks.banks[0].frozen.utterances[0].candidates[0].hypothesis == "same hypothesis"
    assert len(banks.banks[0].frozen.utterances) == 6


def test_historical_bank_loader_rejects_bad_paired_order_and_non_fp32_logs(tmp_path: Path) -> None:
    root = tmp_path / "banks"
    _write_banks(root)
    path = root / "cycle-003.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["utterances"][1]["condition"] = "clean"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="clean/white_train"):
        load_historical_banks(root)

    payload = _bank_payload(3)
    payload["utterances"][0]["candidates"][0]["old_token_log_probs"][0] = 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="FP32"):
        load_historical_banks(root)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda payload: payload["utterances"][0]["candidates"][0]["token_ids"].__setitem__(0, 2.7), "integer"),
        (lambda payload: payload["utterances"][0]["candidates"][0]["token_ids"].__setitem__(0, True), "integer"),
        (lambda payload: payload["utterances"][0]["candidates"][0]["token_mask"].__setitem__(0, 1), "boolean"),
        (lambda payload: payload["utterances"][0]["candidates"][0]["token_mask"].__setitem__(0, "false"), "boolean"),
        (lambda payload: payload["utterances"][0]["candidates"][0].update({"extra": 1}), "unexpected"),
    ],
)
def test_historical_bank_loader_rejects_raw_json_coercions(
    tmp_path: Path,
    mutate,
    error: str,
) -> None:
    root = tmp_path / "banks"
    _write_banks(root)
    path = root / "cycle-006.json"
    payload = _bank_payload(6)
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=error):
        load_historical_banks(root)


def test_historical_bank_loader_rejects_pair_field_and_snr_drift(tmp_path: Path) -> None:
    root = tmp_path / "banks"
    _write_banks(root)
    path = root / "cycle-007.json"
    payload = _bank_payload(7)
    payload["utterances"][1]["reference"] = "changed"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pair identity"):
        load_historical_banks(root)

    payload = _bank_payload(7)
    payload["utterances"][1]["utterance_id"] = "u-7-0@white-12.0db"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SNR grammar"):
        load_historical_banks(root)


def test_k3_decomposition_is_token_weighted_thresholded_and_matches_h6_term() -> None:
    current = torch.tensor([[[0.0, 0.0], [0.2, 0.0]]], dtype=torch.float32)
    reference = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]], dtype=torch.float32)
    mask = torch.tensor([[[True, True], [True, False]]])

    result = decompose_sampled_k3(
        current_token_log_probs=current,
        reference_token_log_probs=reference,
        token_mask=mask,
        families=("Dravidian",),
        conditions=("clean",),
    )

    expected = sampled_k3_reference_kl(current, reference, mask)
    torch.testing.assert_close(result.bank.k3_per_token, expected)
    assert result.bank.valid_token_count == 3
    assert result.bank.threshold is K3Threshold.AT_OR_ABOVE_LIMIT
    assert result.candidates[0].valid_token_count == 2
    assert result.utterances[0].valid_token_count == 3
    assert result.groups[("Dravidian", "clean")].valid_token_count == 3


def test_k3_decomposition_recomposes_all_six_groups_by_valid_token_count() -> None:
    differences = torch.tensor([0.1, 0.2, 0.3, 0.417, -0.5, -0.6], dtype=torch.float32)
    current = torch.zeros((6, 1, 3), dtype=torch.float32)
    reference = differences.view(6, 1, 1).expand(-1, -1, 3).clone()
    mask = torch.tensor(
        [
            [[True, False, False]],
            [[True, True, False]],
            [[True, True, True]],
            [[True, False, False]],
            [[True, True, False]],
            [[True, True, True]],
        ]
    )
    families = ("Dravidian", "Dravidian", "Indo-Aryan", "Indo-Aryan", "Sino-Tibetan", "Sino-Tibetan")
    conditions = ("clean", "white_train", "clean", "white_train", "clean", "white_train")

    result = decompose_sampled_k3(
        current_token_log_probs=current,
        reference_token_log_probs=reference,
        token_mask=mask,
        families=families,
        conditions=conditions,
    )

    assert len(result.groups) == 6
    assert [summary.valid_token_count for summary in result.utterances] == [1, 2, 3, 1, 2, 3]
    recomposed = sum(
        summary.k3_per_token * summary.valid_token_count for summary in result.groups.values()
    ) / result.bank.valid_token_count
    torch.testing.assert_close(recomposed, result.bank.k3_per_token)
    torch.testing.assert_close(
        result.bank.k3_per_token,
        sampled_k3_reference_kl(current, reference, mask),
    )
    assert classify_k3(torch.tensor(0.1, dtype=torch.float32)) is K3Threshold.AT_OR_ABOVE_LIMIT
    assert classify_k3(torch.nextafter(torch.tensor(0.1), torch.tensor(0.0))) is K3Threshold.BELOW_LIMIT


def test_k3_decomposition_rejects_invalid_selected_terms_and_deterministic_manifest(tmp_path: Path) -> None:
    current = torch.zeros((1, 1, 1), dtype=torch.float32)
    reference = torch.full((1, 1, 1), 21.0, dtype=torch.float32)
    with pytest.raises(FloatingPointError, match="safety bound"):
        decompose_sampled_k3(
            current_token_log_probs=current,
            reference_token_log_probs=reference,
            token_mask=torch.ones((1, 1, 1), dtype=torch.bool),
            families=("Dravidian",),
            conditions=("clean",),
        )

    with pytest.raises(ValueError, match="current token log-probabilities must use FP32"):
        decompose_sampled_k3(
            current_token_log_probs=current.double(),
            reference_token_log_probs=torch.zeros_like(reference),
            token_mask=torch.ones((1, 1, 1), dtype=torch.bool),
            families=("Dravidian",),
            conditions=("clean",),
        )

    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")
    manifest_one = build_content_manifest((second, first), root=tmp_path)
    manifest_two = build_content_manifest((first, second), root=tmp_path)
    assert manifest_one.to_dict() == manifest_two.to_dict()


def test_locked_h6_manifest_has_analyzer_compatible_payload_and_digest(tmp_path: Path) -> None:
    arm = "h6_s2028_beta0"
    arm_root = tmp_path / arm
    entries = []
    for kind, prefix in (("diagnostics", "d"), ("rollouts", "r")):
        directory = arm_root / kind
        directory.mkdir(parents=True)
        for cycle in range(28):
            path = directory / f"cycle-{cycle:03d}.json"
            path.write_text(f"{prefix}{cycle}", encoding="utf-8")
            entries.append(
                {
                    "path": f"{kind}/cycle-{cycle:03d}.json",
                    "sha256": hashlib.sha256(f"{prefix}{cycle}".encode()).hexdigest(),
                    "size_bytes": len(f"{prefix}{cycle}".encode()),
                }
            )
    volume_path = "/artifacts/profile-h6-refkl-beta0-s2028-20260812/h6-beta0-fr-cispo"
    manifest = build_locked_h6_input_manifest(tmp_path, arm=arm, modal_volume_path=volume_path)
    payload = {"arm": arm, "modal_volume_path": volume_path, "files": entries}
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()

    assert manifest.canonical_payload() == payload
    assert manifest.manifest_sha256 == expected
    assert manifest.to_dict()["diagnostic_file_count"] == 28
    assert manifest.to_dict()["rollout_file_count"] == 28
    assert assert_locked_manifest_digest(manifest, expected) is manifest
    with pytest.raises(ValueError, match="92c5b7"):
        assert_locked_manifest_digest(manifest)


def test_saved_target_scorer_repeats_a_full_h7_bank_without_cross_bank_batching() -> None:
    model = _SavedTokenModel()
    saved_ids = torch.arange(2, 2 + 6 * 4 * 3, dtype=torch.long).view(6, 4, 3)
    saved_mask = torch.tensor([[[True, True, False]] * 4] * 6)
    acoustic = torch.ones((6, 2, 3))
    attention = torch.ones((6, 3), dtype=torch.long)

    first = score_saved_target_tokens(
        model,
        input_features=acoustic,
        attention_mask=attention,
        saved_token_ids=saved_ids,
        saved_token_mask=saved_mask,
        prefix_token_ids=(1,),
        pad_token_id=0,
    )
    second = score_saved_target_tokens(
        model,
        input_features=acoustic,
        attention_mask=attention,
        saved_token_ids=saved_ids,
        saved_token_mask=saved_mask,
        prefix_token_ids=(1,),
        pad_token_id=0,
    )

    assert [call["input_features"].shape[0] for call in model.calls] == [24, 24]
    torch.testing.assert_close(first.token_ids, saved_ids)
    torch.testing.assert_close(first.token_mask, saved_mask)
    torch.testing.assert_close(first.token_log_probs, second.token_log_probs, rtol=0, atol=0)


def test_historical_bank_loader_rejects_missing_candidate_and_duplicate_id(tmp_path: Path) -> None:
    root = tmp_path / "banks"
    _write_banks(root)
    path = root / "cycle-005.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["utterances"][0]["candidates"] = payload["utterances"][0]["candidates"][:3]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate count"):
        load_historical_banks(root)

    payload = _bank_payload(5)
    payload["utterances"][1]["utterance_id"] = payload["utterances"][0]["utterance_id"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate utterance IDs"):
        load_historical_banks(root)
