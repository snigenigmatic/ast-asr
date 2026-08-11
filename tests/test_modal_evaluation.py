from __future__ import annotations

from pathlib import Path

import pytest

from ast_asr.modal_evaluation import EVALUATION_ARMS, evaluation_checkpoint


def test_zero_shot_keeps_base_checkpoint_behavior() -> None:
    assert (
        evaluation_checkpoint(
            "zero-shot",
            output_root=Path("/artifacts"),
            checkpoint_run_name="",
            checkpoint_output_name="",
            checkpoint_name="",
        )
        == "base"
    )


@pytest.mark.parametrize("arm", ("sft", "fr-cispo"))
def test_adapter_arms_preserve_their_label_and_require_coordinates(arm: str) -> None:
    checkpoint = evaluation_checkpoint(
        arm,
        output_root=Path("/artifacts"),
        checkpoint_run_name="historical-run",
        checkpoint_output_name="historical-output",
        checkpoint_name="checkpoint-final",
    )

    assert arm in EVALUATION_ARMS
    assert checkpoint == str(
        Path("/artifacts")
        / "historical-run"
        / "historical-output"
        / "checkpoint-final"
    )

    with pytest.raises(ValueError, match="three single-component"):
        evaluation_checkpoint(
            arm,
            output_root=Path("/artifacts"),
            checkpoint_run_name="historical-run",
            checkpoint_output_name="",
            checkpoint_name="checkpoint-final",
        )


def test_adapter_coordinates_must_be_single_path_components() -> None:
    with pytest.raises(ValueError, match="three single-component"):
        evaluation_checkpoint(
            "fr-cispo",
            output_root=Path("/artifacts"),
            checkpoint_run_name="historical/run",
            checkpoint_output_name="historical-output",
            checkpoint_name="checkpoint-final",
        )


def test_unknown_evaluation_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="evaluation arm"):
        evaluation_checkpoint(
            "invented-arm",
            output_root=Path("/artifacts"),
            checkpoint_run_name="",
            checkpoint_output_name="",
            checkpoint_name="",
        )
