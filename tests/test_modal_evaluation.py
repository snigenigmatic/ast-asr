from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ast_asr.modal_evaluation import EVALUATION_ARMS, evaluation_checkpoint, main


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


def test_package_cli_resolves_zero_shot_inside_the_project_environment(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "resolve-checkpoint",
                "--arm",
                "zero-shot",
                "--output-root",
                "/artifacts",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == "base\n"


def test_package_cli_preserves_adapter_arm_coordinates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "resolve-checkpoint",
                "--arm",
                "fr-cispo",
                "--output-root",
                "/artifacts",
                "--checkpoint-run-name",
                "historical-run",
                "--checkpoint-output-name",
                "historical-output",
                "--checkpoint-name",
                "checkpoint-final",
            ]
        )
        == 0
    )
    expected = (
        Path("/artifacts")
        / "historical-run"
        / "historical-output"
        / "checkpoint-final"
    )
    assert capsys.readouterr().out == f"{expected}\n"


def test_modal_wrapper_resolves_checkpoint_only_via_uv_project_environment() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "modal_fr_cispo.py"
    )
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    direct_package_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "ast_asr.modal_evaluation"
    ]

    assert direct_package_imports == []
    assert '"ast_asr.modal_evaluation"' in source
    assert '"resolve-checkpoint"' in source
    assert '"uv"' in source
    assert '"run"' in source
