from __future__ import annotations

import json
from pathlib import Path

import pytest

from ast_asr.modal_policy_config import (
    main,
    safe_artifact_component,
    validated_reference_kl_beta,
    write_run_specific_policy_config,
)


def _base_config(path: Path) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "policy": {"reference_kl_beta": 0.0, "inner_updates": 4},
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def test_h5_beta_configs_are_run_specific_and_leave_the_volume_config_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "immutable-volume-config.json"
    original = _base_config(source)

    beta0 = write_run_specific_policy_config(
        immutable_config=source,
        run_artifact_root=tmp_path / "artifacts" / "h5-paired",
        output_name="profile-h5-beta0",
        reference_kl_beta=0.0,
    )
    beta04 = write_run_specific_policy_config(
        immutable_config=source,
        run_artifact_root=tmp_path / "artifacts" / "h5-paired",
        output_name="profile-h5-beta04",
        reference_kl_beta=0.04,
    )

    assert json.loads(source.read_text(encoding="utf-8")) == original
    assert json.loads(beta0.read_text(encoding="utf-8"))["policy"][
        "reference_kl_beta"
    ] == 0.0
    assert json.loads(beta04.read_text(encoding="utf-8"))["policy"][
        "reference_kl_beta"
    ] == 0.04
    assert beta0 != beta04


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -0.01))
def test_reference_kl_beta_must_be_finite_and_nonnegative(value: float) -> None:
    with pytest.raises(ValueError, match="reference_kl_beta"):
        validated_reference_kl_beta(value)


@pytest.mark.parametrize("value", ("", ".", "..", "h5/beta04", "h5\\beta04"))
def test_artifact_components_cannot_escape_the_run_root(value: str) -> None:
    with pytest.raises(ValueError, match="single path component"):
        safe_artifact_component(value, field="output_name")


def test_conflicting_beta_cannot_overwrite_an_existing_h5_output_config(
    tmp_path: Path,
) -> None:
    source = tmp_path / "immutable-volume-config.json"
    _base_config(source)
    kwargs = {
        "immutable_config": source,
        "run_artifact_root": tmp_path / "artifacts" / "h5-paired",
        "output_name": "profile-h5-beta0",
    }
    write_run_specific_policy_config(**kwargs, reference_kl_beta=0.0)
    with pytest.raises(FileExistsError, match="immutable artifact"):
        write_run_specific_policy_config(**kwargs, reference_kl_beta=0.04)


def test_policy_config_cli_uses_the_same_validated_derivation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "immutable-volume-config.json"
    _base_config(source)
    run_root = tmp_path / "run"

    assert main(
        [
            "derive",
            "--immutable-config",
            str(source),
            "--run-artifact-root",
            str(run_root),
            "--output-name",
            "h5-beta004-fr-cispo",
            "--reference-kl-beta",
            "0.04",
        ]
    ) == 0

    destination = Path(capsys.readouterr().out.strip())
    assert destination.is_file()
    assert json.loads(destination.read_text())["policy"]["reference_kl_beta"] == 0.04
