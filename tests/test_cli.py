from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cli_exposes_reproducible_experiment_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "ast_asr", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
    )

    for command in (
        "prepare-data",
        "train-sft",
        "train-policy",
        "evaluate-fold",
        "aggregate-oof",
    ):
        assert command in completed.stdout
    assert "reference_kl_beta" in subprocess.run(
        [sys.executable, "-m", "ast_asr", "train-policy", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
    ).stdout
