"""Validation for Modal evaluation-arm labels and checkpoint coordinates."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .ladder import TRAINING_LADDER

ZERO_SHOT_ARM = "zero-shot"
SFT_ARM = "sft"
ADAPTER_EVALUATION_ARMS = frozenset((SFT_ARM, *TRAINING_LADDER))
EVALUATION_ARMS = frozenset((ZERO_SHOT_ARM, *ADAPTER_EVALUATION_ARMS))


def evaluation_checkpoint(
    arm: str,
    *,
    output_root: Path,
    checkpoint_run_name: str,
    checkpoint_output_name: str,
    checkpoint_name: str,
) -> str:
    """Resolve a base model or require a safe adapter checkpoint path."""
    if arm not in EVALUATION_ARMS:
        allowed = ", ".join(sorted(EVALUATION_ARMS))
        raise ValueError(f"evaluation arm must be one of: {allowed}")
    if arm == ZERO_SHOT_ARM:
        return "base"

    components = (checkpoint_run_name, checkpoint_output_name, checkpoint_name)
    if any(not value or Path(value).name != value for value in components):
        raise ValueError("adapter evaluation requires three single-component paths")
    return str(output_root.joinpath(*components))


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve an evaluation checkpoint from the installed project environment.

    Modal imports its launcher with a Python interpreter that is separate from
    the project's ``uv run`` environment. Keeping this small command-line
    entry point in the package lets the launcher reuse the one arm/checkpoint
    validation implementation without importing ``ast_asr`` directly.
    """
    parser = argparse.ArgumentParser(
        description="Resolve a validated Modal evaluation checkpoint."
    )
    parser.add_argument("command", choices=("resolve-checkpoint",))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-run-name", default="")
    parser.add_argument("--checkpoint-output-name", default="")
    parser.add_argument("--checkpoint-name", default="")
    args = parser.parse_args(argv)
    try:
        checkpoint = evaluation_checkpoint(
            args.arm,
            output_root=Path(args.output_root),
            checkpoint_run_name=args.checkpoint_run_name,
            checkpoint_output_name=args.checkpoint_output_name,
            checkpoint_name=args.checkpoint_name,
        )
    except ValueError as error:
        parser.error(str(error))
    print(checkpoint)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through uv on Modal.
    raise SystemExit(main())
