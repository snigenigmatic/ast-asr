"""Validation for Modal evaluation-arm labels and checkpoint coordinates."""

from __future__ import annotations

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
