"""Pure helpers for immutable, run-specific Modal policy configurations.

The Modal launcher itself deliberately does not import the project package: it is
loaded before the remote image has installed the package.  Keeping this logic
here makes the safety contract testable without importing Modal or any remote
resource.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .artifacts import write_immutable_json


def safe_artifact_component(value: str, *, field: str) -> str:
    """Return one safe artifact path component or raise a clear error."""
    candidate = str(value)
    if (
        not candidate
        or candidate in {".", ".."}
        or "/" in candidate
        or "\\" in candidate
        or Path(candidate).name != candidate
    ):
        raise ValueError(f"{field} must be a non-empty single path component")
    return candidate


def validated_reference_kl_beta(value: float) -> float:
    """Validate the scalar before creating any run artifact."""
    beta = float(value)
    if not math.isfinite(beta):
        raise ValueError("reference_kl_beta must be finite")
    if beta < 0:
        raise ValueError("reference_kl_beta must be nonnegative")
    return beta


def write_run_specific_policy_config(
    *,
    immutable_config: Path,
    run_artifact_root: Path,
    output_name: str,
    reference_kl_beta: float,
) -> Path:
    """Copy a pinned volume configuration and set the one H5 run parameter.

    The derived configuration lives under the output volume's *run* directory,
    rather than beside the shared prepared-data config.  ``write_immutable_json``
    makes retrying the same command idempotent and refuses a conflicting reuse
    of an output name.
    """
    output_component = safe_artifact_component(output_name, field="output_name")
    beta = validated_reference_kl_beta(reference_kl_beta)
    raw: dict[str, Any] = json.loads(immutable_config.read_text(encoding="utf-8"))
    policy = raw.get("policy")
    if not isinstance(policy, dict):
        raise TypeError("immutable configuration has no policy object")
    policy["reference_kl_beta"] = beta

    destination = (
        run_artifact_root / "resolved-policy-configs" / f"{output_component}.json"
    )
    write_immutable_json(destination, raw)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    """Create one validated run config inside the installed project runtime."""
    parser = argparse.ArgumentParser(
        description="Derive an immutable Modal policy config for one run."
    )
    parser.add_argument("command", choices=("derive",))
    parser.add_argument("--immutable-config", type=Path, required=True)
    parser.add_argument("--run-artifact-root", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--reference-kl-beta", type=float, required=True)
    args = parser.parse_args(argv)
    destination = write_run_specific_policy_config(
        immutable_config=args.immutable_config,
        run_artifact_root=args.run_artifact_root,
        output_name=args.output_name,
        reference_kl_beta=args.reference_kl_beta,
    )
    print(destination)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through Modal uv run.
    raise SystemExit(main())
