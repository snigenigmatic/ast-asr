"""Paired speaker-clustered comparison of two SPIRE evaluations.

Runs inside the project virtualenv (invoked by modal_spire_eval via `uv run`),
because ast_asr is installed there rather than in the container interpreter.

Registered contract: experiments/SPIRE-crosscorpus/protocol.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spire_crosscorpus as contract


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--control-run-name", required=True)
    parser.add_argument("--treatment-run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _load(eval_root: Path, run_name: str) -> tuple[list, dict[str, object]]:
    directory = eval_root / run_name
    predictions = directory / "predictions.jsonl"
    metrics_path = directory / "metrics.json"
    for path in (predictions, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing evaluation artifact: {path}")
    results = [
        contract.result_from_record(json.loads(line))
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not results:
        raise RuntimeError(f"no predictions in {predictions}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    provenance = {
        "run_name": run_name,
        "arm": metrics.get("arm"),
        "adapter_checkpoint": metrics.get("adapter_checkpoint"),
        "adapter_checkpoint_revision": metrics.get("adapter_checkpoint_revision"),
        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        "predictions_sha256_recorded_by_evaluation": metrics.get("predictions_sha256"),
        "manifest_sha256": metrics.get("manifest_sha256"),
        "decoding": metrics.get("decoding"),
    }
    recorded = provenance["predictions_sha256_recorded_by_evaluation"]
    if recorded and recorded != provenance["predictions_sha256"]:
        raise RuntimeError(
            f"predictions for {run_name} changed since evaluation: "
            f"{provenance['predictions_sha256']} != {recorded}"
        )
    return results, provenance


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {args.output}")

    control, control_provenance = _load(args.eval_root, args.control_run_name)
    treatment, treatment_provenance = _load(args.eval_root, args.treatment_run_name)
    if control_provenance["manifest_sha256"] != treatment_provenance["manifest_sha256"]:
        raise RuntimeError("arms were evaluated against different manifests")

    bootstrap = contract.paired_speaker_bootstrap(
        control,
        treatment,
        resamples=args.resamples,
        seed=args.seed,
    )
    control_summary = contract.summarize_arm(control)
    treatment_summary = contract.summarize_arm(treatment)
    report: dict[str, object] = {
        "artifact_kind": "spire_crosscorpus_paired_comparison",
        "corpus": "spire-sies",
        "split": "val",
        "condition": "clean",
        "cluster_unit": "corpus_speaker",
        "endpoint_note": (
            "Worst-family WER here is over two families on clean speech only. It "
            "is NOT the Svarah worst family-by-condition endpoint and must not "
            "be compared with it as the same quantity."
        ),
        "control": control_provenance,
        "treatment": treatment_provenance,
        "control_endpoints": control_summary,
        "treatment_endpoints": treatment_summary,
        "point_deltas": {
            "overall_wer": treatment_summary["overall_wer"]
            - control_summary["overall_wer"],
            "worst_family_wer": treatment_summary["worst_family_wer"]
            - control_summary["worst_family_wer"],
            "worst_20_percent_speaker_wer": treatment_summary[
                "worst_20_percent_speaker_wer"
            ]
            - control_summary["worst_20_percent_speaker_wer"],
            "family_gap": treatment_summary["family_gap"]
            - control_summary["family_gap"],
        },
        "bootstrap": bootstrap,
        "worst_family_interval_includes_harm": contract.interval_includes_harm(
            bootstrap["worst_family_wer_delta"]
        ),
        "overall_interval_includes_harm": contract.interval_includes_harm(
            bootstrap["overall_wer_delta"]
        ),
        "worst_family_label_changed_between_arms": (
            control_summary["worst_family"] != treatment_summary["worst_family"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "point_deltas": report["point_deltas"],
                "bootstrap": bootstrap,
                "worst_family_label_changed_between_arms": report[
                    "worst_family_label_changed_between_arms"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
