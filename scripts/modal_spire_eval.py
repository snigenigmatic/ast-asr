"""Cross-corpus SPIRE-SIES evaluation on Modal.

Evaluation only: no checkpoint is ever trained on this corpus.

Registered contract: experiments/SPIRE-crosscorpus/protocol.md

The image definition matches scripts/modal_fr_cispo.py exactly, so Modal reuses
the already-built project image instead of rebuilding CUDA and torch.

Zero-shot baseline:

    $env:PYTHONIOENCODING='utf-8'
    uvx modal run --detach scripts/modal_spire_eval.py::evaluate_spire \
        --run-name spire-eval-zeroshot-20260820 --arm zero-shot

An adapter checkpoint produced by the FR-CISPO runs:

    uvx modal run --detach scripts/modal_spire_eval.py::evaluate_spire \
        --run-name spire-eval-h5-beta004-20260820 --arm h5-beta004 \
        --checkpoint-run-name profile-h5-refkl-beta004-s2026-20260812 \
        --checkpoint-output-name h5-beta004-fr-cispo \
        --checkpoint-name checkpoint-last-safe
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

app = modal.App("ast-asr-spire-eval")

cache_volume = modal.Volume.from_name("ast-asr-cache", create_if_missing=False)
spire_volume = modal.Volume.from_name("ast-asr-spire", create_if_missing=False)
runs_volume = modal.Volume.from_name("ast-asr-fr-cispo-runs", create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])

PROJECT_ROOT = "/root/fr-cispo"
CACHE_DIR = "/cache"
SPIRE_DIR = "/spire"
RUNS_DIR = "/artifacts"

CACHE_ENV = {
    "HF_HOME": f"{CACHE_DIR}/hf",
    "HF_HUB_CACHE": f"{CACHE_DIR}/hf/hub",
    "HUGGINGFACE_HUB_CACHE": f"{CACHE_DIR}/hf/hub",
    "HF_DATASETS_CACHE": f"{CACHE_DIR}/ai4bharat___svarah",
    "TRANSFORMERS_CACHE": f"{CACHE_DIR}/hf/hub",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "UV_LINK_MODE": "copy",
}

# Identical to modal_fr_cispo.image so the built layers are shared, then the
# SPIRE drivers are appended as a final thin layer.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("uv>=0.8,<0.9")
    .env(CACHE_ENV)
    .add_local_file(
        "pyproject.toml",
        remote_path=f"{PROJECT_ROOT}/pyproject.toml",
        copy=True,
    )
    .add_local_file("uv.lock", remote_path=f"{PROJECT_ROOT}/uv.lock", copy=True)
    .run_commands(
        f"cd {PROJECT_ROOT} && uv sync --frozen --no-dev --no-install-project"
    )
    .add_local_dir("src", remote_path=f"{PROJECT_ROOT}/src", copy=True)
    .add_local_dir("configs", remote_path=f"{PROJECT_ROOT}/configs", copy=True)
    .run_commands(
        f"cd {PROJECT_ROOT} && uv sync --frozen --offline --no-build-isolation"
    )
    .add_local_file(
        "scripts/spire_crosscorpus.py",
        remote_path=f"{PROJECT_ROOT}/scripts/spire_crosscorpus.py",
        copy=True,
    )
    .add_local_file(
        "scripts/spire_eval_entry.py",
        remote_path=f"{PROJECT_ROOT}/scripts/spire_eval_entry.py",
        copy=True,
    )
)

VOLUMES = {
    CACHE_DIR: cache_volume,
    SPIRE_DIR: spire_volume,
    RUNS_DIR: runs_volume,
}


def _resolve_checkpoint(
    checkpoint_run_name: str,
    checkpoint_output_name: str,
    checkpoint_name: str,
) -> Path | None:
    """Resolve an adapter checkpoint, or None for the zero-shot base model."""
    provided = [checkpoint_run_name, checkpoint_output_name, checkpoint_name]
    if not any(provided):
        return None
    if not all(provided):
        raise ValueError(
            "supply all three of checkpoint run name, output name, and "
            "checkpoint name, or none of them for zero-shot"
        )
    checkpoint = (
        Path(RUNS_DIR) / checkpoint_run_name / checkpoint_output_name / checkpoint_name
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"adapter checkpoint is missing: {checkpoint}")
    return checkpoint


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=32768,
    timeout=4 * 60 * 60,
    retries=0,
    volumes=VOLUMES,
    secrets=[hf_secret],
)
def evaluate_spire(
    run_name: str,
    arm: str,
    checkpoint_run_name: str = "",
    checkpoint_output_name: str = "",
    checkpoint_name: str = "",
    batch_size: int = 0,
    limit: int = 0,
) -> dict[str, object]:
    """Greedy FP32 evaluation of one checkpoint on the SPIRE validation split."""
    manifest = Path(SPIRE_DIR) / "val" / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"missing {manifest}; run prepare_spire_val before evaluating"
        )
    checkpoint = _resolve_checkpoint(
        checkpoint_run_name,
        checkpoint_output_name,
        checkpoint_name,
    )
    output = Path(SPIRE_DIR) / "eval" / run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite evaluation run: {output}")

    command = [
        "uv",
        "run",
        "--frozen",
        "python",
        "scripts/spire_eval_entry.py",
        "--config",
        f"{PROJECT_ROOT}/configs/fr_cispo_tiny.json",
        "--manifest",
        str(manifest),
        "--audio-root",
        str(Path(SPIRE_DIR) / "val"),
        "--output-dir",
        str(output),
        "--arm",
        arm,
    ]
    if checkpoint is not None:
        command += ["--adapter-checkpoint", str(checkpoint)]
    if batch_size:
        command += ["--batch-size", str(batch_size)]
    if limit:
        command += ["--limit", str(limit)]

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    spire_volume.commit()
    return metrics


@app.function(
    image=image,
    cpu=4.0,
    memory=16384,
    timeout=60 * 60,
    retries=0,
    volumes=VOLUMES,
)
def compare_spire_arms(
    control_run_name: str,
    treatment_run_name: str,
    resamples: int = 10_000,
    seed: int = 2026,
) -> dict[str, object]:
    """Paired speaker-clustered bootstrap between two completed evaluations."""
    import sys

    sys.path.insert(0, f"{PROJECT_ROOT}/scripts")
    import spire_crosscorpus as contract

    from ast_asr.metrics import EditCounts

    def _load(run_name: str) -> list:
        path = Path(SPIRE_DIR) / "eval" / run_name / "predictions.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing predictions: {path}")
        results = []
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            results.append(
                contract.UtteranceResult(
                    uid=record["uid"],
                    speaker_id=record["speaker_id"],
                    family=record["family"],
                    gender=record["gender"],
                    counts=EditCounts(
                        substitutions=record["substitutions"],
                        deletions=record["deletions"],
                        insertions=record["insertions"],
                        reference_words=record["reference_words"],
                    ),
                )
            )
        return results

    control = _load(control_run_name)
    treatment = _load(treatment_run_name)
    report = {
        "artifact_kind": "spire_crosscorpus_paired_comparison",
        "control_run": control_run_name,
        "treatment_run": treatment_run_name,
        "control_endpoints": contract.summarize_arm(control),
        "treatment_endpoints": contract.summarize_arm(treatment),
        "bootstrap": contract.paired_speaker_bootstrap(
            control,
            treatment,
            resamples=resamples,
            seed=seed,
        ),
    }
    report["worst_family_interval_includes_harm"] = contract.interval_includes_harm(
        report["bootstrap"]["worst_family_wer_delta"]
    )
    destination = (
        Path(SPIRE_DIR)
        / "eval"
        / f"compare-{control_run_name}-vs-{treatment_run_name}.json"
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {destination}")
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    spire_volume.commit()
    return report
