"""Command-line interface for the reproducible FR-CISPO experiment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def _prepare_data(args: argparse.Namespace) -> None:
    from .artifacts import write_immutable_json
    from .data import prepare_svarah_archive
    from .folds import build_outer_fold_manifests
    from .taxonomy import SVARAH_LANGUAGE_FAMILIES

    prepared = prepare_svarah_archive(
        args.archive_root,
        dataset_revision=args.dataset_revision,
        family_mapping=SVARAH_LANGUAGE_FAMILIES,
        expected_speakers=args.expected_speakers,
        speaker_key_mode=args.speaker_key_mode,
    )
    output = args.output_dir
    write_immutable_json(output / "dataset_manifest.json", prepared.to_dict())
    folds = build_outer_fold_manifests(
        prepared.speaker_profiles,
        seed=args.fold_seed,
        expected_speakers=args.expected_speakers,
    )
    for fold in folds:
        write_immutable_json(output / "folds" / f"fold-{fold.fold}.json", fold.to_dict())


def _train_sft(args: argparse.Namespace) -> None:
    from .sft import train_sft

    train_sft(args)


def _train_policy(args: argparse.Namespace) -> None:
    from .policy_training import train_policy

    train_policy(args)


def _evaluate_fold(args: argparse.Namespace) -> None:
    from .evaluation import evaluate_fold

    evaluate_fold(args)


def _aggregate_oof(args: argparse.Namespace) -> None:
    from .analysis import aggregate_oof

    aggregate_oof(args)


def _diagnose_invariance(args: argparse.Namespace) -> None:
    from .invariance_diagnostics import diagnose_batch_invariance

    diagnose_batch_invariance(args)


def _measure_sentinel_kl(args: argparse.Namespace) -> None:
    from .h7_runner import run_h7_cuda

    run_h7_cuda(
        config_path=args.config,
        bank_root=args.bank_root,
        archive_root=args.archive_root,
        policy_checkpoint=args.policy_checkpoint,
        reference_checkpoint=args.reference_checkpoint,
        output_dir=args.output_dir,
        input_lock=args.input_lock,
        expected_policy_revision=args.expected_policy_revision,
        expected_reference_revision=args.expected_reference_revision,
        expected_cycle27_model_revision=args.expected_cycle27_model_revision,
        expected_config_sha256=args.expected_config_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ast-asr",
        description="FR-CISPO fair and robust post-training for Whisper-tiny",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-data", help="build immutable Svarah folds")
    prepare.add_argument("--archive-root", type=Path, required=True)
    prepare.add_argument("--dataset-revision", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--fold-seed", type=int, default=2026)
    prepare.add_argument("--expected-speakers", type=int, default=117)
    prepare.add_argument(
        "--speaker-key-mode",
        choices=("authoritative", "demographic_profile"),
        default="authoritative",
        help="opt-in development fallback when official speaker IDs are unavailable",
    )
    prepare.set_defaults(handler=_prepare_data)

    sft = commands.add_parser("train-sft", help="train one fold-specific LoRA SFT adapter")
    sft.add_argument("--config", type=Path, required=True)
    sft.add_argument("--fold", type=int, choices=range(5), required=True)
    sft.add_argument("--seed", type=int, required=True)
    sft.add_argument("--output-dir", type=Path, required=True)
    sft.add_argument("--development-gate", type=Path)
    sft.add_argument("--maximum-epochs", type=int)
    sft.add_argument("--max-train-examples", type=int)
    sft.add_argument("--max-validation-examples", type=int)
    sft.add_argument("--max-optimizer-steps", type=int)
    sft.add_argument("--maximum-new-tokens", type=int)
    sft.set_defaults(handler=_train_sft)

    policy = commands.add_parser("train-policy", help="run one post-training ladder arm")
    policy.add_argument(
        "--config",
        type=Path,
        required=True,
        help="pinned policy settings, including reference_kl_beta",
    )
    policy.add_argument("--fold", type=int, choices=range(5), required=True)
    policy.add_argument("--seed", type=int, required=True)
    policy.add_argument("--arm", required=True)
    policy.add_argument("--sft-checkpoint", type=Path, required=True)
    policy.add_argument("--learning-rate", type=float, required=True)
    policy.add_argument("--output-dir", type=Path, required=True)
    policy.add_argument("--development-gate", type=Path)
    policy.add_argument("--rollout-cycles", type=int)
    policy.add_argument("--probe-examples", type=int)
    policy.add_argument("--maximum-new-tokens", type=int)
    policy.set_defaults(handler=_train_policy)

    evaluate = commands.add_parser("evaluate-fold", help="evaluate clean and noisy fold conditions")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--fold", type=int, choices=range(5), required=True)
    evaluate.add_argument("--arm", required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.set_defaults(handler=_evaluate_fold)

    aggregate = commands.add_parser("aggregate-oof", help="aggregate five-fold predictions")
    aggregate.add_argument("--predictions", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10_000)
    aggregate.add_argument("--seed", type=int, default=2026)
    aggregate.set_defaults(handler=_aggregate_oof)

    invariance = commands.add_parser(
        "diagnose-invariance",
        help="compare solo and batched decoding for a saved adapter",
    )
    invariance.add_argument("--config", type=Path, required=True)
    invariance.add_argument("--checkpoint", type=Path, required=True)
    invariance.add_argument("--fold", type=int, choices=range(5), required=True)
    invariance.add_argument("--output", type=Path, required=True)
    invariance.add_argument("--probe-examples", type=int, default=8)
    invariance.add_argument("--batch-size", type=int, default=8)
    invariance.set_defaults(handler=_diagnose_invariance)

    h7 = commands.add_parser(
        "measure-sentinel-kl",
        help="H7 fixed-policy score-only measurement; requires a separately authorized CUDA launch",
    )
    h7.add_argument("--config", type=Path, required=True)
    h7.add_argument("--bank-root", type=Path, required=True)
    h7.add_argument("--archive-root", type=Path, required=True)
    h7.add_argument("--policy-checkpoint", type=Path, required=True)
    h7.add_argument("--reference-checkpoint", type=Path, required=True)
    h7.add_argument("--output-dir", type=Path, required=True)
    h7.add_argument("--input-lock", type=Path, required=True)
    h7.add_argument("--expected-policy-revision", required=True)
    h7.add_argument("--expected-reference-revision", required=True)
    h7.add_argument("--expected-cycle27-model-revision", required=True)
    h7.add_argument("--expected-config-sha256", required=True)
    h7.set_defaults(handler=_measure_sentinel_kl)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)
