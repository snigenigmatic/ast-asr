"""
pipeline.py
Main entry point for the fairness evaluation pipeline.

Usage (from project root):
    python src/pipeline.py --model whisper-tiny --max-samples 200 --snr 0

Full run (all of Svarah, all models):
    python src/pipeline.py --model whisper-small --output outputs/results.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from data_loader      import load_svarah
from asr_inference    import run_inference
from noise_augment    import add_noise
from fairness_metrics import (
    delta_dp, delta_eo, delta_noise,
    poisson_significance, print_fairness_report,
)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    model_name:  str   = "whisper-tiny",
    max_samples: int   = None,
    snr_db:      float = 0.0,
    noise_type:  str   = "white",
    group_col:   str   = "language_family",
    output_csv:  str   = None,
    cache_dir:   str   = "cache",
    model_path:  str   = None,
) -> dict:
    """
    Full evaluation pipeline:
      1. Load Svarah
      2. Run ASR inference (clean)
      3. Run ASR inference (noisy, for Δg_noise)
      4. Compute ΔDP, ΔEO, Δg_noise
      5. Run Poisson significance test
      6. Save results
    """

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1 — Loading Svarah")
    logger.info("=" * 55)
    df = load_svarah(max_samples=max_samples, cache_dir=cache_dir)

    # ── 2. Inference on clean audio ───────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 2 — ASR inference (clean) with %s", model_name)
    logger.info("=" * 55)
    df_clean = run_inference(df, model_name=model_name, model_path=model_path)

    # ── 3. Inference on noisy audio ───────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 3 — Adding %s noise at %d dB SNR", noise_type, snr_db)
    logger.info("=" * 55)
    df_noisy_audio = add_noise(df, snr_db=snr_db, noise_type=noise_type)
    df_noisy = run_inference(df_noisy_audio, model_name=model_name, model_path=model_path)

    # ── 4. Fairness metrics ───────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 4 — Computing fairness metrics")
    logger.info("=" * 55)

    dp_result      = delta_dp(df_clean, group_col=group_col)
    eo_result      = delta_eo(df_clean, group_col=group_col)
    noise_result   = delta_noise(df_clean, df_noisy, group_col=group_col)
    poisson_result = poisson_significance(df_clean, group_col=group_col)

    print_fairness_report(
        dp_result, eo_result, poisson_result,
        noise_result=noise_result,
        model_name=model_name,
    )

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    results_dir = Path(output_csv).parent if output_csv else Path("outputs")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Per-utterance results (clean)
    utt_path = output_csv or str(results_dir / f"results_{model_name}_clean.csv")
    df_clean[["uid", "accent", "language_family", "gender", "age",
              "reference", "hypothesis", "model"]].to_csv(utt_path, index=False)
    logger.info("Per-utterance results saved → %s", utt_path)

    # Summary metrics
    summary = {
        "model":           model_name,
        "n_utterances":    len(df_clean),
        "overall_wer":     _overall_wer(df_clean),
        "delta_dp":        dp_result["delta_dp"],
        "delta_eo":        eo_result["delta_eo"],
        "max_noise_gap":   noise_result["max_noise_gap"],
        "poisson_p":       poisson_result["p_value"],
        "systematic_gap":  poisson_result["systematic"],
        "worst_dp_pair":   " vs ".join(str(x) for x in dp_result["worst_pair"]),
    }
    # Add per-group WERs
    for grp, w in dp_result["wer_by_group"].items():
        summary[f"wer_{grp.replace('-', '_').lower()}"] = w

    summary_path = str(results_dir / f"summary_{model_name}.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    logger.info("Summary saved → %s", summary_path)

    return {
        "df_clean":    df_clean,
        "df_noisy":    df_noisy,
        "dp":          dp_result,
        "eo":          eo_result,
        "noise":       noise_result,
        "poisson":     poisson_result,
        "summary":     summary,
    }


def _overall_wer(df: pd.DataFrame) -> float:
    from jiwer import wer
    return wer([r.lower() for r in df["reference"]], [h.lower() for h in df["hypothesis"]])


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fairness evaluation pipeline for Indian English ASR"
    )
    parser.add_argument(
        "--model", default="whisper-tiny",
        choices=["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium",
                 "wav2vec2-base", "wav2vec2-large", "hubert-large", "hybrid-w2v2-grl"],
        help="Model to evaluate",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Override checkpoint directory (used for hybrid-w2v2-grl variants)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap utterances for quick runs (None = full dataset)",
    )
    parser.add_argument(
        "--snr", type=float, default=10.0,
        help="SNR in dB for noise robustness evaluation (default: 10 dB)",
    )
    parser.add_argument(
        "--noise-type", default="white",
        choices=["white", "pink", "babble"],
        help="Noise type for Δg_noise",
    )
    parser.add_argument(
        "--group", default="language_family",
        choices=["language_family", "accent", "gender"],
        help="Protected attribute to disaggregate on",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path for per-utterance CSV output",
    )
    parser.add_argument(
        "--cache-dir", default="cache",
        help="HuggingFace dataset cache directory",
    )
    args = parser.parse_args()

    run_pipeline(
        model_name  = args.model,
        max_samples = args.max_samples,
        snr_db      = args.snr,
        noise_type  = args.noise_type,
        group_col   = args.group,
        output_csv  = args.output,
        cache_dir   = args.cache_dir,
        model_path  = args.model_path,
    )


if __name__ == "__main__":
    main()