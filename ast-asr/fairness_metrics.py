"""
fairness_metrics.py
Implements the three fairness metrics from the position paper
(Section 7, "Towards a Fair Evaluation Protocol") plus the
Poisson regression significance test.

Metrics
-------
  delta_dp(df)          → Demographic Parity Gap (Eq. 2)
  delta_eo(df)          → Equal Opportunity Gap (Eq. 3)
  delta_noise(df_clean, df_noisy) → Per-Group Noise Robustness Gap (Eq. 4)

Significance test
-----------------
  poisson_significance(df, group_col) → p-value & whether gap is systematic
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from jiwer import wer as compute_wer

logger = logging.getLogger(__name__)

_THRESHOLD_DP = 0.05   # 5 pp — "requires explanation" per the paper
_ALPHA        = 0.05   # Poisson significance level


# ── WER helpers ───────────────────────────────────────────────────────────────

def _group_wer(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Compute WER per group. Returns a Series indexed by group value."""
    results = {}
    for grp, sub in df.groupby(group_col):
        refs  = [r.lower() for r in sub["reference"].tolist()]
        hyps  = [h.lower() for h in sub["hypothesis"].tolist()]
        # jiwer expects lists of strings; normalize case for fair comparison
        results[grp] = compute_wer(refs, hyps)
    return pd.Series(results, name="wer")


def _group_tpr(df: pd.DataFrame, group_col: str) -> pd.Series:
    """
    Word-level True Positive Rate per group.
    TPR = correctly recognised words / total reference words
        = 1 – WER  (when substitutions+deletions+insertions = errors)
    We compute it explicitly via jiwer's word-level alignment.
    """
    from jiwer import process_words

    results = {}
    for grp, sub in df.groupby(group_col):
        refs = [r.lower() for r in sub["reference"].tolist()]
        hyps = [h.lower() for h in sub["hypothesis"].tolist()]
        out  = process_words(refs, hyps)
        # hits = reference words that are correct (not substituted or deleted)
        total_ref_words = sum(len(r.split()) for r in refs)
        errors = out.substitutions + out.deletions + out.insertions
        hits   = max(0, total_ref_words - (out.substitutions + out.deletions))
        results[grp] = hits / total_ref_words if total_ref_words > 0 else 0.0
    return pd.Series(results, name="tpr")


# ── Metric 1: Demographic Parity Gap ─────────────────────────────────────────

def delta_dp(
    df: pd.DataFrame,
    group_col: str = "language_family",
) -> dict:
    """
    ΔDP = max_{i,j ∈ G} |WER(gᵢ) − WER(gⱼ)|   (Equation 2)

    Returns
    -------
    {
        "delta_dp": float,
        "wer_by_group": pd.Series,
        "worst_pair": (group_i, group_j),
        "flagged": bool,         # True if ΔDP > 5 pp
    }
    """
    wer_by_group = _group_wer(df, group_col)
    groups = wer_by_group.index.tolist()

    max_gap   = 0.0
    worst_pair = (None, None)

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gap = abs(wer_by_group[groups[i]] - wer_by_group[groups[j]])
            if gap > max_gap:
                max_gap    = gap
                worst_pair = (groups[i], groups[j])

    flagged = max_gap > _THRESHOLD_DP

    logger.info(
        "ΔDP = %.4f (%.1f pp) | worst pair: %s vs %s | flagged: %s",
        max_gap, max_gap * 100, worst_pair[0], worst_pair[1], flagged,
    )
    if flagged:
        logger.warning(
            "ΔDP exceeds 5 pp threshold — this disparity requires explanation."
        )

    return {
        "delta_dp":      max_gap,
        "wer_by_group":  wer_by_group,
        "worst_pair":    worst_pair,
        "flagged":       flagged,
    }


# ── Metric 2: Equal Opportunity Gap ──────────────────────────────────────────

def delta_eo(
    df: pd.DataFrame,
    group_col: str = "language_family",
) -> dict:
    """
    ΔEO = max_{i,j ∈ G} |TPR(gᵢ) − TPR(gⱼ)|   (Equation 3)

    Returns
    -------
    {
        "delta_eo": float,
        "tpr_by_group": pd.Series,
        "worst_pair": (group_i, group_j),
    }
    """
    tpr_by_group = _group_tpr(df, group_col)
    groups = tpr_by_group.index.tolist()

    max_gap    = 0.0
    worst_pair = (None, None)

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gap = abs(tpr_by_group[groups[i]] - tpr_by_group[groups[j]])
            if gap > max_gap:
                max_gap    = gap
                worst_pair = (groups[i], groups[j])

    logger.info(
        "ΔEO = %.4f (%.1f pp) | worst pair: %s vs %s",
        max_gap, max_gap * 100, worst_pair[0], worst_pair[1],
    )

    return {
        "delta_eo":      max_gap,
        "tpr_by_group":  tpr_by_group,
        "worst_pair":    worst_pair,
    }


# ── Metric 3: Per-Group Noise Robustness Gap ──────────────────────────────────

def delta_noise(
    df_clean: pd.DataFrame,
    df_noisy: pd.DataFrame,
    group_col: str = "language_family",
) -> dict:
    """
    Δnoise(gᵢ) = WER(gᵢ, noisy) − WER(gᵢ, clean)   (Equation 4)

    Both DataFrames must have identical uid ordering.

    Returns
    -------
    {
        "noise_gap_by_group": pd.Series,    # Δnoise per group
        "wer_clean":          pd.Series,
        "wer_noisy":          pd.Series,
        "max_noise_gap":      float,        # max across groups
        "most_affected_group": str,
    }
    """
    wer_clean = _group_wer(df_clean, group_col)
    wer_noisy = _group_wer(df_noisy, group_col)

    # Align on common groups
    common = wer_clean.index.intersection(wer_noisy.index)
    noise_gap = wer_noisy[common] - wer_clean[common]

    most_affected = noise_gap.idxmax()
    logger.info("Noise robustness gap per group:\n%s", noise_gap.to_string())
    logger.info("Most noise-sensitive group: %s (Δ=%.4f)", most_affected, noise_gap.max())

    return {
        "noise_gap_by_group":  noise_gap,
        "wer_clean":           wer_clean,
        "wer_noisy":           wer_noisy,
        "max_noise_gap":       noise_gap.max(),
        "most_affected_group": most_affected,
    }


# ── Significance Test: Poisson Regression Drop-in-Deviance ───────────────────

def poisson_significance(
    df: pd.DataFrame,
    group_col: str = "language_family",
    alpha: float = _ALPHA,
) -> dict:
    """
    Drop-in-deviance test using Poisson GLM on word error counts.
    Follows Jahan (2025) and Rai et al. (2025).

    Model:
        error_count ~ Poisson(μ)
        log(μ) = β₀ + β_group · group + β_len · utterance_length   (full)
        log(μ) = β₀ + β_len · utterance_length                      (null)

    p-value from χ² test on deviance difference (df = n_groups − 1).

    Returns
    -------
    {
        "p_value":    float,
        "systematic": bool,    # True if p < alpha
        "deviance_diff": float,
        "df_diff": int,
        "group_coefficients": pd.Series,  # log-rate ratios per group
    }
    """
    import statsmodels.formula.api as smf
    from scipy.stats import chi2

    # Build per-utterance error counts
    rows = []
    for _, row in df.iterrows():
        ref_words = len(row["reference"].split())
        if ref_words == 0:
            continue
        # error count = WER × ref_length  (approximate integer errors)
        utt_wer   = compute_wer(row["reference"], row["hypothesis"])
        err_count = max(0, round(utt_wer * ref_words))
        rows.append(
            {
                "error_count":  err_count,
                "utt_length":   ref_words,
                group_col:      row[group_col],
            }
        )

    glm_df = pd.DataFrame(rows)

    # Encode group as categorical
    glm_df[group_col] = glm_df[group_col].astype("category")

    # Null model: length only (offset absorbs exposure)
    null_formula = f"error_count ~ utt_length"
    full_formula = f"error_count ~ utt_length + C({group_col})"

    null_model = smf.glm(null_formula, data=glm_df, family=_poisson_family()).fit()
    full_model = smf.glm(full_formula, data=glm_df, family=_poisson_family()).fit()

    deviance_null = null_model.deviance
    deviance_full = full_model.deviance
    deviance_diff = deviance_null - deviance_full
    df_diff       = null_model.df_resid - full_model.df_resid

    p_value   = 1 - chi2.cdf(deviance_diff, df=df_diff)
    systematic = p_value < alpha

    logger.info(
        "Poisson drop-in-deviance: Δdeviance=%.3f (df=%d) → p=%.4f | systematic=%s",
        deviance_diff, df_diff, p_value, systematic,
    )

    # Extract group log-rate coefficients
    group_coefs = {
        k: v
        for k, v in full_model.params.items()
        if group_col in k
    }

    return {
        "p_value":             p_value,
        "systematic":          systematic,
        "deviance_diff":       deviance_diff,
        "df_diff":             df_diff,
        "group_coefficients":  pd.Series(group_coefs),
    }


def _poisson_family():
    import statsmodels.api as sm
    return sm.families.Poisson()


# ── Summary printer ───────────────────────────────────────────────────────────

def print_fairness_report(
    dp_result:    dict,
    eo_result:    dict,
    poisson_result: dict,
    noise_result: Optional[dict] = None,
    model_name:   str = "",
) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  FAIRNESS REPORT  {model_name}")
    print(sep)

    print("\n[Metric 1] Demographic Parity Gap (ΔDP)")
    for grp, w in dp_result["wer_by_group"].items():
        print(f"    {grp:<20}  WER = {w:.4f} ({w*100:.1f}%)")
    print(f"  → ΔDP = {dp_result['delta_dp']:.4f} ({dp_result['delta_dp']*100:.1f} pp)"
          f"  {'⚠ FLAGGED' if dp_result['flagged'] else '✓ within threshold'}")

    print("\n[Metric 2] Equal Opportunity Gap (ΔEO)")
    for grp, t in eo_result["tpr_by_group"].items():
        print(f"    {grp:<20}  TPR = {t:.4f} ({t*100:.1f}%)")
    print(f"  → ΔEO = {eo_result['delta_eo']:.4f} ({eo_result['delta_eo']*100:.1f} pp)")

    if noise_result:
        print("\n[Metric 3] Per-Group Noise Robustness Gap (Δg_noise)")
        for grp, gap in noise_result["noise_gap_by_group"].items():
            print(f"    {grp:<20}  Δnoise = {gap:+.4f} ({gap*100:+.1f} pp)")
        print(f"  → Most affected: {noise_result['most_affected_group']}")

    print("\n[Significance] Poisson Drop-in-Deviance")
    print(f"  p-value = {poisson_result['p_value']:.4f}"
          f"  {'→ SYSTEMATIC disparity (p < 0.05)' if poisson_result['systematic'] else '→ not statistically significant'}")

    print(f"\n{sep}\n")