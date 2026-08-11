# Prior-results benchmark audit

**Purpose.** This ledger separates a number in a prior review slide from a number that is an appropriate target for a new Whisper-tiny, speaker-disjoint experiment.

**Audit date:** 2026-08-11
**Scope:** read-only inspection of `C:\Kaustubh\ast-asr`; no historical artifact was changed.

## Sources and provenance

| Source | What it contains | Use in this audit |
| --- | --- | --- |
| `PW25_BJD_05_Phase2_Review2.html`, slides "Result" and "What's still broken" (HTML lines 709-821) | Main historical Whisper-small GRPO claim, group WERs, and white-noise results | Primary presentation claim |
| `outputs/whisper_rl_results/summary_whisper-small.csv` | Historical zero-shot metric summary, 1,998 utterances | Numeric export matching the slides |
| `outputs/whisper_rl_results/summary_whisper-small-rl.csv` and `whisper-rl-fair_eval.csv` | Historical post-training metric summary and per-utterance predictions | Numeric export matching the slides |
| `outputs/whisper_rl_results/whisper-rl-fair_summary.txt` | Evaluation log, family counts, white-noise 10 dB evaluation | Records condition details and inference warnings |
| `outputs/summary_ft-w2v2.csv`, `summary_hybrid-lam*.csv` | Four 500-utterance Wav2Vec2/GRL summaries | Supports the negative GRL ablation claim |
| `PW25_BJD_05_Phase_2-ISA-1-2.pdf`, pp. 3, 7-18, 32 | Earlier Phase-2 review: motivation, cited literature results, and planned evaluation | Context only; relevant pages were rendered and visually checked |
| Historical `docs/mu_degeneracy_and_paper_claim.md`, pp. 51-83 and 105-111 | Later mathematical audit: one-update clipping/ratio machinery is inert, while dropout can forge apparent ratio movement | Invalidates causal attribution to historical ratio/clipping machinery |
| Current-worktree `docs/development-evidence-20260810.md` and `docs/speaker-identity-audit.md` | Corrected inference audit and public-Svarah speaker-ID audit | Explains why historical values are not publication-valid comparators |

The PDF is not the source of the historical 18.4-to-16.4 result. It predates that result: it describes a proposed multi-strategy project and reports literature or non-ASR benchmark numbers. The HTML is the presentation artifact that makes the project-result claims.

## Metric definitions used by the presentation

| Name | Presentation definition | Interpretation risk |
| --- | --- | --- |
| Overall WER | Word errors divided by reference words | Comparable only with identical text normalization, split, decoding, and model size. |
| `Delta_DP` | `max_i,j |WER(g_i)-WER(g_j)|` across language families | A worst-family WER spread, not demographic parity in the usual classification sense. Smaller is better. |
| `Delta_EO` | Maximum difference in word-level TPR across families | Descriptive secondary metric; no historical target was presented. |
| `Delta_noise(g)` | `WER(g, noisy)-WER(g, clean)` | Metric slide says **0 dB SNR** and SPIRE-SIES naturalistic noise; result slide says **white noise at 10 dB** over Svarah. Not the same endpoint. |
| Poisson p-value | Drop-in-deviance test on word-error counts with utterance length covariate | A p-value is not a fairness score. `p >= 0.05` does not prove equal performance, especially with only 121 Sino-Tibetan utterances. |

## Historical project results

### A. Wav2Vec2 + GRL negative ablation

All four summaries use a 500-utterance Svarah evaluation. The HTML reports that WER remained in a 41-42% band and that Sino-Tibetan was the worst affected family. The raw exports agree numerically.

| Arm | Lambda | Overall WER | Delta_DP | Delta_EO | max noise gap | Poisson p | Status in slides |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ft-w2v2` LoRA, no GRL | 0 | 41.34% | 8.96 pp | 10.77 pp | 53.54 pp | 0.00784 | baseline |
| Hybrid GRL | 0.05 | 42.20% | 10.65 pp | 11.58 pp | 54.23 pp | 0.00929 | worse than baseline |
| Hybrid GRL | 0.10 | 41.35% | 12.46 pp | 13.81 pp | 54.08 pp | 0.00552 | worst Delta_DP |
| Hybrid GRL | 0.30 | 41.65% | 9.69 pp | 12.12 pp | 52.69 pp | 0.00612 | worse than baseline |

**Supported conclusion:** on this 500-utterance export, tested GRL arms did not reduce the WER-spread metric relative to the LoRA-only arm.
**Not established:** a general causal claim about adversarial debiasing. The slides attribute failure to zero Sino-Tibetan coverage in SPIRE-SIES, but this audit did not find a pinned data manifest/seed bundle demonstrating that claim end-to-end.

### B. Historical Whisper-small GRPO/CISPO presentation result

| Metric | `openai/whisper-small` zero-shot | `whisper-small-rl`, reported as 1,500 GRPO steps | Arithmetic change | Provenance |
| --- | ---: | ---: | ---: | --- |
| Evaluation utterances | 1,998 | 1,998 | -- | HTML "Result" slide; both CSV exports |
| Sino-Tibetan utterances | 121 | 121 | -- | HTML result note; evaluation log |
| Overall WER | 18.359% | 16.386% | **-1.973 pp** | HTML lines 718-754; summary CSVs |
| Dravidian WER | 17.024% | 16.163% | -0.862 pp | HTML result note; summary CSVs |
| Indo-Aryan WER | 18.848% | 16.271% | -2.577 pp | HTML result note; summary CSVs |
| Sino-Tibetan WER | 19.056% | 18.803% | -0.253 pp | HTML result note; summary CSVs |
| Delta_DP | 2.031 pp | 2.640 pp | **+0.609 pp** (worse) | HTML lines 724-749; summary CSVs |
| Delta_EO | 2.483 pp | 3.391 pp | +0.907 pp (worse) | summary CSVs; omitted from result slide |
| Poisson p-value | 0.0200 | 0.0780 | crosses stated 0.05 convention | HTML lines 729-753; summary CSVs |
| White-noise 10 dB Delta_noise, Dravidian | not presented | +6.560 pp | -- | HTML lines 774-789; post-training log |
| White-noise 10 dB Delta_noise, Indo-Aryan | not presented | +15.318 pp | -- | same |
| White-noise 10 dB Delta_noise, Sino-Tibetan | not presented | +16.863 pp | -- | same |

The HTML specifies `alpha_fair = 2`, batch `3 x 4` gradient accumulation, learning rate `2e-5`, bf16/CUDA, and 1,500 steps (lines 700-705 and 736-754). It describes a family-balanced sampler and a composite WER/CER plus fairness-scaled reward (lines 590-704).

## What the historical evidence does and does not support

### Numerically supported, but legacy

The CSV summaries and saved prediction exports support that the historical code produced the numbers above under its then-current preprocessing and decoding. The post-training evaluation log records 1,998 utterances (1,337 Indo-Aryan, 540 Dravidian, 121 Sino-Tibetan) and white noise at 10 dB.

### Not a valid publication or method-comparison result

1. **Wrong model class for the current target.** The headline uses `whisper-small`; FR-CISPO is restricted to `whisper-tiny`. A smaller WER by a smaller model on a different split is not an apples-to-apples beat.
2. **No verifiable speaker-disjoint split.** The historical report calls the 70/30 split speaker-stratified, but the current data audit finds no utterance-to-speaker IDs in public Parquet releases. That claim cannot be independently checked or used for speaker-clustered uncertainty.
3. **Inference warning.** The historical Whisper evaluation log says the attention mask was not set and could not be inferred because pad and EOS tokens are the same. It cannot certify correctly masked greedy inference.
4. **Objective attribution is invalid.** The later theory audit proves that at `mu = 1` with dropout disabled, PPO/CISPO clipping is inert; with dropout, ratios can appear to move without a policy update. The result cannot establish that the ratio or clipping machinery caused the WER change.
5. **Fairness interpretation overreaches.** The reported WER spread rises from 2.03 to 2.64 pp. The p-value changing from 0.020 to 0.078 is an uncertainty statement, not evidence that disparity was eliminated.
6. **Noise endpoint drift.** The metric-definition slide declares 0 dB/SPIRE-SIES, while the result uses white 10 dB/Svarah. Its Delta_noise values must not be compared to an unseen-noise or 0 dB result without relabeling.

## Explicit targets to beat

There are two targets: a **legacy replication target** that applies only after reproducing the exact historical model/split/decoder, and the **publication-valid FR-CISPO target**. Do not collapse them into one leaderboard.

| Track | Endpoint | Historical number | Minimum “beat” threshold | Comparability requirement |
| --- | --- | ---: | ---: | --- |
| Legacy replication | Overall WER, Whisper-small, historical 1,998 utterances | 16.386% | **< 16.386%** | Same Whisper-small revision, exact 1,998 IDs, normalization, decoding, and white-noise generator. Not a Whisper-tiny target. |
| Legacy replication | Dravidian / Indo-Aryan / ST clean WER | 16.163 / 16.271 / 18.803% | lower in all three groups | Same groups and historical split; report paired deltas, not rounded values. |
| Legacy replication | Worst clean-family WER | 18.803% (ST) | **< 18.803%** | Same caveat. More meaningful than the p-value. |
| Legacy replication | White 10 dB ST noise amplification | +16.863 pp | **< +16.863 pp** | Same white-noise seed/SNR and clean prediction set; not equivalent to MUSAN or 0 dB. |
| Legacy replication | Delta_DP | 2.640 pp after RL; 2.031 pp before RL | **< 2.031 pp** | Use this explicit lower-is-better target; do not call `p >= 0.05` a win. |
| FR-CISPO confirmatory | Worst family x condition WER | no valid historical comparable number | **at least 2.00 absolute pp lower than matched Tiny SFT**, with clean overall WER no more than **+1.00 pp** worse | Exact frozen objective, authoritative 117-speaker folds, concatenated OOF predictions, clean/white-10dB/MUSAN-babble-10dB conditions. |
| FR-CISPO safety | KL and ratio movement | historical evidence insufficient | KL/token **< 0.1**, ratio p99 **< 2**, finite steps, checkpoint and solo/batch reproduction | Required before any WER claim. |

The final two rows are the scientifically meaningful go/no-go targets. They do **not** promise lower raw WER than Whisper-small; they demand a credible improvement from the same Whisper-tiny/SFT reference on the intended fairness-and-robustness endpoint.

## Results from the Phase-2 PDF: related-work context, not benchmarks

| PDF page | Reported result | Why it cannot be a target for FR-CISPO |
| ---: | --- | --- |
| 3 | 20-40% accuracy drop for Indian/non-Western English speakers | Unattributed literature-range claim; no shared data or WER protocol. |
| 8 | Jahan dissertation: seven ASR variants; “competitive WER” and disparity reduction | Exact ASR WER deltas are not on the slide. |
| 10 | SSAST: AudioSet-20K mAP 0.148 to 0.310; speaker ID 30.1% to 64.2% | Classification/speaker identification, not ASR WER. |
| 12 | CPE: AudioSet mAP 0.313 to 0.343; ESC-50 accuracy 87.5% to 91.4% | Audio classification, not ASR. |
| 14 | AST: AudioSet mAP 0.485; ESC-50 95.6%; Speech Commands 98.1% | Classification/command recognition, not Indian-English transcription. |
| 15 | Speech-emotion testing: 11 models, 2,029 tests; more than half of background-noise tests fail for most models | SER behavioral testing, not ASR WER. |
| 18 | BiLSTM-Transformer: 95.6% accuracy and about 180 ms latency on VCTK | Curated VCTK speaking-assessment result; no Indian-English subgroup WER. |
| 32 | Expected output: subgroup-aware Indian-English/noise evaluation | Aspiration, not a measured result. |

## Operational conclusion

The 16.386% historical value is useful as a **reproduction audit fixture**, not as the main score to chase. The next credible result needs authoritative speaker IDs, a frozen Tiny baseline, correct masks/FP32-invariant inference, a live-ratio objective with movement diagnostics, and out-of-fold predictions. Only then should FR-CISPO claim to beat a baseline or prior method.
