# Model-Based Approach for Measuring the Fairness in ASR

- **Authors/year:** Zhe Liu, Irina-Elena Veliche, Fuchun Peng (2022; arXiv 2021).
- **Stable record:** [arXiv:2109.09061](https://arxiv.org/abs/2109.09061); [DOI:10.1109/ICASSP43922.2022.9747654](https://doi.org/10.1109/ICASSP43922.2022.9747654).
- **Verification/BibTeX:** arXiv plus IEEE ICASSP proceedings record; BibTeX fetched from DOI content negotiation.
- **Verified claim:** proposes mixed-effects Poisson regression for ASR WER disparities, explicitly addressing nuisance factors and unobserved speaker heterogeneity. It demonstrates the approach on synthetic and real-world speech data.
- **Method/results and reported metrics/datasets:** measurement methodology, not an adaptation algorithm; demonstrates on synthetic and real speech data and warns that uncontrolled subgroup analyses can yield misleading conclusions.
- **Relevance:** supports speaker-clustered uncertainty, preserved covariates, and cautious interpretation of family/accent results.
- **Copy / do not copy:** copy speaker-aware statistical analysis as a supplementary audit. Do **not** claim a family weight or non-significant p-value proves fairness.
