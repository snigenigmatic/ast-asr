# Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization

- **Authors/year:** Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang (2020; arXiv 2019).
- **Stable record:** [arXiv:1911.08731](https://arxiv.org/abs/1911.08731); [ICLR/OpenReview record](https://openreview.net/forum?id=ryxGuJrFvS).
- **Verification/BibTeX:** arXiv plus accepted ICLR record; BibTeX fetched from `https://arxiv.org/bibtex/1911.08731`.
- **Verified claim:** group DRO minimizes worst-case training loss over predefined groups, but naive group DRO can fail in overparameterized models. Stronger regularization or early stopping materially improves worst-group generalization; reported gains are 10–40 percentage points on the paper's NLI and vision tasks.
- **Method/results and reported metrics/datasets:** non-ASR group-shift work, centered on worst-group generalization rather than average accuracy, using NLI and vision tasks.
- **Relevance:** motivates the FR-CISPO worst family × condition endpoint and an explicit KL/early-stop safety gate.
- **Copy / do not copy:** copy predefined groups, worst-group reporting, and regularization/early stopping. Do **not** assume exponentiated dual weights automatically improve test fairness or replace a validation gate.
