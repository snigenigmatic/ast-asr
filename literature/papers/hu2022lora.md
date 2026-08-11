# LoRA: Low-Rank Adaptation of Large Language Models

- **Authors/year:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2022; arXiv 2021).
- **Stable record:** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685); [ICLR/OpenReview paper](https://openreview.net/forum?id=nZeVKeeFYf9).
- **Verification/BibTeX:** arXiv plus accepted ICLR paper record; BibTeX fetched from `https://arxiv.org/bibtex/2106.09685`.
- **Verified claim:** LoRA freezes pretrained weights and injects trainable low-rank update matrices into Transformer layers, reducing the number of trainable parameters for downstream adaptation.
- **Method/results and reported metrics/datasets:** efficient adaptation study on language-model tasks, not a fairness or ASR paper.
- **Relevance:** supports the project's rank-8 adapter experiment as a compute-controlled adaptation mechanism.
- **Copy / do not copy:** copy frozen-base adapters and save adapters/checkpoint metadata for reproducibility. Do **not** present LoRA itself as a fairness intervention or infer that a rank/target-module configuration is optimal without validation.
